"""Durable human-approval records for tool calls -- the single source of
truth for "may this step run?" when a spec's tool policy requires approval
(see app/factory/tool_policy.py).

Why durable, in the shared SQL database, rather than in memory like the
older `app/runtime/hil.py` store: the run that pauses and the person who
decides are often in different processes -- an agent pauses inside `adk web`
while an approver clicks Approve in the UI served by the FastAPI app. Both
read and write the same rows here, and a pending approval survives either
process restarting.

Lifecycle: pending -> approved | denied | expired, and an approved record
becomes `consumed` the moment the step it approved actually runs. Approval
is single-use: it covers one execution of one step with exactly those
arguments (matched by `fingerprint`), not the tool forever.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

PENDING, APPROVED, DENIED, EXPIRED, CONSUMED = "pending", "approved", "denied", "expired", "consumed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    # SQLite hands timestamps back naive; treat them as the UTC they were written as.
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class ApprovalRow(Base):
    __tablename__ = "tool_approvals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, index=True)
    requested_by: Mapped[str] = mapped_column(String, default="")
    spec_id: Mapped[str] = mapped_column(String, index=True, default="")
    spec_name: Mapped[str] = mapped_column(String, default="")
    step_id: Mapped[str] = mapped_column(String, default="")
    tool_name: Mapped[str] = mapped_column(String, default="")
    capability_id: Mapped[str] = mapped_column(String, default="")
    arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    fingerprint: Mapped[str] = mapped_column(String, index=True)
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    rules: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_factors: Mapped[list] = mapped_column(JSON, default=list)
    channel: Mapped[str] = mapped_column(String, default="chat")
    status: Mapped[str] = mapped_column(String, index=True, default=PENDING)
    function_call_id: Mapped[str] = mapped_column(String, index=True, default="")
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by: Mapped[str] = mapped_column(String, default="")
    decided_via: Mapped[str] = mapped_column(String, default="")
    note: Mapped[str] = mapped_column(String, default="")
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def fingerprint(*, tenant_id: str, spec_id: str, step_id: str, tool_name: str, arguments: dict[str, Any]) -> str:
    """Identifies one specific call: same spec, step, tool and arguments."""
    payload = json.dumps([tenant_id, spec_id, step_id, tool_name, arguments], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def to_dict(row: ApprovalRow) -> dict[str, Any]:
    return {
        "approval_id": row.id,
        "tenant_id": row.tenant_id,
        "requested_by": row.requested_by,
        "spec_id": row.spec_id,
        "spec_name": row.spec_name,
        "step_id": row.step_id,
        "tool_name": row.tool_name,
        "capability_id": row.capability_id,
        "arguments": row.arguments,
        "reasons": row.reasons,
        "rules": row.rules,
        "confidence": row.confidence,
        "confidence_factors": row.confidence_factors,
        "channel": row.channel,
        "status": row.status,
        "requested_at": _aware(row.requested_at).isoformat() if row.requested_at else None,
        "expires_at": _aware(row.expires_at).isoformat() if row.expires_at else None,
        "decided_at": _aware(row.decided_at).isoformat() if row.decided_at else None,
        "decided_by": row.decided_by,
        "decided_via": row.decided_via,
        "note": row.note,
    }


def _expire_if_due(row: ApprovalRow) -> ApprovalRow:
    if row.status == PENDING and _aware(row.expires_at) <= _now():
        row.status = EXPIRED
        row.decided_at = _now()
        row.decided_via = "expiry"
    return row


async def request(
    session: AsyncSession,
    *,
    tenant_id: str,
    requested_by: str,
    spec_id: str,
    spec_name: str,
    step_id: str,
    tool_name: str,
    capability_id: str,
    arguments: dict[str, Any],
    reasons: list[str],
    rules: list[str],
    confidence: float | None,
    confidence_factors: list[str],
    channel: str,
    ttl_seconds: int,
    function_call_id: str = "",
) -> ApprovalRow:
    """Return the pending request for this exact call if one exists (a
    retried call shouldn't stack up duplicates in the inbox), else file a
    new one."""
    fp = fingerprint(tenant_id=tenant_id, spec_id=spec_id, step_id=step_id, tool_name=tool_name, arguments=arguments)
    existing = (
        await session.execute(
            select(ApprovalRow).where(ApprovalRow.fingerprint == fp, ApprovalRow.status == PENDING)
        )
    ).scalars().first()
    if existing is not None and _expire_if_due(existing).status == PENDING:
        if function_call_id:
            existing.function_call_id = function_call_id
        existing.reasons, existing.rules = reasons, rules
        existing.confidence, existing.confidence_factors = confidence, confidence_factors
        await session.flush()
        return existing

    row = ApprovalRow(
        id=f"apr_{uuid.uuid4().hex[:12]}",
        tenant_id=tenant_id,
        requested_by=requested_by,
        spec_id=spec_id,
        spec_name=spec_name,
        step_id=step_id,
        tool_name=tool_name,
        capability_id=capability_id,
        arguments=arguments,
        fingerprint=fp,
        reasons=reasons,
        rules=rules,
        confidence=confidence,
        confidence_factors=confidence_factors,
        channel=channel,
        status=PENDING,
        function_call_id=function_call_id,
        requested_at=_now(),
        expires_at=_now() + timedelta(seconds=ttl_seconds),
    )
    session.add(row)
    await session.flush()
    return row


async def get(session: AsyncSession, approval_id: str) -> ApprovalRow | None:
    row = await session.get(ApprovalRow, approval_id)
    return _expire_if_due(row) if row is not None else None


async def find_by_call(session: AsyncSession, function_call_id: str) -> ApprovalRow | None:
    if not function_call_id:
        return None
    row = (
        await session.execute(select(ApprovalRow).where(ApprovalRow.function_call_id == function_call_id))
    ).scalars().first()
    return _expire_if_due(row) if row is not None else None


async def latest_for_call(session: AsyncSession, fp: str) -> ApprovalRow | None:
    """The most recent decision-bearing record for this exact call: an
    unconsumed approval to use, or a denial to honor, or a pending request.
    Consumed and expired records don't count -- a new call needs a new
    approval."""
    rows = (
        await session.execute(
            select(ApprovalRow).where(ApprovalRow.fingerprint == fp).order_by(ApprovalRow.requested_at.desc())
        )
    ).scalars().all()
    for row in rows:
        _expire_if_due(row)
        if row.status in (PENDING, APPROVED, DENIED):
            return row
    return None


async def decide(
    session: AsyncSession, approval_id: str, *, approve: bool, decided_by: str, via: str, note: str = ""
) -> ApprovalRow:
    """Record a decision. Only a pending request can be decided -- the first
    decision is final. `via` is "chat" or "inbox"; an inbox-channel request
    can only be decided via the inbox."""
    row = await get(session, approval_id)
    if row is None:
        raise KeyError(f"no such approval: {approval_id}")
    if row.status != PENDING:
        raise ValueError(f"approval {approval_id} is already {row.status}")
    if row.channel == "inbox" and via != "inbox":
        raise PermissionError(f"approval {approval_id} must be decided by an approver in the approvals inbox")
    row.status = APPROVED if approve else DENIED
    row.decided_at = _now()
    row.decided_by = decided_by
    row.decided_via = via
    row.note = note
    await session.flush()
    return row


async def consume(session: AsyncSession, approval_id: str) -> None:
    row = await session.get(ApprovalRow, approval_id)
    if row is not None and row.status == APPROVED:
        row.status = CONSUMED
        row.consumed_at = _now()
        await session.flush()


async def list_for_tenant(session: AsyncSession, tenant_id: str, *, status: str | None = None, limit: int = 200) -> list[ApprovalRow]:
    stmt = select(ApprovalRow).where(ApprovalRow.tenant_id == tenant_id).order_by(ApprovalRow.requested_at.desc()).limit(limit)
    rows = [_expire_if_due(r) for r in (await session.execute(stmt)).scalars().all()]
    await session.flush()
    return [r for r in rows if status is None or r.status == status]
