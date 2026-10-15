"""Publish/request/grant workflow + the invocation-tracking ledger. See the
architecture doc's "Marketplace mechanics" section."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.marketplace.models import (
    AgentDefinitionRow,
    CapabilityGrantRow,
    InvocationRecordRow,
    PlanRecordRow,
)
from app.planner.schemas import ExecutionPlan


async def request_grant(
    session: AsyncSession, *, capability_id: str, grantor_tenant_id: str, grantee_tenant_id: str, scope_limits: list[str] | None = None
) -> CapabilityGrantRow:
    """Idempotent per (capability, grantee): if a request is already pending
    or approved, return it instead of filing a duplicate -- otherwise every
    extra click of "Request access" adds another row to the grantor's
    inbox. A denied or revoked grant can be re-requested."""
    existing = (
        await session.execute(
            select(CapabilityGrantRow).where(
                CapabilityGrantRow.capability_id == capability_id,
                CapabilityGrantRow.grantee_tenant_id == grantee_tenant_id,
                CapabilityGrantRow.status.in_(("pending", "approved")),
            )
        )
    ).scalars().first()
    if existing is not None:
        return existing

    grant = CapabilityGrantRow(
        capability_id=capability_id,
        grantor_tenant_id=grantor_tenant_id,
        grantee_tenant_id=grantee_tenant_id,
        scope_limits=scope_limits or [],
        status="pending",
    )
    session.add(grant)
    await session.flush()
    return grant


async def decide_grant(session: AsyncSession, grant_id: str, *, approve: bool) -> CapabilityGrantRow:
    grant = await session.get(CapabilityGrantRow, grant_id)
    if grant is None:
        raise KeyError(f"no such grant: {grant_id}")
    grant.status = "approved" if approve else "denied"
    grant.decided_at = datetime.now(timezone.utc)
    await session.flush()
    return grant


async def grants_involving(session: AsyncSession, tenant_id: str) -> dict[str, list[CapabilityGrantRow]]:
    """Every grant where this tenant is the grantor ("incoming" -- requests
    for its capabilities, which it decides) or the grantee ("outgoing" --
    access it asked for), newest first."""
    incoming = select(CapabilityGrantRow).where(CapabilityGrantRow.grantor_tenant_id == tenant_id)
    outgoing = select(CapabilityGrantRow).where(CapabilityGrantRow.grantee_tenant_id == tenant_id)
    order = CapabilityGrantRow.requested_at.desc()
    return {
        "incoming": list((await session.execute(incoming.order_by(order))).scalars().all()),
        "outgoing": list((await session.execute(outgoing.order_by(order))).scalars().all()),
    }


async def granted_capability_ids(session: AsyncSession, tenant_id: str) -> set[str]:
    stmt = select(CapabilityGrantRow.capability_id).where(
        CapabilityGrantRow.grantee_tenant_id == tenant_id, CapabilityGrantRow.status == "approved"
    )
    return {row for row in (await session.execute(stmt)).scalars().all()}


async def save_plan(session: AsyncSession, tenant_id: str, plan: ExecutionPlan) -> None:
    # `merge`, not `add`: a plan returned by POST /plans and then handed
    # straight to POST /runs gets saved twice with the same plan_id -- this
    # makes that idempotent instead of a unique-constraint error.
    await session.merge(
        PlanRecordRow(plan_id=plan.plan_id, tenant_id=tenant_id, goal=plan.goal, data=plan.model_dump(mode="json"))
    )
    await session.flush()


async def save_agent_definition(
    session: AsyncSession, *, plan_id: str, tenant_id: str, capability_id: str | None = None
) -> AgentDefinitionRow:
    row = AgentDefinitionRow(plan_id=plan_id, tenant_id=tenant_id, capability_id=capability_id)
    session.add(row)
    await session.flush()
    return row


async def open_invocation(
    session: AsyncSession, *, run_id: str, tenant_id: str, user_id: str, plan_id: str | None, trace_id: str | None
) -> InvocationRecordRow:
    row = InvocationRecordRow(
        run_id=run_id, tenant_id=tenant_id, user_id=user_id, plan_id=plan_id, status="running", trace_id=trace_id
    )
    session.add(row)
    await session.flush()
    return row


async def record_step_invocation(
    session: AsyncSession, *, run_id: str, capability_id: str, grant_id: str | None = None
) -> None:
    """One `InvocationRecord` per capability used -- called per step rather
    than once per run, so a multi-step plan's per-capability metering/billing
    is attributable to the exact capability, not just the run."""
    row = await session.get(InvocationRecordRow, run_id)
    if row is not None and row.capability_id is None:
        row.capability_id = capability_id
        row.grant_id = grant_id
        await session.flush()


async def close_invocation(session: AsyncSession, *, run_id: str, status: str) -> None:
    row = await session.get(InvocationRecordRow, run_id)
    if row is None:
        return
    row.status = status
    row.completed_at = datetime.now(timezone.utc)
    await session.flush()


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"
