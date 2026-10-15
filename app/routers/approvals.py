"""The approvals inbox: human-in-the-loop requests filed by compiled spec
steps (app/factory/tool_policy.py, stored by app/runtime/approvals.py),
tenant-scoped.

    GET  /approvals?status=pending     what's waiting (or any status / all)
    GET  /approvals/{id}               one request, with its reasons,
                                       arguments and confidence breakdown
    POST /approvals/{id}/decide        {"approve": true|false, "note": "..."}

Deciding here is the "inbox" channel: it can decide any request, including
those whose spec requires inbox-only approval (separation of duties -- the
person chatting with the agent can't approve their own risky call). The
paused run picks the decision up when it resumes: in `adk web`, by
confirming the pending prompt (which re-checks); under the supervisor, by
running the spec again.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_identity
from app.auth.models import IdentityContext
from app.db import SessionLocal
from app.runtime import approvals as approvals_store

router = APIRouter(prefix="/approvals", tags=["approvals"])


async def _db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


class DecideBody(BaseModel):
    approve: bool
    note: str = ""


@router.get("")
async def list_approvals(
    status: str | None = "pending",
    identity: IdentityContext = Depends(get_identity),
    session: AsyncSession = Depends(_db),
):
    """`status=pending` by default; pass `status=` (empty) for every status."""
    rows = await approvals_store.list_for_tenant(session, identity.tenant_id, status=status or None)
    await session.commit()  # persist any pending -> expired transitions noticed while listing
    return [approvals_store.to_dict(r) for r in rows]


@router.get("/{approval_id}")
async def get_approval(approval_id: str, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)):
    row = await approvals_store.get(session, approval_id)
    if row is None or row.tenant_id != identity.tenant_id:
        raise HTTPException(status_code=404, detail="approval not found")
    return approvals_store.to_dict(row)


@router.post("/{approval_id}/decide")
async def decide_approval(
    approval_id: str, body: DecideBody, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
):
    row = await approvals_store.get(session, approval_id)
    if row is None or row.tenant_id != identity.tenant_id:
        raise HTTPException(status_code=404, detail="approval not found")
    try:
        row = await approvals_store.decide(
            session, approval_id, approve=body.approve, decided_by=identity.user_id, via="inbox", note=body.note
        )
    except ValueError as exc:  # already decided / expired
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    await session.commit()
    return approvals_store.to_dict(row)
