"""Execute a plan and stream its progress. `GET /runs/{run_id}/events` is the
offline-resume endpoint from the architecture doc: pass the same `consumer`
id on reconnect and you only get what you haven't acked yet."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_identity
from app.auth.models import IdentityContext
from app.db import SessionLocal
from app.marketplace.models import PlanRecordRow
from app.planner.schemas import ExecutionPlan
from app.runtime import redis_streams
from app.runtime.hil import approval_store
from app.runtime.run_service import start_run

router = APIRouter(prefix="/runs", tags=["runs"])


async def _db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


class StartRunRequest(BaseModel):
    plan: ExecutionPlan | None = None
    plan_id: str | None = None


class StartRunResponse(BaseModel):
    run_id: str


class ApprovalRequest(BaseModel):
    approved: bool
    note: str | None = None


@router.post("", response_model=StartRunResponse)
async def start(
    body: StartRunRequest, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
) -> StartRunResponse:
    plan = body.plan
    if plan is None and body.plan_id:
        row = await session.get(PlanRecordRow, body.plan_id)
        if row is None:
            raise HTTPException(status_code=404, detail="plan not found")
        plan = ExecutionPlan.model_validate(row.data)
    if plan is None:
        raise HTTPException(status_code=400, detail="either plan or plan_id is required")

    run_id = await start_run(session, identity, plan)
    return StartRunResponse(run_id=run_id)


@router.get("/{run_id}/events")
async def stream_events(run_id: str, consumer: str = "default-consumer"):
    async def _gen():
        while True:
            saw_event = False
            async for entry_id, event in redis_streams.read_events(run_id, consumer=consumer):
                saw_event = True
                yield f"data: {json.dumps(event)}\n\n"
                await redis_streams.ack(run_id, entry_id)
                if event.get("kind") == "run.finished":
                    return
            if not saw_event:
                # read_events timed out with nothing new; keep the connection
                # open (a real client would also handle a plain disconnect
                # and reconnect later with the same `consumer` id).
                continue

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.post("/{run_id}/steps/{step_id}/approve")
async def approve_step(run_id: str, step_id: str, body: ApprovalRequest, identity: IdentityContext = Depends(get_identity)):
    try:
        request = approval_store.resolve_approval(
            run_id, step_id, approved=body.approved, resolved_by=identity.user_id, note=body.note
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"status": request.status}
