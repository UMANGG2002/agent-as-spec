"""NL intent -> typed ExecutionPlan. This is the brief's "planner agent
generated typed execution flow" endpoint, returned as reviewable data before
anything executes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_identity
from app.auth.models import IdentityContext
from app.db import SessionLocal
from app.marketplace import service as marketplace_service
from app.planner.planner_agent import create_plan
from app.planner.schemas import ExecutionPlan

router = APIRouter(prefix="/plans", tags=["plans"])


async def _db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


class PlanRequest(BaseModel):
    intent: str
    persist_as_capability: bool = False


@router.post("", response_model=ExecutionPlan)
async def create_execution_plan(
    body: PlanRequest, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
) -> ExecutionPlan:
    plan = await create_plan(session, identity, body.intent)
    await marketplace_service.save_plan(session, identity.tenant_id, plan)
    if body.persist_as_capability:
        await marketplace_service.save_agent_definition(session, plan_id=plan.plan_id, tenant_id=identity.tenant_id)
    await session.commit()
    return plan
