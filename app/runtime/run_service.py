"""Ties the whole request flow together end to end, per the architecture
doc's "Request shape, end to end" paragraph:

  intent -> plan -> compile -> execute -> stream events -> close ledger -> notify

`start_run` returns immediately with a `run_id`; execution happens in a
background task against its own DB session (the HTTP request's session is
gone by the time a long-running plan finishes), and a client subscribes to
that run's Redis stream (runtime/redis_streams.py) to watch it live or
resume after being offline.
"""

from __future__ import annotations

import asyncio
import logging

from opentelemetry import trace
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import delegation
from app.auth.models import IdentityContext
from app.db import session_scope
from app.factory.agent_factory import compile_plan
from app.factory.tool_wrapper import ToolCallContext
from app.guardrails.policy import get_policy
from app.marketplace import service as marketplace_service
from app.notifications.composio_dispatcher import dispatcher
from app.planner.schemas import ExecutionPlan, StepKind
from app.registry import store as registry_store
from app.runtime import redis_streams

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("capability_marketplace.runtime")


async def start_run(session: AsyncSession, identity: IdentityContext, plan: ExecutionPlan) -> str:
    run_id = marketplace_service.new_run_id()
    await marketplace_service.save_plan(session, identity.tenant_id, plan)
    await marketplace_service.open_invocation(
        session, run_id=run_id, tenant_id=identity.tenant_id, user_id=identity.user_id, plan_id=plan.plan_id, trace_id=None
    )
    await session.commit()

    asyncio.create_task(_execute(identity, plan, run_id))
    return run_id


async def _execute(identity: IdentityContext, plan: ExecutionPlan, run_id: str) -> None:
    with tracer.start_as_current_span(
        "agent.run",
        attributes={
            "run_id": run_id,
            "tenant_id": identity.tenant_id,
            "brand_id": identity.brand_id,
            "user_id": identity.user_id,
            "plan_id": plan.plan_id,
        },
    ):
        await redis_streams.publish_event(run_id, {"kind": "run.started", "plan_id": plan.plan_id})
        overall_status = "succeeded"
        try:
            async with session_scope() as session:
                plan_scopes = tuple({s for step in plan.steps for s in step.required_scopes})
                run_token = delegation.delegate_to_run(identity, run_id=run_id, plan_scopes=plan_scopes)

                capabilities = {}
                granted_ids = await marketplace_service.granted_capability_ids(session, identity.tenant_id)
                visible = await registry_store.list_for_tenant(
                    session, identity.tenant_id, granted_capability_ids=granted_ids
                )
                by_id = {c.id: c for c in visible}
                for step in plan.steps:
                    if step.kind == StepKind.TOOL_CALL and step.capability_id in by_id:
                        capabilities[step.capability_id] = by_id[step.capability_id]

                ctx = ToolCallContext(
                    identity=identity, run_id=run_id, run_token=run_token, policy=get_policy("default")
                )
                compiled = compile_plan(plan, capabilities, ctx)

                async for event in compiled.run():
                    await redis_streams.publish_event(
                        run_id,
                        {
                            "kind": event.kind,
                            "step_id": event.step_id,
                            "capability_id": event.capability_id,
                            "ok": event.result.ok if event.result else None,
                            "error": event.result.error if event.result else None,
                        },
                    )
                    if event.capability_id:
                        await marketplace_service.record_step_invocation(
                            session, run_id=run_id, capability_id=event.capability_id
                        )
                    if event.kind == "step.blocked":
                        overall_status = "blocked"

                await marketplace_service.close_invocation(session, run_id=run_id, status=overall_status)
                await dispatcher.dispatch(
                    session,
                    tenant_id=identity.tenant_id,
                    event_type="run_completed" if overall_status == "succeeded" else "run_failed",
                    payload={"run_id": run_id, "status": overall_status},
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("run %s failed", run_id)
            overall_status = "failed"
            async with session_scope() as session:
                await marketplace_service.close_invocation(session, run_id=run_id, status="failed")
                await dispatcher.dispatch(
                    session,
                    tenant_id=identity.tenant_id,
                    event_type="run_failed",
                    payload={"run_id": run_id, "error": str(exc)},
                )
        finally:
            await redis_streams.publish_event(run_id, {"kind": "run.finished", "status": overall_status})
            # Deliberately not trimming the stream here: a client may not
            # have read "run.finished" yet, and trimming races the whole
            # point of offline resume. A production build runs a separate
            # reaper that calls `redis_streams.close_stream` once a run has
            # been terminal for some grace period (e.g. 24h), not inline here.
