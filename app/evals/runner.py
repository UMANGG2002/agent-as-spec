"""Run one eval case in-process and return a `Trace` of what the agent did.

This is the same compiled `App` the production runner executes (same plugins:
guardrails, memory, usage; same step tools, policy and approval gate), driven
through ADK's `Runner` with an in-memory session, so an eval needs no queue,
no API and no stored session -- and leaves none behind. Only the model is
swappable (`model=`): the scripted model for deterministic CI, the live model
for real quality numbers.

A simulated approver answers tool-confirmation requests per
`EvalCase.confirmation`: "approve" and "reject" exercise both branches of the
approval flow; "leave_pending" stops at the pause and reports
`awaiting_approval`.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from google.genai import types

from app.auth.models import IdentityContext
from app.core import context
from app.evals.schemas import EvalCase, ToolCallRecord, Trace

MAX_CONFIRM_ROUNDS = 5
BLOCK_STATUSES = {"denied", "denied_by_policy", "rejected"}


def identity_state(identity: IdentityContext) -> dict:
    return {"tenant_id": identity.tenant_id, "brand_id": identity.brand_id, "user_id": identity.user_id,
            "scopes": list(identity.scopes), "act_chain": list(identity.act_chain)}


async def _load_app(case: EvalCase, identity: IdentityContext, capabilities: dict, model: Any):
    from app.sessions.agent_cache import get_app, get_router_app
    from app.specs import store as specs_store

    tenant = identity.tenant_id
    if case.target.kind == "router":
        specs = await specs_store.list_specs_cached(tenant)
        return await get_router_app(tenant, specs, capabilities, model=model), 100
    if not case.target.spec_id:
        raise ValueError("a spec target needs spec_id")
    if case.target.spec_version:
        spec = await specs_store.get_spec_version(case.target.spec_id, case.target.spec_version, tenant_id=tenant)
    else:
        spec = await specs_store.get_spec(case.target.spec_id, tenant_id=tenant)
    if spec is None:
        raise ValueError(f"spec {case.target.spec_id} v{case.target.spec_version or 'head'} not found")
    return await get_app(spec, capabilities, model=model), spec.runtime.max_llm_calls or 50


def _absorb(trace: Trace, payload: dict, calls_by_id: dict[str, ToolCallRecord]) -> None:
    trace.n_events += 1
    if "usage" in payload:
        trace.llm_calls += 1
        trace.tokens_in += int(payload["usage"].get("in") or 0)
        trace.tokens_out += int(payload["usage"].get("out") or 0)
    if payload.get("transfer_to"):
        trace.transfers.append(payload["transfer_to"])
    for call in payload.get("function_calls", []):
        if call["name"] in ("transfer_to_agent", "adk_request_confirmation"):
            continue
        rec = ToolCallRecord(name=call["name"], agent=payload.get("author", ""),
                             args=call["args"] if isinstance(call["args"], dict) else {})
        trace.tool_calls.append(rec)
        if call.get("id"):
            calls_by_id[call["id"]] = rec
    for resp in payload.get("function_responses", []):
        rec = calls_by_id.get(resp.get("id"))
        out = resp.get("response")
        if rec is None or not isinstance(out, dict):
            continue
        rec.step_id = out.get("step_id")
        rec.status = out.get("status")
        rec.ok = out.get("ok")
        rec.output = out.get("output", out if rec.step_id is None else None)
        rec.error = out.get("error")
        if out.get("blocked_by") and not rec.status:
            rec.status = "blocked"
    if payload.get("final") and payload.get("text") and not payload.get("partial"):
        trace.final_text = payload["text"]


def _derive_outcome(trace: Trace, awaiting: bool) -> str:
    if trace.error:
        return "error"
    if awaiting:
        return "awaiting_approval"
    if any((c.status in BLOCK_STATUSES) or (c.ok is False and c.status == "blocked") for c in trace.tool_calls):
        return "blocked"
    if trace.final_text.startswith("Request blocked"):
        return "blocked"
    return "completed"


async def run_case(case: EvalCase, identity: IdentityContext, *, model: Any = None,
                   capabilities: dict | None = None, timeout_s: float = 180.0) -> Trace:
    """Execute `case` and return the trace. Never raises for an agent failure:
    the error is captured on the trace so the metrics can fail the case."""
    trace = Trace()
    started = time.monotonic()
    sid = f"eval_{uuid.uuid4().hex[:12]}"
    awaiting = False
    try:
        with context.bind(tenant_id=identity.tenant_id, user_id=identity.user_id, session_id=sid,
                          invocation_id=sid):
            from google.adk.agents.run_config import RunConfig, StreamingMode
            from google.adk.runners import Runner
            from google.adk.sessions import InMemorySessionService

            from app.registry import store as registry_store
            from app.sessions.runner import serialize_event
            from app.sessions.session_store import adk_user_id

            caps = capabilities if capabilities is not None else await registry_store.visible_capabilities(
                identity.tenant_id)
            app, max_calls = await _load_app(case, identity, caps, model)
            uid = adk_user_id(identity.tenant_id, identity.user_id)
            service = InMemorySessionService()
            await service.create_session(app_name=app.name, user_id=uid, session_id=sid)
            runner = Runner(app=app, session_service=service)
            state = {"__identity__": identity_state(identity)}
            run_config = RunConfig(streaming_mode=StreamingMode.NONE, max_llm_calls=max(1, max_calls))
            calls_by_id: dict[str, ToolCallRecord] = {}

            async def _drive(message: types.Content) -> list[str]:
                asked: list[str] = []
                async for event in runner.run_async(user_id=uid, session_id=sid, new_message=message,
                                                    state_delta=state, run_config=run_config):
                    payload = serialize_event(event)
                    _absorb(trace, payload, calls_by_id)
                    asked += [c["id"] for c in payload.get("function_calls", [])
                              if c["name"] == "adk_request_confirmation" and c.get("id")]
                return asked

            async def _all_turns() -> None:
                nonlocal awaiting
                for text in case.messages:
                    message = types.Content(role="user", parts=[types.Part(text=text)])
                    for _ in range(MAX_CONFIRM_ROUNDS):
                        asked = await _drive(message)
                        if not asked:
                            break
                        trace.confirmations_requested += len(asked)
                        if case.confirmation == "leave_pending":
                            awaiting = True
                            return
                        approve = case.confirmation == "approve"
                        message = types.Content(role="user", parts=[
                            types.Part(function_response=types.FunctionResponse(
                                id=i, name="adk_request_confirmation",
                                response={"confirmed": approve, "payload": None})) for i in asked])
                    trace.answers_by_turn.append(trace.final_text)

            await asyncio.wait_for(_all_turns(), timeout=timeout_s)
    except asyncio.TimeoutError:
        trace.error = f"timed out after {timeout_s:.0f}s"
    except Exception as exc:  # noqa: BLE001 -- an agent failure is a result, not a crash
        trace.error = f"{type(exc).__name__}: {exc}"[:500]
    trace.latency_ms = int((time.monotonic() - started) * 1000)
    trace.outcome = _derive_outcome(trace, awaiting)  # type: ignore[assignment]
    return trace
