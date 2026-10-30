"""End to end: API -> SessionManager -> (queue) -> runner -> ADK Runner ->
Redis event stream -> resumable SSE, with a scripted model and the real
test worker behind every tool call."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from google.adk.models import LlmResponse
from google.genai import types

from app.auth.models import IdentityContext
from app.config import settings
from app.main import app
from app.planner.schemas import ExecutionPlan, PlanStep, StepKind
from app.sessions import agent_cache, manager, queue, streams
from app.sessions import runner as runner_mod
from app.specs import service as specs_service
from app.specs.schemas import AgentRuntimeConfig, ToolPolicyConfig
from app.testing import worker
from app.testing.scripted_llm import ScriptedToolCallingLlm

TENANT = "acme"
_ID = IdentityContext(tenant_id=TENANT, brand_id="b", user_id="alice", scopes=("*",))


def _plan() -> ExecutionPlan:
    return ExecutionPlan(plan_id="plan_sess", goal="quote 4 x SKU-3", steps=[
        PlanStep(step_id="quote", kind=StepKind.TOOL_CALL, capability_id=worker.mcp_capability_id("inventory-tools"),
                 tool_name="price_quote", arguments={"sku": "SKU-3", "quantity": 4}, instruction="Quote 4 x SKU-3"),
    ])


@pytest.fixture
async def spec(session):
    base_url = worker.ensure_running()
    await worker.register_all(session, TENANT, base_url)
    await session.commit()
    runner_mod.model_override = ScriptedToolCallingLlm()
    return await specs_service.approve_plan(session, _ID, name="quote-agent", plan=_plan())


@pytest.fixture
async def gated_spec(session):
    base_url = worker.ensure_running()
    await worker.register_all(session, TENANT, base_url)
    await session.commit()
    runner_mod.model_override = ScriptedToolCallingLlm()
    runtime = AgentRuntimeConfig(tool_policy=ToolPolicyConfig(require_approval_before=["quote"]))
    return await specs_service.approve_plan(session, _ID, name="gated-agent", plan=_plan(), runtime=runtime)


async def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=60)


async def _token(client, user="alice", tenant=TENANT) -> dict:
    tok = (await client.post("/auth/dev-token", json={"tenant_id": tenant, "user_id": user})).json()["access_token"]
    return {"Authorization": f"Bearer {tok}"}


def _parse_sse(raw: str) -> list[tuple[str, str, dict]]:
    import json

    out = []
    for block in raw.split("\n\n"):
        fields = {}
        for line in block.splitlines():
            if line.startswith(":") or ":" not in line:
                continue
            k, _, v = line.partition(": ")
            fields[k] = v
        if "data" in fields:
            out.append((fields.get("id"), fields.get("event"), json.loads(fields["data"])))
    return out


async def _run(client, headers, session_id, body, *, extra_headers=None):
    resp = await client.post(f"/v1/sessions/{session_id}/messages", json=body, headers={**headers, **(extra_headers or {})})
    assert resp.status_code == 202, resp.text
    inv = resp.json()
    stream = await client.get(inv["stream_url"], headers=headers)
    return inv, _parse_sse(stream.text)


async def test_message_runs_on_the_runner_and_streams_every_event(spec):
    async with await _client() as client:
        h = await _token(client)
        sid = (await client.post("/v1/sessions", json={"spec_id": spec.spec_id}, headers=h)).json()["session_id"]
        inv, events = await _run(client, h, sid, {"text": "quote please"})
        types_ = [e[1] for e in events]
        assert types_[0] == "invocation.queued" and types_[1] == "invocation.started"
        assert types_[-1] == "invocation.finished" and events[-1][2]["status"] == "succeeded"
        calls = [c["name"] for _, _, e in events for c in e.get("function_calls", [])]
        assert calls == ["step_quote"]
        responses = [r for _, _, e in events for r in e.get("function_responses", [])]
        assert responses[0]["response"]["ok"] is True
        ids = [e[0] for e in events]
        assert ids == sorted(ids, key=lambda i: tuple(map(int, i.split("-"))))  # strictly ordered ids
        status = (await client.get(f"/v1/sessions/{sid}/invocations/{inv['invocation_id']}", headers=h)).json()
        assert status["status"] == "succeeded" and status["events"] >= 3


async def test_reconnect_with_last_event_id_resumes_exactly_after_it(spec):
    async with await _client() as client:
        h = await _token(client)
        sid = (await client.post("/v1/sessions", json={"spec_id": spec.spec_id}, headers=h)).json()["session_id"]
        inv, full = await _run(client, h, sid, {"text": "quote please"})
        cut = len(full) // 2
        resumed = await client.get(inv["stream_url"], headers={**h, "Last-Event-ID": full[cut - 1][0]})
        rest = _parse_sse(resumed.text)
        assert [e[0] for e in rest] == [e[0] for e in full[cut:]]  # no gap, no duplicate
        via_query = _parse_sse((await client.get(inv["stream_url"] + f"?after={full[cut - 1][0]}", headers=h)).text)
        assert [e[0] for e in via_query] == [e[0] for e in rest]


async def test_live_tail_through_the_stream_hub_while_the_writer_publishes():
    received: list[str] = []

    async def reader():
        async for eid, event in streams.follow("ses_hub", "inv_hub", None, heartbeat=0.2):
            if eid:
                received.append(event["n"])

    readers = [asyncio.create_task(reader()) for _ in range(25)]  # 25 tabs, one hub, one connection
    await asyncio.sleep(0.05)
    for n in range(5):
        await streams.publish("ses_hub", "inv_hub", {"type": "event", "n": n})
        await asyncio.sleep(0.01)
    await streams.publish("ses_hub", "inv_hub", {"type": streams.TERMINAL, "n": "end"})
    await asyncio.wait_for(asyncio.gather(*readers), timeout=5)
    assert received.count(0) == 25 and received.count("end") == 25


async def test_idempotency_key_makes_a_retried_post_return_the_same_invocation(spec):
    async with await _client() as client:
        h = await _token(client)
        sid = (await client.post("/v1/sessions", json={"spec_id": spec.spec_id}, headers=h)).json()["session_id"]
        body = {"text": "quote please"}
        a = (await client.post(f"/v1/sessions/{sid}/messages", json=body, headers={**h, "Idempotency-Key": "k1"})).json()
        b = (await client.post(f"/v1/sessions/{sid}/messages", json=body, headers={**h, "Idempotency-Key": "k1"})).json()
        assert a["invocation_id"] == b["invocation_id"]
        await client.get(a["stream_url"], headers=h)  # let it finish


async def test_sessions_are_private_to_their_user_and_tenant(spec):
    async with await _client() as client:
        alice = await _token(client)
        sid = (await client.post("/v1/sessions", json={"spec_id": spec.spec_id}, headers=alice)).json()["session_id"]
        bob = await _token(client, user="bob")
        mallory = await _token(client, user="alice", tenant="globex")
        for h in (bob, mallory):
            r = await client.post(f"/v1/sessions/{sid}/messages", json={"text": "hi"}, headers=h)
            assert r.status_code == 404
        r = await client.post("/v1/sessions", json={"spec_id": spec.spec_id}, headers=mallory)
        assert r.status_code == 404  # another tenant's spec doesn't exist for globex


async def test_tool_confirmation_round_trip_over_the_api(gated_spec):
    from app.db import session_scope
    from app.runtime import approvals

    async with await _client() as client:
        h = await _token(client)
        sid = (await client.post("/v1/sessions", json={"spec_id": gated_spec.spec_id}, headers=h)).json()["session_id"]
        _, events = await _run(client, h, sid, {"text": "quote please"})
        assert events[-1][2]["status"] == "awaiting_input"
        request = next(e["confirmation_request"] for _, _, e in events if "confirmation_request" in e)
        assert "Approval needed before step 'quote'" in request["hint"]

        async with session_scope(tenant_id=TENANT) as db:
            row = await approvals.get(db, request["payload"]["approval_id"])
            assert row.requested_by == "alice"  # the caller -- not baked into the shared compiled agent

        _, events = await _run(client, h, sid, {"confirmations": [
            {"function_call_id": request["function_call_id"], "confirmed": True}]})
        assert events[-1][2]["status"] == "succeeded"
        quote = [r for _, _, e in events for r in e.get("function_responses", []) if r["name"] == "step_quote"]
        assert quote and quote[-1]["response"]["ok"] is True


async def test_tenant_concurrency_limit_defers_instead_of_failing(spec):
    from app.tenancy import service as tenancy

    await tenancy.upsert_tenant(TENANT, tier="internal", limits_override={"concurrent_invocations": 1})
    async with await _client() as client:
        h = await _token(client)
        sids = [(await client.post("/v1/sessions", json={"spec_id": spec.spec_id}, headers=h)).json()["session_id"]
                for _ in range(3)]
        invs = [(await client.post(f"/v1/sessions/{s}/messages", json={"text": "go"}, headers=h)).json() for s in sids]
        finals = [(_parse_sse((await client.get(i["stream_url"], headers=h)).text))[-1][2]["status"] for i in invs]
    assert finals == ["succeeded"] * 3


async def test_queue_mode_worker_runs_jobs_and_reclaims_a_crashed_workers_job(spec, monkeypatch):
    monkeypatch.setattr(settings, "runner_mode", "queue")
    monkeypatch.setattr(settings, "work_queue_shards", 4)
    await queue.ensure_groups()
    s1 = await manager.create_session(_ID, spec.spec_id)
    s2 = await manager.create_session(_ID, spec.spec_id)

    # Worker A takes job 1 and "crashes" (never acks).
    crashed = await manager.send_message(_ID, s1["session_id"], text="go")
    taken = await queue.read("worker-A", count=1, block_ms=100)
    assert taken and taken[0][2]["invocation_id"] == crashed["invocation_id"]

    # Job 2 runs normally on worker B's loop.
    normal = await manager.send_message(_ID, s2["session_id"], text="go")
    monkeypatch.setattr(settings, "runner_visibility_timeout_ms", 50)
    await asyncio.sleep(0.1)
    w = runner_mod.Worker(concurrency=4)
    task = asyncio.create_task(w.run())
    try:
        for inv, sid in ((normal, s2["session_id"]), (crashed, s1["session_id"])):
            final = None
            async for eid, event in streams.follow(sid, inv["invocation_id"], None, heartbeat=0.5):
                if event and event["type"] == streams.TERMINAL:
                    final = event["status"]
            assert final == "succeeded", inv
        for _ in range(50):  # the ack lands right after the terminal event
            if await queue.depth() == 0:
                break
            await asyncio.sleep(0.05)
        assert await queue.depth() == 0
    finally:
        w.stopping.set()
        await asyncio.wait_for(task, timeout=5)


async def test_memory_from_an_earlier_session_reaches_the_model(spec):
    seen: list[str] = []

    class RecordingLlm(ScriptedToolCallingLlm):
        async def generate_content_async(self, llm_request, stream=False):
            seen.append(self._system_instruction(llm_request))
            async for r in super().generate_content_async(llm_request, stream):
                yield r

    runner_mod.model_override = RecordingLlm()
    agent_cache.clear()
    a = await manager.create_session(_ID, spec.spec_id)
    first = await manager.send_message(_ID, a["session_id"], text="remember that my warehouse is Pune")
    async for _ in streams.follow(a["session_id"], first["invocation_id"], None):
        pass
    await asyncio.sleep(0.1)  # background memory ingestion
    b = await manager.create_session(_ID, spec.spec_id)
    seen.clear()
    second = await manager.send_message(_ID, b["session_id"], text="which warehouse is mine?")
    async for _ in streams.follow(b["session_id"], second["invocation_id"], None):
        pass
    assert any("my warehouse is Pune" in s for s in seen)


async def test_long_context_is_compacted_by_token_threshold(spec, monkeypatch):
    monkeypatch.setattr(settings, "context_compaction_token_threshold", 50)
    monkeypatch.setattr(settings, "context_compaction_retain_events", 2)
    monkeypatch.setattr(settings, "context_compaction_interval", None)

    class BigPromptLlm(ScriptedToolCallingLlm):
        async def generate_content_async(self, llm_request, stream=False):
            async for r in super().generate_content_async(llm_request, stream):
                r.usage_metadata = types.GenerateContentResponseUsageMetadata(
                    prompt_token_count=5_000, candidates_token_count=10, total_token_count=5_010)
                yield r

    runner_mod.model_override = BigPromptLlm()
    agent_cache.clear()
    app_ = agent_cache.build_app(spec, {}, model=runner_mod.model_override)
    assert app_.events_compaction_config.token_threshold == 50

    s = await manager.create_session(_ID, spec.spec_id)
    for text in ("first question", "second question"):
        inv = await manager.send_message(_ID, s["session_id"], text=text)
        async for _ in streams.follow(s["session_id"], inv["invocation_id"], None):
            pass
    from app.sessions.session_store import adk_user_id, get_session_service

    session = await get_session_service().get_session(app_name=agent_cache.app_name(spec.spec_id),
                                                      user_id=adk_user_id(TENANT, "alice"), session_id=s["session_id"])
    assert any(e.actions and e.actions.compaction for e in session.events)


async def test_cancel_stops_a_running_invocation(spec):
    class SlowForeverLlm(ScriptedToolCallingLlm):
        # Slow enough that the run is still going when cancel lands
        # (a fast runaway model would hit max_llm_calls first).
        async def generate_content_async(self, llm_request, stream=False):
            await asyncio.sleep(0.1)
            async for r in super().generate_content_async(llm_request, stream):
                yield r

    runner_mod.model_override = SlowForeverLlm(loop_forever=True)
    agent_cache.clear()
    s = await manager.create_session(_ID, spec.spec_id)
    inv = await manager.send_message(_ID, s["session_id"], text="go forever")
    await asyncio.sleep(0.3)
    await manager.cancel(_ID, s["session_id"], inv["invocation_id"])
    final = None
    async for _, event in streams.follow(s["session_id"], inv["invocation_id"], None, heartbeat=1):
        if event and event["type"] == streams.TERMINAL:
            final = event["status"]
    assert final in ("cancelled", "failed")


async def test_daily_token_budget_stops_model_calls(spec):
    from app.core import ratelimit
    from app.sessions.plugins import UsagePlugin
    from app.tenancy import service as tenancy

    await tenancy.upsert_tenant(TENANT, tier="internal", limits_override={"daily_token_budget": 100})
    await ratelimit.consume_budget(TENANT, 500, day=UsagePlugin._day())
    s = await manager.create_session(_ID, spec.spec_id)
    inv = await manager.send_message(_ID, s["session_id"], text="go")
    texts = [e.get("text", "") async for _, e in streams.follow(s["session_id"], inv["invocation_id"], None) if e]
    assert any("model budget" in t for t in texts)


async def test_chat_with_spec_test_agent_routes_to_an_approved_spec(spec):
    """The UI chat target `spec_test_agent`: a router with every approved
    spec mounted as a sub-agent, run through the same production runtime."""
    from app.factory.spec_agents import _safe_identifier

    sub_agent = f"spec_{_safe_identifier(spec.name, fallback='agent')}_{spec.spec_id[-6:]}"
    runner_mod.model_override = ScriptedToolCallingLlm(transfer_to=[sub_agent])
    agent_cache.clear()
    async with await _client() as client:
        h = await _token(client)
        targets = (await client.get("/v1/agents", headers=h)).json()
        assert targets[0]["id"] == "spec_test_agent" and any(t["id"] == spec.spec_id for t in targets)
        sid = (await client.post("/v1/sessions", json={"spec_id": "spec_test_agent"}, headers=h)).json()["session_id"]
        _, events = await _run(client, h, sid, {"text": "get me a quote"})
        assert events[-1][2]["status"] == "succeeded"
        assert any(e.get("transfer_to") == sub_agent for _, _, e in events)
        quote = [r for _, _, e in events for r in e.get("function_responses", []) if r["name"] == "step_quote"]
        assert quote and quote[0]["response"]["ok"] is True
        listed = (await client.get("/v1/sessions", headers=h)).json()
        assert listed[0]["spec_id"] == "spec_test_agent"
