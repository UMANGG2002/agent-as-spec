"""Dynamic policy enforcement + human-in-the-loop before tool use
(`app/factory/tool_policy.py`, `app/runtime/approvals.py`, the step tool in
`app/factory/spec_agents.py`, `app/routers/approvals.py`).

Unit layer: selectors, confidence, rule semantics, precedence, validation.
End to end: through ADK's real runner with a scripted model, every step
really calling the test worker -- the run pauses on ADK's own tool
confirmation, the decision lands in the durable store, the run resumes.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.auth.models import IdentityContext
from app.factory import spec_agents, tool_policy as tp
from app.guardrails import policy as guardrail_policy
from app.planner.schemas import ExecutionPlan, PlanStep, StepKind
from app.registry import store as registry_store
from app.runtime import approvals
from app.specs import service as specs_service
from app.specs.schemas import AgentRuntimeConfig, AgentSpec, PolicyCondition, PolicyEffect, PolicyRule, ToolPolicyConfig
from app.testing import worker
from app.testing.scripted_llm import ScriptedToolCallingLlm

_ID = IdentityContext(tenant_id="acme", brand_id="acme-brand", user_id="policy-tests", scopes=("*",))


def _steps() -> list[PlanStep]:
    mcp, a2a = worker.mcp_capability_id, worker.a2a_capability_id
    return [
        PlanStep(step_id="quote", kind=StepKind.TOOL_CALL, capability_id=mcp("inventory-tools"),
                 tool_name="price_quote", arguments={"sku": "SKU-3", "quantity": 4}, instruction="Quote 4 x SKU-3"),
        PlanStep(step_id="route", kind=StepKind.TOOL_CALL, capability_id=a2a("ticket-triage-agent"),
                 tool_name="route_ticket", instruction="my invoice payment failed", depends_on=["quote"]),
    ]


def _spec(policy: ToolPolicyConfig, steps: list[PlanStep] | None = None) -> AgentSpec:
    return AgentSpec(spec_id="spec_policy000001", tenant_id=_ID.tenant_id, name="policy-agent", goal="quote then route",
                     source_plan_id="plan_policy", steps=steps or _steps(), approved_by=_ID.user_id,
                     runtime=AgentRuntimeConfig(tool_policy=policy))


# -- unit: selectors, confidence, rules ------------------------------------------------


def test_selectors_match_step_id_tool_names_and_capability():
    step = _steps()[0]
    for selector in ("quote", "step_quote", "price_quote", worker.mcp_capability_id("inventory-tools"), "*"):
        assert tp.tools_match([selector], step, None), selector
    assert not tp.tools_match(["route"], step, None)


def test_signal_confidence_explains_each_penalty():
    step = _steps()[1]  # depends on "quote"
    score, factors = tp.signal_confidence(step, tp.RunLog())
    assert score == 0.5 and "depends on 'quote', which has not run yet" in factors[0]

    log = tp.RunLog()
    log.record("quote", ok=False)
    score, factors = tp.signal_confidence(step, log, guardrail_blocks=2)
    assert score == pytest.approx(1 - 0.5 - 0.15 - 0.2)
    assert any("failed" in f for f in factors) and any("guardrail" in f for f in factors)

    log = tp.RunLog()
    log.record("quote", ok=True)
    assert tp.signal_confidence(step, log) == (1.0, ["no risk signals observed"])


def test_combined_confidence_is_the_lower_of_model_and_signals():
    policy = ToolPolicyConfig(confidence_threshold=0.6)
    step = _steps()[0]
    assert tp.compute_confidence(policy, step, tp.RunLog(), self_reported=0.3)["confidence"] == 0.3
    step2 = _steps()[1]
    assert tp.compute_confidence(policy, step2, tp.RunLog(), self_reported=0.95)["confidence"] == 0.5  # signals lower
    signals_only = ToolPolicyConfig(confidence_threshold=0.6, confidence_method="signals")
    assert tp.compute_confidence(signals_only, step, tp.RunLog(), self_reported=0.1)["confidence"] == 1.0


def _facts(step, **overrides):
    facts = tp.gather_facts(step=step, capability=None, tool_name=step.tool_name, arguments=step.arguments,
                            runlog=tp.RunLog(), confidence={"confidence": 0.9, "self_confidence": 0.9, "signal_confidence": 1.0},
                            tenant_id="acme", user_id="u")
    facts.update(overrides)
    return facts


def test_most_restrictive_matching_effect_wins_and_reasons_are_reported():
    step = _steps()[0]
    policy = ToolPolicyConfig(
        require_approval_before=["quote"],
        rules=[
            PolicyRule(name="ok", tools=["*"], effect=PolicyEffect.ALLOW),
            PolicyRule(name="too-many", tools=["quote"], effect=PolicyEffect.DENY, reason="max 3 per order",
                       when=[PolicyCondition(fact="args.quantity", op="gt", value=3)]),
        ],
    )
    decision = tp.evaluate(policy, step, None, _facts(step))
    assert decision.effect == PolicyEffect.DENY
    assert decision.rules == ["too-many"] and decision.reasons == ["max 3 per order"]

    step.arguments["quantity"] = 2
    decision = tp.evaluate(policy, step, None, _facts(step))
    assert decision.effect == PolicyEffect.REQUIRE_APPROVAL and decision.rules == ["require_approval_before"]


def test_default_deny_turns_rules_into_an_allowlist():
    quote, route = _steps()
    policy = ToolPolicyConfig(default_effect="deny", rules=[PolicyRule(name="only-quotes", tools=["quote"], effect=PolicyEffect.ALLOW)])
    assert tp.evaluate(policy, quote, None, _facts(quote)).effect == PolicyEffect.ALLOW
    assert tp.evaluate(policy, route, None, _facts(route)).effect == PolicyEffect.DENY


def test_time_regex_and_membership_conditions(monkeypatch):
    step = _steps()[0]
    monkeypatch.setattr(tp, "clock", lambda: datetime(2026, 10, 17, 23, 30, tzinfo=timezone.utc))  # a Saturday night
    after_hours = PolicyRule(name="after-hours", effect=PolicyEffect.REQUIRE_APPROVAL, when=[
        PolicyCondition(fact="hour_utc", op="gte", value=22)])
    weekend = PolicyRule(name="weekend", effect=PolicyEffect.REQUIRE_APPROVAL, when=[
        PolicyCondition(fact="weekday", op="in", value=["saturday", "sunday"])])
    sku = PolicyRule(name="sku3", effect=PolicyEffect.REQUIRE_APPROVAL, when=[
        PolicyCondition(fact="args.sku", op="matches", value=r"^SKU-\d$")])
    for rule in (after_hours, weekend, sku):
        policy = ToolPolicyConfig(rules=[rule])
        assert tp.evaluate(policy, step, None, tp.gather_facts(
            step=step, capability=None, tool_name="price_quote", arguments=step.arguments, runlog=tp.RunLog(),
            confidence={"confidence": 1, "self_confidence": None, "signal_confidence": 1}, tenant_id="acme", user_id="u",
        )).rules == [rule.name]
    # A comparison against a missing or incomparable fact simply doesn't hold -- no crash.
    odd = ToolPolicyConfig(rules=[PolicyRule(name="odd", effect=PolicyEffect.DENY, when=[
        PolicyCondition(fact="args.sku", op="gt", value=3)])])
    assert tp.evaluate(odd, step, None, _facts(step)).effect == PolicyEffect.ALLOW


@pytest.mark.parametrize(("policy", "fragment"), [
    (ToolPolicyConfig(require_approval_before=["qoute"]), "'qoute' matches no step"),
    (ToolPolicyConfig(rules=[PolicyRule(name="x", effect=PolicyEffect.DENY, when=[PolicyCondition(fact="nope", op="eq", value=1)])]), "unknown fact"),
    (ToolPolicyConfig(rules=[PolicyRule(name="x", effect=PolicyEffect.DENY, when=[PolicyCondition(fact="args.sku", op="matches", value="(")])]), "invalid regex"),
    (ToolPolicyConfig(rules=[PolicyRule(name="x", effect=PolicyEffect.DENY, when=[PolicyCondition(fact="attempt", op="gt", value="2")])]), "numeric value"),
    (ToolPolicyConfig(rules=[PolicyRule(name="x", effect=PolicyEffect.DENY), PolicyRule(name="x", effect=PolicyEffect.ALLOW)]), "duplicate rule name"),
], ids=["typo-selector", "unknown-fact", "bad-regex", "non-numeric-comparison", "duplicate-name"])
def test_invalid_policies_are_rejected(policy, fragment):
    with pytest.raises(ValueError, match=fragment):
        tp.validate_tool_policy(policy, _steps(), {})


# -- end to end ------------------------------------------------------------------


@pytest.fixture
async def caps(session):
    base_url = worker.ensure_running()
    await worker.register_all(session, _ID.tenant_id, base_url)
    await session.commit()
    return {c.id: c for c in await registry_store.list_for_tenant(session, _ID.tenant_id)}


async def _drive(agent, text: str, answers: list):
    """Run a conversation, answering each tool-confirmation pause with the
    next entry of `answers` (True/False, or an async callable returning
    one -- e.g. one that decides in the inbox first)."""
    runner = InMemoryRunner(agent=agent, app_name="policy-tests")
    session = await runner.session_service.create_session(app_name="policy-tests", user_id="u")
    message = types.Content(role="user", parts=[types.Part(text=text)])
    log = {"calls": [], "responses": {}, "pauses": []}
    for _ in range(10):
        pause = None
        async for event in runner.run_async(user_id="u", session_id=session.id, new_message=message):
            for part in (event.content.parts if event.content else None) or []:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    pause = part.function_call
                    log["pauses"].append(dict(part.function_call.args or {}))
                elif part.function_call:
                    log["calls"].append(part.function_call.name)
                if part.function_response and part.function_response.name != "adk_request_confirmation":
                    log["responses"][part.function_response.name] = part.function_response.response
        if pause is None or not answers:
            break
        answer = answers.pop(0)
        if callable(answer):
            answer = await answer(pause)
        message = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id=pause.id, name="adk_request_confirmation", response={"confirmed": answer}))])
    return log


async def _approval(session, approval_id):
    # The app's sessions use expire_on_commit=False, so this test session would
    # keep serving its cached copy; expire it to read what the run committed.
    await session.rollback()
    session.expire_all()
    return await approvals.get(session, approval_id)


def _agent(policy: ToolPolicyConfig, caps, *, confidence: float = 0.9, steps=None):
    return spec_agents.compile_spec_to_agent(_spec(policy, steps), caps, _ID, model=ScriptedToolCallingLlm(confidence=confidence))


@pytest.mark.asyncio
async def test_require_approval_before_a_tool_pauses_then_runs_after_chat_approval(session, caps):
    log = await _drive(_agent(ToolPolicyConfig(require_approval_before=["quote"]), caps), "go", [True])

    assert len(log["pauses"]) == 1
    hint = log["pauses"][0]["toolConfirmation"]["hint"]
    assert "Approval needed before step 'quote'" in hint and "configured to require approval" in hint
    quote = log["responses"]["step_quote"]
    assert quote["ok"] is True and quote["output"]["structuredContent"]["total"] == 29.0  # ran after approval
    assert log["responses"]["step_route"]["output"] == "team=billing"  # the rest of the run continued
    row = await _approval(session, quote["approval_id"])
    assert (row.status, row.decided_via) == (approvals.CONSUMED, "chat")


@pytest.mark.asyncio
async def test_rejecting_in_chat_stops_that_step_without_calling_the_tool(session, caps):
    log = await _drive(_agent(ToolPolicyConfig(require_approval_before=["quote"]), caps), "go", [False])
    quote = log["responses"]["step_quote"]
    assert quote["status"] == "rejected" and "output" not in quote
    assert (await _approval(session, quote["approval_id"])).status == approvals.DENIED


@pytest.mark.asyncio
async def test_inbox_channel_ignores_chat_confirmation_until_an_approver_decides(session, caps):
    policy = ToolPolicyConfig(require_approval_before=["quote"], approval_channel="inbox")

    # The person chatting "confirms" -- they're not allowed to decide, so the
    # step stays pending and says who has to.
    log = await _drive(_agent(policy, caps), "go", [True])
    assert len(log["pauses"]) == 1
    assert "only re-checks the inbox decision" in log["pauses"][0]["toolConfirmation"]["hint"]
    quote = log["responses"]["step_quote"]
    assert quote["status"] == "awaiting_approval" and "must be decided by an approver" in quote["message"]
    assert (await _approval(session, quote["approval_id"])).status == approvals.PENDING

    # An approver decides in the inbox; the next run of that step proceeds.
    await approvals.decide(session, quote["approval_id"], approve=True, decided_by="approver", via="inbox", note="ok")
    await session.commit()
    retry = await _drive(_agent(policy, caps), "retry", [])
    assert retry["pauses"] == [] and retry["responses"]["step_quote"]["ok"] is True
    row = await _approval(session, quote["approval_id"])
    assert (row.status, row.decided_by, row.decided_via) == (approvals.CONSUMED, "approver", "inbox")


@pytest.mark.asyncio
async def test_low_confidence_pauses_with_a_breakdown_high_confidence_runs(session, caps):
    policy = ToolPolicyConfig(confidence_threshold=0.6, confidence_tools=["quote"])
    unsure = await _drive(_agent(policy, caps, confidence=0.3), "go", [True])
    assert len(unsure["pauses"]) == 1
    hint = unsure["pauses"][0]["toolConfirmation"]["hint"]
    assert "confidence 0.30 is below the threshold 0.60" in hint and "model reported 0.30" in hint
    row = await _approval(session, unsure["responses"]["step_quote"]["approval_id"])
    assert row.confidence == 0.3 and row.rules == ["confidence_threshold"]

    sure = await _drive(_agent(policy, caps, confidence=0.95), "go", [])
    assert sure["pauses"] == [] and sure["responses"]["step_quote"]["ok"] is True
    assert sure["responses"]["step_quote"]["confidence"] == 0.95


@pytest.mark.asyncio
async def test_signal_confidence_gates_a_step_whose_dependency_failed(session, caps):
    """No self-report: the run's own evidence. 'route' depends on 'quote';
    quote is denied by policy, so route's signal confidence drops to 0.5."""
    policy = ToolPolicyConfig(
        confidence_threshold=0.6, confidence_tools=["route"], confidence_method="signals",
        rules=[PolicyRule(name="no-quotes-today", tools=["quote"], effect=PolicyEffect.DENY, reason="quotes are frozen")],
    )
    log = await _drive(_agent(policy, caps), "go", [False])
    assert log["responses"]["step_quote"]["status"] == "denied_by_policy"
    assert log["responses"]["step_quote"]["error"] == "quotes are frozen"
    hint = log["pauses"][0]["toolConfirmation"]["hint"]
    assert "depends on 'quote', which has not run yet" in hint and "below the threshold" in hint


@pytest.mark.asyncio
async def test_deny_rule_on_arguments_blocks_before_the_tool_is_called(caps):
    policy = ToolPolicyConfig(rules=[PolicyRule(name="bulk", tools=["quote"], effect=PolicyEffect.DENY, reason="bulk orders go through sales",
                                                when=[PolicyCondition(fact="args.quantity", op="gte", value=4)])])
    log = await _drive(_agent(policy, caps), "go", [])
    quote = log["responses"]["step_quote"]
    assert quote["status"] == "denied_by_policy" and quote["blocked_by"] == "policy:bulk" and quote["output"] is None


@pytest.mark.asyncio
async def test_platform_approval_rule_pauses_instead_of_hanging(session, caps, monkeypatch):
    """An in-code ApprovalRule used to make call_capability wait up to an
    hour on an in-memory future nobody in adk web could resolve."""
    monkeypatch.setattr(guardrail_policy, "_APPROVAL_RULES", {})
    guardrail_policy.register_approval_rule(guardrail_policy.ApprovalRule(
        tenant_id="*", capability_pattern="inventory", tool_pattern="price_quote", reason="pricing needs sign-off"))
    log = await _drive(_agent(ToolPolicyConfig(), caps), "go", [True])
    assert "pricing needs sign-off" in log["pauses"][0]["toolConfirmation"]["hint"]
    assert log["responses"]["step_quote"]["ok"] is True


@pytest.mark.asyncio
async def test_under_an_agent_tool_the_step_reports_awaiting_and_runs_once_after_approval(session, caps):
    """The supervisor calls specs through AgentTool, which doesn't surface a
    nested confirmation -- so the step reports awaiting_approval with an id;
    approve it, run again, and it proceeds exactly once."""
    tool = spec_agents.compile_spec_to_tool(_spec(ToolPolicyConfig(require_approval_before=["quote"])), caps, _ID,
                                            model=ScriptedToolCallingLlm())
    supervisor = LlmAgent(name="sup", model=ScriptedToolCallingLlm(), instruction="run it", tools=[tool])

    first = await _drive(supervisor, "go", [])
    pending = await approvals.list_for_tenant(session, _ID.tenant_id, status="pending")
    assert len(pending) == 1 and pending[0].step_id == "quote" and first["pauses"] == []

    await approvals.decide(session, pending[0].id, approve=True, decided_by="boss", via="chat")
    await session.commit()
    await _drive(supervisor, "go again", [])
    assert (await _approval(session, pending[0].id)).status == approvals.CONSUMED

    await _drive(supervisor, "and again", [])
    assert len(await approvals.list_for_tenant(session, _ID.tenant_id, status="pending")) == 1  # single use


@pytest.mark.asyncio
async def test_approval_rejects_a_policy_with_a_typo_before_saving(session, caps):
    plan = ExecutionPlan(plan_id="p", goal="g", steps=_steps())
    with pytest.raises(ValueError, match="'qoute' matches no step"):
        await specs_service.approve_plan(session, _ID, name="typo", plan=plan,
                                         runtime=AgentRuntimeConfig(tool_policy=ToolPolicyConfig(require_approval_before=["qoute"])))


@pytest.mark.asyncio
async def test_inbox_api_lists_and_decides_tenant_scoped(session, caps):
    from app.main import app

    tool = spec_agents.compile_spec_to_agent(_spec(ToolPolicyConfig(require_approval_before=["quote"])), caps, _ID).tools[0]
    waiting = await tool.func()
    assert waiting["status"] == "awaiting_approval"

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async def token(tenant):
            r = await client.post("/auth/dev-token", json={"tenant_id": tenant, "user_id": "approver", "scopes": ["*"]})
            return {"Authorization": f"Bearer {r.json()['access_token']}"}

        acme, globex = await token("acme"), await token("globex")
        listed = (await client.get("/approvals", headers=acme)).json()
        assert [a["approval_id"] for a in listed] == [waiting["approval_id"]]
        assert listed[0]["reasons"] == ["configured to require approval before 'quote'"]
        assert (await client.get("/approvals", headers=globex)).json() == []
        assert (await client.post(f"/approvals/{waiting['approval_id']}/decide", json={"approve": True}, headers=globex)).status_code == 404

        decided = await client.post(f"/approvals/{waiting['approval_id']}/decide", json={"approve": True, "note": "fine"}, headers=acme)
        assert decided.json()["status"] == "approved" and decided.json()["decided_via"] == "inbox"
        again = await client.post(f"/approvals/{waiting['approval_id']}/decide", json={"approve": False}, headers=acme)
        assert again.status_code == 409

    ran = await tool.func()
    assert ran["ok"] is True and ran["approval_id"] == waiting["approval_id"]
