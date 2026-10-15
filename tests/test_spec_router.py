"""`app/factory/spec_router.py` + `agents/spec_test_agent/agent.py`: every
approved spec mounted as a real ADK sub-agent under one router.

The end-to-end tests run through ADK's real runner with a scripted model:
the router transfers to a spec, the spec runs its own pattern (its steps
really calling the test worker's MCP servers and A2A agents), and the next
message comes back to the router, which can then transfer to a different
spec -- multi-turn, one session.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.flows.llm_flows.extensions._agent_transfer import _get_transfer_targets
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

from app.auth.models import IdentityContext
from app.config import settings
from app.factory import spec_router
from app.planner.schemas import PlanStep, StepKind
from app.registry import store as registry_store
from app.specs import store as specs_store
from app.specs.schemas import AgentPattern, AgentRuntimeConfig, AgentSpec, CallbackConfig, CallbackHook
from app.testing import worker
from app.testing.scripted_llm import ScriptedToolCallingLlm

_IDENTITY = IdentityContext(tenant_id="acme", brand_id="acme-brand", user_id="router-tests", scopes=("*",))


def _spec(spec_id: str, name: str, pattern: AgentPattern, steps: list[PlanStep], **runtime) -> AgentSpec:
    return AgentSpec(
        spec_id=spec_id,
        tenant_id=_IDENTITY.tenant_id,
        name=name,
        goal=f"{name} goal",
        source_plan_id=f"plan_{spec_id}",
        steps=steps,
        approved_by=_IDENTITY.user_id,
        runtime=AgentRuntimeConfig(pattern=pattern, **runtime),
    )


def _quote_spec() -> AgentSpec:
    mcp = worker.mcp_capability_id
    return _spec("spec_quote00000001", "order-quote", AgentPattern.SEQUENTIAL, [
        PlanStep(step_id="quote", kind=StepKind.TOOL_CALL, capability_id=mcp("inventory-tools"),
                 tool_name="price_quote", arguments={"sku": "SKU-3", "quantity": 4}, instruction="quote"),
        PlanStep(step_id="double", kind=StepKind.TOOL_CALL, capability_id=mcp("math-tools"),
                 tool_name="multiply", arguments={"a": 29, "b": 2}, instruction="double", depends_on=["quote"]),
    ])


def _support_spec() -> AgentSpec:
    a2a = worker.a2a_capability_id
    return _spec("spec_support000002", "support-desk", AgentPattern.SINGLE, [
        PlanStep(step_id="route", kind=StepKind.TOOL_CALL, capability_id=a2a("ticket-triage-agent"),
                 tool_name="route_ticket", instruction="my invoice payment failed"),
        PlanStep(step_id="reply", kind=StepKind.TOOL_CALL, capability_id=a2a("email-drafter-agent"),
                 instruction="We've passed this to billing.", depends_on=["route"]),
    ])


@pytest.fixture
async def visible(session):
    base_url = worker.ensure_running()
    await worker.register_all(session, _IDENTITY.tenant_id, base_url)
    await session.commit()
    return {c.id: c for c in await registry_store.list_for_tenant(session, _IDENTITY.tenant_id)}


def _router(agents, *, transfer_to: list[str]):
    state = {"loaded": {}, "skipped": {}}
    return LlmAgent(
        name=spec_router.ROUTER_NAME,
        model=ScriptedToolCallingLlm(transfer_to=transfer_to),
        instruction=spec_router.RouterInstruction(state),
        sub_agents=agents,
    )


async def _turns(agent, messages: list[str]):
    """Run several user turns in one session; per turn, return (authors that
    produced events, function-call names, function responses)."""
    runner = InMemoryRunner(agent=agent, app_name="router-tests")
    session = await runner.session_service.create_session(app_name="router-tests", user_id="u")
    out = []
    for text in messages:
        authors, calls, responses = [], [], {}
        async for event in runner.run_async(user_id="u", session_id=session.id,
                                            new_message=types.Content(role="user", parts=[types.Part(text=text)])):
            if event.author and event.author not in authors:
                authors.append(event.author)
            for part in (event.content.parts if event.content else None) or []:
                if part.function_call:
                    calls.append(part.function_call.name)
                if part.function_response:
                    responses[part.function_response.name] = part.function_response.response
        out.append((authors, calls, responses))
    return out


def test_compile_sub_agents_isolates_each_spec_and_keeps_its_pattern(visible):
    agents, loaded, skipped = spec_router.compile_sub_agents(
        [_quote_spec(), _support_spec()], visible, _IDENTITY, model=ScriptedToolCallingLlm())
    assert skipped == {}
    assert [type(a) for a in agents] == [SequentialAgent, LlmAgent]
    assert {info["pattern"] for info in loaded.values()} == {"sequential", "single"}

    router = _router(agents, transfer_to=[])
    assert [a.parent_agent for a in agents] == [router, router]
    # The router can transfer to both specs...
    assert {t.name for t in _get_transfer_targets(router)} == {a.name for a in agents}
    # ...but a mounted spec can't transfer anywhere: not back up, not sideways
    # into another spec, and its own step agents can't either.
    support = agents[1]
    assert support.disallow_transfer_to_parent and support.disallow_transfer_to_peers
    assert _get_transfer_targets(support) == []
    assert all(_get_transfer_targets(step) == [] for step in agents[0].sub_agents)


@pytest.mark.asyncio
async def test_router_transfers_to_each_spec_across_turns_and_each_runs_for_real(visible):
    agents, _, _ = spec_router.compile_sub_agents(
        [_quote_spec(), _support_spec()], visible, _IDENTITY, model=ScriptedToolCallingLlm())
    quote_agent, support_agent = agents
    router = _router(agents, transfer_to=[quote_agent.name, support_agent.name])

    (t1_authors, t1_calls, t1_resp), (t2_authors, t2_calls, t2_resp) = await _turns(
        router, ["Quote 4 doohickeys and double it.", "Now handle my billing ticket."])

    # Turn 1: router -> quote spec (sequential: one step agent per step).
    assert t1_authors[0] == spec_router.ROUTER_NAME
    assert t1_calls == ["transfer_to_agent", "step_quote", "step_double"]
    assert t1_resp["step_quote"]["output"]["structuredContent"]["total"] == 29.0
    assert t1_resp["step_double"]["output"]["structuredContent"] == {"result": 58.0}

    # Turn 2 starts at the router again (the spec gave control back), which
    # transfers to the other spec -- an A2A-backed single-pattern agent.
    assert t2_authors[0] == spec_router.ROUTER_NAME
    assert t2_calls == ["transfer_to_agent", "step_route", "step_reply"]
    assert t2_resp["step_route"]["output"] == "team=billing"
    assert t2_resp["step_reply"]["output"].startswith("Hi there,")


@pytest.mark.asyncio
async def test_a_spec_keeps_its_own_callbacks_when_mounted(visible):
    guarded = _spec("spec_guarded000003", "guarded", AgentPattern.SINGLE, _support_spec().steps, callbacks=[
        CallbackConfig(hook=CallbackHook.BEFORE_MODEL, kind="block_keywords", params={"keywords": ["wire transfer"]}),
    ])
    agents, _, _ = spec_router.compile_sub_agents([guarded], visible, _IDENTITY, model=ScriptedToolCallingLlm())
    router = _router(agents, transfer_to=[agents[0].name])
    [(_, calls, _)] = await _turns(router, ["Handle this and also do a wire transfer."])
    assert calls == ["transfer_to_agent"]  # transferred, then the spec's own guardrail stopped it before any step


def test_a_spec_that_cannot_compile_is_skipped_not_fatal(visible, monkeypatch):
    custom = _spec("spec_custom0000004", "needs-custom", AgentPattern.SINGLE, _support_spec().steps, callbacks=[
        CallbackConfig(hook=CallbackHook.AFTER_MODEL, kind="custom",
                       params={"path": "app.callbacks.examples:append_disclaimer", "params": {"text": "x"}}),
    ])
    monkeypatch.setattr(settings, "allow_custom_callbacks", False)
    agents, loaded, skipped = spec_router.compile_sub_agents([custom, _quote_spec()], visible, _IDENTITY)
    assert [info["spec_name"] for info in loaded.values()] == ["order-quote"]
    assert "custom callbacks are disabled" in skipped["spec_custom0000004"]


def test_mount_sub_agents_swaps_the_tree_in_place(visible):
    first, _, _ = spec_router.compile_sub_agents([_quote_spec()], visible, _IDENTITY)
    router = _router(first, transfer_to=[])
    second, _, _ = spec_router.compile_sub_agents([_quote_spec(), _support_spec()], visible, _IDENTITY)
    spec_router.mount_sub_agents(router, second)
    assert router.sub_agents == second
    assert all(a.parent_agent is router for a in second)
    assert router.find_agent(second[1].name) is second[1]
    assert router.find_agent(first[0].name) is not first[0]  # the old instance is gone from the tree


def test_filter_specs_by_name_or_id():
    specs = [_quote_spec(), _support_spec()]
    assert [s.name for s in spec_router.filter_specs(specs, None)] == ["order-quote", "support-desk"]
    assert [s.name for s in spec_router.filter_specs(specs, {"support-desk"})] == ["support-desk"]
    assert [s.name for s in spec_router.filter_specs(specs, {"spec_quote00000001"})] == ["order-quote"]


# -- the adk web app module itself ----------------------------------------------


def _import_fresh_app():
    sys.modules.pop("agents.spec_test_agent.agent", None)
    sys.modules.pop("agents.spec_test_agent", None)
    return importlib.import_module("agents.spec_test_agent.agent")


@pytest.mark.asyncio
async def test_adk_web_app_mounts_every_approved_spec_and_reloads_new_ones(visible, monkeypatch):
    monkeypatch.delenv("TEST_SPECS", raising=False)
    monkeypatch.delenv("TEST_SPEC_NAME", raising=False)
    monkeypatch.delenv("TEST_SPEC_ID", raising=False)
    await specs_store.save_spec(_quote_spec())

    app = _import_fresh_app()
    assert [a.name for a in app.root_agent.sub_agents] == ["spec_order_quote_000001"]
    assert "order-quote" in app.root_agent.instruction
    listed = await app.list_loaded_specs()
    assert listed["loaded"][0]["pattern"] == "sequential" and listed["skipped"] == {}

    # Approved after startup: picked up by reload_specs, no restart.
    await specs_store.save_spec(_support_spec())
    reloaded = await app.reload_specs()
    assert sorted(reloaded["mounted"]) == ["spec_order_quote_000001", "spec_support_desk_000002"]
    assert "support-desk" in app.root_agent.instruction
    assert all(a.parent_agent is app.root_agent for a in app.root_agent.sub_agents)

    # And the reloaded tree runs: route to the newly mounted spec.
    app.root_agent.model = ScriptedToolCallingLlm(transfer_to=["spec_support_desk_000002"])
    for agent in app.root_agent.sub_agents:
        for node in [agent, *getattr(agent, "sub_agents", [])]:
            if isinstance(node, LlmAgent):
                node.model = ScriptedToolCallingLlm()
    app.root_agent.tools[:] = [t for t in app.root_agent.tools if not isinstance(t, FunctionTool)]  # router: transfer only
    [(_, calls, responses)] = await _turns(app.root_agent, ["Handle my billing ticket."])
    assert calls == ["transfer_to_agent", "step_route", "step_reply"]
    assert responses["step_route"]["output"] == "team=billing"


@pytest.mark.asyncio
async def test_adk_web_app_honors_the_spec_filter(visible, monkeypatch):
    await specs_store.save_spec(_quote_spec())
    await specs_store.save_spec(_support_spec())
    monkeypatch.setenv("TEST_SPECS", "support-desk")
    monkeypatch.delenv("TEST_SPEC_NAME", raising=False)
    monkeypatch.delenv("TEST_SPEC_ID", raising=False)
    app = _import_fresh_app()
    assert [a.name for a in app.root_agent.sub_agents] == ["spec_support_desk_000002"]
