"""Regression suite for the test worker (`app/testing/worker.py`): every MCP
tool and every A2A skill in `app/testing/worker_catalog.py`, called over its
real protocol through this project's real client code, checked against the
catalog's own expected output -- then registration into the registry, and a
compiled multi-tool agent whose steps span several MCP servers and A2A
agents, run step by step through `tool_wrapper.call_capability`.

Nothing here is mocked: MCP calls go through ADK's `McpToolset` over SSE to
real FastMCP servers; A2A calls go through `agent_card_adapter` (card fetch,
then JSON-RPC `message/send`). Adding a catalog entry adds coverage here
automatically, because the tests are parametrized over the catalog.

    pytest tests/test_worker_regression.py -v
"""

from __future__ import annotations

import httpx
import pytest

from app.auth.models import IdentityContext
from app.factory import spec_agents
from app.planner.schemas import ExecutionPlan, PlanStep, StepKind
from app.registry import store as registry_store
from app.registry.adapters import agent_card_adapter
from app.registry.schemas import CapabilityKind
from app.specs import service as specs_service
from app.testing import worker
from app.testing.worker_catalog import A2A_AGENTS, MCP_SERVERS

_IDENTITY = IdentityContext(tenant_id="acme", brand_id="acme-brand", user_id="worker-regression", scopes=("*",))


@pytest.fixture(scope="module")
def base_url() -> str:
    return worker.ensure_running()


def test_manifest_exposes_at_least_twenty_mcp_servers_and_twenty_a2a_agents(base_url):
    manifest = httpx.get(f"{base_url}/").json()
    assert manifest["counts"]["mcp_servers"] >= 20
    assert manifest["counts"]["a2a_agents"] >= 20
    assert manifest["counts"]["mcp_tools"] == sum(len(s.tools) for s in MCP_SERVERS)

    for server in manifest["mcp_servers"]:
        assert server["sse_url"] == f"{base_url}/mcp/{server['name']}/sse"
        for tool in server["tools"]:
            # FastMCP derived a real JSON schema from the function signature.
            assert tool["input_schema"]["type"] == "object"
    for agent in manifest["a2a_agents"]:
        card = httpx.get(agent["agent_card_url"]).json()
        assert card["name"] == agent["name"]
        assert card["url"] == agent["rpc_url"]
        assert {s["id"] for s in card["skills"]} == {s["id"] for s in agent["skills"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("server_def", MCP_SERVERS, ids=[s.name for s in MCP_SERVERS])
async def test_every_mcp_tool_over_real_sse(base_url, server_def):
    from google.adk.tools.mcp_tool import McpToolset, SseConnectionParams

    toolset = McpToolset(connection_params=SseConnectionParams(url=f"{base_url}/mcp/{server_def.name}/sse"))
    try:
        tools = {t.name: t for t in await toolset.get_tools()}
        assert set(tools) == {t.name for t in server_def.tools}
        for tool_def in server_def.tools:
            result = await tools[tool_def.name].run_async(args=tool_def.example_args, tool_context=None)
            assert result["isError"] is False, result
            assert result["structuredContent"] == tool_def.expected
    finally:
        await toolset.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_def", A2A_AGENTS, ids=[a.name for a in A2A_AGENTS])
async def test_every_a2a_skill_over_real_json_rpc(base_url, agent_def):
    card = await agent_card_adapter.fetch_agent_card(f"{base_url}/a2a/{agent_def.name}")
    capability = agent_card_adapter.capability_from_agent_card(card, tenant_id=_IDENTITY.tenant_id)
    for skill in agent_def.skills:
        output = await agent_card_adapter.invoke_remote_skill(capability, skill_id=skill.id, input_text=skill.example_input)
        assert output == skill.expected


@pytest.mark.asyncio
async def test_a2a_unknown_skill_is_a_json_rpc_error(base_url):
    resp = httpx.post(
        f"{base_url}/a2a/faq-agent",
        json={"jsonrpc": "2.0", "id": "1", "method": "message/send",
              "params": {"message": {"role": "user", "parts": [{"kind": "text", "text": "hi"}], "metadata": {"skill_id": "nope"}}}},
    )
    assert resp.json()["error"]["code"] == -32602


@pytest.mark.asyncio
async def test_register_all_is_idempotent_and_uses_real_onboarding(session, base_url):
    first = await worker.register_all(session, _IDENTITY.tenant_id, base_url)
    await session.commit()
    second = await worker.register_all(session, _IDENTITY.tenant_id, base_url)
    await session.commit()

    assert len(first) == len(second) == len(MCP_SERVERS) + len(A2A_AGENTS)
    stored = await registry_store.list_for_tenant(session, _IDENTITY.tenant_id)
    assert len(stored) == len(first)  # same stable ids -> updated in place, no duplicates

    by_id = {c.id: c for c in stored}
    math_cap = by_id[worker.mcp_capability_id("math-tools")]
    assert math_cap.kind == CapabilityKind.MCP
    assert math_cap.endpoint_ref.sse_url == f"{base_url}/mcp/math-tools/sse"
    # Input schemas were introspected over the wire, not hand-written.
    add_schema = next(t for t in math_cap.tools if t.name == "add").input_schema
    assert add_schema["required"] == ["a", "b"]

    triage_cap = by_id[worker.a2a_capability_id("ticket-triage-agent")]
    assert triage_cap.kind == CapabilityKind.AGENT_CARD
    assert {t.name for t in triage_cap.tools} == {"triage_ticket", "route_ticket"}


@pytest.mark.asyncio
async def test_worker_capabilities_are_registered_publicly(session, base_url):
    """Public = marketplace visibility: any tenant can discover every worker
    capability and request access, but -- like any marketplace capability
    -- can't bind a plan to one until a grant is approved. Only the owner
    can use them immediately."""
    from app.marketplace import service as marketplace_service
    from app.registry.schemas import Visibility

    saved = await worker.register_all(session, _IDENTITY.tenant_id, base_url)
    await session.commit()
    assert {c.visibility for c in saved} == {Visibility.MARKETPLACE}

    discovered = {r.capability.id for r in await registry_store.discover_marketplace(session, limit=1000)}
    assert {c.id for c in saved} <= discovered

    # Another tenant: discoverable, not usable until granted.
    assert await registry_store.list_for_tenant(session, "globex") == []
    math_id = worker.mcp_capability_id("math-tools")
    grant = await marketplace_service.request_grant(
        session, capability_id=math_id, grantor_tenant_id=_IDENTITY.tenant_id, grantee_tenant_id="globex"
    )
    await marketplace_service.decide_grant(session, grant.id, approve=True)
    await session.commit()
    granted = await marketplace_service.granted_capability_ids(session, "globex")
    usable = await registry_store.list_for_tenant(session, "globex", granted_capability_ids=granted)
    assert [c.id for c in usable] == [math_id]


def _multi_tool_plan() -> ExecutionPlan:
    """Six steps, four MCP servers, two A2A agents, typed args, a tool past
    the first on a multi-tool server, and a dependency chain."""
    mcp = worker.mcp_capability_id
    a2a = worker.a2a_capability_id
    return ExecutionPlan(
        plan_id="plan_worker_multi_tool",
        goal="Quote an order, check stock, convert the delivery date, and handle the customer's ticket",
        steps=[
            PlanStep(step_id="quote", kind=StepKind.TOOL_CALL, capability_id=mcp("inventory-tools"),
                     tool_name="price_quote", arguments={"sku": "SKU-3", "quantity": 4},
                     instruction="Quote 4 units of SKU-3"),
            PlanStep(step_id="stock", kind=StepKind.TOOL_CALL, capability_id=mcp("inventory-tools"),
                     tool_name="stock_level", arguments={"sku": "SKU-3"},
                     instruction="Check SKU-3 stock", depends_on=["quote"]),
            PlanStep(step_id="eta", kind=StepKind.TOOL_CALL, capability_id=mcp("date-tools"),
                     tool_name="add_days", arguments={"date_iso": "2026-10-16", "days": 5},
                     instruction="Delivery is 5 days out", depends_on=["stock"]),
            PlanStep(step_id="total", kind=StepKind.TOOL_CALL, capability_id=mcp("math-tools"),
                     tool_name="multiply", arguments={"a": 7.25, "b": 4},
                     instruction="Double-check the quote total"),
            PlanStep(step_id="triage", kind=StepKind.TOOL_CALL, capability_id=a2a("ticket-triage-agent"),
                     tool_name="route_ticket", instruction="Customer says: my invoice payment failed"),
            PlanStep(step_id="reply", kind=StepKind.TOOL_CALL, capability_id=a2a("email-drafter-agent"),
                     instruction="Your order of 4 Doohickeys ships on 2026-10-21.", depends_on=["triage", "eta"]),
        ],
    )


@pytest.mark.asyncio
async def test_compiled_multi_tool_agent_runs_every_step_against_the_worker(session, base_url):
    await worker.register_all(session, _IDENTITY.tenant_id, base_url)
    await session.commit()

    plan = _multi_tool_plan()
    spec = await specs_service.approve_plan(session, _IDENTITY, name="worker-multi-tool-agent", plan=plan)
    assert len(spec.capability_snapshots) == 5  # inventory, date, math, triage, email-drafter

    capabilities = {c.id: c for c in await registry_store.list_for_tenant(session, _IDENTITY.tenant_id)}
    agent = spec_agents.compile_spec_to_agent(spec, capabilities, _IDENTITY)
    tools = {t.name: t for t in agent.tools}
    assert len(tools) == 6

    results = {}
    for step in plan.steps:  # already in dependency order
        result = await tools[f"step_{step.step_id}"].func()
        assert result["ok"] is True, result
        assert result["tool_name"] == (step.tool_name or result["tool_name"])
        results[step.step_id] = result["output"]

    assert results["quote"]["structuredContent"] == {"sku": "SKU-3", "quantity": 4, "unit_price": 7.25, "total": 29.0}
    assert results["stock"]["structuredContent"]["in_stock"] is True
    assert results["eta"]["structuredContent"] == {"date": "2026-10-21"}
    assert results["total"]["structuredContent"] == {"result": 29.0}
    assert results["triage"] == "team=billing"
    assert results["reply"] == "Hi there,\n\nYour order of 4 Doohickeys ships on 2026-10-21.\n\nBest regards,\nThe Team"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("step_kwargs", "fragment"),
    [
        ({"tool_name": "divide", "arguments": {"a": 1, "b": 2}}, "has no tool 'divide'"),
        ({"tool_name": "add", "arguments": {"a": 1}}, "'b' is a required property"),
        ({"tool_name": "add", "arguments": {"a": "one", "b": 2}}, "invalid a"),
        ({"tool_name": "add"}, "no arguments set"),
    ],
    ids=["unknown-tool", "missing-arg", "wrong-type", "typed-tool-without-arguments"],
)
async def test_approval_rejects_steps_that_could_never_run(session, base_url, step_kwargs, fragment):
    await worker.register_all(session, _IDENTITY.tenant_id, base_url)
    await session.commit()
    plan = ExecutionPlan(
        plan_id="plan_bad_step",
        goal="call math-tools wrongly",
        steps=[PlanStep(step_id="bad", kind=StepKind.TOOL_CALL, capability_id=worker.mcp_capability_id("math-tools"),
                        instruction="do math", **step_kwargs)],
    )
    with pytest.raises(ValueError, match=fragment):
        await specs_service.approve_plan(session, _IDENTITY, name="bad-step-agent", plan=plan)


@pytest.mark.asyncio
async def test_a_tool_removed_after_approval_fails_that_step_cleanly_at_run_time(session, base_url):
    """Approval checked the tool existed; the capability can still change
    afterwards. The run-time check reports it instead of crashing."""
    await worker.register_all(session, _IDENTITY.tenant_id, base_url)
    await session.commit()
    plan = ExecutionPlan(
        plan_id="plan_tool_removed",
        goal="multiply",
        steps=[PlanStep(step_id="mul", kind=StepKind.TOOL_CALL, capability_id=worker.mcp_capability_id("math-tools"),
                        tool_name="multiply", arguments={"a": 2, "b": 3}, instruction="multiply")],
    )
    spec = await specs_service.approve_plan(session, _IDENTITY, name="tool-removed-agent", plan=plan)

    capabilities = {c.id: c for c in await registry_store.list_for_tenant(session, _IDENTITY.tenant_id)}
    math_cap = capabilities[worker.mcp_capability_id("math-tools")]
    math_cap.tools = [t for t in math_cap.tools if t.name != "multiply"]  # the server dropped it since approval
    agent = spec_agents.compile_spec_to_agent(spec, capabilities, _IDENTITY)
    result = await agent.tools[0].func()
    assert result["ok"] is False
    assert "multiply" in result["error"]
