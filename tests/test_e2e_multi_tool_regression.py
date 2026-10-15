"""The one worker that runs a diverse, genuinely multi-tool agent
end to end, against real backends of every capability kind this project
supports -- OPENAPI, MCP, and AGENT_CARD (see `app/registry/schemas.py`'s
`CapabilityKind`) -- so compiling and running an agent that spans several
different kinds of tool is covered by an actual regression test, not just
reasoned about.

Three real local servers back the three capabilities (`app/testing/`):
  - `echo_server`       -- OPENAPI: a plain REST backend, hit directly.
  - `mcp_demo_server`   -- MCP: a real FastMCP server speaking the genuine
                           MCP wire protocol over SSE, itself forwarding to
                           `echo_server` underneath (the "OpenAPI -> MCP"
                           translation `openapi_adapter.build_mcp_app`
                           exists for).
  - `agent_card_server` -- AGENT_CARD: a toy remote "agent" serving its own
                           `/.well-known/agent-card.json` and a minimal A2A
                           `message/send` JSON-RPC endpoint.

None of these are stubs or monkeypatches of `_dispatch` -- every step tool
call below goes through the real `app/factory/tool_wrapper.call_capability`,
which really dispatches by `capability.kind` to the real adapter, which
really talks to one of these three servers over its real protocol. The only
thing not exercised here is the live LLM itself (same `_force_offline_planner`
fixture as everywhere else keeps this network-free and deterministic) --
`tests/test_supervisor_agent.py` already covers the model-facing
plan -> approve -> refresh -> run loop; this file is the complementary
"does a compiled multi-kind agent actually run its steps for real" check.

Run just this file to regression-test the whole multi-tool compile+run
pattern on its own: `pytest tests/test_e2e_multi_tool_regression.py -v`.
"""

from __future__ import annotations

import pytest

from app.auth.models import IdentityContext
from app.factory import spec_agents
from app.planner.schemas import ExecutionPlan, PlanStep, StepKind
from app.registry import store as registry_store
from app.specs import service as specs_service
from app.testing import agent_card_server, echo_server, mcp_demo_server

_IDENTITY = IdentityContext(tenant_id="acme", brand_id="acme-brand", user_id="e2e-worker", scopes=("*",))


async def _register_all_capabilities(session):
    """Starts all three demo backends and registers one capability per kind
    against the real registry DB -- the same `registry_store.save` every
    other capability in this project goes through, not a shortcut."""
    echo_server.ensure_running()
    mcp_demo_server.ensure_running()
    agent_card_server.ensure_running()

    openapi_cap = echo_server.capability_for(_IDENTITY.tenant_id)
    mcp_cap = mcp_demo_server.capability_for(_IDENTITY.tenant_id)
    agent_card_cap = await agent_card_server.capability_for(_IDENTITY.tenant_id)

    for cap in (openapi_cap, mcp_cap, agent_card_cap):
        await registry_store.save(session, cap)
    await session.commit()

    return openapi_cap, mcp_cap, agent_card_cap


def _build_plan(openapi_cap, mcp_cap, agent_card_cap) -> ExecutionPlan:
    """A genuinely multi-tool plan: three steps, three different capability
    kinds, chained with `depends_on` so step ordering is exercised too --
    not just "three independent tools that happen to be attached to one
    agent"."""
    return ExecutionPlan(
        plan_id="plan_e2e_multi_tool",
        goal="ingest content, summarize it over MCP, then have the remote agent draft a reply",
        steps=[
            PlanStep(
                step_id="step_ingest",
                kind=StepKind.TOOL_CALL,
                capability_id=openapi_cap.id,
                instruction="Ingest the raw content",
            ),
            PlanStep(
                step_id="step_summarize",
                kind=StepKind.TOOL_CALL,
                capability_id=mcp_cap.id,
                instruction="Summarize the ingested content",
                depends_on=["step_ingest"],
            ),
            PlanStep(
                step_id="step_draft_reply",
                kind=StepKind.TOOL_CALL,
                capability_id=agent_card_cap.id,
                instruction="Draft a reply based on the summary",
                depends_on=["step_summarize"],
            ),
        ],
    )


@pytest.mark.asyncio
async def test_compile_and_run_a_genuinely_multi_tool_multi_kind_agent(session):
    openapi_cap, mcp_cap, agent_card_cap = await _register_all_capabilities(session)
    plan = _build_plan(openapi_cap, mcp_cap, agent_card_cap)

    # Approve it for real -- capability snapshotting, instruction building
    # (and its no-live-model fallback), versioning: the same path
    # POST /agent-specs/approve and the supervisor's approve_last_plan both
    # go through (app/specs/service.py).
    spec = await specs_service.approve_plan(
        session, _IDENTITY, name="multi-tool-e2e-agent", plan=plan, author_notes="be concise at every step"
    )
    assert set(spec.capability_snapshots.keys()) == {openapi_cap.id, mcp_cap.id, agent_card_cap.id}
    assert spec.capability_snapshots[mcp_cap.id].kind.value == "mcp"
    assert spec.capability_snapshots[agent_card_cap.id].kind.value == "agent_card"
    assert spec.capability_snapshots[openapi_cap.id].kind.value == "openapi"

    # Compile it -- the same compile step the supervisor uses to turn an
    # approved spec into a real, callable sub-agent.
    capabilities = {openapi_cap.id: openapi_cap, mcp_cap.id: mcp_cap, agent_card_cap.id: agent_card_cap}
    agent = spec_agents.compile_spec_to_agent(spec, capabilities, _IDENTITY)
    assert agent.output_schema is None
    assert agent.planner is None
    assert {t.name for t in agent.tools} == {"step_step_ingest", "step_step_summarize", "step_step_draft_reply"}

    # Also confirm it wraps into a callable AgentTool, exactly as
    # compile_spec_to_tool hands the supervisor -- not exercising the
    # wrapping here would miss a regression in that one extra layer.
    tool = spec_agents.compile_spec_to_tool(spec, capabilities, _IDENTITY)
    assert tool.agent is agent or tool.agent.name == agent.name

    # Run every step for real, in dependency order, through the real
    # tool_wrapper.call_capability path against the three real backends.
    tools_by_name = {t.name: t for t in agent.tools}

    ingest_result = await tools_by_name["step_step_ingest"].func()
    assert ingest_result["ok"] is True
    assert ingest_result["output"]["path"] == "/ingest_content"

    summarize_result = await tools_by_name["step_step_summarize"].func()
    assert summarize_result["ok"] is True
    # The MCP tool call round-tripped through FastMCP -> echo_server for
    # real. An MCP tool result comes back in MCP's own envelope
    # (`content` / `structuredContent` / `isError`), with echo_server's JSON
    # inside `structuredContent` -- that envelope is itself proof this went
    # over the real MCP protocol rather than straight to the REST backend.
    assert summarize_result["output"]["isError"] is False
    structured = summarize_result["output"]["structuredContent"]
    assert structured["path"] == "/summarize_text"
    assert structured["received"]["instruction"] == "Summarize the ingested content"

    draft_result = await tools_by_name["step_step_draft_reply"].func()
    assert draft_result["ok"] is True
    assert "demo-remote-agent" in draft_result["output"]
    assert "Draft a reply based on the summary" in draft_result["output"]


@pytest.mark.asyncio
async def test_a_step_with_a_revoked_capability_fails_that_step_only(session):
    """A capability can vanish between approval and a later compile (see
    `_capabilities_for_spec` in the supervisor and
    `compile_spec_to_agent`'s own docstring) -- confirm that's still true in
    a multi-kind plan: only the step missing its capability fails, the
    others still run against their real backends."""
    openapi_cap, mcp_cap, agent_card_cap = await _register_all_capabilities(session)
    plan = _build_plan(openapi_cap, mcp_cap, agent_card_cap)
    spec = await specs_service.approve_plan(session, _IDENTITY, name="multi-tool-e2e-agent-2", plan=plan)

    # Simulate the MCP capability having been revoked since approval: it's
    # simply missing from the live-resolved capabilities map passed to
    # compile, same as `_capabilities_for_spec` would produce.
    capabilities = {openapi_cap.id: openapi_cap, agent_card_cap.id: agent_card_cap}
    agent = spec_agents.compile_spec_to_agent(spec, capabilities, _IDENTITY)
    tools_by_name = {t.name: t for t in agent.tools}

    ingest_result = await tools_by_name["step_step_ingest"].func()
    assert ingest_result["ok"] is True

    summarize_result = await tools_by_name["step_step_summarize"].func()
    assert summarize_result["ok"] is False
    assert mcp_cap.id in summarize_result["error"]

    draft_result = await tools_by_name["step_step_draft_reply"].func()
    assert draft_result["ok"] is True
