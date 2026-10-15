"""End-to-end exercise of the single supervisor agent's own
plan -> approve -> refresh -> run loop (`agents/capability_marketplace/agent.py`)
-- the same thing `app/routers/agent_specs.py` exposes over HTTP, but
reachable directly from an `adk web` chat with this one agent, via the
`approve_last_plan` tool and `app/specs/service.py`'s shared approval logic.

Doesn't go through a live model -- the autouse `_force_offline_planner`
fixture in `conftest.py` keeps `make_plan` on the deterministic offline
planner for the whole test session, and every tool function here is called
directly, the same way `test_spec_agents.py` exercises the compiled step
tools without a live model."""

from __future__ import annotations

import pytest

from agents.capability_marketplace import agent as agent_mod
from app.registry import store as registry_store
from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility

_FIXED_TOOL_NAMES = {
    "list_capabilities",
    "register_demo_capability",
    "make_plan",
    "run_last_plan",
    "approve_last_plan",
    "describe_agent_runtime_options",
    "list_pending_approvals",
    "decide_approval",
    "list_agent_specs",
    "list_active_tools",
    "refresh_agent_spec_tools",
}


class _FakeToolContext:
    """Duck-types the one thing these tool functions actually touch on a
    real ADK `ToolContext` -- a plain `.state` dict -- so tests don't need
    to construct a real `InvocationContext`."""

    def __init__(self):
        self.state: dict = {}


@pytest.fixture(autouse=True)
def _reset_supervisor_module_state():
    """`agent.py` keeps process-level state (`_ready`, `_registered_spec_tools`,
    and spec tools appended onto `root_agent.tools`) that would otherwise
    leak between tests, since `root_agent` is a single module-level object
    shared across the whole test session. Reset it before and after every
    test so each one starts from (and leaves behind) just the 7 fixed
    tools."""
    def _strip_to_fixed():
        agent_mod.root_agent.tools[:] = [t for t in agent_mod.root_agent.tools if t.name in _FIXED_TOOL_NAMES]

    agent_mod._ready = False
    agent_mod._registered_spec_tools.clear()
    _strip_to_fixed()
    yield
    agent_mod._ready = False
    agent_mod._registered_spec_tools.clear()
    _strip_to_fixed()


async def _register_capability(session) -> Capability:
    cap = Capability(
        id="cap_super_demo",
        tenant_id=agent_mod.DEV_IDENTITY.tenant_id,
        kind=CapabilityKind.OPENAPI,
        name="ingestion-tool",
        description="ingest content into the pipeline",
        tools=[ToolSchema(name="ingest_content", description="ingest it")],
        visibility=Visibility.PRIVATE,
        endpoint_ref=EndpointRef(openapi_base_url="http://127.0.0.1:1"),
    )
    await registry_store.save(session, cap)
    await session.commit()
    return cap


@pytest.mark.asyncio
async def test_plan_approve_refresh_run_loop_stays_on_one_supervisor(session):
    await _register_capability(session)
    ctx = _FakeToolContext()

    plan_result = await agent_mod.make_plan("ingest some content", ctx)
    assert plan_result["steps"], "offline planner should have bound the registered capability"

    approved = await agent_mod.approve_last_plan("ingestion-agent-demo", ctx)
    assert approved["name"] == "ingestion-agent-demo"
    assert approved["version"] == 1
    assert "cap_super_demo" in approved["capability_snapshots"]

    # Not auto-attached: _ensure_ready already ran once (inside make_plan),
    # so this fresh approval needs an explicit refresh, exactly like a human
    # approving through the HTTP API mid-conversation.
    before = await agent_mod.list_active_tools()
    assert not any(t["name"].startswith("spec_ingestion_agent_demo") for t in before["tools"])

    refreshed = await agent_mod.refresh_agent_spec_tools()
    assert len(refreshed["active_spec_tools"]) == 1
    spec_tool_name = refreshed["active_spec_tools"][0]
    assert spec_tool_name.startswith("spec_ingestion_agent_demo")

    after = await agent_mod.list_active_tools()
    names_to_entries = {t["name"]: t for t in after["tools"]}
    assert spec_tool_name in names_to_entries
    # AgentTool's fixed calling convention -- a single free-text "request",
    # never structured per-step args (see spec_agents.py's module docstring).
    assert names_to_entries[spec_tool_name]["parameters"] == {
        "type": "object",
        "properties": {"request": {"type": "string"}},
        "required": ["request"],
    }

    # Re-approving under the same name bumps the version in place; refreshing
    # again swaps the tool for the new one instead of leaving a duplicate.
    await agent_mod.make_plan("ingest some content again", ctx)
    approved_again = await agent_mod.approve_last_plan("ingestion-agent-demo", ctx)
    assert approved_again["version"] == 2

    refreshed_again = await agent_mod.refresh_agent_spec_tools()
    assert len(refreshed_again["active_spec_tools"]) == 1


@pytest.mark.asyncio
async def test_approve_last_plan_without_a_plan_yet_reports_an_error():
    ctx = _FakeToolContext()
    result = await agent_mod.approve_last_plan("too-soon", ctx)
    assert "error" in result


@pytest.mark.asyncio
async def test_approve_last_plan_with_an_empty_plan_reports_an_error(session):
    # No capability registered -- the offline planner returns a goal with no
    # steps rather than guessing, and approving that shouldn't produce a
    # spec with nothing to run.
    ctx = _FakeToolContext()
    plan_result = await agent_mod.make_plan("do something nothing can handle", ctx)
    assert plan_result["steps"] == []

    result = await agent_mod.approve_last_plan("empty-agent", ctx)
    assert "error" in result


@pytest.mark.asyncio
async def test_approve_last_plan_accepts_a_pattern_and_callbacks_and_refresh_compiles_it(session):
    from google.adk.agents import SequentialAgent

    await _register_capability(session)
    ctx = _FakeToolContext()
    await agent_mod.make_plan("ingest some content", ctx)

    approved = await agent_mod.approve_last_plan(
        "sequential-ingest",
        ctx,
        pattern="sequential",
        callbacks=[
            {"hook": "before_tool", "kind": "validate_args_schema"},
            {"hook": "before_model", "kind": "block_keywords", "params": {"keywords": ["wire transfer"]}},
        ],
    )
    assert approved["pattern"] == "sequential"
    assert approved["callbacks"] == ["before_tool:validate_args_schema", "before_model:block_keywords"]

    refreshed = await agent_mod.refresh_agent_spec_tools()
    assert refreshed["skipped"] == {}
    tool = next(t for t in agent_mod.root_agent.tools if t.name == refreshed["active_spec_tools"][0])
    assert isinstance(tool.agent, SequentialAgent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"pattern": "not-a-pattern"}, "invalid runtime configuration"),
        ({"callbacks": [{"hook": "before_model", "kind": "nope"}]}, "unknown callback kind"),
        ({"pattern": "loop", "max_iterations": 0}, "invalid runtime configuration"),
    ],
    ids=["unknown-pattern", "unknown-callback", "zero-iterations"],
)
async def test_approve_last_plan_rejects_bad_runtime_config_with_a_reason(session, kwargs, fragment):
    await _register_capability(session)
    ctx = _FakeToolContext()
    await agent_mod.make_plan("ingest some content", ctx)
    result = await agent_mod.approve_last_plan("bad-config", ctx, **kwargs)
    assert fragment in result["error"]
    assert (await agent_mod.list_agent_specs())["specs"] == []  # nothing saved


@pytest.mark.asyncio
async def test_one_uncompilable_spec_does_not_block_the_others(session, monkeypatch):
    """A spec saved while custom callbacks were allowed, loaded after they
    were turned off: that one spec is skipped with a reason; every other
    spec still becomes a tool."""
    from app.config import settings

    await _register_capability(session)
    ctx = _FakeToolContext()
    await agent_mod.make_plan("ingest some content", ctx)
    assert "error" not in await agent_mod.approve_last_plan("plain-agent", ctx)

    monkeypatch.setattr(settings, "allow_custom_callbacks", True)
    custom = {"hook": "after_model", "kind": "custom",
              "params": {"path": "app.callbacks.examples:append_disclaimer", "params": {"text": "Not advice."}}}
    assert "error" not in await agent_mod.approve_last_plan("custom-agent", ctx, callbacks=[custom])
    monkeypatch.setattr(settings, "allow_custom_callbacks", False)

    refreshed = await agent_mod.refresh_agent_spec_tools()
    assert len(refreshed["active_spec_tools"]) == 1
    assert refreshed["active_spec_tools"][0].startswith("spec_plain_agent")
    assert len(refreshed["skipped"]) == 1
    assert "custom callbacks are disabled" in next(iter(refreshed["skipped"].values()))


@pytest.mark.asyncio
async def test_describe_agent_runtime_options_lists_patterns_and_callbacks():
    options = await agent_mod.describe_agent_runtime_options()
    assert set(options["patterns"]) == {"single", "sequential", "parallel", "loop", "hierarchical", "react", "plan_execute"}
    assert "validate_args_schema" in options["callbacks_by_hook"]["before_tool"]
