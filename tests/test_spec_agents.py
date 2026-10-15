"""`app/factory/spec_agents.py` -- an approved `AgentSpec` compiled into a
real ADK sub-agent the supervisor can call. Exercises the compile step and
each step tool's own logic directly (not through a live model -- see
`conftest.py`'s `_force_offline_planner`), the same way `test_guardrails_and_policy.py`
and `test_idempotency.py` exercise `tool_wrapper.call_capability` without a
live model either."""

from __future__ import annotations

import pytest

from app.auth.models import IdentityContext
from app.config import settings
from app.factory import spec_agents
from app.factory.tool_wrapper import ToolCallResult
from app.planner.schemas import PlanStep, StepKind
from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility
from app.specs.schemas import AgentSpec


def _capability() -> Capability:
    return Capability(
        id="cap_demo",
        tenant_id="acme",
        name="demo-ingestion-tool",
        kind=CapabilityKind.OPENAPI,
        visibility=Visibility.PRIVATE,
        endpoint_ref=EndpointRef(openapi_base_url="http://127.0.0.1:1"),
        tools=[ToolSchema(name="ingest_content")],
    )


def _spec() -> AgentSpec:
    steps = [
        PlanStep(
            step_id="step_1",
            kind=StepKind.TOOL_CALL,
            capability_id="cap_demo",
            instruction="Use the ingestion tool to ingest content",
        ),
        PlanStep(
            step_id="step_2",
            kind=StepKind.HUMAN_APPROVAL,
            capability_id="",
            instruction="Make sure a human signs off before publishing",
            depends_on=["step_1"],
        ),
    ]
    return AgentSpec(
        spec_id="spec_abc123def456",
        tenant_id="acme",
        version=1,
        name="Demo Ingestion Agent!",
        goal="ingest content using my ingestion tool",
        source_plan_id="plan_test123",
        steps=steps,
        approved_by="demo-user",
    )


def _identity() -> IdentityContext:
    return IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))


def test_safe_identifier_sanitizes_punctuation_and_leading_digits():
    assert spec_agents._safe_identifier("Demo Ingestion Agent!", fallback="agent") == "demo_ingestion_agent"
    assert spec_agents._safe_identifier("123abc", fallback="agent") == "agent_123abc"
    assert spec_agents._safe_identifier("   ", fallback="agent") == "agent"


def test_compile_spec_to_agent_builds_one_tool_per_step_with_no_output_schema_or_planner():
    spec = _spec()
    agent = spec_agents.compile_spec_to_agent(spec, {"cap_demo": _capability()}, _identity())

    # No output_schema, no planner=: this is exactly the combination whose
    # absence fixed the live planner's infinite-loop bug (app/config.py's
    # _GatewayLiteLlm docstring) -- a regression here would reintroduce it.
    assert agent.output_schema is None
    assert agent.planner is None

    assert len(agent.tools) == 2
    tool_names = {t.name for t in agent.tools}
    assert tool_names == {"step_step_1", "step_step_2"}
    assert agent.name.startswith("spec_demo_ingestion_agent_")
    assert spec.goal in agent.description
    assert "step_1" in agent.instruction and "step_2" in agent.instruction
    assert "(after: step_1)" in agent.instruction  # dependency spelled out in plain text


@pytest.mark.asyncio
async def test_tool_call_step_runs_through_call_capability(monkeypatch):
    spec = _spec()
    capability = _capability()
    agent = spec_agents.compile_spec_to_agent(spec, {"cap_demo": capability}, _identity())
    step_tool = next(t for t in agent.tools if t.name == "step_step_1")

    seen_args = {}

    async def _fake_call_capability(capability_arg, tool_name, args, ctx, *, step_id=None, cost_estimate=None, approval_handled=False):
        seen_args["capability_id"] = capability_arg.id
        seen_args["tool_name"] = tool_name
        seen_args["args"] = args
        seen_args["step_id"] = step_id
        return ToolCallResult(ok=True, output={"status": "ingested"})

    monkeypatch.setattr(spec_agents, "call_capability", _fake_call_capability)

    result = await step_tool.func()
    assert result == {
        "step_id": "step_1",
        "capability_id": "cap_demo",
        "tool_name": "ingest_content",
        "ok": True,
        "output": {"status": "ingested"},
        "error": None,
        "blocked_by": None,
    }
    assert seen_args == {
        "capability_id": "cap_demo",
        "tool_name": "ingest_content",
        "args": {"instruction": "Use the ingestion tool to ingest content"},
        "step_id": "step_1",
    }


@pytest.mark.asyncio
async def test_tool_call_step_reports_missing_capability_instead_of_crashing():
    spec = _spec()
    # Capability map doesn't include "cap_demo" -- simulates it being revoked
    # or ungranted after the spec was approved against it.
    agent = spec_agents.compile_spec_to_agent(spec, {}, _identity())
    step_tool = next(t for t in agent.tools if t.name == "step_step_1")

    result = await step_tool.func()
    assert result["ok"] is False
    assert "cap_demo" in result["error"]


@pytest.mark.asyncio
async def test_human_approval_step_waits_on_a_durable_approval(session):
    """A `human_approval` step files a durable approval request and reports
    `awaiting_approval` instead of blocking (the old in-memory wait hung for
    an hour whenever the approver was in another process). Once approved,
    the next call goes through -- once: the approval is consumed."""
    from app.runtime import approvals

    spec = _spec()
    agent = spec_agents.compile_spec_to_agent(spec, {"cap_demo": _capability()}, _identity())
    step_tool = next(t for t in agent.tools if t.name == "step_step_2")

    waiting = await step_tool.func()
    assert waiting["status"] == "awaiting_approval"
    assert waiting["reasons"] == ["Make sure a human signs off before publishing"]

    await approvals.decide(session, waiting["approval_id"], approve=True, decided_by="boss", via="inbox")
    await session.commit()

    approved = await step_tool.func()
    assert approved == {"step_id": "step_2", "kind": "human_approval", "ok": True, "status": "approved",
                        "approval_id": waiting["approval_id"]}
    assert (await step_tool.func())["status"] == "awaiting_approval"  # single use: a new call needs a new approval


def test_compile_spec_to_tool_wraps_agent_as_an_agent_tool():
    spec = _spec()
    tool = spec_agents.compile_spec_to_tool(spec, {"cap_demo": _capability()}, _identity())
    assert tool.name == tool.agent.name
    assert tool.agent.name.startswith("spec_demo_ingestion_agent_")


# -- polish_spec_instruction: the model-rewrite pass that upgrades the
# mechanical, template-built draft into a senior-engineer-quality
# instruction -- and its safety nets, none of which should ever let an
# approval fail or silently lose a step_id the runtime depends on.


@pytest.mark.asyncio
async def test_polish_returns_mechanical_draft_unchanged_without_a_live_model():
    """The common case for tests and any environment without a model key --
    `settings.planner_is_live` is False (see conftest.py's
    `_force_offline_planner`), so this must be a pure no-op, not attempt a
    network call at all."""
    spec = _spec()
    draft = "- step_1: demo-ingestion-tool -- do it\n- step_2: -- sign off"
    result = await spec_agents.polish_spec_instruction(draft, spec.steps, author_notes="be extra careful")
    assert result == draft


@pytest.mark.asyncio
async def test_polish_uses_the_rewritten_text_when_it_preserves_every_step_id(monkeypatch):
    spec = _spec()
    draft = "- step_1: demo-ingestion-tool -- do it\n- step_2: -- sign off"
    seen_args = {}

    async def _fake_polisher(mechanical_instruction, author_notes):
        seen_args["mechanical_instruction"] = mechanical_instruction
        seen_args["author_notes"] = author_notes
        return "A polished instruction mentioning step_1 and step_2, properly ordered."

    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://fake.example.test")
    monkeypatch.setattr(spec_agents, "_run_instruction_polisher", _fake_polisher)

    result = await spec_agents.polish_spec_instruction(draft, spec.steps, author_notes="be extra careful")

    assert result == "A polished instruction mentioning step_1 and step_2, properly ordered."
    assert seen_args == {"mechanical_instruction": draft, "author_notes": "be extra careful"}


@pytest.mark.asyncio
async def test_polish_falls_back_to_mechanical_draft_if_rewrite_drops_a_step_id(monkeypatch):
    spec = _spec()
    draft = "- step_1: demo-ingestion-tool -- do it\n- step_2: -- sign off"

    async def _fake_polisher(mechanical_instruction, author_notes):
        return "A polished instruction that only mentions step_1, forgetting the other one entirely."

    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://fake.example.test")
    monkeypatch.setattr(spec_agents, "_run_instruction_polisher", _fake_polisher)

    result = await spec_agents.polish_spec_instruction(draft, spec.steps)
    assert result == draft  # discarded the bad rewrite, kept the safe mechanical draft


@pytest.mark.asyncio
async def test_polish_falls_back_to_mechanical_draft_if_rewrite_call_raises(monkeypatch):
    spec = _spec()
    draft = "- step_1: demo-ingestion-tool -- do it\n- step_2: -- sign off"

    async def _fake_polisher(mechanical_instruction, author_notes):
        raise RuntimeError("the gateway had a bad day")

    monkeypatch.setattr(settings, "llm_api_key", "fake-key")
    monkeypatch.setattr(settings, "llm_base_url", "https://fake.example.test")
    monkeypatch.setattr(spec_agents, "_run_instruction_polisher", _fake_polisher)

    result = await spec_agents.polish_spec_instruction(draft, spec.steps)
    assert result == draft
