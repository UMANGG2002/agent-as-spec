"""Shared "approve a plan into an AgentSpec" logic.

Used by both `POST /agent-specs/approve` (`app/routers/agent_specs.py`) and
the supervisor's own `approve_last_plan` tool
(`agents/capability_marketplace/agent.py`), so a plan can be approved either
over HTTP (the config-plane UI's flow) or directly from a chat with `adk
web` -- without re-implementing the versioning, capability-snapshotting, and
instruction-building logic twice in two different places that would
inevitably drift apart.

Putting this here (rather than leaving it inline in the router) is also what
makes it possible to test and exercise the whole plan -> approve -> refresh
-> run loop against a single supervisor agent through `adk web` alone: the
supervisor doesn't need to make an HTTP call back into its own FastAPI
sibling (a separate process it has no handle to anyway) to approve a spec --
it just calls this function directly, against the same DB session it already
has open.

Also where the compiled agent's instruction gets its final polish: the
mechanical, template-built draft (`build_spec_instruction`) goes through
`polish_spec_instruction` before being persisted, so `AgentSpec.instruction`
reads like a senior engineer actually wrote it -- folding in any human
`author_notes` given at approval time -- rather than a bullet-list template.
See `app/factory/spec_agents.py`'s docstrings for the rewrite prompt and the
safety nets (falls back to the mechanical draft untouched if no live model
is configured, the rewrite call fails, or it drops a step_id the runtime
depends on)."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import IdentityContext
from app.factory.spec_agents import build_spec_instruction, polish_spec_instruction
from app.planner.schemas import ExecutionPlan, StepKind
from app.registry import store as registry_store
from app.specs import store as specs_store
from app.specs.schemas import AgentRuntimeConfig, AgentSpec


async def approve_plan(
    session: AsyncSession,
    identity: IdentityContext,
    *,
    name: str,
    plan: ExecutionPlan,
    guardrail_policy_id: str = "default",
    author_notes: str = "",
    runtime: AgentRuntimeConfig | None = None,
) -> AgentSpec:
    """Graduate a reviewed `ExecutionPlan` into a named, versioned
    `AgentSpec`. Re-approving the same `name` bumps `version` in place
    rather than creating a parallel spec_id -- an AgentSpec is meant to be
    the one current-and-versioned artifact for a given agent, not a pile of
    one-offs (see `app/routers/agent_specs.py`'s module docstring).

    `author_notes` -- optional free text from whoever is approving this
    (tone, edge cases, error-handling preferences, anything else they want
    the compiled agent's instruction to reflect) -- is folded into the
    instruction-polishing pass below, not stored separately: once it's
    woven into `AgentSpec.instruction`, that's the one place to look for
    what the agent was told to do.

    `runtime` -- the agent pattern and configured callbacks (see
    `AgentRuntimeConfig`) -- is validated here, before anything is saved or
    any model is called: an impossible pattern for this plan (e.g.
    'parallel' over steps with dependencies), a dependency cycle, an
    unknown callback kind, or a disallowed custom callback raises
    ValueError now, while a human is approving, rather than surfacing later
    when another process tries to compile the spec."""
    from app.factory import agent_patterns

    runtime = runtime or AgentRuntimeConfig()
    agent_patterns.validate_runtime_config(plan.steps, runtime)

    existing = [s for s in await specs_store.list_specs(identity.tenant_id) if s.name == name]
    version = max((s.version for s in existing), default=0) + 1
    spec_id = existing[0].spec_id if existing else f"spec_{uuid.uuid4().hex[:12]}"

    capability_ids = {s.capability_id for s in plan.steps if s.kind != StepKind.HUMAN_APPROVAL and s.capability_id}
    snapshots = {}
    for capability_id in capability_ids:
        capability = await registry_store.get(session, capability_id)
        if capability is not None:
            snapshots[capability_id] = capability

    _validate_step_calls(plan, snapshots)
    from app.factory import tool_policy

    tool_policy.validate_tool_policy(runtime.tool_policy, plan.steps, snapshots)

    mechanical_instruction = build_spec_instruction(plan.goal, plan.steps, snapshots)
    instruction = await polish_spec_instruction(mechanical_instruction, plan.steps, author_notes=author_notes)

    spec = AgentSpec(
        spec_id=spec_id,
        tenant_id=identity.tenant_id,
        version=version,
        name=name,
        goal=plan.goal,
        source_plan_id=plan.plan_id,
        steps=plan.steps,
        capability_snapshots=snapshots,
        instruction=instruction,
        runtime=runtime,
        guardrail_policy_id=guardrail_policy_id,
        approved_by=identity.user_id,
    )
    return await specs_store.save_spec(spec)


def _validate_step_calls(plan: ExecutionPlan, snapshots: dict) -> None:
    """Check each tool-call step against the capability it binds to, the
    same way it will be called at run time (`spec_agents._step_tool_function`):
    the named tool must exist on the capability, and the arguments it will
    be called with -- `step.arguments`, or `{"instruction": ...}` when none
    are set -- must satisfy that tool's declared JSON input schema. A
    mismatch raises ValueError now, at approval, instead of every run of
    the approved agent failing on that step later.

    Steps whose capability isn't resolvable right now are left alone (see
    `approve_plan`: an unresolvable capability is snapshotted as absent and
    reported at run time, not treated as an approval error), and tools that
    declare no schema -- A2A skills, which take free text -- accept any
    arguments."""
    import jsonschema

    for step in plan.steps:
        if step.kind == StepKind.HUMAN_APPROVAL:
            continue
        capability = snapshots.get(step.capability_id)
        if capability is None or not capability.tools:
            continue
        tool_name = step.tool_name or capability.tools[0].name
        tool = next((t for t in capability.tools if t.name == tool_name), None)
        if tool is None:
            available = [t.name for t in capability.tools]
            raise ValueError(f"step '{step.step_id}': capability '{capability.name}' has no tool '{tool_name}' (available: {available})")
        schema = tool.input_schema or {}
        if not schema.get("properties") and not schema.get("required"):
            continue
        args = dict(step.arguments) if step.arguments else {"instruction": step.instruction}
        try:
            jsonschema.validate(instance=args, schema=schema)
        except jsonschema.ValidationError as exc:
            where = "/".join(str(p) for p in exc.absolute_path) or "arguments"
            hint = "" if step.arguments else " (no arguments set, so it would be called with {\"instruction\": ...})"
            raise ValueError(
                f"step '{step.step_id}' calls '{capability.name}.{tool_name}' with invalid {where}: {exc.message}{hint}"
            ) from exc
