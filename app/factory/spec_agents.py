"""Compiles an approved `AgentSpec` into a real ADK sub-agent the supervisor
(`agents/capability_marketplace/agent.py`'s `root_agent`) can delegate to --
closing the gap `app/factory/agent_factory.py`'s own docstring flagged as not
yet built (it names a `compile_plan_to_adk_agent` that didn't exist) and the
gap in `app/routers/agent_specs.py`: approving a plan only ever *persisted*
the spec, it never became anything the supervisor could actually call.

Approving a plan can't reach into a separate, already-running `adk web`
process's memory to hand it a new tool -- the FastAPI app and `adk web` are
two different clients of the same store (see `README.md`), not the same
process. So the supervisor loads approved specs itself: once at startup
(`_ensure_ready` in `agent.py`), and again whenever `refresh_agent_spec_tools`
is called, so specs approved after it started show up without a restart.

Each spec becomes one `LlmAgent` -- no `output_schema`, no explicit
`planner=` (see `app/config.py`'s `_GatewayLiteLlm` docstring for exactly why
stacking those together with tools caused the live planner's earlier
infinite-loop bug; this is a plain tools-only agent, the same shape as
`root_agent` itself, which is already known to behave). It gets one fixed,
non-negotiable tool per plan step: the model decides *when* to call each
step (respecting `depends_on`, spelled out in its instruction), never *what*
that step does -- the capability, tool name, and instruction are exactly
what a human approved, not something the model can improvise. The whole
thing is wrapped in ADK's `AgentTool` so the supervisor calls it like any
other function tool, and it runs as its own nested conversation per call --
`AgentTool` is the documented way to expose any agent as a callable from a
parent `LlmAgent` outside of a static `sub_agents=[...]` tree, which only
wires up at the parent's own construction time: too early for a spec
approved after the supervisor process has already started.

Every step tool call still passes through `tool_wrapper.call_capability` --
guardrails, idempotency, OAuth delegation, and telemetry all run exactly as
they do for the HTTP API's deterministic executor (`agent_factory.py`). Two
simplifications here, worth calling out rather than hiding:

  - Each step mints its own run-scoped delegation token, narrowed to just
    that step's own `required_scopes`, rather than sharing one run-token
    across the whole spec-agent invocation the way `run_service.start_run`
    does for the HTTP path. Simpler, and arguably tighter-scoped, at the
    cost of one shared `run_id` tying the whole invocation together for
    tracing/idempotency grouping.
  - This path doesn't open/close an entry in the invocation ledger or fire
    Composio notifications the way `run_service` does for an HTTP-triggered
    run. A reasonable follow-up, not wired here.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from app.auth import delegation
from app.auth.models import IdentityContext
from app.factory.tool_wrapper import ToolCallContext, call_capability
from app.guardrails.policy import resolve_policy
from app.marketplace import service as marketplace_service
from app.planner.schemas import PlanStep, StepKind
from app.registry.schemas import Capability
from app.specs.schemas import AgentSpec

logger = logging.getLogger(__name__)


def _safe_identifier(text: str, *, fallback: str) -> str:
    """ADK agent/tool names must be valid Python identifiers. Lowercases,
    replaces anything else with `_`, and falls back to a generated name if
    that leaves nothing usable."""
    cleaned = re.sub(r"\W+", "_", text.strip().lower()).strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned


def _step_tool_function(
    step: PlanStep,
    capability: Capability | None,
    identity: IdentityContext,
    *,
    spec_id: str = "",
    spec_name: str = "",
    tool_policy: "ToolPolicyConfig | None" = None,
):
    """Builds the one fixed tool function for a single plan step -- a
    closure over exactly what the human approved, so the model has nothing
    to improvise except *when* to call it.

    It is also where the spec's tool policy is enforced, right before
    dispatch (see app/factory/tool_policy.py): facts about the call and the
    run are gathered, confidence computed, rules evaluated, and the call is
    then denied, paused for human approval, or executed. Approval goes
    through the durable store (app/runtime/approvals.py) and, when the
    agent runs inside an ADK runner, through ADK's own tool confirmation --
    the run pauses, the person approves or rejects, ADK re-invokes this same
    call, and it proceeds or stops.

    Parameters the model sees: none -- unless this step is confidence-gated
    with a self-reported method, in which case it must pass `confidence`
    (0-1) and may pass a short `rationale`. `tool_context` is injected by
    ADK and hidden from the model; it's None when called directly (tests,
    scripts), in which case approval can't pause a run -- the call reports
    `awaiting_approval` and a later call with the approval granted proceeds."""
    from app.factory import tool_policy as tp
    from app.specs.schemas import PolicyEffect, ToolPolicyConfig

    policy_cfg = tool_policy or ToolPolicyConfig()
    state_key = f"spec_runlog:{spec_id or 'adhoc'}"
    capability_label = capability.name if capability else step.capability_id

    async def _gate(decision, *, tool_name: str, args: dict, confidence: dict | None, rationale: str, tool_context) -> dict:
        """Resolve a require-approval decision. Returns {"status": "approved", "approval_id"}
        or a not-approved outcome for the step to return as-is."""
        from app.db import session_scope
        from app.runtime import approvals as approvals_store

        fc_id = getattr(tool_context, "function_call_id", "") or "" if tool_context is not None else ""
        confirmation = getattr(tool_context, "tool_confirmation", None) if tool_context is not None else None
        decider = getattr(tool_context, "user_id", None) or identity.user_id
        fp = approvals_store.fingerprint(tenant_id=identity.tenant_id, spec_id=spec_id, step_id=step.step_id,
                                         tool_name=tool_name, arguments=args)
        async with session_scope() as session:
            row = None
            if confirmation is not None:
                # Resuming the same call after the person answered in chat.
                row = await approvals_store.find_by_call(session, fc_id)
                payload = getattr(confirmation, "payload", None) or {}
                if row is None and isinstance(payload, dict) and payload.get("approval_id"):
                    row = await approvals_store.get(session, payload["approval_id"])
                if row is not None and row.status == approvals_store.PENDING and row.channel == "chat":
                    row = await approvals_store.decide(
                        session, row.id, approve=bool(confirmation.confirmed), decided_by=decider, via="chat",
                        note="approved in chat" if confirmation.confirmed else "rejected in chat",
                    )
            if row is None:
                row = await approvals_store.latest_for_call(session, fp)
            if row is not None and row.status == approvals_store.APPROVED:
                return {"status": "approved", "approval_id": row.id}
            if row is not None and row.status == approvals_store.DENIED and row.consumed_at is None:
                row.consumed_at = approvals_store._now()  # report a denial once; a later call may ask again
                return {
                    "step_id": step.step_id, "ok": False, "status": "rejected", "approval_id": row.id,
                    "error": f"not approved by {row.decided_by or 'the approver'}" + (f": {row.note}" if row.note else ""),
                }
            row = await approvals_store.request(
                session,
                tenant_id=identity.tenant_id, requested_by=identity.user_id, spec_id=spec_id, spec_name=spec_name,
                step_id=step.step_id, tool_name=tool_name, capability_id=step.capability_id, arguments=args,
                reasons=decision.reasons + ([f"model's rationale: {rationale}"] if rationale else []),
                rules=decision.rules,
                confidence=confidence["confidence"] if confidence else None,
                confidence_factors=confidence["factors"] if confidence else [],
                channel=policy_cfg.approval_channel, ttl_seconds=policy_cfg.approval_ttl_seconds,
                function_call_id=fc_id,
            )
            approval_id, channel = row.id, row.channel

        if confirmation is not None:
            # The person answered in chat, but this request can only be decided
            # in the inbox (and isn't yet). ADK won't pause the same resumed call
            # a second time, so say so plainly and let the model relay it; the
            # next call of this step picks up the approver's decision.
            return {
                "step_id": step.step_id, "ok": False, "status": "awaiting_approval", "approval_id": approval_id,
                "reasons": decision.reasons, "channel": channel,
                "message": (f"Still waiting: approval {approval_id} for step '{step.step_id}' must be decided by an "
                            "approver in the approvals inbox -- confirming in chat can't approve it. Ask me to run "
                            "this step again once it's approved."),
            }

        waiting_on = ("an approver in the approvals inbox" if channel == "inbox"
                      else "you (approve or reject here), or an approver in the approvals inbox")
        hint = (
            f"Approval needed before step '{step.step_id}' runs {capability_label}.{tool_name}"
            f" with {args}. Why: {'; '.join(decision.reasons)}."
            + (f" Confidence {confidence['confidence']:.2f} ({'; '.join(confidence['factors'])})." if confidence else "")
            + f" Waiting on {waiting_on}. Approval id {approval_id}."
            + (" Confirming here only re-checks the inbox decision." if channel == "inbox" else "")
        )
        if tool_context is not None and fc_id:
            tool_context.request_confirmation(hint=hint, payload={"approval_id": approval_id})
            tool_context.actions.skip_summarization = True
        return {
            "step_id": step.step_id, "ok": False, "status": "awaiting_approval", "approval_id": approval_id,
            "reasons": decision.reasons, "channel": channel, "message": hint,
        }

    async def _execute(self_confidence: float | None, rationale: str, tool_context) -> dict[str, Any]:
        from app.db import session_scope
        from app.guardrails import policy as guardrail_policy_mod
        from app.guardrails.plugin import guardrail_engine
        from app.runtime import approvals as approvals_store

        if step.kind == StepKind.HUMAN_APPROVAL:
            decision = tp.PolicyDecision(
                effect=PolicyEffect.REQUIRE_APPROVAL,
                reasons=[step.instruction or "this step is a human sign-off"],
                rules=["human_approval_step"],
            )
            outcome = await _gate(decision, tool_name="human_approval", args={}, confidence=None, rationale="",
                                  tool_context=tool_context)
            if outcome["status"] != "approved":
                return {"kind": "human_approval", **outcome}
            async with session_scope() as session:
                await approvals_store.consume(session, outcome["approval_id"])
            return {"step_id": step.step_id, "kind": "human_approval", "ok": True, "status": "approved",
                    "approval_id": outcome["approval_id"]}

        if capability is None:
            return {
                "step_id": step.step_id,
                "ok": False,
                "error": f"capability '{step.capability_id}' is no longer available to this tenant",
            }

        tool_name = step.tool_name or (capability.tools[0].name if capability.tools else capability.name)
        if capability.tools and tool_name not in {t.name for t in capability.tools}:
            return {
                "step_id": step.step_id,
                "ok": False,
                "error": f"capability '{capability.id}' has no tool named '{tool_name}'",
            }
        args = dict(step.arguments) if step.arguments else {"instruction": step.instruction}

        policy = resolve_policy(
            tenant_id=identity.tenant_id,
            brand_id=identity.brand_id,
            capability_id=capability.id,
            base_policy_id=step.guardrail_policy_id,
        )
        state = getattr(tool_context, "state", None) if tool_context is not None else None
        runlog = tp.RunLog.from_state(state.get(state_key) if state is not None else None)
        gated = tp.confidence_gated(policy_cfg, step, capability)
        confidence = tp.compute_confidence(
            policy_cfg, step, runlog, self_reported=self_confidence,
            guardrail_blocks=guardrail_policy_mod.block_count(identity.tenant_id, capability.id),
        )
        facts = tp.gather_facts(step=step, capability=capability, tool_name=tool_name, arguments=args, runlog=runlog,
                                confidence=confidence, tenant_id=identity.tenant_id, user_id=identity.user_id)
        platform_needed, platform_reason = guardrail_engine.needs_approval(
            policy=policy, capability=capability, tool_name=tool_name, identity=identity
        )
        decision = tp.evaluate(policy_cfg, step, capability, facts,
                               platform_requirement=platform_reason if platform_needed else None)

        base = {"step_id": step.step_id, "capability_id": capability.id, "tool_name": tool_name}
        if gated:
            base["confidence"] = confidence["confidence"]
        if decision.effect == PolicyEffect.DENY:
            return {**base, "ok": False, "status": "denied_by_policy", "output": None,
                    "error": "; ".join(decision.reasons), "blocked_by": "policy:" + ",".join(decision.rules)}

        approval_id = None
        if decision.effect == PolicyEffect.REQUIRE_APPROVAL:
            outcome = await _gate(decision, tool_name=tool_name, args=args, confidence=confidence if gated else None,
                                  rationale=rationale, tool_context=tool_context)
            if outcome["status"] != "approved":
                return {**base, **outcome}
            approval_id = outcome["approval_id"]

        run_id = marketplace_service.new_run_id()
        run_token = delegation.delegate_to_run(identity, run_id=run_id, plan_scopes=tuple(step.required_scopes))
        ctx = ToolCallContext(identity=identity, run_id=run_id, run_token=run_token, policy=policy)
        result = await call_capability(capability, tool_name, args, ctx, step_id=step.step_id, approval_handled=True)

        if approval_id:
            async with session_scope() as session:
                await approvals_store.consume(session, approval_id)
        runlog.record(step.step_id, result.ok)
        if state is not None:
            state[state_key] = runlog.to_state()

        out = {
            **base,
            "ok": result.ok,
            "output": result.output,
            "error": result.error,
            "blocked_by": result.blocked_by,
        }
        if approval_id:
            out["approval_id"] = approval_id
        return out

    if tp.needs_self_report(policy_cfg, step, capability):
        async def _run_step(confidence: float, rationale: str = "", tool_context=None) -> dict[str, Any]:
            return await _execute(confidence, rationale, tool_context)

        confidence_doc = (
            " Pass `confidence`: your honest estimate (0-1) that running this step now, with what you have "
            "seen so far, is the right action -- low if inputs look wrong, an earlier step failed or "
            "returned something unexpected. Add a one-line `rationale`. Low confidence pauses for human approval."
        )
    else:
        async def _run_step(tool_context=None) -> dict[str, Any]:
            return await _execute(None, "", tool_context)

        confidence_doc = ""

    _run_step.__name__ = f"step_{_safe_identifier(step.step_id, fallback='step')}"
    _run_step.__doc__ = (
        f"Run the approved step '{step.step_id}' (capability: {capability_label}). "
        f"Fixed instruction, not model-editable: {step.instruction}" + confidence_doc
    )
    return _run_step


def build_spec_instruction(goal: str, steps: list[PlanStep], capabilities: dict[str, Capability]) -> str:
    """Build the full instruction text a compiled spec-agent runs with: the
    goal, plus one line per step naming the capability it's bound to and its
    fixed instruction, with `depends_on` spelled out in plain text so the
    model knows what order to call things in.

    Called once, from `POST /agent-specs/approve`
    (`app/routers/agent_specs.py`), against the same capability snapshot
    just taken for `AgentSpec.capability_snapshots` -- the result is then
    persisted as `AgentSpec.instruction`, so the saved spec is fully
    self-describing (the exact prompt its agent runs with, not just the
    ingredients to rebuild one) and that prompt can't silently drift later
    if a bound capability gets renamed. `compile_spec_to_agent` below only
    calls this itself as a fallback, for specs saved before this field
    existed -- there it's called against *live*-resolved capabilities
    instead, which was the original (pre-snapshot) behavior."""
    step_lines = []
    for step in steps:
        deps = f" (after: {', '.join(step.depends_on)})" if step.depends_on else ""
        cap = capabilities.get(step.capability_id)
        cap_label = cap.name if cap else step.capability_id
        step_lines.append(f"- {step.step_id}: {cap_label} -- {step.instruction}{deps}")

    return (
        f"You are the approved, compiled agent for: {goal}\n\n"
        "This exact plan was reviewed and approved by a human; you run it, you "
        "don't redesign it. Call each step's tool exactly once, respecting the "
        "dependencies noted below -- a step marked '(after: ...)' must wait for "
        "those step_ids to finish first. Report what happened at each step "
        "plainly, including anything a guardrail blocked or that's still "
        "waiting on approval -- don't just say \"done\".\n\nSteps:\n"
        + "\n".join(step_lines)
    )


_INSTRUCTION_POLISH_SYSTEM_PROMPT = """You are a senior software engineer \
writing the system prompt for a production AI agent, in the style you'd use \
if you were hand-authoring it yourself rather than generating it from a \
template.

You will be given a mechanical, auto-generated draft instruction -- a goal \
plus a bare bullet list of fixed steps, each naming the capability/tool it \
calls, its instruction, and (sometimes) which other steps it must wait for \
-- and, optionally, notes from the human who approved this agent (tone, \
edge cases to call out, anything else they want folded in).

Rewrite it into a clear, well-organized, professional instruction a senior \
engineer would actually ship: explain the agent's purpose and what it is (and \
is not) trusted to do, describe the steps in connected, readable prose \
instead of reciting a bullet list, and weave in any human notes naturally \
rather than appending them as an afterthought.

Non-negotiable constraints, because this text is the only thing telling the \
model what to do at runtime -- nothing downstream re-checks your output \
against the plan, so dropping or softening any of this changes what the \
agent actually does:
  - Preserve every single step_id from the draft, verbatim, at least once in \
your output (e.g. "step_a1b2c3d4") -- each one names a specific fixed tool \
call the model must still be able to find and invoke.
  - Preserve every ordering dependency the draft states (a step marked \
"after: X" must still be described as waiting for X).
  - Keep the instruction that each step's tool is called exactly once, and \
that the model must report what happened at each step plainly -- including \
anything a guardrail blocked or that's still waiting on human approval -- \
rather than just declaring the job "done".
  - Never invent a step, tool, or capability that wasn't in the draft, and \
never remove one that was.

Output only the rewritten instruction text itself -- no preamble, no \
markdown headers, no commentary about what you changed."""


async def polish_spec_instruction(
    mechanical_instruction: str,
    steps: list[PlanStep],
    *,
    author_notes: str = "",
) -> str:
    """Take `build_spec_instruction`'s mechanical draft and have the
    configured live model rewrite it into a well-organized, deliberately
    written instruction -- the quality bar a senior engineer hand-authoring
    an agent's system prompt would aim for, rather than a template's bullet
    dump. Optional `author_notes` from the approving human (tone, edge
    cases, error-handling preferences) go into the same rewrite call, not
    appended separately -- the model is asked to weave them in, not just
    concatenate them on.

    Called from `app/specs/service.py`'s `approve_plan`, which persists
    whatever this returns as `AgentSpec.instruction` -- so the polish only
    ever runs once, at approval time, never on every compile.

    Degrades to returning `mechanical_instruction` unchanged, rather than
    raising, in two cases: no live model is configured
    (`settings.planner_is_live` is False -- the common case for tests and
    any environment without a model key), or the rewrite call fails for any
    reason (timeout, malformed response, quota). A third safety net: if the
    model's rewrite silently drops a `step_id` the mechanical draft named --
    checked directly against the `steps` this spec actually has, not by
    re-parsing the rendered text -- that's treated as a bad rewrite and
    discarded too, same as an exception. An approval should never fail, or
    silently produce an instruction missing a step the runtime depends on,
    just because this one polish step had a bad day; the mechanical draft is
    always a safe, honest fallback."""
    from app.config import settings

    if not settings.planner_is_live:
        return mechanical_instruction

    try:
        polished = await _run_instruction_polisher(mechanical_instruction, author_notes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("instruction polish failed (%s); keeping the mechanical draft", exc)
        return mechanical_instruction

    missing = [step.step_id for step in steps if step.step_id not in polished]
    if missing:
        logger.warning(
            "instruction polish dropped step_id(s) %s; keeping the mechanical draft instead", missing
        )
        return mechanical_instruction

    return polished


async def _run_instruction_polisher(mechanical_instruction: str, author_notes: str) -> str:
    """One-shot, tool-free, schema-free chat turn with the configured live
    model -- deliberately the simplest possible agent shape (no tools, no
    output_schema, no planner=), the same safe combination
    `compile_spec_to_agent` itself uses and `app/config.py`'s
    `_GatewayLiteLlm` docstring explains the reasoning for. Mirrors
    `app/planner/planner_agent.py`'s `_LivePlanner.plan()` -- a fresh
    `LlmAgent` + `InMemoryRunner` per call, no state kept between calls."""
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from app.config import settings

    message = mechanical_instruction
    if author_notes.strip():
        message += f"\n\nNotes from the approving human, fold these in:\n{author_notes.strip()}"

    agent = LlmAgent(
        name="instruction_polisher",
        model=settings.live_model,
        instruction=_INSTRUCTION_POLISH_SYSTEM_PROMPT,
    )
    runner = InMemoryRunner(agent=agent, app_name="capability-marketplace-polish")
    session_id = f"polish-session-{uuid.uuid4().hex[:8]}"
    await runner.session_service.create_session(
        app_name="capability-marketplace-polish", user_id="polisher", session_id=session_id
    )

    final_text: str | None = None
    async for event in runner.run_async(
        user_id="polisher",
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = event.content.parts[0].text

    if not final_text or not final_text.strip():
        raise RuntimeError("instruction polisher produced no final response")
    return final_text.strip()


def compile_spec_to_agent(
    spec: AgentSpec,
    capabilities: dict[str, Capability],
    identity: IdentityContext,
    *,
    model: Any = None,
):
    """Build the agent tree for one approved spec, shaped by
    `spec.runtime.pattern` and carrying `spec.runtime.callbacks` -- see
    `app/factory/agent_patterns.py` for every pattern. `capabilities` maps
    capability_id -> Capability for every id the spec's steps reference that
    this tenant can still actually use -- a step whose capability has since
    been revoked gets a tool that reports that plainly instead of crashing
    the whole agent (see `_step_tool_function`).

    The root's name is the same for every pattern
    (`spec_<name>_<id suffix>`), so the supervisor's tool for a spec keeps
    its name if the spec is re-approved under a different pattern.

    `instruction` is `spec.instruction` -- computed once at approval time,
    not re-stitched here (see `build_spec_instruction`'s docstring) --
    except for specs saved before that field existed, where it falls back
    to building it live. `model` defaults to the configured live model;
    tests pass a scripted one (`app/testing/scripted_llm.py`)."""
    from app.config import settings
    from app.factory import agent_patterns

    step_tools = {
        step.step_id: _step_tool_function(
            step, capabilities.get(step.capability_id), identity,
            spec_id=spec.spec_id, spec_name=spec.name, tool_policy=spec.runtime.tool_policy,
        )
        for step in spec.steps
    }
    capability_labels = {
        step.step_id: (capabilities[step.capability_id].name if step.capability_id in capabilities else step.capability_id)
        for step in spec.steps
    }
    instruction = spec.instruction or build_spec_instruction(spec.goal, spec.steps, capabilities)

    return agent_patterns.build_agent_tree(
        root_name=f"spec_{_safe_identifier(spec.name, fallback='agent')}_{spec.spec_id[-6:]}",
        description=f"Approved agent spec '{spec.name}' (v{spec.version}, {spec.runtime.pattern.value}): {spec.goal}",
        goal=spec.goal,
        instruction=instruction,
        steps=spec.steps,
        step_tools=step_tools,
        capability_labels=capability_labels,
        runtime=spec.runtime,
        model=model if model is not None else settings.live_model,
    )


def compile_spec_to_tool(
    spec: AgentSpec, capabilities: dict[str, Capability], identity: IdentityContext, *, model: Any = None
):
    """Wrap the compiled spec-agent as a tool the supervisor can call -- see
    the module docstring for why `AgentTool` rather than a static
    `sub_agents=[...]` entry. Works for every pattern: AgentTool accepts any
    BaseAgent, including the Sequential/Parallel/Loop workflow agents."""
    from google.adk.tools.agent_tool import AgentTool

    agent = compile_spec_to_agent(spec, capabilities, identity, model=model)
    return AgentTool(agent=agent)
