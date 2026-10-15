"""Root agent for `adk web` / `adk run` / `scripts/run_agent.py` -- a single
conversational orchestrator sitting in front of the actual marketplace stack
in `app/` (registry, planner, agent factory, guardrails, OAuth delegation,
idempotency, OpenTelemetry). This is the "talk directly to my orchestration
layer" entry point the architecture doc describes, wired up for local
interactive testing rather than the HTTP API `scripts/demo.py` exercises.

Discovered by `adk web agents` / `adk run agents/capability_marketplace`
because this package has an `agent.py` exposing `root_agent` -- see the ADK
CLI's own `adk create` scaffold for the convention this follows.

A single fixed dev identity stands in for the real OAuth front door (see
`app/auth/deps.py` for what that looks like over HTTP): `adk web` has no
login flow of its own, so this is the one place in the repo that is
deliberately *not* behind real auth -- scoped to local testing only, never
meant to be deployed as-is.

This is also the supervisor agent every approved AgentSpec graduates into a
sub-agent of: `_load_or_refresh_spec_tools` (called once at startup, and
again by the `refresh_agent_spec_tools` tool) compiles each approved spec
into a real ADK agent and attaches it to `root_agent.tools` -- see
`app/factory/spec_agents.py` for how a spec becomes an agent, and why
`POST /agent-specs/approve` itself can't do this wiring directly (it runs in
a different process).

One supervisor, one page: there's exactly one `root_agent` defined here, and
every compiled spec-agent is attached to *this same* agent's `tools` list --
never a separate supervisor per spec, never a second root agent. A spec can
even be approved without leaving this process at all, via the
`approve_last_plan` tool below (shared approval logic in
`app/specs/service.py`), which is what makes it possible to exercise the
whole make_plan -> approve_last_plan -> refresh_agent_spec_tools -> call-it
loop against a single supervisor agent through `adk web` alone -- no FastAPI
process or manual HTTP call required to test an agent spec end to end.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

# `adk web`/`adk run` only ever add the `agents/` directory itself to
# sys.path (see google.adk.cli.utils.agent_loader._perform_load), never its
# parent -- so loading this module through the ADK CLI leaves the project
# root, where `app/` lives, off the path entirely. `scripts/run_agent.py`
# works around this for itself the same way; this module needs its own copy
# since the CLI imports it directly, not through that script. Without this,
# every `from app... import ...` below raises `ModuleNotFoundError: No
# module named 'app'` -- but only on the *first real chat turn* against this
# agent, not when `adk web` merely lists it at startup, since discovery
# doesn't import the module and nobody had actually chatted with it through
# the CLI until that turn.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from google.adk.agents.llm_agent import Agent
from google.adk.tools import FunctionTool, ToolContext

from app.auth.models import IdentityContext
from app.config import settings
from app.db import init_db, session_scope
from app.factory import spec_agents
from app.marketplace import service as marketplace_service
from app.planner.planner_agent import create_plan
from app.planner.schemas import ExecutionPlan, StepKind
from app.registry import store as registry_store
from app.registry.adapters import openapi_adapter
from app.registry.schemas import Capability
from app.runtime import redis_streams, run_service
from app.specs import service as specs_service
from app.specs import store as specs_store
from app.telemetry.setup import configure_telemetry
from app.testing import echo_server

DEV_IDENTITY = IdentityContext(
    tenant_id="acme", brand_id="acme-brand", user_id="adk-web-user", scopes=("*",)
)
DEMO_CAPABILITY_NAME = "demo-ingestion-tool"

_ready_lock = asyncio.Lock()
_ready = False

# spec_id -> the name of the tool currently attached to root_agent.tools for
# it, so a re-approval (a version bump, same spec_id) replaces that tool
# instead of leaving a stale duplicate alongside the new one.
_registered_spec_tools: dict[str, str] = {}

# spec_id -> why it couldn't be compiled on the last refresh (see
# _load_or_refresh_spec_tools); reported by refresh_agent_spec_tools.
_skipped_spec_tools: dict[str, str] = {}

logger = logging.getLogger(__name__)


async def _ensure_ready() -> None:
    """Runs the same startup the FastAPI app does in its lifespan
    (`app/main.py`) -- telemetry + table creation -- lazily and exactly
    once, since nothing here drives a FastAPI lifespan for us. Also loads
    every already-approved AgentSpec as a tool, so specs approved before
    this process started are available from the first message, not just
    after an explicit `refresh_agent_spec_tools` call."""
    global _ready
    if _ready:
        return
    async with _ready_lock:
        if _ready:
            return
        configure_telemetry()
        await init_db()
        await _load_or_refresh_spec_tools()
        _ready = True


async def _capabilities_for_spec(spec) -> dict[str, Capability]:
    """Resolve the Capability objects a spec's steps reference, the same way
    `run_service._execute` does for the HTTP run path -- filtered to what
    this tenant can still actually use, since a capability can be revoked or
    ungranted after a spec was approved against it."""
    async with session_scope() as session:
        granted = await marketplace_service.granted_capability_ids(session, DEV_IDENTITY.tenant_id)
        visible = await registry_store.list_for_tenant(session, DEV_IDENTITY.tenant_id, granted_capability_ids=granted)
    by_id = {c.id: c for c in visible}
    needed = {step.capability_id for step in spec.steps if step.kind != StepKind.HUMAN_APPROVAL}
    return {cap_id: by_id[cap_id] for cap_id in needed if cap_id in by_id}


async def _load_or_refresh_spec_tools() -> list[str]:
    """Load every approved AgentSpec for the demo tenant and make sure each
    has exactly one matching tool on `root_agent.tools` -- see
    `app/factory/spec_agents.py` for what "approved spec becomes a tool"
    actually builds and why it has to happen here, in the supervisor's own
    process, rather than at approval time."""
    specs = await specs_store.list_specs(DEV_IDENTITY.tenant_id)
    active_names: list[str] = []
    _skipped_spec_tools.clear()
    for spec in specs:
        # One spec that can't compile any more (say, a custom callback that
        # settings no longer allow) must not stop every *other* spec from
        # loading -- skip it, remember why, and keep going.
        try:
            capabilities = await _capabilities_for_spec(spec)
            tool = spec_agents.compile_spec_to_tool(spec, capabilities, DEV_IDENTITY)
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping spec %s (%s): %s", spec.spec_id, spec.name, exc)
            _skipped_spec_tools[spec.spec_id] = f"{spec.name}: {exc}"
            continue

        old_name = _registered_spec_tools.get(spec.spec_id)
        if old_name:
            root_agent.tools[:] = [t for t in root_agent.tools if getattr(t, "name", None) != old_name]

        root_agent.tools.append(tool)
        _registered_spec_tools[spec.spec_id] = tool.name
        active_names.append(tool.name)
    return active_names


async def refresh_agent_spec_tools() -> dict:
    """Call this after approving a new agent spec (or changing which
    capabilities this tenant is granted) so the running conversation picks
    it up as a callable tool right away, without restarting `adk web`.
    Reports any spec that couldn't be compiled, and why, under "skipped"."""
    await _ensure_ready()
    active = await _load_or_refresh_spec_tools()
    return {"active_spec_tools": active, "skipped": dict(_skipped_spec_tools)}


async def list_pending_approvals() -> dict:
    """List tool calls waiting for human approval for this tenant -- which
    spec and step, the tool and exact arguments, why it paused (rules
    matched, confidence and its breakdown), and whether it can be decided
    here in chat or only by an approver in the inbox."""
    from app.runtime import approvals as approvals_store

    await _ensure_ready()
    async with session_scope() as session:
        rows = await approvals_store.list_for_tenant(session, DEV_IDENTITY.tenant_id, status="pending")
        return {"pending": [approvals_store.to_dict(r) for r in rows]}


async def decide_approval(approval_id: str, approve: bool, note: str = "") -> dict:
    """Approve or reject a pending tool-call approval on the person's
    behalf -- only when they've clearly said so. Requests whose spec
    requires inbox-only approval are refused here (separation of duties).
    After approving, run the spec again: the approved step then proceeds,
    once."""
    from app.runtime import approvals as approvals_store

    await _ensure_ready()
    try:
        async with session_scope() as session:
            row = await approvals_store.decide(
                session, approval_id, approve=approve, decided_by=DEV_IDENTITY.user_id, via="chat", note=note
            )
            return approvals_store.to_dict(row)
    except (KeyError, ValueError, PermissionError) as exc:
        return {"error": str(exc)}


async def describe_agent_runtime_options() -> dict:
    """List every agent pattern (single, sequential, parallel, loop,
    hierarchical, react, plan_execute) and every configurable callback kind
    per lifecycle hook (before_model, after_model, before_tool, after_tool,
    before_agent, after_agent) -- what approve_last_plan's `pattern` and
    `callbacks` arguments accept. Call this before approving if the person
    wants a specific execution pattern, guardrails, or validation."""
    from app.factory import agent_patterns, callback_registry
    from app.specs.schemas import POLICY_FACTS

    return {
        "patterns": agent_patterns.describe_patterns(),
        "callbacks_by_hook": callback_registry.catalog(),
        "callback_entry_shape": {"hook": "<hook>", "kind": "<kind>", "params": {"...": "..."}},
        "examples": [
            {"hook": "before_tool", "kind": "validate_args_schema"},
            {"hook": "before_model", "kind": "block_keywords", "params": {"keywords": ["wire transfer"]}},
            {"hook": "before_model", "kind": "guardrail_policy", "params": {"policy_id": "default"}},
            {"hook": "after_tool", "kind": "redact_pii"},
            {"hook": "after_tool", "kind": "truncate_output", "params": {"max_chars": 2000}},
        ],
        "tool_policy": {
            "shape": {
                "require_approval_before": ["<step id / tool / capability>", "or *"],
                "confidence_threshold": "0-1 or null",
                "confidence_tools": ["*"],
                "confidence_method": "combined | signals | self_reported",
                "rules": [{"name": "...", "tools": ["*"], "when": [{"fact": "...", "op": "...", "value": "..."}],
                           "effect": "allow | require_approval | deny", "reason": "..."}],
                "default_effect": "allow | deny",
                "approval_channel": "chat | inbox",
            },
            "facts": sorted(POLICY_FACTS) + ["args.<name>"],
            "ops": ["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains", "matches", "exists"],
        },
    }


async def list_agent_specs() -> dict:
    """List every approved AgentSpec for the demo tenant -- read-only, so
    you can see what's been approved without touching the live tool list
    (call refresh_agent_spec_tools to actually attach any new ones as
    tools)."""
    await _ensure_ready()
    specs = await specs_store.list_specs(DEV_IDENTITY.tenant_id)
    return {
        "specs": [
            {"spec_id": s.spec_id, "name": s.name, "version": s.version, "goal": s.goal, "steps": len(s.steps)}
            for s in specs
        ]
    }


def _tool_parameters_schema(tool) -> dict | None:
    """Pull the JSON schema for a tool's call arguments out of the same
    `FunctionDeclaration` ADK itself builds to hand the model the tool's
    calling convention -- `BaseTool._get_declaration()`. Works for both the
    fixed `FunctionTool`s (schema derived from the wrapped function's own
    signature + docstring) and the per-spec `AgentTool`s (always a single
    `request: str` field -- see `app/factory/spec_agents.py`'s
    `compile_spec_to_tool`: the whole point of an AgentTool is that the
    caller hands it a natural-language instruction, not structured args, and
    the wrapped agent's own fixed, non-negotiable per-step tools are what
    actually constrain what happens next).

    `parameters_json_schema` is ADK's newer, plain-JSON-Schema field; older
    ADK builds (or models without the JSON_SCHEMA_FOR_FUNC_DECL feature) may
    instead populate `.parameters` as a `genai.types.Schema` object, so fall
    back to dumping that. Returns None if a tool declares no callable
    declaration at all (shouldn't happen for anything in root_agent.tools,
    but this is a read-only inspection tool, not something worth crashing
    the agent over)."""
    try:
        declaration = tool._get_declaration()
    except Exception:
        return None
    if declaration is None:
        return None
    if declaration.parameters_json_schema is not None:
        return declaration.parameters_json_schema
    if declaration.parameters is not None:
        return declaration.parameters.model_dump(mode="json", exclude_none=True)
    return None


async def list_active_tools() -> dict:
    """List every tool this agent can actually call right now -- the fixed
    ones (list_capabilities, make_plan, ...) plus one per approved AgentSpec
    currently attached to it, each named `spec_<name>_<id>` -- including each
    tool's parameter schema (what args it expects, and their types), not just
    its name and description. Purely read-only, unlike
    refresh_agent_spec_tools (which also attaches any newly-approved specs as
    a side effect): call this one just to see what's there, that one to
    actually pick up something new."""
    await _ensure_ready()
    tools_info = []
    for t in root_agent.tools:
        entry = {"name": t.name, "description": getattr(t, "description", "")}
        schema = _tool_parameters_schema(t)
        if schema is not None:
            entry["parameters"] = schema
        tools_info.append(entry)
    return {"tools": tools_info}


async def list_capabilities() -> dict:
    """List the capabilities the demo tenant can plan against right now
    (its own, plus anything already granted to it). Call this first if
    you're not sure what's available before planning."""
    await _ensure_ready()
    async with session_scope() as session:
        granted = await marketplace_service.granted_capability_ids(session, DEV_IDENTITY.tenant_id)
        caps = await registry_store.list_for_tenant(session, DEV_IDENTITY.tenant_id, granted_capability_ids=granted)
    return {
        "capabilities": [
            {"id": c.id, "name": c.name, "kind": c.kind.value, "tools": [t.name for t in c.tools]} for c in caps
        ]
    }


async def register_demo_capability() -> dict:
    """Register a small demo capability backed by a local echo server, so
    there's something to plan against without needing a real MCP server,
    agent card, or external API. Idempotent -- safe to call more than once;
    it reuses the existing capability record instead of registering a
    duplicate, but always repoints it at *this process's* echo server (the
    server is per-process and ephemeral, while the registry row persists in
    SQLite across restarts -- without this, a capability registered by a
    previous run would point at a port nothing is listening on anymore)."""
    await _ensure_ready()
    port = echo_server.ensure_running()
    base_url = f"http://127.0.0.1:{port}"

    async with session_scope() as session:
        existing = await registry_store.list_for_tenant(session, DEV_IDENTITY.tenant_id)
        for cap in existing:
            if cap.name == DEMO_CAPABILITY_NAME:
                if cap.endpoint_ref.openapi_base_url != base_url:
                    cap.endpoint_ref.openapi_base_url = base_url
                    await registry_store.save(session, cap)
                return {
                    "capability_id": cap.id,
                    "tools": [t.name for t in cap.tools],
                    "already_registered": True,
                }

        spec = {
            "info": {"description": "a toy content-ingestion API for local testing"},
            "paths": {
                "/ingest_content": {
                    "post": {
                        "operationId": "ingest_content",
                        "summary": "Ingest generated content into the pipeline",
                    }
                }
            },
        }
        capability = openapi_adapter.capability_from_openapi(
            spec,
            tenant_id=DEV_IDENTITY.tenant_id,
            name=DEMO_CAPABILITY_NAME,
            base_url=base_url,
        )
        await registry_store.save(session, capability)
    return {
        "capability_id": capability.id,
        "tools": [t.name for t in capability.tools],
        "already_registered": False,
    }


async def make_plan(intent: str, tool_context: ToolContext) -> dict:
    """Turn a natural-language goal into a typed ExecutionPlan, using the
    real planner -- the live Gemini planner if GOOGLE_API_KEY is set,
    otherwise the deterministic offline planner -- bound only to
    capabilities this tenant is actually allowed to use. Stores the plan on
    the conversation so `run_last_plan` can run it without you repeating it.
    """
    await _ensure_ready()
    async with session_scope() as session:
        plan = await create_plan(session, DEV_IDENTITY, intent)
    tool_context.state["last_plan"] = plan.model_dump(mode="json")
    return {
        "plan_id": plan.plan_id,
        "goal": plan.goal,
        "steps": [
            {"step_id": s.step_id, "capability_id": s.capability_id, "instruction": s.instruction}
            for s in plan.steps
        ],
    }


async def run_last_plan(tool_context: ToolContext) -> dict:
    """Execute the most recent plan `make_plan` produced in this
    conversation -- the real run path: OAuth delegation, guardrails,
    idempotency, and OpenTelemetry spans all run for real, exactly as they
    would over the HTTP API. Waits a few seconds for the run to finish and
    returns what the stream reported at each step."""
    await _ensure_ready()
    raw_plan = tool_context.state.get("last_plan")
    if not raw_plan:
        return {"error": "no plan yet -- call make_plan with an intent first"}
    plan = ExecutionPlan.model_validate(raw_plan)
    if not plan.steps:
        return {"error": "that plan has no steps (the planner found no usable capability) -- nothing to run"}

    async with session_scope() as session:
        run_id = await run_service.start_run(session, DEV_IDENTITY, plan)

    events: list[dict] = []
    status = "unknown"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and status == "unknown":
        async for entry_id, event in redis_streams.read_events(run_id, consumer="adk-web", block_ms=1000):
            events.append(event)
            await redis_streams.ack(run_id, entry_id)
            if event.get("kind") == "run.finished":
                status = event.get("status", "unknown")
    return {"run_id": run_id, "status": status, "events": events}


async def approve_last_plan(
    name: str,
    tool_context: ToolContext,
    guardrail_policy_id: str = "default",
    author_notes: str = "",
    pattern: str = "single",
    max_iterations: int = 3,
    callbacks: list[dict] | None = None,
    tool_policy: dict | None = None,
) -> dict:
    """Approve the most recent plan `make_plan` produced in this conversation
    into a named, versioned AgentSpec -- the same thing `POST
    /agent-specs/approve` does over HTTP (see `app/specs/service.py`'s
    `approve_plan`, shared by both paths), called here directly against this
    process's own DB session instead of making an HTTP round-trip to the
    FastAPI app (a separate process this supervisor has no handle to
    anyway).

    This is what lets the whole plan -> approve -> refresh -> run loop
    happen in one `adk web` conversation with this one supervisor agent,
    with no second process or manual API call needed to test an agent spec
    end to end. Re-approving under the same `name` bumps the version in
    place, exactly like the HTTP endpoint. The new spec is saved, but -- like
    any other approval -- not yet attached as a callable tool here; call
    refresh_agent_spec_tools afterward to pick it up.

    `author_notes` -- anything the person approving wants the compiled
    agent's instruction to actually reflect (tone, edge cases, how it should
    handle a guardrail block, whatever) -- gets woven into that instruction
    by a real rewrite pass (`spec_agents.polish_spec_instruction`), not just
    tacked onto the end of the mechanical step list. If you want the
    compiled agent's own system prompt to read like it, pass it here rather
    than only telling it to me conversationally -- I don't get a say in what
    ends up in a sub-agent's instruction, only in what I do myself.

    `pattern` -- how the compiled agent runs its steps: single, sequential,
    parallel, loop, hierarchical, react, or plan_execute (loop also uses
    `max_iterations`). `callbacks` -- a list of {"hook", "kind", "params"}
    entries attaching guardrails / validation / redaction / logging at
    before_model, after_model, before_tool, after_tool, before_agent or
    after_agent. Call describe_agent_runtime_options for every pattern and
    callback kind. `tool_policy` -- dynamic policy enforcement and human
    approval before tool use, e.g. {"require_approval_before": ["pay"],
    "confidence_threshold": 0.6, "rules": [{"name": "big-orders", "tools":
    ["quote"], "when": [{"fact": "args.quantity", "op": "gt", "value": 100}],
    "effect": "require_approval"}], "approval_channel": "chat"}. An invalid
    combination (e.g. parallel over dependent steps, an unknown callback
    kind, a policy selector that matches no step) is rejected here with an
    explanation, nothing is saved."""
    from pydantic import ValidationError

    from app.specs.schemas import AgentRuntimeConfig

    await _ensure_ready()
    try:
        runtime = AgentRuntimeConfig(
            pattern=pattern, max_iterations=max_iterations, callbacks=callbacks or [], tool_policy=tool_policy or {}
        )
    except ValidationError as exc:
        return {"error": f"invalid runtime configuration: {exc.errors()}"}

    raw_plan = tool_context.state.get("last_plan")
    if not raw_plan:
        return {"error": "no plan yet -- call make_plan with an intent first"}
    plan = ExecutionPlan.model_validate(raw_plan)
    if not plan.steps:
        return {"error": "that plan has no steps (the planner found no usable capability) -- nothing to approve"}

    try:
        async with session_scope() as session:
            spec = await specs_service.approve_plan(
                session,
                DEV_IDENTITY,
                name=name,
                plan=plan,
                guardrail_policy_id=guardrail_policy_id,
                author_notes=author_notes,
                runtime=runtime,
            )
    except ValueError as exc:
        return {"error": f"not approved: {exc}"}
    return {
        "spec_id": spec.spec_id,
        "name": spec.name,
        "version": spec.version,
        "pattern": spec.runtime.pattern.value,
        "callbacks": [f"{c.hook.value}:{c.kind}" for c in spec.runtime.callbacks],
        "tool_policy_active": spec.runtime.tool_policy.is_active(),
        "capability_snapshots": list(spec.capability_snapshots.keys()),
        "instruction": spec.instruction,
    }


root_agent = Agent(
    name="capability_marketplace_orchestrator",
    model=settings.live_model,
    description="Talks to the capability marketplace: registers capabilities, plans, and runs them.",
    instruction="""You are the chat front door to a multi-tenant capability \
marketplace built on Google ADK. You are talking on behalf of one fixed demo \
tenant ("acme") for local testing -- there is no login here.

When the person asks what's available, call list_capabilities. If it comes \
back empty and they haven't mentioned a capability of their own, offer to \
call register_demo_capability to set up a small local demo capability they \
can plan against.

When they describe something they want done, call make_plan with their \
intent in their own words. Show them the resulting plan -- its steps and \
which capability_id each is bound to -- and ask whether to run it, unless \
they already said to just do it, in which case call run_last_plan right \
after make_plan without asking again.

When asked to run it, call run_last_plan and report the run's status and \
what happened at each step from the returned events, including anything a \
guardrail blocked or that needed approval -- don't just say "done".

Separately from the plan/run flow above: every AgentSpec a human has \
approved -- through the configuration UI (chat -> plan -> approve over \
HTTP), or by approving right here with approve_last_plan -- shows up here \
as its own callable tool, named after the spec -- calling it runs that \
exact approved plan for real, through the same guardrails/delegation/ \
telemetry path as everything else. You, this one agent, can run the entire \
loop yourself in a single conversation: make_plan, then approve_last_plan \
with a name for it, then refresh_agent_spec_tools to attach it, then call \
it. If the person tells you anything about how the approved agent should \
behave -- tone, edge cases, how to handle something going wrong -- pass it \
to approve_last_plan's author_notes so it actually ends up in that agent's \
own instruction, not just in this conversation. If they want a particular \
execution pattern (sequential, parallel, loop, hierarchical, react, \
plan_execute) or guardrails / argument validation / redaction, call \
describe_agent_runtime_options and pass the matching pattern and callbacks \
to approve_last_plan; if it rejects the combination, explain why. For human \
approval before tool use or rules that allow/deny/require approval for calls \
(by tool, arguments, confidence, time...), pass tool_policy. When a spec \
reports a step "awaiting_approval", tell the person exactly what is waiting and \
why; if they approve or reject it in chat, call decide_approval, then run the \
spec again so the approved step proceeds. list_pending_approvals shows what's \
waiting. Call list_agent_specs to \
see what's been approved (a database listing \
-- not necessarily callable yet), list_active_tools to see what's actually \
callable on you right now (the fixed tools plus one per attached spec), and \
refresh_agent_spec_tools if a spec (yours or approved elsewhere, through \
the HTTP API) isn't in that list yet (new approvals don't appear \
automatically in an already-running conversation).""",
    tools=[
        FunctionTool(list_capabilities),
        FunctionTool(register_demo_capability),
        FunctionTool(make_plan),
        FunctionTool(run_last_plan),
        FunctionTool(approve_last_plan),
        FunctionTool(describe_agent_runtime_options),
        FunctionTool(list_pending_approvals),
        FunctionTool(decide_approval),
        FunctionTool(list_agent_specs),
        FunctionTool(list_active_tools),
        FunctionTool(refresh_agent_spec_tools),
    ],
)
