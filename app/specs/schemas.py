"""The declarative `AgentSpec` -- the thing a chat-with-the-planner session
produces once a human approves the plan it came up with. This is the
"AgentDefinition" the architecture doc's target-state section flagged as not
yet built: today, an `ExecutionPlan` is transient (compiled and run once);
an `AgentSpec` is the named, versioned, re-runnable artifact a plan graduates
into once someone signs off on it. Stored in MongoDB (`app/specs/store.py`),
deliberately separate from the SQLAlchemy-backed registry/marketplace
tables -- this is the one piece of state in the project that's a document,
not a relational row, and MongoDB is a reasonable place for a
schema-flexible, version-stamped spec that will keep growing new optional
fields as the config plane grows."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.planner.schemas import PlanStep
from app.registry.schemas import Capability


class AgentPattern(str, Enum):
    """How a compiled spec's steps are arranged into an ADK agent tree --
    see `app/factory/agent_patterns.py` for exactly what each one builds and
    which plan shapes it accepts. `single` is the original (and default)
    behavior: one LlmAgent holding every step's tool."""

    SINGLE = "single"  # one LlmAgent, all step tools, model decides order from the instruction
    SEQUENTIAL = "sequential"  # SequentialAgent: one sub-agent per step, run in depends_on order
    PARALLEL = "parallel"  # ParallelAgent: one sub-agent per step, all at once (no depends_on allowed)
    LOOP = "loop"  # LoopAgent over the sequential chain, until exit_loop or max_iterations
    HIERARCHICAL = "hierarchical"  # coordinator LlmAgent delegating to one specialist sub-agent per step
    REACT = "react"  # single LlmAgent + PlanReActPlanner (explicit reason/act/observe protocol)
    PLAN_EXECUTE = "plan_execute"  # planner sub-agent writes a plan, executor sub-agent carries it out


class CallbackHook(str, Enum):
    """The ADK agent lifecycle points a configured callback can attach to --
    one-to-one with `LlmAgent`'s own `*_callback` fields."""

    BEFORE_AGENT = "before_agent"
    AFTER_AGENT = "after_agent"
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"


class CallbackConfig(BaseModel):
    """One configured callback: which lifecycle `hook` it attaches to, which
    registered implementation (`kind`) runs there, and that
    implementation's `params`. Resolved against
    `app/factory/callback_registry.py` -- and validated at approval time, so
    an unknown kind or a kind attached to the wrong hook is rejected when
    the spec is approved, not discovered later when some other process
    tries to compile it."""

    hook: CallbackHook
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)


class PolicyEffect(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


# Facts a policy condition can test, evaluated live right before each step's
# tool call (see app/factory/tool_policy.py for how each is computed). Plus
# `args.<path>` for any argument the call would be made with.
POLICY_FACTS = {
    "step_id", "tool", "tool_name", "capability_id", "capability_kind",
    "confidence", "self_confidence", "signal_confidence",
    "attempt", "failed_steps", "calls_in_run",
    "hour_utc", "weekday", "tenant_id", "user_id",
}

PolicyOp = Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains", "matches", "exists"]


class PolicyCondition(BaseModel):
    """`fact op value`, e.g. {"fact": "args.quantity", "op": "gt", "value": 10}."""

    fact: str
    op: PolicyOp
    value: Any = None


class PolicyRule(BaseModel):
    """When every condition holds for a call to one of `tools`, apply
    `effect`. Across all matching rules the most restrictive effect wins:
    deny > require_approval > allow."""

    name: str
    tools: list[str] = Field(default_factory=lambda: ["*"])
    when: list[PolicyCondition] = Field(default_factory=list)
    effect: PolicyEffect
    reason: str = ""


class ToolPolicyConfig(BaseModel):
    """Dynamic policy enforcement and human-in-the-loop, evaluated before
    every step's tool call. Tool selectors (in `require_approval_before`,
    `confidence_tools` and each rule's `tools`) match a step id, its tool
    name as the model sees it (`step_<id>`), the capability's tool name, the
    capability id or name -- or "*" for every step.

    The first two fields are shorthand for the most common rules:
      - `require_approval_before`: always pause for approval before these.
      - `confidence_threshold`: pause when the computed confidence for a
        call is below it (0-1), for steps in `confidence_tools`.
    `rules` cover everything else -- deny, allow, or require approval on any
    combination of facts (arguments, confidence, attempts, failures so far,
    time, tenant, user...)."""

    require_approval_before: list[str] = Field(default_factory=list)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_tools: list[str] = Field(default_factory=lambda: ["*"])
    # How confidence is computed: the model's own estimate (it must pass a
    # `confidence` argument to gated tools), signals observed in the run
    # (unmet dependencies, earlier failures, retries, guardrail blocks), or
    # the lower of the two.
    confidence_method: Literal["self_reported", "signals", "combined"] = "combined"
    rules: list[PolicyRule] = Field(default_factory=list)
    # When no rule matches. "deny" turns the rules into an allowlist.
    default_effect: Literal["allow", "deny"] = "allow"
    # Who may decide: "chat" -- the person in the conversation (inline
    # confirm), or any approver in the inbox; "inbox" -- only an approver in
    # the approvals inbox (UI / API), for separation of duties.
    approval_channel: Literal["chat", "inbox"] = "chat"
    # A pending approval nobody decides within this long expires (= denied).
    approval_ttl_seconds: int = Field(default=86400, ge=60)

    def is_active(self) -> bool:
        return bool(self.require_approval_before or self.confidence_threshold is not None or self.rules
                    or self.default_effect == "deny")


class AgentRuntimeConfig(BaseModel):
    """Everything about *how* a compiled spec runs that isn't the plan
    itself: the agent pattern, its knobs, and the callbacks attached at each
    lifecycle hook. Persisted on the spec, so the same approved spec always
    compiles to the same agent tree in every process that loads it."""

    pattern: AgentPattern = AgentPattern.SINGLE
    # LOOP only: hard cap on iterations, on top of the model calling exit_loop.
    max_iterations: int = Field(default=3, ge=1, le=50)
    # A safety net every compiled LlmAgent gets automatically (as a
    # before_model callback), regardless of pattern: ADK's own run loop has
    # no iteration cap (see app/config.py's _GatewayLiteLlm docstring for
    # the infinite-loop bug that taught us this), so a model that never
    # produces a final answer gets stopped here instead of billing forever.
    # Per invocation, per agent tree; None disables it (not recommended).
    max_llm_calls: int | None = Field(default=25, ge=1)
    callbacks: list[CallbackConfig] = Field(default_factory=list)
    # Dynamic policy enforcement + human approval before tool use.
    tool_policy: ToolPolicyConfig = Field(default_factory=ToolPolicyConfig)


class AgentSpec(BaseModel):
    spec_id: str
    tenant_id: str
    version: int = 1
    name: str
    goal: str
    source_plan_id: str
    steps: list[PlanStep]
    # A snapshot of every capability a step above binds to, keyed by
    # capability_id, as the registry had it *at the moment this was
    # approved* -- not a live reference. Filled in by `POST
    # /agent-specs/approve` (see app/routers/agent_specs.py). This is what
    # makes an approved spec self-describing: reading the saved JSON (or
    # Mongo document) shows exactly which tools/schema/endpoint each step
    # was bound to, without cross-referencing the separate registry DB --
    # which matters because that DB row can change or be deleted later,
    # while this snapshot can't. Execution itself does NOT trust this
    # snapshot: `agents/capability_marketplace/agent.py`'s
    # `_capabilities_for_spec` re-resolves capabilities live, against
    # current grants, every time a compiled spec-agent runs a step -- so a
    # capability revoked after approval still can't be called, even though
    # its snapshot is sitting right here. Empty for a `human_approval` step
    # (it has no capability_id to snapshot).
    capability_snapshots: dict[str, Capability] = Field(default_factory=dict)
    # The full instruction text the compiled spec-agent runs with (goal +
    # one line per step, dependencies spelled out) -- built once, here, at
    # approval time (see `app/factory/spec_agents.py`'s
    # `build_spec_instruction`, called from `POST /agent-specs/approve`)
    # against the `capability_snapshots` above, rather than re-stitched
    # every time `compile_spec_to_agent` runs. That makes the saved spec
    # fully self-describing: reading the JSON shows the literal prompt the
    # agent executes, not just the ingredients to reconstruct one -- and it
    # means that prompt can never silently drift if a bound capability is
    # later renamed, since it's baked in against the snapshot, not a live
    # lookup. Defaults to "" for specs saved before this field existed;
    # `compile_spec_to_agent` falls back to building it live (against
    # *live*-resolved capabilities, the old behavior) whenever it's empty.
    instruction: str = ""
    # How this spec compiles and runs -- pattern + callbacks. Defaults to the
    # original single-LlmAgent shape with no extra callbacks (beyond the
    # automatic max_llm_calls safety net), so specs saved before this field
    # existed compile exactly as they always did.
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)
    guardrail_policy_id: str = "default"
    status: Literal["approved"] = "approved"
    approved_by: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
