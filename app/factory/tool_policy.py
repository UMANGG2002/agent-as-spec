"""Dynamic policy enforcement and human-in-the-loop for compiled spec steps.

Evaluated inside every step tool (`spec_agents._step_tool_function`), right
before the call is dispatched -- so it applies identically in every agent
pattern, under the supervisor, and under the spec_test_agent router. Three
sources feed one decision:

  1. The spec's `runtime.tool_policy` (see `ToolPolicyConfig`):
     `require_approval_before`, the confidence gate, and `rules`.
  2. Platform rules the guardrail layer already had -- an in-code
     `ApprovalRule` match, or adaptive escalation after repeated guardrail
     blocks (`GuardrailEngine.needs_approval`). These used to *block* inside
     `tool_wrapper.call_capability` on an in-memory future for up to an
     hour, which nothing in `adk web` could ever resolve; now they become
     ordinary approval requests in the same durable flow.
  3. `default_effect` when nothing matched.

The most restrictive effect wins: deny > require_approval > allow. Every
decision carries its reasons and rule names, so an approver sees *why* a
call paused, and a denied call tells the model why it can't proceed.

Confidence (0-1) is computed per call, never assumed:
  - self-reported: the model passes `confidence` (and a `rationale`) as
    arguments to gated tools -- its own estimate that this call, now, with
    what it has seen, is right;
  - signals: observed facts about the run -- dependencies that haven't
    succeeded, earlier failed steps, failed attempts of this step, guardrail
    blocks on this capability -- each lowering the score, each reported as
    a factor;
  - combined (default): the lower of the two. A gate should trip if either
    the model or the evidence is unsure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.planner.schemas import PlanStep, StepKind
from app.registry.schemas import Capability
from app.specs.schemas import POLICY_FACTS, PolicyCondition, PolicyEffect, PolicyRule, ToolPolicyConfig

_RANK = {PolicyEffect.ALLOW: 0, PolicyEffect.REQUIRE_APPROVAL: 1, PolicyEffect.DENY: 2}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_CONFIDENCE_FACTS = {"confidence", "self_confidence", "signal_confidence"}

# Swappable for tests (time-of-day rules).
clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)  # noqa: E731


def _safe(step_id: str) -> str:
    from app.factory.spec_agents import _safe_identifier

    return _safe_identifier(step_id, fallback="step")


def step_tool_name(step: PlanStep) -> str:
    return f"step_{_safe(step.step_id)}"


def selector_names(step: PlanStep, capability: Capability | None) -> set[str]:
    names = {step.step_id, step_tool_name(step)}
    if step.capability_id:
        names.add(step.capability_id)
    if step.tool_name:
        names.add(step.tool_name)
    if capability is not None:
        names.update({capability.id, capability.name})
        if not step.tool_name and capability.tools:
            names.add(capability.tools[0].name)
    return names


def tools_match(selectors: list[str], step: PlanStep, capability: Capability | None) -> bool:
    if "*" in selectors:
        return True
    return bool(set(selectors) & selector_names(step, capability))


def _rule_uses_confidence(rule: PolicyRule) -> bool:
    return any(c.fact in _CONFIDENCE_FACTS for c in rule.when)


def confidence_gated(policy: ToolPolicyConfig, step: PlanStep, capability: Capability | None) -> bool:
    if policy.confidence_threshold is not None and tools_match(policy.confidence_tools, step, capability):
        return True
    return any(_rule_uses_confidence(r) and tools_match(r.tools, step, capability) for r in policy.rules)


def needs_self_report(policy: ToolPolicyConfig, step: PlanStep, capability: Capability | None) -> bool:
    """Whether this step's tool should ask the model for a `confidence`
    argument: only when confidence matters for it and the method uses the
    model's own estimate."""
    return policy.confidence_method in ("self_reported", "combined") and confidence_gated(policy, step, capability)


# -- run log & confidence -----------------------------------------------------------


@dataclass
class RunLog:
    """What has happened so far in this conversation for one spec, kept in
    session state by the step tools themselves (so it survives a pause for
    approval and the resume after it)."""

    results: dict[str, bool] = field(default_factory=dict)  # step_id -> last outcome
    attempts: dict[str, int] = field(default_factory=dict)  # step_id -> executions
    failures: dict[str, int] = field(default_factory=dict)  # step_id -> failed executions
    calls: int = 0

    @classmethod
    def from_state(cls, raw: dict | None) -> "RunLog":
        raw = raw or {}
        return cls(
            results=dict(raw.get("results", {})),
            attempts=dict(raw.get("attempts", {})),
            failures=dict(raw.get("failures", {})),
            calls=int(raw.get("calls", 0)),
        )

    def record(self, step_id: str, ok: bool) -> None:
        self.results[step_id] = ok
        self.attempts[step_id] = self.attempts.get(step_id, 0) + 1
        if not ok:
            self.failures[step_id] = self.failures.get(step_id, 0) + 1
        self.calls += 1

    def to_state(self) -> dict:
        return {"results": self.results, "attempts": self.attempts, "failures": self.failures, "calls": self.calls}


def signal_confidence(step: PlanStep, runlog: RunLog, *, guardrail_blocks: int = 0) -> tuple[float, list[str]]:
    score, factors = 1.0, []
    for dep in step.depends_on:
        if runlog.results.get(dep) is not True:
            score -= 0.5
            state = "failed" if runlog.results.get(dep) is False else "not run yet"
            factors.append(f"-0.50 depends on '{dep}', which has {state}")
    other_failures = sum(1 for sid, ok in runlog.results.items() if ok is False and sid != step.step_id)
    if other_failures:
        penalty = min(0.15 * other_failures, 0.45)
        score -= penalty
        factors.append(f"-{penalty:.2f} {other_failures} other step(s) failed earlier in this run")
    own_failures = runlog.failures.get(step.step_id, 0)
    if own_failures:
        penalty = 0.2 * own_failures
        score -= penalty
        factors.append(f"-{penalty:.2f} this step already failed {own_failures} time(s)")
    if guardrail_blocks:
        penalty = min(0.1 * guardrail_blocks, 0.3)
        score -= penalty
        factors.append(f"-{penalty:.2f} {guardrail_blocks} recent guardrail block(s) on this capability")
    if not factors:
        factors.append("no risk signals observed")
    return max(0.0, min(1.0, round(score, 4))), factors


def compute_confidence(
    policy: ToolPolicyConfig,
    step: PlanStep,
    runlog: RunLog,
    *,
    self_reported: float | None,
    guardrail_blocks: int = 0,
) -> dict[str, Any]:
    signal, factors = signal_confidence(step, runlog, guardrail_blocks=guardrail_blocks)
    self_value = None if self_reported is None else max(0.0, min(1.0, float(self_reported)))
    if policy.confidence_method == "signals":
        value = signal
    elif policy.confidence_method == "self_reported":
        value = self_value if self_value is not None else 0.0
        factors = [f"model reported {self_value:.2f}" if self_value is not None else "model reported no confidence"]
    else:
        if self_value is None:
            value = signal
            factors = factors + ["model reported no confidence"]
        else:
            value = min(signal, self_value)
            factors = [f"model reported {self_value:.2f}", f"signals {signal:.2f}"] + factors
    return {"confidence": round(value, 4), "self_confidence": self_value, "signal_confidence": signal, "factors": factors}


# -- facts & evaluation ------------------------------------------------------------


def gather_facts(
    *,
    step: PlanStep,
    capability: Capability | None,
    tool_name: str,
    arguments: dict[str, Any],
    runlog: RunLog,
    confidence: dict[str, Any],
    tenant_id: str,
    user_id: str,
) -> dict[str, Any]:
    now = clock()
    return {
        "step_id": step.step_id,
        "tool": step_tool_name(step),
        "tool_name": tool_name,
        "capability_id": step.capability_id,
        "capability_kind": capability.kind.value if capability is not None else "",
        "confidence": confidence["confidence"],
        "self_confidence": confidence["self_confidence"],
        "signal_confidence": confidence["signal_confidence"],
        "attempt": runlog.attempts.get(step.step_id, 0),
        "failed_steps": sum(1 for ok in runlog.results.values() if ok is False),
        "calls_in_run": runlog.calls,
        "hour_utc": now.hour,
        "weekday": _WEEKDAYS[now.weekday()],
        "tenant_id": tenant_id,
        "user_id": user_id,
        "args": arguments,
    }


_MISSING = object()


def _fact(facts: dict[str, Any], name: str) -> Any:
    if name.startswith("args."):
        node: Any = facts.get("args", {})
        for key in name[5:].split("."):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return _MISSING
        return node
    return facts.get(name, _MISSING)


def _holds(cond: PolicyCondition, facts: dict[str, Any]) -> bool:
    value = _fact(facts, cond.fact)
    if cond.op == "exists":
        return (value is not _MISSING and value is not None) == bool(cond.value if cond.value is not None else True)
    if value is _MISSING or value is None:
        return False
    try:
        if cond.op == "eq":
            return value == cond.value
        if cond.op == "ne":
            return value != cond.value
        if cond.op == "gt":
            return value > cond.value
        if cond.op == "gte":
            return value >= cond.value
        if cond.op == "lt":
            return value < cond.value
        if cond.op == "lte":
            return value <= cond.value
        if cond.op == "in":
            return value in (cond.value or [])
        if cond.op == "not_in":
            return value not in (cond.value or [])
        if cond.op == "contains":
            return cond.value in value
        if cond.op == "matches":
            return re.search(str(cond.value), str(value)) is not None
    except TypeError:
        return False  # incomparable types: the condition doesn't hold
    return False


@dataclass
class PolicyDecision:
    effect: PolicyEffect
    reasons: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)


def evaluate(
    policy: ToolPolicyConfig,
    step: PlanStep,
    capability: Capability | None,
    facts: dict[str, Any],
    *,
    platform_requirement: str | None = None,
) -> PolicyDecision:
    matched: list[tuple[PolicyEffect, str, str]] = []  # (effect, rule name, reason)

    if policy.require_approval_before and tools_match(policy.require_approval_before, step, capability):
        matched.append((PolicyEffect.REQUIRE_APPROVAL, "require_approval_before",
                        f"configured to require approval before '{step.step_id}'"))
    threshold = policy.confidence_threshold
    if threshold is not None and tools_match(policy.confidence_tools, step, capability) and facts["confidence"] < threshold:
        matched.append((PolicyEffect.REQUIRE_APPROVAL, "confidence_threshold",
                        f"confidence {facts['confidence']:.2f} is below the threshold {threshold:.2f}"))
    for rule in policy.rules:
        if tools_match(rule.tools, step, capability) and all(_holds(c, facts) for c in rule.when):
            reason = rule.reason or f"rule '{rule.name}' matched"
            matched.append((rule.effect, rule.name, reason))
    if platform_requirement:
        matched.append((PolicyEffect.REQUIRE_APPROVAL, "platform", platform_requirement))

    if not matched:
        effect = PolicyEffect(policy.default_effect)
        reason = "no rule allows this tool (default: deny)" if effect == PolicyEffect.DENY else ""
        return PolicyDecision(effect=effect, reasons=[reason] if reason else [], rules=["default"] if reason else [])

    effect = max((m[0] for m in matched), key=lambda e: _RANK[e])
    winners = [m for m in matched if m[0] == effect]
    return PolicyDecision(effect=effect, reasons=[m[2] for m in winners], rules=[m[1] for m in winners])


# -- approval-time validation --------------------------------------------------------


def validate_tool_policy(policy: ToolPolicyConfig, steps: list[PlanStep], capabilities: dict[str, Capability]) -> None:
    """Reject a policy that can't mean what its author intended: a selector
    that matches no step (usually a typo -- which would silently disable the
    approval it was meant to require), an unknown fact, a regex that doesn't
    compile, a malformed comparison, duplicate rule names."""
    tool_steps = [s for s in steps if s.kind != StepKind.HUMAN_APPROVAL]

    def _check_selectors(selectors: list[str], where: str) -> None:
        for selector in selectors:
            if selector == "*":
                continue
            if not any(selector in selector_names(s, capabilities.get(s.capability_id)) for s in tool_steps):
                known = sorted({n for s in tool_steps for n in selector_names(s, capabilities.get(s.capability_id))})
                raise ValueError(f"tool_policy.{where}: '{selector}' matches no step in this plan (known: {known})")

    _check_selectors(policy.require_approval_before, "require_approval_before")
    _check_selectors(policy.confidence_tools, "confidence_tools")

    names: set[str] = set()
    for index, rule in enumerate(policy.rules):
        where = f"rules[{index}] ('{rule.name}')"
        if rule.name in names:
            raise ValueError(f"tool_policy.{where}: duplicate rule name")
        names.add(rule.name)
        _check_selectors(rule.tools, f"{where}.tools")
        for cond in rule.when:
            if cond.fact not in POLICY_FACTS and not cond.fact.startswith("args."):
                raise ValueError(
                    f"tool_policy.{where}: unknown fact '{cond.fact}' (use one of {sorted(POLICY_FACTS)} or args.<name>)"
                )
            if cond.op == "matches":
                try:
                    re.compile(str(cond.value))
                except re.error as exc:
                    raise ValueError(f"tool_policy.{where}: invalid regex {cond.value!r}: {exc}") from exc
            if cond.op in ("in", "not_in") and not isinstance(cond.value, list):
                raise ValueError(f"tool_policy.{where}: '{cond.op}' needs a list value")
            if cond.op in ("gt", "gte", "lt", "lte") and not isinstance(cond.value, (int, float)):
                raise ValueError(f"tool_policy.{where}: '{cond.op}' needs a numeric value")
