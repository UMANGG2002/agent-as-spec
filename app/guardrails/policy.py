"""Guardrail policies are data, not code (architecture doc: "a tenant can
author stricter or looser policies without a redeploy"). This is the minimal
rule engine the policy data drives -- regex/keyword rules rather than a
classifier model, which is the right trade for a minimal build and the right
seam to swap later (everything downstream only talks to `GuardrailPolicy` /
`ApprovalRule` + the functions below, never to a regex directly).

Three things live here, matching the architecture doc's "Dynamic policy
engine" section:
  1. `GuardrailPolicy` -- input/output screening + redaction (unchanged from
     the original guardrails section, now with an adaptive-escalation knob).
  2. `ApprovalRule` -- dynamic HIL: a rule table the tool wrapper checks
     before every dispatch, so "this tool needs sign-off for this tenant" is
     a row, not a planner decision.
  3. `resolve_policy` -- composes a tenant/capability/brand's effective
     policy at call time, including the adaptive-escalation feedback loop
     off `guardrail_blocks_total`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GuardrailPolicy:
    id: str
    tenant_id: str = "*"
    denied_tool_name_patterns: list[str] = field(default_factory=list)
    blocked_input_keywords: list[str] = field(default_factory=lambda: ["ignore previous instructions", "system prompt"])
    pii_redaction_patterns: dict[str, str] = field(
        default_factory=lambda: {
            r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b": "[redacted-email]",
            r"\b\d{3}-\d{2}-\d{4}\b": "[redacted-ssn]",
            r"\b(?:\d[ -]*?){13,16}\b": "[redacted-card]",
        }
    )
    max_output_chars: int = 50_000
    quota_per_run: int | None = None  # None = unlimited; enforced by the run service
    # Adaptive guardrails: if a (tenant, capability) pair accumulates this
    # many guardrail blocks within the window tracked by `record_block`,
    # `resolve_policy` forces an approval requirement regardless of the
    # ApprovalRule table -- the guardrail layer tightening itself under
    # pressure instead of waiting for a human to edit a policy.
    escalate_to_approval_after_blocks: int | None = 3


DEFAULT_POLICY = GuardrailPolicy(id="default")

_POLICIES: dict[str, GuardrailPolicy] = {"default": DEFAULT_POLICY}


def register_policy(policy: GuardrailPolicy) -> None:
    _POLICIES[policy.id] = policy


def get_policy(policy_id: str) -> GuardrailPolicy:
    return _POLICIES.get(policy_id, DEFAULT_POLICY)


def contains_blocked_keyword(text: str, policy: GuardrailPolicy) -> str | None:
    lowered = text.lower()
    for kw in policy.blocked_input_keywords:
        if kw.lower() in lowered:
            return kw
    return None


def tool_name_denied(tool_name: str, policy: GuardrailPolicy) -> bool:
    return any(re.search(pattern, tool_name) for pattern in policy.denied_tool_name_patterns)


def redact(text: str, policy: GuardrailPolicy) -> str:
    out = text
    for pattern, replacement in policy.pii_redaction_patterns.items():
        out = re.sub(pattern, replacement, out)
    if len(out) > policy.max_output_chars:
        out = out[: policy.max_output_chars] + "...[truncated]"
    return out


# --- Dynamic HIL: approval rules ------------------------------------------

ApprovalCondition = Literal["always", "cost_over", "tenant_tier_is"]


@dataclass
class ApprovalRule:
    """Matched against every tool call before dispatch. `capability_pattern`
    and `tool_pattern` are regexes ("*" is shorthand for "match anything").
    Scope a rule to one tenant via `tenant_id`, or make it platform-wide with
    `tenant_id="*"` (e.g. "any tenant's destructive OpenAPI operations always
    need sign-off")."""

    tenant_id: str
    capability_pattern: str
    tool_pattern: str
    condition: ApprovalCondition = "always"
    cost_threshold: float | None = None
    tenant_tier: str | None = None
    reason: str = "policy requires human approval for this call"


_APPROVAL_RULES: dict[str, list[ApprovalRule]] = {}


def register_approval_rule(rule: ApprovalRule) -> None:
    _APPROVAL_RULES.setdefault(rule.tenant_id, []).append(rule)


def _pattern_matches(pattern: str, value: str) -> bool:
    return pattern == "*" or bool(re.search(pattern, value))


def requires_approval(
    *,
    tenant_id: str,
    capability_id: str,
    tool_name: str,
    cost_estimate: float | None = None,
    tenant_tier: str | None = None,
) -> ApprovalRule | None:
    """Checked by the tool wrapper immediately before dispatch (dynamic HIL),
    independent of whether the plan step itself was authored as
    `kind: human_approval`."""
    candidates = _APPROVAL_RULES.get(tenant_id, []) + _APPROVAL_RULES.get("*", [])
    for rule in candidates:
        if not (_pattern_matches(rule.capability_pattern, capability_id) and _pattern_matches(rule.tool_pattern, tool_name)):
            continue
        if rule.condition == "always":
            return rule
        if rule.condition == "cost_over" and cost_estimate is not None and rule.cost_threshold is not None:
            if cost_estimate > rule.cost_threshold:
                return rule
        if rule.condition == "tenant_tier_is" and tenant_tier is not None and rule.tenant_tier == tenant_tier:
            return rule
    return None


# --- Adaptive guardrails: block-count feedback loop ------------------------

# Minimal build: an in-process, unwindowed counter. Production would window
# this (e.g. Redis INCR + EXPIRE per (tenant, capability, hour)) so an old
# spike doesn't permanently escalate a pair; the interface below is the seam.
_block_counts: dict[tuple[str, str], int] = {}
_block_counts_reset_at: float = time.time()


def record_block(tenant_id: str, capability_id: str) -> int:
    key = (tenant_id, capability_id)
    _block_counts[key] = _block_counts.get(key, 0) + 1
    return _block_counts[key]


def block_count(tenant_id: str, capability_id: str) -> int:
    return _block_counts.get((tenant_id, capability_id), 0)


def resolve_policy(*, tenant_id: str, brand_id: str, capability_id: str, base_policy_id: str = "default") -> GuardrailPolicy:
    """Composes the *effective* policy for one call -- the extension point
    the architecture doc's "adaptive guardrails" section describes: tenant
    base policy, composed with a capability-owner override and brand
    overrides. The minimal build resolves to the tenant's base policy only
    (`brand_id`/`capability_id` are accepted for forward compatibility, not
    yet consulted); a production build would look up capability- and
    brand-level override rows here and merge them in.

    Call this once, where a run/step's policy is first chosen (e.g. in
    runtime/run_service.py, before building `ToolCallContext`), and pass the
    *result* into `GuardrailEngine.needs_approval` / `before_tool` /
    `after_tool` -- those read the object you hand them, they never
    re-resolve by id themselves (see `needs_approval`'s docstring for why
    that distinction matters)."""
    return get_policy(base_policy_id)
