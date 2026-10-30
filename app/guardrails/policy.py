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
from functools import lru_cache
from typing import Literal


@dataclass
class GuardrailPolicy:
    id: str
    tenant_id: str = "*"
    denied_tool_name_patterns: list[str] = field(default_factory=list)
    blocked_input_keywords: list[str] = field(default_factory=lambda: [
        "ignore previous instructions", "ignore all previous instructions", "disregard previous instructions",
        "system prompt", "reveal your instructions",
    ])
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


_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def normalize(text: str) -> str:
    """Defeat trivial obfuscation before keyword screening: Unicode
    compatibility folding (full-width letters etc.), zero-width characters
    removed, case-folded, whitespace collapsed."""
    import unicodedata

    folded = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).casefold()
    return " ".join(folded.split())


@lru_cache(maxsize=4096)
def _compiled(pattern: str) -> re.Pattern:
    return re.compile(pattern)


def contains_blocked_keyword(text: str, policy: GuardrailPolicy) -> str | None:
    lowered = normalize(text[:200_000])
    for kw in policy.blocked_input_keywords:
        if normalize(kw) in lowered:
            return kw
    return None


def tool_name_denied(tool_name: str, policy: GuardrailPolicy) -> bool:
    return any(_compiled(pattern).search(tool_name) for pattern in policy.denied_tool_name_patterns)


def redact(text: str, policy: GuardrailPolicy) -> str:
    out = text[: policy.max_output_chars * 2]  # bound regex work on huge outputs
    for pattern, replacement in policy.pii_redaction_patterns.items():
        out = _compiled(pattern).sub(replacement, out)
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
    return pattern == "*" or bool(_compiled(pattern).search(value))


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

# Windowed, fleet-wide counters: each block is INCRed into an hourly Redis
# bucket (`gb:{tenant}:<capability>:<hour>`, expiring after two windows), and
# the count is a sliding-window estimate -- current bucket + the previous one
# weighted by how much of it still overlaps the window. So escalation sees
# blocks from every replica, and an old spike stops escalating after an hour
# instead of forever. A local copy of the same windows backs the synchronous
# API and covers Redis being briefly unavailable.
WINDOW_SECONDS = 3600
_local_buckets: dict[tuple[str, str, int], int] = {}


def _bucket(now: float | None = None) -> int:
    return int((now or time.time()) // WINDOW_SECONDS)


def _sliding(current: int, previous: int, now: float | None = None) -> int:
    elapsed = ((now or time.time()) % WINDOW_SECONDS) / WINDOW_SECONDS
    return int(current + previous * (1 - elapsed))


def _redis_key(tenant_id: str, capability_id: str, bucket: int) -> str:
    from app.core.redis import key

    return key("gb", tenant_id, capability_id, str(bucket))


async def _record_block_redis(tenant_id: str, capability_id: str, bucket: int) -> None:
    try:
        from app.core.redis import get_redis

        client = await get_redis()
        k = _redis_key(tenant_id, capability_id, bucket)
        pipe = client.pipeline(transaction=False)
        pipe.incr(k)
        pipe.expire(k, 2 * WINDOW_SECONDS)
        await pipe.execute()
    except Exception:  # noqa: BLE001 - counters are advisory
        pass


def record_block(tenant_id: str, capability_id: str) -> int:
    b = _bucket()
    key_ = (tenant_id, capability_id, b)
    _local_buckets[key_] = _local_buckets.get(key_, 0) + 1
    for stale in [k for k in _local_buckets if k[2] < b - 1]:
        _local_buckets.pop(stale, None)
    try:
        import asyncio

        asyncio.get_running_loop().create_task(_record_block_redis(tenant_id, capability_id, b))
    except RuntimeError:
        pass
    return block_count(tenant_id, capability_id)


def block_count(tenant_id: str, capability_id: str) -> int:
    b = _bucket()
    return _sliding(_local_buckets.get((tenant_id, capability_id, b), 0),
                    _local_buckets.get((tenant_id, capability_id, b - 1), 0))


async def block_count_async(tenant_id: str, capability_id: str) -> int:
    local = block_count(tenant_id, capability_id)
    try:
        from app.core.redis import get_redis

        client = await get_redis()
        b = _bucket()
        cur, prev = await client.mget(_redis_key(tenant_id, capability_id, b),
                                      _redis_key(tenant_id, capability_id, b - 1))
        return max(local, _sliding(int(cur or 0), int(prev or 0)))
    except Exception:  # noqa: BLE001
        return local


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
