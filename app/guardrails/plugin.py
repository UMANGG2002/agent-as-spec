"""The six-hook guardrail engine from the architecture doc's table, plus an
ADK `BasePlugin` adapter so the exact same engine governs both the real ADK
agent-tree execution path and the deterministic executor path (tool_wrapper
calls `guardrail_engine.before_tool`/`after_tool` directly; the Plugin below
is what makes the same checks run automatically for an ADK-compiled agent's
own model/agent-level turns, which the deterministic path doesn't have).

Registered once on the `App` (see runtime/run_service.py), not per-agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentelemetry import trace

from app.auth.models import IdentityContext
from app.guardrails import policy as policy_mod
from app.guardrails.policy import GuardrailPolicy
from app.registry.schemas import Capability
from app.telemetry.setup import guardrail_blocks_total

tracer = trace.get_tracer("capability_marketplace.guardrails")


@dataclass
class GuardrailDecision:
    blocked: bool
    reason: str | None = None
    redacted_output: Any | None = None


class GuardrailEngine:
    """Pure logic, framework-agnostic -- callable from an ADK callback, from
    the deterministic executor, or from a unit test with no ADK involved."""

    def before_agent(self, *, identity: IdentityContext, capability: Capability | None) -> GuardrailDecision:
        if capability and not any(identity.has_scope(s) for s in (capability.required_scopes or ["*"])):
            self._record_block("before_agent", "default")
            return GuardrailDecision(blocked=True, reason="caller lacks required scope for this capability")
        return GuardrailDecision(blocked=False)

    def before_model(self, *, text: str, policy: GuardrailPolicy) -> GuardrailDecision:
        hit = policy_mod.contains_blocked_keyword(text, policy)
        if hit:
            self._record_block("before_model", policy.id)
            return GuardrailDecision(blocked=True, reason=f"blocked input pattern: '{hit}'")
        return GuardrailDecision(blocked=False)

    def after_model(self, *, text: str, policy: GuardrailPolicy) -> GuardrailDecision:
        redacted = policy_mod.redact(text, policy)
        return GuardrailDecision(blocked=False, redacted_output=redacted if redacted != text else None)

    def before_tool(
        self,
        *,
        policy: GuardrailPolicy,
        capability: Capability,
        tool_name: str,
        args: dict[str, Any],
        identity: IdentityContext,
    ) -> GuardrailDecision:
        if policy_mod.tool_name_denied(tool_name, policy):
            self._record_block("before_tool", policy.id, tenant_id=identity.tenant_id, capability_id=capability.id)
            return GuardrailDecision(blocked=True, reason=f"tool '{tool_name}' is denylisted by policy {policy.id}")
        required = capability.required_scopes
        if required and not any(identity.has_scope(s) for s in required):
            self._record_block("before_tool", policy.id, tenant_id=identity.tenant_id, capability_id=capability.id)
            return GuardrailDecision(blocked=True, reason="scope check failed at tool boundary")
        return GuardrailDecision(blocked=False)

    def after_tool(self, *, policy: GuardrailPolicy, capability: Capability, output: Any) -> GuardrailDecision:
        if isinstance(output, str):
            redacted = policy_mod.redact(output, policy)
            return GuardrailDecision(blocked=False, redacted_output=redacted if redacted != output else None)
        return GuardrailDecision(blocked=False)

    def needs_approval(
        self,
        *,
        policy: GuardrailPolicy,
        capability: Capability,
        tool_name: str,
        identity: IdentityContext,
        cost_estimate: float | None = None,
    ) -> tuple[bool, str | None]:
        """Dynamic HIL check: a static `ApprovalRule` match, or an adaptive
        escalation triggered by `policy.escalate_to_approval_after_blocks`
        recent guardrail blocks on this (tenant, capability) pair. Checked by
        `tool_wrapper.call_capability` immediately before dispatch, in
        addition to -- not instead of -- any plan step explicitly authored as
        `kind: human_approval`.

        Note this reads the threshold off the `policy` object passed in, not
        a re-fetch by id -- `policy` may already be the effective, per-call
        policy a caller resolved (e.g. via `policy_mod.resolve_policy` at a
        higher layer), and re-fetching by id here would silently discard
        that and fall back to whatever is registered globally under the same
        id.
        """
        threshold = policy.escalate_to_approval_after_blocks
        if threshold is not None and policy_mod.block_count(identity.tenant_id, capability.id) >= threshold:
            return True, f"escalated: repeated guardrail blocks on {capability.id} for {identity.tenant_id}"
        rule = policy_mod.requires_approval(
            tenant_id=identity.tenant_id, capability_id=capability.id, tool_name=tool_name, cost_estimate=cost_estimate
        )
        if rule:
            return True, rule.reason
        return False, None

    async def needs_approval_async(
        self,
        *,
        policy: GuardrailPolicy,
        capability: Capability,
        tool_name: str,
        identity: IdentityContext,
        cost_estimate: float | None = None,
    ) -> tuple[bool, str | None]:
        """`needs_approval` with the fleet-wide (Redis) block count."""
        threshold = policy.escalate_to_approval_after_blocks
        if threshold is not None and await policy_mod.block_count_async(identity.tenant_id, capability.id) >= threshold:
            return True, f"escalated: repeated guardrail blocks on {capability.id} for {identity.tenant_id}"
        rule = policy_mod.requires_approval(
            tenant_id=identity.tenant_id, capability_id=capability.id, tool_name=tool_name, cost_estimate=cost_estimate
        )
        if rule:
            return True, rule.reason
        return False, None

    def _record_block(self, stage: str, policy_id: str, *, tenant_id: str | None = None, capability_id: str | None = None) -> None:
        guardrail_blocks_total.add(1, {"stage": stage, "policy_id": policy_id})
        if tenant_id and capability_id:
            policy_mod.record_block(tenant_id, capability_id)


guardrail_engine = GuardrailEngine()


def _build_base_plugin_class():
    """Deferred import: the ADK `BasePlugin` class is only needed when the
    real ADK agent-tree execution path is used."""
    from google.adk.plugins.base_plugin import BasePlugin
    from google.genai import types

    class GuardrailPlugin(BasePlugin):
        """Registered once per `App`; reads `identity`/`policy_id` out of
        session state (set by runtime/agent_factory when a run starts)."""

        def __init__(self) -> None:
            super().__init__(name="guardrail_plugin")

        async def before_model_callback(self, *, callback_context, llm_request):  # noqa: ANN001
            policy = policy_mod.get_policy(callback_context.state.get("policy_id", "default"))
            # The person's own message for this invocation, plus the latest
            # user-role text: after a transfer_to_agent, the latter is ADK's
            # injected "For context: ..." note, not what the person typed.
            user_content = getattr(callback_context, "user_content", None)
            own = " ".join(p.text for p in (user_content.parts or []) if p.text) if user_content else ""
            text = f"{own} {_last_user_text(llm_request)}".strip()
            decision = guardrail_engine.before_model(text=text, policy=policy)
            if decision.blocked:
                from google.adk.models import LlmResponse

                return LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text=f"Request blocked: {decision.reason}")])
                )
            return None

        async def after_model_callback(self, *, callback_context, llm_response):  # noqa: ANN001
            policy = policy_mod.get_policy(callback_context.state.get("policy_id", "default"))
            if not llm_response.content or not llm_response.content.parts:
                return None
            text = llm_response.content.parts[0].text or ""
            decision = guardrail_engine.after_model(text=text, policy=policy)
            if decision.redacted_output is not None:
                llm_response.content.parts[0].text = decision.redacted_output
                return llm_response
            return None

        async def before_tool_callback(self, *, tool, tool_args, tool_context):  # noqa: ANN001
            policy = policy_mod.get_policy(tool_context.state.get("policy_id", "default"))
            if policy_mod.tool_name_denied(tool.name, policy):
                guardrail_engine._record_block("before_tool", policy.id)
                return {"error": f"tool '{tool.name}' is denylisted by policy {policy.id}"}
            return None

        async def after_tool_callback(self, *, tool, tool_args, tool_context, result):  # noqa: ANN001
            policy = policy_mod.get_policy(tool_context.state.get("policy_id", "default"))
            if isinstance(result, dict) and "output" in result and isinstance(result["output"], str):
                redacted = policy_mod.redact(result["output"], policy)
                if redacted != result["output"]:
                    return {**result, "output": redacted}
            return None

    def _last_user_text(llm_request) -> str:  # noqa: ANN001
        for content in reversed(llm_request.contents or []):
            if content.role == "user" and content.parts:
                return content.parts[0].text or ""
        return ""

    return GuardrailPlugin


def build_guardrail_plugin():
    """Call this (not the class directly) so importing this module never
    requires google-adk to be installed."""
    return _build_base_plugin_class()()
