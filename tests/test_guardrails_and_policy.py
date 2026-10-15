from __future__ import annotations

from app.auth.models import IdentityContext
from app.guardrails import policy as policy_mod
from app.guardrails.plugin import guardrail_engine
from app.guardrails.policy import ApprovalRule, GuardrailPolicy
from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility


def _capability(required_scopes=None) -> Capability:
    return Capability(
        id="cap_test",
        tenant_id="acme",
        kind=CapabilityKind.OPENAPI,
        name="test-cap",
        tools=[ToolSchema(name="delete_everything")],
        required_scopes=required_scopes or [],
        visibility=Visibility.PRIVATE,
        endpoint_ref=EndpointRef(openapi_base_url="http://x.test"),
    )


def test_tool_denylist_blocks():
    policy = GuardrailPolicy(id="strict", denied_tool_name_patterns=["delete_.*"])
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))
    decision = guardrail_engine.before_tool(
        policy=policy, capability=_capability(), tool_name="delete_everything", args={}, identity=identity
    )
    assert decision.blocked
    assert "denylisted" in decision.reason


def test_scope_check_blocks_when_identity_lacks_scope():
    policy = GuardrailPolicy(id="default")
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("read:only",))
    decision = guardrail_engine.before_tool(
        policy=policy, capability=_capability(required_scopes=["write:admin"]), tool_name="run", args={}, identity=identity
    )
    assert decision.blocked


def test_redaction_masks_pii():
    policy = GuardrailPolicy(id="default")
    text = "contact me at jane@example.com"
    redacted = policy_mod.redact(text, policy)
    assert "jane@example.com" not in redacted
    assert "[redacted-email]" in redacted


def test_approval_rule_always_matches():
    policy_mod.register_approval_rule(
        ApprovalRule(tenant_id="acme", capability_pattern="cap_test", tool_pattern="delete_.*", condition="always")
    )
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))
    needs, reason = guardrail_engine.needs_approval(
        policy=GuardrailPolicy(id="default", escalate_to_approval_after_blocks=None),
        capability=_capability(),
        tool_name="delete_everything",
        identity=identity,
    )
    assert needs is True
    assert reason


def test_adaptive_escalation_forces_approval_after_repeated_blocks():
    policy = GuardrailPolicy(id="escalating", escalate_to_approval_after_blocks=2)
    identity = IdentityContext(tenant_id="acme2", brand_id="default", user_id="u1", scopes=("*",))
    cap = _capability()
    cap.id = "cap_escalation_test"

    needs, _ = guardrail_engine.needs_approval(policy=policy, capability=cap, tool_name="run", identity=identity)
    assert needs is False  # no blocks recorded yet

    policy_mod.record_block("acme2", "cap_escalation_test")
    policy_mod.record_block("acme2", "cap_escalation_test")

    needs, reason = guardrail_engine.needs_approval(policy=policy, capability=cap, tool_name="run", identity=identity)
    assert needs is True
    assert "escalated" in reason
