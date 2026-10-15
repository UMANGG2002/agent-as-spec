from __future__ import annotations

import pytest

from app.auth import delegation
from app.auth.models import IdentityContext


def test_unscoped_identity_keeps_scopes_through_run_delegation():
    """Regression test for the bug the smoke test caught: a plan whose steps
    declare no required_scopes must NOT narrow an unrestricted identity down
    to zero scopes."""
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))
    run_token = delegation.delegate_to_run(identity, run_id="run_1", plan_scopes=())
    assert run_token.identity.scopes == ("*",)


def test_plan_scopes_narrow_when_declared():
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*", "read:kb"))
    run_token = delegation.delegate_to_run(identity, run_id="run_1", plan_scopes=("read:kb",))
    assert run_token.identity.scopes == ("read:kb",)


def test_exchange_for_tool_call_with_no_required_scope_does_not_raise():
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))
    run_token = delegation.delegate_to_run(identity, run_id="run_1", plan_scopes=())
    tool_token = delegation.exchange_for_tool_call(run_token, capability_id="cap_1", required_scope=None)
    assert tool_token.identity.act_chain[-1] == "tool:cap_1"


def test_exchange_for_tool_call_rejects_missing_scope():
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("read:only",))
    run_token = delegation.delegate_to_run(identity, run_id="run_1", plan_scopes=("read:only",))
    with pytest.raises(PermissionError):
        delegation.exchange_for_tool_call(run_token, capability_id="cap_1", required_scope="write:admin")
