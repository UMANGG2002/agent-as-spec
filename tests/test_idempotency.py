from __future__ import annotations

import pytest

from app.auth import delegation
from app.auth.models import IdentityContext
from app.factory import tool_wrapper
from app.factory.tool_wrapper import ToolCallContext
from app.guardrails.policy import GuardrailPolicy
from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility


def _identity_and_run_token():
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))
    run_token = delegation.delegate_to_run(identity, run_id="run_idem", plan_scopes=())
    return identity, run_token


@pytest.mark.asyncio
async def test_non_idempotent_call_is_not_re_dispatched_on_retry(monkeypatch):
    calls = []

    async def fake_dispatch(capability, tool_name, args, *, bearer_token):
        calls.append(1)
        return {"result": "charged once"}

    monkeypatch.setattr(tool_wrapper, "_dispatch", fake_dispatch)

    capability = Capability(
        id="cap_charge",
        tenant_id="acme",
        kind=CapabilityKind.OPENAPI,
        name="billing",
        tools=[ToolSchema(name="charge_card", idempotent=False)],
        visibility=Visibility.PRIVATE,
        endpoint_ref=EndpointRef(openapi_base_url="http://billing.test"),
    )
    identity, run_token = _identity_and_run_token()
    ctx = ToolCallContext(identity=identity, run_id="run_idem", run_token=run_token, policy=GuardrailPolicy(id="default"))

    args = {"amount": 100}
    first = await tool_wrapper.call_capability(capability, "charge_card", args, ctx, step_id="step_1")
    second = await tool_wrapper.call_capability(capability, "charge_card", args, ctx, step_id="step_1")

    assert first.ok and second.ok
    assert first.output == second.output
    assert len(calls) == 1  # the second call hit the idempotency cache, not the dispatcher


@pytest.mark.asyncio
async def test_idempotent_call_dispatches_every_time(monkeypatch):
    calls = []

    async def fake_dispatch(capability, tool_name, args, *, bearer_token):
        calls.append(1)
        return {"value": len(calls)}

    monkeypatch.setattr(tool_wrapper, "_dispatch", fake_dispatch)

    capability = Capability(
        id="cap_lookup",
        tenant_id="acme",
        kind=CapabilityKind.OPENAPI,
        name="lookup",
        tools=[ToolSchema(name="get_status", idempotent=True)],
        visibility=Visibility.PRIVATE,
        endpoint_ref=EndpointRef(openapi_base_url="http://lookup.test"),
    )
    identity, run_token = _identity_and_run_token()
    ctx = ToolCallContext(identity=identity, run_id="run_idem2", run_token=run_token, policy=GuardrailPolicy(id="default"))

    await tool_wrapper.call_capability(capability, "get_status", {}, ctx, step_id="step_1")
    await tool_wrapper.call_capability(capability, "get_status", {}, ctx, step_id="step_1")

    assert len(calls) == 2  # idempotent tools are never cached -- always re-dispatched
