"""The one place every capability invocation passes through, regardless of
source kind (MCP / OpenAPI / agent card) or whether it's reached via a real
ADK `FunctionTool` or the deterministic executor. This is what makes "OAuth
delegation + guardrails + telemetry on every call" true by construction
rather than by every call site remembering to do it -- see the architecture
doc's guardrails and OAuth sections.

Call order per invocation, matching the doc's "Dynamic policy engine"
section:
  before_tool guardrail -> idempotency check (return cached result, skip the
    rest, if this exact run+step+args already executed) -> dynamic-HIL check
    (ApprovalRule match or adaptive escalation -> suspend for approval) ->
    scope re-check -> token exchange -> span open -> actual call -> store
    idempotent result -> after_tool guardrail (redact/sanitize) -> span close
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from opentelemetry import trace

from app.auth import delegation
from app.auth.models import DelegatedToken, IdentityContext
from app.guardrails.plugin import GuardrailDecision, guardrail_engine
from app.guardrails.policy import GuardrailPolicy
from app.registry.schemas import Capability, CapabilityKind
from app.runtime import cache as result_cache
from app.runtime.hil import ApprovalStatus, approval_store
from app.telemetry.setup import tool_latency_ms

tracer = trace.get_tracer("capability_marketplace.tool")


@dataclass
class ToolCallContext:
    identity: IdentityContext
    run_id: str
    run_token: DelegatedToken
    policy: GuardrailPolicy


@dataclass
class ToolCallResult:
    ok: bool
    output: Any = None
    error: str | None = None
    blocked_by: str | None = None  # guardrail policy id, when blocked


async def call_capability(
    capability: Capability,
    tool_name: str,
    args: dict[str, Any],
    ctx: ToolCallContext,
    *,
    step_id: str | None = None,
    cost_estimate: float | None = None,
    approval_handled: bool = False,
) -> ToolCallResult:
    """`approval_handled=True` means the caller already evaluated approval
    requirements (including the platform ApprovalRule / adaptive escalation
    checked below) and obtained any approval through the durable approvals
    flow -- compiled spec steps do this (app/factory/tool_policy.py), since
    the in-memory wait below can't be resolved from another process. The
    deterministic HTTP executor still uses the in-memory path."""
    step_id = step_id or tool_name
    # None, not a placeholder scope string: a capability that declares no
    # required_scopes needs no scope check at all (see delegation.py's
    # exchange_for_tool_call) -- treating "none declared" as a scope named
    # "default" was a real bug: it demanded a scope no identity actually has.
    required_scope = capability.required_scopes[0] if capability.required_scopes else None

    before: GuardrailDecision = guardrail_engine.before_tool(
        policy=ctx.policy, capability=capability, tool_name=tool_name, args=args, identity=ctx.identity
    )
    if before.blocked:
        return ToolCallResult(ok=False, blocked_by=ctx.policy.id, error=before.reason)

    tool_schema = next((t for t in capability.tools if t.name == tool_name), None)
    is_idempotent = bool(tool_schema and tool_schema.idempotent)
    idem_key = None
    idem_lock = None
    if not is_idempotent:
        # Exactly-once *effect* for side-effecting calls: a cached result
        # short-circuits a retry/redelivery/resume; a short Redis lock stops
        # two workers executing the same (run, step, args) concurrently.
        idem_key = result_cache.idempotency_key(ctx.run_id, step_id, args)
        cached = await result_cache.get(idem_key)
        if cached is not None:
            return ToolCallResult(ok=True, output=cached)
        idem_lock = await result_cache.acquire_execution_lock(idem_key)
        if idem_lock is None:
            cached = await result_cache.wait_for_result(idem_key)
            if cached is not None:
                return ToolCallResult(ok=True, output=cached)
            return ToolCallResult(ok=False, error="the same call is already executing; retry shortly")

    needs_approval, reason = (False, None) if approval_handled else await guardrail_engine.needs_approval_async(
        policy=ctx.policy, capability=capability, tool_name=tool_name, identity=ctx.identity, cost_estimate=cost_estimate
    )
    if needs_approval:
        status = await approval_store.request_approval(ctx.run_id, step_id, reason=reason or "approval required")
        if status != ApprovalStatus.APPROVED:
            await result_cache.release_execution_lock(idem_key, idem_lock)
            return ToolCallResult(ok=False, blocked_by=ctx.policy.id, error=f"approval not granted ({status.value}): {reason}")

    try:
        tool_token = delegation.exchange_for_tool_call(
            ctx.run_token, capability_id=capability.id, required_scope=required_scope
        )
    except PermissionError as exc:
        return ToolCallResult(ok=False, error=str(exc))

    start = time.perf_counter()
    with tracer.start_as_current_span(
        "tool.call",
        attributes={
            "capability.id": capability.id,
            "capability.kind": capability.kind.value,
            "tool.name": tool_name,
            "tenant.id": ctx.identity.tenant_id,
            "brand.id": ctx.identity.brand_id,
            "user.id": ctx.identity.user_id,
            "run.id": ctx.run_id,
            "delegation.act_chain": "->".join(tool_token.identity.act_chain),
        },
    ) as span:
        try:
            output = await _resilient_dispatch(capability, tool_name, args, bearer_token=tool_token.encoded,
                                               idempotent=is_idempotent)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a failed step, not a crash
            span.record_exception(exc)
            await result_cache.release_execution_lock(idem_key, idem_lock)
            return ToolCallResult(ok=False, error=f"{type(exc).__name__}: {exc}"[:2000])
        finally:
            tool_latency_ms.record(
                (time.perf_counter() - start) * 1000,
                {"capability_kind": capability.kind.value},
            )

    after = guardrail_engine.after_tool(policy=ctx.policy, capability=capability, output=output)
    if after.blocked:
        await result_cache.release_execution_lock(idem_key, idem_lock)
        return ToolCallResult(ok=False, blocked_by=ctx.policy.id, error=after.reason)

    final_output = after.redacted_output if after.redacted_output is not None else output
    if idem_key is not None:
        await result_cache.set(idem_key, final_output, ttl_seconds=86_400)
        await result_cache.release_execution_lock(idem_key, idem_lock)
    return ToolCallResult(ok=True, output=final_output)


async def _resilient_dispatch(capability: Capability, tool_name: str, args: dict[str, Any], *, bearer_token: str,
                              idempotent: bool) -> Any:
    """Every outbound call: bulkhead (bounded concurrency per capability) ->
    circuit breaker (fail fast while the endpoint is down) -> timeout ->
    retries with jittered backoff, only for idempotent tools and only on
    transport errors (never re-send a side-effecting call blindly)."""
    import asyncio

    import httpx

    from app.config import settings
    from app.core import resilience

    cb = resilience.breaker(f"cap:{capability.id}")

    async def _once() -> Any:
        return await asyncio.wait_for(_dispatch(capability, tool_name, args, bearer_token=bearer_token),
                                      timeout=settings.tool_call_timeout_seconds)

    async def _guarded() -> Any:
        if not idempotent:
            return await cb.call(_once)
        return await cb.call(lambda: resilience.retry(
            _once, attempts=settings.tool_call_max_retries,
            retry_on=(httpx.TransportError, asyncio.TimeoutError, ConnectionError)))

    return await resilience.with_bulkhead(f"cap:{capability.id}", _guarded)


async def _dispatch(capability: Capability, tool_name: str, args: dict[str, Any], *, bearer_token: str) -> Any:
    """Source-specific invocation. Each branch is a thin adapter call -- the
    guardrail/auth/telemetry wrapping above is identical regardless of kind."""
    if capability.kind == CapabilityKind.MCP:
        from app.registry.adapters import mcp_pool

        # Pooled, long-lived MCP connections (the old code connected, listed
        # tools and disconnected on every single call).
        return await mcp_pool.call_tool(capability, tool_name, args)

    if capability.kind == CapabilityKind.OPENAPI:
        from app.core import http as http_core

        base_url = (capability.endpoint_ref.openapi_base_url or "").rstrip("/")
        await http_core.check_url(base_url)
        headers = {capability.auth_header_name or "Authorization": f"Bearer {bearer_token}"}
        # The operationId names the tool; a fuller adapter keeps the parsed
        # method+path mapping from openapi_adapter.py.
        resp = await http_core.client().post(f"{base_url}/{tool_name}", json=args, headers=headers)
        resp.raise_for_status()
        return resp.json()

    if capability.kind == CapabilityKind.AGENT_CARD:
        import json

        from app.registry.adapters import agent_card_adapter

        # A2A skills take a text message. Send the step's text as-is when
        # there is one (previously this sent str(args) -- a Python dict repr
        # -- whenever args had no "input" key, which is every step that uses
        # the default {"instruction": ...} args); fall back to JSON for
        # structured args, which is at least a format the remote agent can
        # parse.
        text_arg = next((args[k] for k in ("input", "text", "instruction") if isinstance(args.get(k), str)), None)
        input_text = text_arg if text_arg is not None else json.dumps(args)
        return await agent_card_adapter.invoke_remote_skill(
            capability, skill_id=tool_name, input_text=input_text, bearer_token=bearer_token
        )

    raise ValueError(f"unsupported capability kind: {capability.kind}")
