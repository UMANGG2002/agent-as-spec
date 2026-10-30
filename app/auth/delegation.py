"""OAuth-style delegation chain: platform -> tenant -> agent-run -> tool-call.

Each hop is a signed, short-lived JWT carrying an `act` claim (RFC 8693 token
exchange's actor-chain idea) naming who is delegating to whom. A tool wrapper
exchanges the run-level token for a tool-call-scoped token immediately before
invocation (`exchange_for_tool_call`) and nothing longer-lived than one tool
call is ever held in agent memory -- see the architecture doc's delegation
section for the rationale.

This module issues/verifies our own internal delegation tokens. It is
deliberately separate from the *external* OAuth authorization-code flow a
BYOK tenant runs against Google/Slack/etc. (that flow produces the refresh
token stored in the vault; this module governs who inside our system is
allowed to ask the vault for it, and for how long).
"""

from __future__ import annotations

import time

from opentelemetry import trace

from app.auth import keys
from app.auth.models import DelegatedToken, IdentityContext
from app.config import settings

tracer = trace.get_tracer("capability_marketplace.auth")



def issue_root_token(identity: IdentityContext) -> DelegatedToken:
    """Mint the first hop, at the API edge, from a verified end-user login."""
    return _issue(identity, scope=" ".join(identity.scopes), ttl=settings.access_token_ttl_seconds)


def delegate_to_run(identity: IdentityContext, *, run_id: str, plan_scopes: tuple[str, ...]) -> DelegatedToken:
    """Narrow the identity's token down to exactly what one agent run needs.

    `plan_scopes` is the union of every step's `required_scopes` across the
    plan. When a plan has no steps that declare a required scope (nothing in
    it is scope-gated), there is nothing to narrow *to* -- narrowing to an
    empty tuple would wipe out even an unrestricted ("*") identity down to
    zero scopes, which is a bug, not a tighter token. In that case the run
    token simply keeps the caller's own scopes unchanged.
    """
    narrowed = identity.delegate(actor=f"run:{run_id}", narrowed_scopes=plan_scopes or identity.scopes)
    with tracer.start_as_current_span("auth.delegate_to_run", attributes={"run_id": run_id}):
        return _issue(narrowed, scope=" ".join(narrowed.scopes), ttl=settings.access_token_ttl_seconds)


def exchange_for_tool_call(run_token: DelegatedToken, *, capability_id: str, required_scope: str | None) -> DelegatedToken:
    """The last, narrowest hop: one token, good for one tool call, discarded
    immediately after use by the caller. `required_scope=None` means the
    capability declared no `required_scopes` at all -- nothing to check or
    narrow to, so the tool-call token simply carries the run token's scopes
    forward unchanged."""
    if run_token.is_expired:
        raise PermissionError("delegation token expired before tool-call exchange")

    if required_scope is None:
        narrowed = run_token.identity.delegate(actor=f"tool:{capability_id}")
        with tracer.start_as_current_span(
            "auth.exchange_for_tool_call", attributes={"capability_id": capability_id, "scope": "*unscoped*"}
        ):
            return _issue(narrowed, scope=" ".join(narrowed.scopes), ttl=settings.delegated_token_ttl_seconds)

    if not run_token.identity.has_scope(required_scope):
        raise PermissionError(f"identity lacks required scope: {required_scope}")

    narrowed = run_token.identity.delegate(actor=f"tool:{capability_id}", narrowed_scopes=(required_scope,))
    with tracer.start_as_current_span(
        "auth.exchange_for_tool_call",
        attributes={"capability_id": capability_id, "scope": required_scope},
    ):
        return _issue(narrowed, scope=required_scope, ttl=settings.delegated_token_ttl_seconds)


def verify(encoded: str) -> IdentityContext:
    """Signature, algorithm, issuer, audience, expiry and required claims
    are all checked by `keys.decode`. Revocation is checked by the caller
    (`app/auth/deps.py`), which has the async context to ask Redis."""
    payload = keys.decode(encoded)
    return IdentityContext(
        tenant_id=payload["tenant_id"],
        brand_id=payload["brand_id"],
        user_id=payload["user_id"],
        scopes=tuple(payload["scope"].split()) if payload["scope"] else (),
        act_chain=tuple(payload.get("act_chain", [])),
    )


def _issue(identity: IdentityContext, *, scope: str, ttl: int) -> DelegatedToken:
    now = time.time()
    jti = DelegatedToken.new_jti()
    expires_at = now + ttl
    payload = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "jti": jti,
        "iat": int(now),
        "exp": int(expires_at),
        "tenant_id": identity.tenant_id,
        "brand_id": identity.brand_id,
        "user_id": identity.user_id,
        "scope": scope,
        "act_chain": list(identity.act_chain),
    }
    encoded = keys.sign(payload)
    return DelegatedToken(jti=jti, identity=identity, scope=scope, expires_at=expires_at, encoded=encoded)
