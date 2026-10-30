"""The API's single authentication + admission choke point. Every
authenticated route depends on `get_identity`, which in order:

  1. verifies the bearer JWT (signature/alg/iss/aud/exp -- `keys.decode`),
     memoising the verified identity in-process for up to 30s (bounded by
     the token's own expiry) so the asymmetric signature check isn't paid
     on every one of a client's requests;
  2. rejects revoked tokens (jti denylist in Redis, checked through a 5s
     in-process cache);
  3. binds tenant/user into the request context -- which is what scopes
     every database transaction to that tenant (row-level security), every
     log line and every span;
  4. rejects suspended tenants and applies the tenant's request-rate limit
     (GCRA), returning 429 with Retry-After.
"""

from __future__ import annotations

import hashlib
import time

from fastapi import Header, HTTPException

from app.auth import delegation
from app.auth.models import IdentityContext
from app.config import settings
from app.core import context
from app.core import redis as redis_core
from app.core.cache import _L1, _MISSING
from app.core.ratelimit import allow

_verified = _L1(100_000)
_revoked_l1 = _L1(100_000)


async def is_revoked(jti: str) -> bool:
    hit = _revoked_l1.get(jti)
    if hit is not _MISSING:
        return hit
    try:
        client = await redis_core.get_redis()
        revoked = bool(await client.exists(redis_core.key("revoked", jti)))
    except Exception:  # noqa: BLE001
        revoked = False
    _revoked_l1.set(jti, revoked, 5.0)
    return revoked


async def revoke(jti: str, *, ttl_seconds: int) -> None:
    client = await redis_core.get_redis()
    await client.set(redis_core.key("revoked", jti), "1", ex=max(1, ttl_seconds))
    _revoked_l1.set(jti, True, 5.0)


def _decode(token: str) -> tuple[IdentityContext, str, float]:
    import jwt as pyjwt

    digest = hashlib.sha256(token.encode()).hexdigest()
    hit = _verified.get(digest)
    if hit is not _MISSING:
        return hit
    identity = delegation.verify(token)
    claims = pyjwt.decode(token, options={"verify_signature": False})
    value = (identity, claims["jti"], float(claims["exp"]))
    _verified.set(digest, value, max(0.1, min(30.0, value[2] - time.time())))
    return value


async def get_identity(authorization: str = Header(...)) -> IdentityContext:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="expected 'Bearer <token>'")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        identity, jti, exp = _decode(token)
    except Exception as exc:  # noqa: BLE001 - surface as 401, not 500
        raise HTTPException(status_code=401, detail=f"invalid token: {exc}") from exc
    if exp <= time.time():
        raise HTTPException(status_code=401, detail="token expired")
    if await is_revoked(jti):
        raise HTTPException(status_code=401, detail="token revoked")

    context.tenant_id_var.set(identity.tenant_id)
    context.user_id_var.set(identity.user_id)

    if settings.rate_limit_enabled:
        from app.tenancy.service import get_limits

        limits = await get_limits(identity.tenant_id)
        if limits.status != "active":
            raise HTTPException(status_code=403, detail=f"tenant is {limits.status}")
        decision = await allow("api", identity.tenant_id, rate_per_second=limits.requests_per_second,
                               burst=limits.burst)
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail="rate limit exceeded",
                headers={"Retry-After": str(max(1, int(decision.retry_after_ms / 1000 + 0.999)))},
            )
    return identity


def enforce_scope(identity: IdentityContext, scope: str) -> None:
    """Call this inline in a route handler once the required scope is known
    (e.g. from the capability being acted on) -- scope requirements are data
    attached to capabilities/plan steps, not static per-route decorators."""
    if not identity.has_scope(scope):
        raise HTTPException(status_code=403, detail=f"missing required scope: {scope}")


def reset_for_tests() -> None:
    _verified.clear()
    _revoked_l1.clear()
