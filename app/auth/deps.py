"""FastAPI dependency that turns an `Authorization: Bearer <jwt>` header into
an IdentityContext, so every route handler gets identity for free instead of
re-deriving it."""

from __future__ import annotations

from fastapi import Header, HTTPException

from app.auth import delegation
from app.auth.models import IdentityContext


async def get_identity(authorization: str = Header(...)) -> IdentityContext:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="expected 'Bearer <token>'")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        return delegation.verify(token)
    except Exception as exc:  # noqa: BLE001 - surface as 401, not 500
        raise HTTPException(status_code=401, detail=f"invalid token: {exc}") from exc


def enforce_scope(identity: IdentityContext, scope: str) -> None:
    """Call this inline in a route handler once the required scope is known
    (e.g. from the capability being acted on) -- scope requirements are data
    attached to capabilities/plan steps, not static per-route decorators."""
    if not identity.has_scope(scope):
        raise HTTPException(status_code=403, detail=f"missing required scope: {scope}")
