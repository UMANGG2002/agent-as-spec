"""Token endpoints.

  - `POST /auth/dev-token` mints a token for any tenant/user -- local and
    test environments ONLY. In staging/production it doesn't exist (404):
    a real login/OIDC flow sits in front of `delegation.issue_root_token`.
  - `GET /.well-known/jwks.json` publishes the public verification keys
    (asymmetric algorithms only) so other services verify our tokens.
  - `POST /auth/revoke` revokes the caller's own token (logout).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.auth import delegation, keys
from app.auth.deps import get_identity, revoke
from app.auth.models import IdentityContext
from app.config import settings

router = APIRouter(tags=["auth"])


class DevTokenRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    brand_id: str = Field(default="default", max_length=64)
    user_id: str = Field(min_length=1, max_length=128)
    scopes: list[str] = ["*"]


class TokenResponse(BaseModel):
    access_token: str
    expires_at: float


@router.post("/auth/dev-token", response_model=TokenResponse)
async def issue_dev_token(body: DevTokenRequest) -> TokenResponse:
    if settings.is_production_like:
        raise HTTPException(status_code=404, detail="not found")
    identity = IdentityContext(
        tenant_id=body.tenant_id, brand_id=body.brand_id, user_id=body.user_id, scopes=tuple(body.scopes)
    )
    token = delegation.issue_root_token(identity)
    return TokenResponse(access_token=token.encoded, expires_at=token.expires_at)


@router.get("/.well-known/jwks.json")
async def jwks() -> dict:
    return keys.keyring().jwks()


@router.post("/auth/revoke")
async def revoke_own_token(
    authorization: str = Header(...), identity: IdentityContext = Depends(get_identity)
) -> dict:
    import jwt as pyjwt

    claims = pyjwt.decode(authorization.removeprefix("Bearer ").strip(), options={"verify_signature": False})
    await revoke(claims["jti"], ttl_seconds=int(claims["exp"] - time.time()) + 5)
    return {"revoked": claims["jti"]}
