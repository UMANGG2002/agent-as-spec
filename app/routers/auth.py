"""Dev-only token issuance. A real deployment puts a proper login/OAuth flow
in front of `delegation.issue_root_token`; this endpoint exists so the rest
of the API is exercisable without standing one up."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.auth import delegation
from app.auth.models import IdentityContext

router = APIRouter(prefix="/auth", tags=["auth"])


class DevTokenRequest(BaseModel):
    tenant_id: str
    brand_id: str = "default"
    user_id: str
    scopes: list[str] = ["*"]


class TokenResponse(BaseModel):
    access_token: str
    expires_at: float


@router.post("/dev-token", response_model=TokenResponse)
async def issue_dev_token(body: DevTokenRequest) -> TokenResponse:
    identity = IdentityContext(
        tenant_id=body.tenant_id, brand_id=body.brand_id, user_id=body.user_id, scopes=tuple(body.scopes)
    )
    token = delegation.issue_root_token(identity)
    return TokenResponse(access_token=token.encoded, expires_at=token.expires_at)
