"""Capability registry endpoints: register (from MCP / OpenAPI / agent card),
search, publish (change visibility). See architecture doc's registry +
marketplace sections."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_identity
from app.auth.models import IdentityContext
from app.db import SessionLocal
from app.marketplace import service as marketplace_service
from app.registry import store as registry_store
from app.registry.adapters import agent_card_adapter, mcp_adapter, openapi_adapter
from app.registry.schemas import Capability, CapabilityKind, Visibility

router = APIRouter(prefix="/capabilities", tags=["capabilities"])


async def _db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


class RegisterMcpRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4000)
    stdio_command: list[str] | None = None
    sse_url: str | None = None
    required_scopes: list[str] = []
    visibility: Visibility = Visibility.PRIVATE


class RegisterOpenApiRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    spec: dict[str, Any]
    base_url: str
    auth_header_name: str = "Authorization"
    required_scopes: list[str] = []
    visibility: Visibility = Visibility.PRIVATE


class RegisterAgentCardRequest(BaseModel):
    agent_base_url: str
    required_scopes: list[str] = []
    visibility: Visibility = Visibility.PRIVATE


@router.post("/mcp", response_model=Capability)
async def register_mcp(
    body: RegisterMcpRequest, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
) -> Capability:
    from app.config import settings
    from app.core.http import UnsafeURL, check_url

    if body.stdio_command and settings.is_production_like:
        # A stdio MCP server is a command run on OUR hosts -- remote code
        # execution by design. Only network MCP endpoints in production.
        raise HTTPException(status_code=400, detail="stdio MCP servers are not allowed; register an SSE/HTTP endpoint")
    if body.sse_url:
        try:
            await check_url(body.sse_url)
        except UnsafeURL as exc:
            raise HTTPException(status_code=400, detail=f"endpoint refused: {exc}") from exc
    capability = await mcp_adapter.introspect_and_register(
        tenant_id=identity.tenant_id,
        name=body.name,
        description=body.description,
        stdio_command=body.stdio_command,
        sse_url=body.sse_url,
        required_scopes=body.required_scopes,
        visibility=body.visibility,
    )
    await registry_store.save(session, capability)
    _audit_registration(session, identity, capability)
    await session.commit()
    return capability


@router.post("/openapi", response_model=Capability)
async def register_openapi(
    body: RegisterOpenApiRequest, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
) -> Capability:
    from app.core.http import UnsafeURL, check_url

    try:
        await check_url(body.base_url)
    except UnsafeURL as exc:
        raise HTTPException(status_code=400, detail=f"endpoint refused: {exc}") from exc
    capability = openapi_adapter.capability_from_openapi(
        body.spec,
        tenant_id=identity.tenant_id,
        name=body.name,
        base_url=body.base_url,
        auth_header_name=body.auth_header_name,
        required_scopes=body.required_scopes,
        visibility=body.visibility,
    )
    await registry_store.save(session, capability)
    _audit_registration(session, identity, capability)
    await session.commit()
    return capability


@router.post("/agent-card", response_model=Capability)
async def register_agent_card(
    body: RegisterAgentCardRequest, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
) -> Capability:
    from app.core.http import UnsafeURL

    try:
        card = await agent_card_adapter.fetch_agent_card(body.agent_base_url)
    except UnsafeURL as exc:
        raise HTTPException(status_code=400, detail=f"endpoint refused: {exc}") from exc
    card.setdefault("url", body.agent_base_url)
    capability = agent_card_adapter.capability_from_agent_card(
        card, tenant_id=identity.tenant_id, required_scopes=body.required_scopes, visibility=body.visibility
    )
    await registry_store.save(session, capability)
    _audit_registration(session, identity, capability)
    await session.commit()
    return capability


def _audit_registration(session: AsyncSession, identity: IdentityContext, capability: Capability) -> None:
    from app.core import audit

    audit.record(session, tenant_id=identity.tenant_id, actor=identity.user_id, action="capability.register",
                 resource_type="capability", resource_id=capability.id,
                 details={"kind": capability.kind.value, "visibility": capability.visibility.value})


@router.get("", response_model=list[Capability])
async def list_capabilities(
    identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
) -> list[Capability]:
    """Everything this tenant can plan against right now -- its own
    capabilities plus anything already granted to it. For "what's registered
    already" in the config-plane UI; `search` below is for narrowing that
    down by keyword once there are more than a handful."""
    # Hot read path: served from the two-tier cache (invalidated on
    # capability save/publish and on grant decisions).
    caps = await registry_store.visible_capabilities(identity.tenant_id)
    return [caps[k] for k in sorted(caps)]


@router.get("/search")
async def search_capabilities(
    q: str = Query(max_length=500),
    kind: CapabilityKind | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    identity: IdentityContext = Depends(get_identity),
    session: AsyncSession = Depends(_db),
):
    """Keyword search over what this tenant can actually use (own +
    granted), matching capability names/descriptions and every tool's
    name/description. For everything public, granted or not, see
    GET /marketplace/discover."""
    granted = await marketplace_service.granted_capability_ids(session, identity.tenant_id)
    results = await registry_store.search(
        session, identity.tenant_id, q, kinds=[kind] if kind else None, granted_capability_ids=granted, limit=limit
    )
    return results


@router.post("/{capability_id}/publish")
async def publish_capability(
    capability_id: str,
    visibility: Visibility,
    identity: IdentityContext = Depends(get_identity),
    session: AsyncSession = Depends(_db),
) -> Capability:
    capability = await registry_store.get(session, capability_id)
    if capability is None or capability.tenant_id != identity.tenant_id:
        raise HTTPException(status_code=404, detail="capability not found")
    updated = capability.model_copy(update={"visibility": visibility})
    await registry_store.save(session, updated)
    from app.core import audit

    audit.record(session, tenant_id=identity.tenant_id, actor=identity.user_id, action="capability.publish",
                 resource_type="capability", resource_id=capability_id, details={"visibility": visibility.value})
    await session.commit()
    return updated
