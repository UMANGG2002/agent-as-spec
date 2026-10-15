"""Cross-tenant request & grant workflow -- architecture doc's "Marketplace
mechanics" section."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_identity
from app.auth.models import IdentityContext
from app.db import SessionLocal
from app.marketplace import service as marketplace_service
from app.marketplace.models import CapabilityGrantRow
from app.registry import store as registry_store

router = APIRouter(prefix="/marketplace", tags=["marketplace"])


async def _db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


class GrantRequestBody(BaseModel):
    capability_id: str
    scope_limits: list[str] = []


class GrantDecisionBody(BaseModel):
    approve: bool


@router.post("/grants")
async def request_grant(
    body: GrantRequestBody, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
):
    capability = await registry_store.get(session, body.capability_id)
    if capability is None:
        raise HTTPException(status_code=404, detail="capability not found")
    if capability.visibility.value != "marketplace":
        raise HTTPException(status_code=403, detail="capability is not marketplace-visible")

    grant = await marketplace_service.request_grant(
        session,
        capability_id=body.capability_id,
        grantor_tenant_id=capability.tenant_id,
        grantee_tenant_id=identity.tenant_id,
        scope_limits=body.scope_limits,
    )
    await session.commit()

    from app.notifications.composio_dispatcher import dispatcher

    await dispatcher.dispatch(
        session,
        tenant_id=capability.tenant_id,
        event_type="grant_requested",
        payload={"grant_id": grant.id, "capability_id": capability.id, "grantee_tenant_id": identity.tenant_id},
    )
    return {"grant_id": grant.id, "status": grant.status}


@router.post("/grants/{grant_id}/decide")
async def decide_grant(
    grant_id: str, body: GrantDecisionBody, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
):
    grant = await session.get(CapabilityGrantRow, grant_id)
    if grant is None:
        raise HTTPException(status_code=404, detail="grant not found")
    if grant.grantor_tenant_id != identity.tenant_id:
        raise HTTPException(status_code=403, detail="only the granting tenant can decide this request")

    updated = await marketplace_service.decide_grant(session, grant_id, approve=body.approve)
    await session.commit()

    if body.approve:
        from app.notifications.composio_dispatcher import dispatcher

        await dispatcher.dispatch(
            session,
            tenant_id=updated.grantee_tenant_id,
            event_type="grant_approved",
            payload={"grant_id": grant_id, "capability_id": updated.capability_id},
        )
    return {"grant_id": updated.id, "status": updated.status}


@router.get("/grants")
async def list_my_grants(identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)):
    granted_ids = await marketplace_service.granted_capability_ids(session, identity.tenant_id)
    return {"granted_capability_ids": sorted(granted_ids)}


@router.get("/grants/requests")
async def list_grant_requests(identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)):
    """Grant requests involving this tenant: `incoming` (others asking for
    this tenant's capabilities -- decide them with POST
    /marketplace/grants/{id}/decide) and `outgoing` (access this tenant
    asked for, with its current status)."""
    grants = await marketplace_service.grants_involving(session, identity.tenant_id)

    def _out(grant) -> dict:  # noqa: ANN001
        return {
            "grant_id": grant.id,
            "capability_id": grant.capability_id,
            "grantor_tenant_id": grant.grantor_tenant_id,
            "grantee_tenant_id": grant.grantee_tenant_id,
            "status": grant.status,
            "requested_at": grant.requested_at.isoformat() if grant.requested_at else None,
            "decided_at": grant.decided_at.isoformat() if grant.decided_at else None,
        }

    return {key: [_out(g) for g in rows] for key, rows in grants.items()}


@router.get("/discover")
async def discover(
    q: str = "",
    limit: int = Query(default=50, ge=1, le=500),
    identity: IdentityContext = Depends(get_identity),
    session: AsyncSession = Depends(_db),
):
    """Browse every marketplace-visibility capability across all tenants --
    deliberately NOT grant-gated (see registry/store.py's `discover_marketplace`
    docstring for why this is a separate function from `/capabilities/search`).
    An empty `q` lists everything (up to `limit`)."""
    return await registry_store.discover_marketplace(session, q, limit=limit)


class RegisterWorkerBody(BaseModel):
    base_url: str = "http://127.0.0.1:9100"
    visibility: str = "marketplace"


@router.post("/register-worker")
async def register_worker(
    body: RegisterWorkerBody, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
):
    """Register every MCP server and A2A agent a running test worker
    (`scripts/run_worker.py`) serves as a capability owned by the calling
    tenant -- public (marketplace) by default -- through the real onboarding
    adapters. Idempotent: stable ids mean re-registering (e.g. after the
    worker moved port) updates the same rows. Same as the worker's own
    startup registration; this is for doing it from the UI or for a
    different tenant.

    The server fetches `base_url` itself, like the existing
    /capabilities/mcp and /capabilities/agent-card registration routes do --
    fine for local use, but don't expose this unauthenticated on a network
    where that's a server-side request forgery concern."""
    from app.registry.schemas import Visibility
    from app.testing import worker

    try:
        visibility = Visibility(body.visibility)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"unknown visibility {body.visibility!r}") from exc
    try:
        saved = await worker.register_all(session, identity.tenant_id, body.base_url, visibility=visibility)
    except Exception as exc:  # noqa: BLE001 - unreachable worker, bad manifest, introspection failure
        raise HTTPException(status_code=502, detail=f"could not register worker at {body.base_url}: {exc}") from exc
    await session.commit()
    return {
        "registered": len(saved),
        "visibility": visibility.value,
        "owner_tenant_id": identity.tenant_id,
        "mcp_servers": sum(1 for c in saved if c.kind.value == "mcp"),
        "a2a_agents": sum(1 for c in saved if c.kind.value == "agent_card"),
        "tools": sum(len(c.tools) for c in saved),
    }
