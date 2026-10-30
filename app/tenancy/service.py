"""Resolve a tenant's effective limits, cached (two-tier) -- read on every
request by the rate limiter, so it must never be a database hit per call."""

from __future__ import annotations

from dataclasses import replace

from app.config import settings
from app.core.cache import cache
from app.db import session_scope
from app.tenancy.models import TIERS, TenantLimits, TenantRow


async def _load(tenant_id: str) -> dict:
    async with session_scope(tenant_id="__system__") as session:
        row = await session.get(TenantRow, tenant_id)
    base = TIERS.get(row.tier if row else settings.default_tenant_tier, TIERS["standard"])
    if row is None:
        return base.to_dict()
    known = {k: v for k, v in (row.limits_override or {}).items() if k in base.to_dict()}
    return replace(base, status=row.status, **known).to_dict()


async def get_limits(tenant_id: str) -> TenantLimits:
    data = await cache.get_or_load(tenant_id, "tenant", "limits", lambda: _load(tenant_id), ttl=30)
    return TenantLimits(**data)


async def upsert_tenant(tenant_id: str, *, tier: str = "standard", name: str = "", status: str = "active",
                        limits_override: dict | None = None) -> None:
    async with session_scope(tenant_id="__system__") as session:
        row = await session.get(TenantRow, tenant_id)
        if row is None:
            session.add(TenantRow(tenant_id=tenant_id, tier=tier, name=name, status=status,
                                  limits_override=limits_override or {}))
        else:
            row.tier, row.status = tier, status
            if name:
                row.name = name
            if limits_override is not None:
                row.limits_override = limits_override
    await cache.invalidate(tenant_id, "tenant")
