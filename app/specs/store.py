"""Approved `AgentSpec`s in Postgres (SQLite locally) -- replaces the old
MongoDB store and its shared-JSON-file fallback.

Two tables:
  - `agent_specs`: the current head of each spec (one row per spec_id),
    the full spec as JSONB, plus the columns we filter on.
  - `agent_spec_versions`: an immutable row per (spec_id, version), written
    in the same transaction -- every version ever approved stays readable
    for audit, diffing and rollback, and a running invocation can pin the
    exact version it started with.

Why not Mongo any more: specs are small, relational to tenants/approvals,
and need the same row-level security and backups as everything else; one
datastore is one fewer thing to scale, secure and operate.

Reads of a single spec go through the two-tier cache (`get_spec_cached`);
writes invalidate it. Functions accept an explicit `session` so callers
with their own engine/loop (adk web's import-time loader) can use them.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, UniqueConstraint, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core.cache import cache
from app.db import Base, session_scope
from app.specs.schemas import AgentSpec

_JSON = JSON().with_variant(JSONB(), "postgresql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AgentSpecRow(Base):
    __tablename__ = "agent_specs"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_agent_specs_tenant_name"),)

    spec_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | archived
    data: Mapped[dict] = mapped_column(_JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class AgentSpecVersionRow(Base):
    __tablename__ = "agent_spec_versions"

    spec_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    data: Mapped[dict] = mapped_column(_JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


def _spec(data: dict) -> AgentSpec:
    return AgentSpec.model_validate(data)


async def _save(session: AsyncSession, spec: AgentSpec) -> AgentSpec:
    doc = spec.model_dump(mode="json")
    row = await session.get(AgentSpecRow, spec.spec_id)
    if row is None:
        session.add(AgentSpecRow(spec_id=spec.spec_id, tenant_id=spec.tenant_id, name=spec.name,
                                 version=spec.version, data=doc))
    else:
        row.name, row.version, row.data, row.updated_at = spec.name, spec.version, doc, _now()
    if await session.get(AgentSpecVersionRow, (spec.spec_id, spec.version)) is None:
        session.add(AgentSpecVersionRow(spec_id=spec.spec_id, version=spec.version, tenant_id=spec.tenant_id, data=doc))
    await session.flush()
    return spec


async def save_spec(spec: AgentSpec, *, session: AsyncSession | None = None) -> AgentSpec:
    """Upsert the head and record the version (idempotent per version)."""
    if session is not None:
        out = await _save(session, spec)
        await cache.invalidate_on_commit(session, spec.spec_id, "spec")
        await cache.invalidate_on_commit(session, spec.tenant_id, "specs")
        return out
    async with session_scope(tenant_id=spec.tenant_id) as s:
        out = await _save(s, spec)
    await cache.invalidate(spec.spec_id, "spec")
    await cache.invalidate(spec.tenant_id, "specs")
    return out


async def get_spec(spec_id: str, *, tenant_id: str | None = None,
                   session: AsyncSession | None = None) -> AgentSpec | None:
    """`tenant_id` scopes the read (row-level security on Postgres); omitted,
    the request/worker context's tenant applies."""
    async def _get(s: AsyncSession) -> AgentSpec | None:
        row = await s.get(AgentSpecRow, spec_id)
        if row is None or (tenant_id and row.tenant_id != tenant_id):
            return None
        return _spec(row.data)

    if session is not None:
        return await _get(session)
    async with session_scope(tenant_id=tenant_id) as s:
        return await _get(s)


async def get_spec_cached(spec_id: str, *, tenant_id: str) -> AgentSpec | None:
    """Tenant is part of the key: what a read returns depends on who reads."""
    return await cache.get_or_load(
        spec_id, "spec", f"head:{tenant_id}", lambda: get_spec(spec_id, tenant_id=tenant_id), ttl=300,
        dump=lambda m: m.model_dump(mode="json"), load=_spec,
    )


async def get_spec_version(spec_id: str, version: int, *, tenant_id: str | None = None,
                           session: AsyncSession | None = None) -> AgentSpec | None:
    async def _get(s: AsyncSession) -> AgentSpec | None:
        row = await s.get(AgentSpecVersionRow, (spec_id, version))
        if row is None or (tenant_id and row.tenant_id != tenant_id):
            return None
        return _spec(row.data)

    if session is not None:
        return await _get(session)
    async with session_scope(tenant_id=tenant_id) as s:
        return await _get(s)


async def list_specs(tenant_id: str, *, session: AsyncSession | None = None, limit: int = 500) -> list[AgentSpec]:
    async def _list(s: AsyncSession) -> list[AgentSpec]:
        stmt = (
            select(AgentSpecRow)
            .where(AgentSpecRow.tenant_id == tenant_id, AgentSpecRow.status == "active")
            .order_by(AgentSpecRow.updated_at.desc())
            .limit(limit)
        )
        return [_spec(r.data) for r in (await s.execute(stmt)).scalars().all()]

    if session is not None:
        return await _list(session)
    async with session_scope(tenant_id=tenant_id) as s:
        return await _list(s)


async def list_specs_cached(tenant_id: str) -> list[AgentSpec]:
    raw = await cache.get_or_load(
        tenant_id, "specs", "list", lambda: list_specs(tenant_id), ttl=60,
        dump=lambda specs: [x.model_dump(mode="json") for x in specs],
        load=lambda rows: [_spec(r) for r in rows],
    )
    return raw


async def find_by_name(tenant_id: str, name: str, *, session: AsyncSession | None = None) -> AgentSpec | None:
    async def _find(s: AsyncSession) -> AgentSpec | None:
        stmt = select(AgentSpecRow).where(AgentSpecRow.tenant_id == tenant_id, AgentSpecRow.name == name)
        row = (await s.execute(stmt)).scalars().first()
        return _spec(row.data) if row else None

    if session is not None:
        return await _find(session)
    async with session_scope(tenant_id=tenant_id) as s:
        return await _find(s)
