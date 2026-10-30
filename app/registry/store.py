"""Persistence + search for the capability registry.

Production changes vs. the original:
  - Authorization-scoped listing is one indexed SQL query
    (`tenant_id = :t OR id IN (:granted)`) instead of loading every
    marketplace row in the platform into Python and filtering there -- the
    old shape was O(total capabilities) per request and the first thing to
    fall over with many tenants.
  - Search narrows in SQL first (Postgres: full-text `search_tsv @@
    websearch_to_tsquery(...)` over a GIN index plus trigram on the name --
    see migrations/sql/0004_search.sql; SQLite: LIKE), then ranks only the
    small candidate set in Python.
  - Marketplace discovery is paginated (`limit`, `after` keyset cursor).
  - The JSON mirror file is written only in the local environment (it
    rewrote the whole file on every save and isn't multi-process safe).
  - `visible_capabilities()` is the cached, hot-path view used by runners
    and the API: two-tier cache keyed by the tenant's grant generation and
    the global catalog generation, invalidated on capability save/publish
    and on grant decisions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import JSON, String, Text, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.config import settings
from app.core.cache import cache
from app.db import Base
from app.registry.schemas import Capability, CapabilityKind, CapabilitySearchResult, Visibility

logger = logging.getLogger(__name__)

CANDIDATE_MULTIPLIER = 5


class CapabilityRow(Base):
    __tablename__ = "capabilities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    visibility: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    # name + description + every tool's name/description, lower-cased: the
    # text the search index is built over.
    search_text: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[dict] = mapped_column(JSON)  # full Capability.model_dump()


def _search_text(capability: Capability) -> str:
    parts = [capability.name, capability.description] + [f"{t.name} {t.description}" for t in capability.tools]
    return " ".join(p for p in parts if p).lower()[:20_000]


def _mirror_path() -> Path:
    return Path(settings.capabilities_json_path).expanduser()


def _write_mirror(capability: Capability) -> None:
    path = _mirror_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text()) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        data = {}
    data[capability.id] = capability.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2))


async def save(session: AsyncSession, capability: Capability) -> None:
    row = CapabilityRow(
        id=capability.id,
        tenant_id=capability.tenant_id,
        kind=capability.kind.value,
        visibility=capability.visibility.value,
        name=capability.name,
        description=capability.description,
        search_text=_search_text(capability),
        data=capability.model_dump(mode="json"),
    )
    await session.merge(row)
    if settings.environment == "local" and settings.capabilities_json_path:
        try:
            _write_mirror(capability)
        except OSError as exc:
            logger.warning("could not write capabilities mirror file %s (%s)", _mirror_path(), exc)
    await invalidate_catalog(session, capability.tenant_id)


async def invalidate_catalog(session: AsyncSession, owner_tenant_id: str) -> None:
    await cache.invalidate_on_commit(session, owner_tenant_id, "visible")
    await cache.invalidate_on_commit(session, "global", "catalog")


async def get(session: AsyncSession, capability_id: str) -> Capability | None:
    row = await session.get(CapabilityRow, capability_id)
    return Capability.model_validate(row.data) if row else None


def _visible_clause(tenant_id: str, granted: set[str]):
    clause = CapabilityRow.tenant_id == tenant_id
    if granted:
        clause = or_(clause, CapabilityRow.id.in_(sorted(granted)))
    return clause


async def list_for_tenant(
    session: AsyncSession, tenant_id: str, *, granted_capability_ids: set[str] | None = None
) -> list[Capability]:
    """Owned capabilities + anything *already granted* to this tenant -- what
    the planner may bind and what a run may execute. A marketplace listing
    this tenant hasn't been granted does NOT appear here (see
    `discover_marketplace`, deliberately a separate, wider function)."""
    granted = granted_capability_ids or set()
    stmt = select(CapabilityRow.data).where(_visible_clause(tenant_id, granted)).order_by(CapabilityRow.id)
    return [Capability.model_validate(d) for d in (await session.execute(stmt)).scalars().all()]


async def visible_capabilities(tenant_id: str) -> dict[str, Capability]:
    """Cached map of everything this tenant may execute right now."""
    from app.db import session_scope
    from app.marketplace import service as marketplace_service

    catalog_gen = await cache.generation("global", "catalog")

    async def _load() -> dict:
        async with session_scope(tenant_id=tenant_id) as session:
            granted = await marketplace_service.granted_capability_ids(session, tenant_id)
            caps = await list_for_tenant(session, tenant_id, granted_capability_ids=granted)
        return {c.id: c.model_dump(mode="json") for c in caps}

    raw = await cache.get_or_load(tenant_id, "visible", f"cat{catalog_gen}", _load, ttl=120)
    return {cid: Capability.model_validate(d) for cid, d in raw.items()}


def _terms(query: str) -> list[str]:
    return [t for t in query.lower().split() if t][:12]


def _text_filter(session: AsyncSession, query: str):
    terms = _terms(query)
    if not terms:
        return None
    if session.bind.dialect.name == "postgresql":
        return text("(search_tsv @@ websearch_to_tsquery('simple', :q) OR name % :qname)").bindparams(
            q=" OR ".join(terms), qname=query.lower()[:200]
        )
    return or_(*[CapabilityRow.search_text.like(f"%{t}%") for t in terms])


async def discover_marketplace(
    session: AsyncSession, query: str = "", *, limit: int = 20, after: str | None = None
) -> list[CapabilitySearchResult]:
    """Browse what's requestable across every tenant (works before any grant
    exists). Never used by the planner or the run service."""
    limit = max(1, min(limit, 500))
    stmt = select(CapabilityRow.data).where(CapabilityRow.visibility == Visibility.MARKETPLACE.value)
    flt = _text_filter(session, query)
    if flt is not None:
        stmt = stmt.where(flt)
    if after:
        stmt = stmt.where(CapabilityRow.id > after)
    stmt = stmt.order_by(CapabilityRow.id).limit(limit * (CANDIDATE_MULTIPLIER if query else 1))
    caps = [Capability.model_validate(d) for d in (await session.execute(stmt)).scalars().all()]
    results = []
    for cap in caps:
        score = _score(query, cap) if query else 1.0
        if score > 0:
            results.append(CapabilitySearchResult(capability=cap, score=score, matched_via="keyword"))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


async def search(
    session: AsyncSession,
    tenant_id: str,
    query: str,
    *,
    kinds: list[CapabilityKind] | None = None,
    granted_capability_ids: set[str] | None = None,
    limit: int = 10,
) -> list[CapabilitySearchResult]:
    """Grant-gated hybrid search: SQL narrows (visibility, kind, text match),
    then a keyword score ranks the small candidate set."""
    limit = max(1, min(limit, 500))
    granted = granted_capability_ids or set()
    stmt = select(CapabilityRow.data).where(_visible_clause(tenant_id, granted))
    if kinds:
        stmt = stmt.where(CapabilityRow.kind.in_([k.value for k in kinds]))
    flt = _text_filter(session, query)
    if flt is not None:
        stmt = stmt.where(flt)
    stmt = stmt.limit(limit * CANDIDATE_MULTIPLIER)
    scored = []
    for data in (await session.execute(stmt)).scalars().all():
        cap = Capability.model_validate(data)
        score = _score(query, cap)
        if score > 0:
            scored.append(CapabilitySearchResult(capability=cap, score=score, matched_via="keyword"))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:limit]


def _score(query: str, capability: Capability) -> float:
    terms = set(_terms(query))
    if not terms:
        return 0.0
    text_ = _search_text(capability)
    hits = sum(1 for term in terms if term in text_)
    return hits / len(terms)
