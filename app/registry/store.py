"""Persistence + search for the capability registry. The SQLAlchemy row
stores the capability as JSON (it's a nested Pydantic shape with a variable
number of tools) plus the handful of columns we filter/search on directly.

The SQL table is the only thing ever read back -- `get`/`list_for_tenant`/
`search`/`discover_marketplace` all query it. Alongside it, `save()` also
write-through-mirrors every capability into a plain JSON file on disk
(`settings.capabilities_json_path`), keyed by capability id, purely so the
registry's full structured schema (tools, kind, endpoint, everything in
`Capability`) can be read directly off disk with a text editor -- the same
reason `app/specs/store.py` keeps AgentSpecs in a JSON file, though here it's
a mirror of the DB rather than a fallback for when the DB is unreachable (the
registry has no "Mongo unavailable" story; SQLite is always there)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from sqlalchemy import JSON, String, select
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import Base
from app.registry.schemas import Capability, CapabilityKind, CapabilitySearchResult, Visibility

logger = logging.getLogger(__name__)


class CapabilityRow(Base):
    __tablename__ = "capabilities"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String, index=True)
    visibility: Mapped[str] = mapped_column(String, index=True)
    name: Mapped[str] = mapped_column(String, index=True)
    description: Mapped[str] = mapped_column(String, default="")
    data: Mapped[dict] = mapped_column(JSON)  # full Capability.model_dump()


def _mirror_path() -> Path:
    return Path(settings.capabilities_json_path).expanduser()


def _read_mirror() -> dict[str, dict]:
    path = _mirror_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("could not read capabilities mirror file %s (%s); treating it as empty", path, exc)
        return {}


def _write_mirror(capability: Capability) -> None:
    path = _mirror_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _read_mirror()
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
        data=capability.model_dump(mode="json"),
    )
    await session.merge(row)
    # Mirror to disk too -- see module docstring. Best-effort: a write
    # failure here (e.g. read-only filesystem) shouldn't fail the save
    # itself, since the SQL row is the real, authoritative write.
    try:
        _write_mirror(capability)
    except OSError as exc:
        logger.warning("could not write capabilities mirror file %s (%s)", _mirror_path(), exc)


async def get(session: AsyncSession, capability_id: str) -> Capability | None:
    row = await session.get(CapabilityRow, capability_id)
    return Capability.model_validate(row.data) if row else None


async def list_for_tenant(
    session: AsyncSession, tenant_id: str, *, granted_capability_ids: set[str] | None = None
) -> list[Capability]:
    """Owned capabilities + anything *already granted* to this tenant. This
    is the authorization-scoped list: what the planner may bind into a plan
    and what the run service may actually execute. A marketplace-visibility
    capability this tenant hasn't been granted does NOT appear here -- see
    `discover_marketplace` for "what could I request," which is a
    deliberately separate, wider function so execution-time authorization
    and marketplace browsing can never be confused for one another."""
    granted = granted_capability_ids or set()
    stmt = select(CapabilityRow).where(
        (CapabilityRow.tenant_id == tenant_id)
        | (CapabilityRow.visibility == Visibility.MARKETPLACE.value)
    )
    rows = (await session.execute(stmt)).scalars().all()
    out = []
    for row in rows:
        cap = Capability.model_validate(row.data)
        if cap.tenant_id == tenant_id or cap.id in granted:
            out.append(cap)
    return out


async def discover_marketplace(session: AsyncSession, query: str = "", *, limit: int = 20) -> list[CapabilitySearchResult]:
    """Browse what's publishable/requestable across every tenant -- the
    "Tenant B finds Tenant A's capability via registry search" half of the
    marketplace mechanics, which must work *before* any grant exists (you
    can't request access to something you can't see). Never used by the
    planner or the run service -- those call `search`/`list_for_tenant`,
    which is grant-gated, exactly so this wider visibility can't leak into
    what a plan is allowed to execute."""
    stmt = select(CapabilityRow).where(CapabilityRow.visibility == Visibility.MARKETPLACE.value)
    rows = (await session.execute(stmt)).scalars().all()
    results = []
    for row in rows:
        cap = Capability.model_validate(row.data)
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
    """Hybrid search: a metadata filter (kind, visible-to-tenant) narrows the
    set, then a cheap keyword/TF-ish ranking orders within it. The
    architecture doc describes this narrowing being backed by a real
    embedding index in production; the ranking function is the one seam to
    swap (`_score`) without touching callers."""
    candidates = await list_for_tenant(session, tenant_id, granted_capability_ids=granted_capability_ids)
    if kinds:
        allowed = {k.value for k in kinds}
        candidates = [c for c in candidates if c.kind.value in allowed]

    scored = []
    for cap in candidates:
        score = _score(query, cap)
        if score > 0:
            scored.append(CapabilitySearchResult(capability=cap, score=score, matched_via="keyword"))
    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:limit]


def _score(query: str, capability: Capability) -> float:
    terms = {t for t in query.lower().split() if t}
    if not terms:
        return 0.0
    haystacks = [capability.name.lower(), capability.description.lower()]
    haystacks += [f"{t.name} {t.description}".lower() for t in capability.tools]
    text = " ".join(haystacks)
    hits = sum(1 for term in terms if term in text)
    return hits / len(terms)
