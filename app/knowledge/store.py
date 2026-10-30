"""Persistence + search for the knowledge base, same shape as
`app/registry/store.py`: one row per document, a cheap keyword score for
ranking. A document is scoped to exactly one tenant -- there is no
marketplace/grant concept here (knowledge isn't a capability you invoke, it's
context a tenant's own planner/agents get grounded against), so this is
simpler than the capability registry on purpose."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base
from app.knowledge.schemas import KnowledgeDocument, KnowledgeSearchResult


class KnowledgeDocumentRow(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


async def add_document(
    session: AsyncSession, *, tenant_id: str, title: str, content: str, tags: list[str] | None = None
) -> KnowledgeDocument:
    row = KnowledgeDocumentRow(
        id=f"kb_{uuid.uuid4().hex[:12]}", tenant_id=tenant_id, title=title, content=content, tags=tags or []
    )
    session.add(row)
    await session.flush()
    return _to_schema(row)


async def list_for_tenant(session: AsyncSession, tenant_id: str, *, limit: int = 500) -> list[KnowledgeDocument]:
    stmt = (select(KnowledgeDocumentRow).where(KnowledgeDocumentRow.tenant_id == tenant_id)
            .order_by(KnowledgeDocumentRow.created_at.desc()).limit(limit))
    rows = (await session.execute(stmt)).scalars().all()
    return [_to_schema(row) for row in rows]


async def search(session: AsyncSession, tenant_id: str, query: str, *, limit: int = 10) -> list[KnowledgeSearchResult]:
    """SQL narrows to documents matching any term (Postgres: full-text index,
    migrations/sql/0006; SQLite: LIKE), Python ranks the small candidate set.
    Previously every document of the tenant was loaded on every search."""
    from sqlalchemy import String, cast, or_, text

    limit = max(1, min(limit, 100))
    terms = [t for t in query.lower().split() if t][:12]
    if not terms:
        return [KnowledgeSearchResult(document=d, score=1.0) for d in await list_for_tenant(session, tenant_id, limit=limit)]
    stmt = select(KnowledgeDocumentRow).where(KnowledgeDocumentRow.tenant_id == tenant_id)
    tag_match = [cast(KnowledgeDocumentRow.tags, String).ilike(f"%{t}%") for t in terms]
    if session.bind.dialect.name == "postgresql":
        stmt = stmt.where(or_(text("search_tsv @@ websearch_to_tsquery('simple', :q)").bindparams(
            q=" OR ".join(terms)), *tag_match))
    else:
        stmt = stmt.where(or_(*[(KnowledgeDocumentRow.title + " " + KnowledgeDocumentRow.content).ilike(f"%{t}%")
                                for t in terms], *tag_match))
    docs = [_to_schema(r) for r in (await session.execute(stmt.limit(limit * 10))).scalars().all()]
    terms = set(terms)
    results = []
    for doc in docs:
        haystack = f"{doc.title} {doc.content} {' '.join(doc.tags)}".lower()
        hits = sum(1 for term in terms if term in haystack)
        if hits:
            results.append(KnowledgeSearchResult(document=doc, score=hits / len(terms)))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


def _to_schema(row: KnowledgeDocumentRow) -> KnowledgeDocument:
    return KnowledgeDocument(
        id=row.id, tenant_id=row.tenant_id, title=row.title, content=row.content, tags=row.tags, created_at=row.created_at
    )
