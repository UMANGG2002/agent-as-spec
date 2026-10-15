"""The per-tenant knowledge base: free-text documents a tenant's agents can
be grounded against -- the "where I add knowledge" half of the
configuration plane. Deliberately simple (one row per document, keyword
search), matching the registry's own "hybrid search... the one seam to
swap" stance in `app/registry/store.py`: the retrieval function is the one
place a real embedding index plugs in later, without the schema or the API
changing."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class KnowledgeDocument(BaseModel):
    id: str
    tenant_id: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeSearchResult(BaseModel):
    document: KnowledgeDocument
    score: float
