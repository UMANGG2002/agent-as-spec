"""Knowledge base endpoints -- the "where I add knowledge" part of the
configuration plane. One tenant's documents, plain keyword search (see
`app/knowledge/store.py`'s docstring for why that's the deliberate scope)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import get_identity
from app.auth.models import IdentityContext
from app.db import SessionLocal
from app.knowledge import store as knowledge_store
from app.knowledge.schemas import KnowledgeDocument, KnowledgeSearchResult

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


async def _db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


class AddDocumentRequest(BaseModel):
    title: str
    content: str
    tags: list[str] = []


@router.post("/documents", response_model=KnowledgeDocument)
async def add_document(
    body: AddDocumentRequest, identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
) -> KnowledgeDocument:
    doc = await knowledge_store.add_document(
        session, tenant_id=identity.tenant_id, title=body.title, content=body.content, tags=body.tags
    )
    await session.commit()
    return doc


@router.get("/documents", response_model=list[KnowledgeDocument])
async def list_documents(
    identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
) -> list[KnowledgeDocument]:
    return await knowledge_store.list_for_tenant(session, identity.tenant_id)


@router.get("/search", response_model=list[KnowledgeSearchResult])
async def search_documents(
    q: str = "", identity: IdentityContext = Depends(get_identity), session: AsyncSession = Depends(_db)
) -> list[KnowledgeSearchResult]:
    return await knowledge_store.search(session, identity.tenant_id, q)
