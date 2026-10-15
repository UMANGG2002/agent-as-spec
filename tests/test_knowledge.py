from __future__ import annotations

import pytest

from app.knowledge import store as knowledge_store


@pytest.mark.asyncio
async def test_documents_are_scoped_per_tenant(session):
    await knowledge_store.add_document(
        session, tenant_id="acme", title="Refund policy", content="Refunds within 30 days, no questions asked.",
        tags=["billing"],
    )
    await knowledge_store.add_document(
        session, tenant_id="globex", title="Globex internal notes", content="Not visible to acme.",
    )
    await session.commit()

    acme_docs = await knowledge_store.list_for_tenant(session, "acme")
    assert {d.title for d in acme_docs} == {"Refund policy"}

    globex_docs = await knowledge_store.list_for_tenant(session, "globex")
    assert {d.title for d in globex_docs} == {"Globex internal notes"}


@pytest.mark.asyncio
async def test_search_ranks_by_keyword_hits(session):
    await knowledge_store.add_document(
        session, tenant_id="acme", title="Refund policy", content="Refunds within 30 days.", tags=["billing", "policy"]
    )
    await knowledge_store.add_document(
        session, tenant_id="acme", title="Shipping FAQ", content="Orders ship within 2 business days.", tags=["shipping"]
    )
    await session.commit()

    results = await knowledge_store.search(session, "acme", "refund")
    assert [r.document.title for r in results] == ["Refund policy"]

    results = await knowledge_store.search(session, "acme", "days")
    assert {r.document.title for r in results} == {"Refund policy", "Shipping FAQ"}

    results = await knowledge_store.search(session, "acme", "nonexistent-term")
    assert results == []
