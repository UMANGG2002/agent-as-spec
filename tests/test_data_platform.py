"""Data-platform guarantees: transactional outbox, audit trail, migrations,
Postgres row-level security and partitions, and mem0 on pgvector.

The Postgres-only tests skip on SQLite; run them with
DATABASE_URL=postgresql+asyncpg://agentspec_app:...  (a NON-superuser role:
superusers bypass RLS) and DATABASE_MIGRATION_URL=<owner role>.
"""

from __future__ import annotations

import os

import orjson
import pytest
from sqlalchemy import text

from app.config import settings
from app.core import outbox
from app.core import redis as redis_core
from app.db import engine, session_scope

PG = engine.dialect.name == "postgresql"
pg_only = pytest.mark.skipif(not PG, reason="Postgres-specific")


async def _stream(topic: str) -> list[dict]:
    client = await redis_core.get_redis()
    return [orjson.loads(f["data"]) for _, f in await client.xrange(outbox.topic_stream(topic))]


async def test_outbox_publishes_committed_events_exactly_once_and_never_rolled_back_ones():
    async with session_scope(tenant_id="acme") as s:
        kept = outbox.emit(s, tenant_id="acme", topic="t.test", payload={"n": 1})
    with pytest.raises(RuntimeError):
        async with session_scope(tenant_id="acme") as s:
            outbox.emit(s, tenant_id="acme", topic="t.test", payload={"n": 2})
            raise RuntimeError("business logic failed -> transaction rolls back")
    async with session_scope(tenant_id="__system__") as s:
        assert await outbox.relay_once(s) == 1
    async with session_scope(tenant_id="__system__") as s:
        assert await outbox.relay_once(s) == 0  # already published
    events = await _stream("t.test")
    assert [e["id"] for e in events] == [kept] and events[0]["payload"] == {"n": 1}


async def test_grant_decision_writes_an_audit_row_in_the_same_transaction(session):
    from app.core.audit import AuditLogRow
    from app.marketplace import service

    grant = await service.request_grant(session, capability_id="cap_x", grantor_tenant_id="acme",
                                        grantee_tenant_id="globex")
    await service.decide_grant(session, grant.id, approve=True)
    await session.commit()
    rows = (await session.execute(text("SELECT action, resource_id FROM audit_log"))).all()
    assert ("grant.approved", grant.id) in [tuple(r) for r in rows]
    assert AuditLogRow.__tablename__ == "audit_log"


@pg_only
async def test_migrations_are_idempotent_and_detect_tampering():
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.core import migrate

    owner = create_async_engine(settings.database_migration_url or settings.database_url)
    try:
        assert await migrate.apply(owner) == []  # already applied by the test harness
        async with owner.begin() as conn:
            await conn.execute(text("UPDATE schema_migrations SET checksum = 'x' WHERE version = '0002_rls'"))
        with pytest.raises(RuntimeError, match="modified after being applied"):
            await migrate.apply(owner)
    finally:
        async with owner.begin() as conn:
            from pathlib import Path
            import hashlib

            body = (migrate.SQL_DIR / "0002_rls.sql").read_text()
            await conn.execute(text("UPDATE schema_migrations SET checksum = :c WHERE version = '0002_rls'"),
                               {"c": hashlib.sha256(body.encode()).hexdigest()})
        await owner.dispose()


@pg_only
async def test_row_level_security_isolates_tenants_even_for_raw_sql():
    from app.knowledge import store as kb

    async with session_scope(tenant_id="acme") as s:
        await kb.add_document(s, tenant_id="acme", title="secret", content="acme only")
    async with session_scope(tenant_id="globex") as s:
        # No WHERE clause at all -- Postgres itself filters.
        assert (await s.execute(text("SELECT count(*) FROM knowledge_documents"))).scalar() == 0
    async with session_scope() as s:  # no tenant bound: sees nothing tenant-scoped
        assert (await s.execute(text("SELECT count(*) FROM knowledge_documents"))).scalar() == 0
    async with session_scope(tenant_id="acme") as s:
        assert (await s.execute(text("SELECT count(*) FROM knowledge_documents"))).scalar() == 1


@pg_only
async def test_rls_rejects_writing_rows_for_another_tenant():
    from sqlalchemy.exc import DBAPIError

    from app.knowledge import store as kb

    with pytest.raises(DBAPIError, match="row-level security"):
        async with session_scope(tenant_id="globex") as s:
            await kb.add_document(s, tenant_id="acme", title="forged", content="x")


@pg_only
async def test_rls_capability_visibility_owned_marketplace_and_granted():
    from app.marketplace import service
    from app.registry import store as registry
    from app.registry.schemas import Capability, CapabilityKind, EndpointRef, Visibility

    def cap(cid, vis):
        return Capability(id=cid, tenant_id="acme", kind=CapabilityKind.OPENAPI, name=cid, visibility=vis,
                          endpoint_ref=EndpointRef(openapi_base_url="https://example.com"))

    async with session_scope(tenant_id="acme") as s:
        await registry.save(s, cap("cap_private", Visibility.PRIVATE))
        await registry.save(s, cap("cap_public", Visibility.MARKETPLACE))
        await registry.save(s, cap("cap_granted", Visibility.PRIVATE))
        grant = await service.request_grant(s, capability_id="cap_granted", grantor_tenant_id="acme",
                                            grantee_tenant_id="globex")
        await service.decide_grant(s, grant.id, approve=True)
    async with session_scope(tenant_id="globex") as s:
        ids = {r[0] for r in (await s.execute(text("SELECT id FROM capabilities"))).all()}
    assert ids == {"cap_public", "cap_granted"}


@pg_only
async def test_append_only_tables_are_monthly_partitioned():
    async with session_scope(tenant_id="__system__") as s:
        parts = {r[0] for r in (await s.execute(text(
            "SELECT inhrelid::regclass::text FROM pg_inherits WHERE inhparent = 'audit_log'::regclass"))).all()}
    assert "audit_log_default" in parts and len(parts) >= 5


def test_mem0_config_is_a_valid_mem0_memory_config(monkeypatch):
    from mem0.configs.base import MemoryConfig

    from app.memory.service import mem0_config

    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://u:p@db:5432/agentspec")
    monkeypatch.setattr(settings, "llm_base_url", "https://gateway/v1")
    monkeypatch.setattr(settings, "llm_api_key", "k")
    monkeypatch.setattr(settings, "llm_model_name", "m")
    cfg = MemoryConfig(**mem0_config())
    assert cfg.vector_store.provider == "pgvector"
    assert cfg.vector_store.config.connection_string.startswith("postgresql://")


@pg_only
async def test_mem0_on_pgvector_keeps_memories_per_tenant_user():
    os.environ["MEM0_TELEMETRY"] = "False"
    from google.adk.events import Event
    from google.genai import types
    from mem0 import AsyncMemory
    from mem0.embeddings.mock import MockEmbeddings

    from app.memory.service import Mem0MemoryService

    dsn = settings.database_url.replace("postgresql+asyncpg", "postgresql")
    memory = AsyncMemory.from_config({
        "vector_store": {"provider": "pgvector", "config": {"connection_string": dsn, "collection_name": "mem_test",
                                                            "embedding_model_dims": 10}},
        "llm": {"provider": "openai", "config": {"model": "offline", "api_key": "offline"}},
        "embedder": {"provider": "openai", "config": {"model": "offline", "api_key": "offline", "embedding_dims": 10}},
    })
    memory.embedding_model = MockEmbeddings()  # no network: fixed vectors
    service = Mem0MemoryService(memory, infer=False)
    event = Event(author="user", content=types.Content(role="user", parts=[types.Part(text="my warehouse is Pune")]))
    await service.add_events_to_memory(app_name="spec_x", user_id="acme:alice", events=[event])
    mine = await service.search_memory(app_name="spec_x", user_id="acme:alice", query="warehouse")
    theirs = await service.search_memory(app_name="spec_x", user_id="globex:alice", query="warehouse")
    assert any("Pune" in m.content.parts[0].text for m in mine.memories)
    assert theirs.memories == []


async def test_memory_search_never_fails_the_turn():
    from app.memory.service import Mem0MemoryService

    class Broken:
        async def search(self, *a, **k):
            raise ConnectionError("vector store down")

    service = Mem0MemoryService(Broken())
    assert (await service.search_memory(app_name="a", user_id="t:u", query="x")).memories == []
