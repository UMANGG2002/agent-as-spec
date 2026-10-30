"""Test harness.

Runs the whole suite against SQLite + the in-process Redis stand-in by
default (no services needed). Point it at real infrastructure to run the
same tests on Postgres (row-level security, migrations, partitions) and a
real Redis:

    DATABASE_URL=postgresql+asyncpg://postgres@127.0.0.1:5433/agentspec \\
    REDIS_URL=redis://127.0.0.1:6380/0 pytest -q

Every test gets a clean database and a clean Redis, and every process-wide
cache/pool is reset so tests can't leak state into each other.
"""

from __future__ import annotations

import os

os.environ.setdefault("MEM0_TELEMETRY", "False")

import pytest
import pytest_asyncio

from app.config import settings

# Before anything else imports the engine-bound modules.
settings.redis_url = os.environ.get("REDIS_URL", "memory://") if os.environ.get("REDIS_URL", "").startswith(
    "redis://") else "memory://"
settings.default_tenant_tier = "internal"  # rate limits are tested explicitly, not by accident
settings.runner_mode = "inline"
settings.memory_backend = "mock"

from app.db import Base, SessionLocal, engine  # noqa: E402

_PG = engine.dialect.name == "postgresql"


@pytest.fixture(autouse=True)
def _force_offline_planner(monkeypatch):
    """Tests stay deterministic and network-free whatever a developer's
    `.env` holds: no live Gemini key or LiteLLM gateway."""
    monkeypatch.setattr(settings, "google_api_key", None)
    monkeypatch.setattr(settings, "llm_api_key", None)


@pytest.fixture(autouse=True)
def _isolated_capabilities_mirror(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "capabilities_json_path", str(tmp_path / "capabilities.json"))


@pytest.fixture(autouse=True)
def _reset_process_state():
    from app.auth import deps, keys
    from app.core import redis as redis_core
    from app.core import resilience
    from app.core.cache import cache
    from app.memory import service as memory_service
    from app.sessions import agent_cache, streams
    from app.sessions import runner as runner_mod

    redis_core.reset_for_tests() if not settings.redis_url.startswith("redis://") else None
    cache.clear_local()
    deps.reset_for_tests()
    keys.reset_for_tests()
    resilience.reset_for_tests()
    memory_service.reset_for_tests()
    agent_cache.clear()
    streams.hub = streams.StreamHub()
    runner_mod.model_override = None
    yield


@pytest_asyncio.fixture(autouse=True)
async def _clean_redis():
    if settings.redis_url.startswith("redis://"):
        from app.core import redis as redis_core

        client = await redis_core.get_redis()
        await client.flushdb()
    yield


async def _pg_reset() -> None:
    from sqlalchemy import text

    from app.core import migrate
    from app.db import _all_models

    _all_models()
    from sqlalchemy.ext.asyncio import create_async_engine

    global _migrated
    owner = create_async_engine(settings.database_migration_url or settings.database_url)
    try:
        if not _migrated:
            async with owner.begin() as conn:
                await conn.execute(text("DROP SCHEMA public CASCADE"))
                await conn.execute(text("CREATE SCHEMA public"))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await migrate.apply(owner)
            _migrated = True
        names = ", ".join(t.name for t in Base.metadata.sorted_tables)
        async with owner.begin() as conn:
            await conn.execute(text(f"TRUNCATE {names} CASCADE"))
    finally:
        await owner.dispose()


_migrated = False


@pytest_asyncio.fixture(autouse=True)
async def _fresh_schema():
    from app.db import _all_models

    _all_models()
    if _PG:
        await _pg_reset()
    else:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def session():
    async with SessionLocal() as s:
        # Direct-session tests act as one tenant (RLS on Postgres) unless
        # they pass explicit tenants through the services.
        s.info["tenant_id"] = "__system__"
        yield s
