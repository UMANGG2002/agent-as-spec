from __future__ import annotations

import pytest
import pytest_asyncio

from app.config import settings
from app.db import Base, SessionLocal, engine


@pytest.fixture(autouse=True)
def _force_offline_planner(monkeypatch):
    """Tests must stay deterministic and network-free no matter what a
    developer's local `.env` has in it -- a real GOOGLE_API_KEY, or an
    LLM_API_KEY/LLM_BASE_URL pointed at a live LiteLLM gateway, would
    otherwise make `settings.planner_is_live` true here too and send every
    "offline planner" test on a real network round trip. Force both off for
    the whole test session; `tests/test_planner.py` covers the live path's
    *parsing* logic directly, against a stubbed runner, not a real model."""
    monkeypatch.setattr(settings, "google_api_key", None)
    monkeypatch.setattr(settings, "llm_api_key", None)


@pytest.fixture(autouse=True)
def _isolated_agent_specs_fallback(monkeypatch, tmp_path):
    """`app/specs/store.py`'s Mongo-unavailable fallback is a JSON file
    shared across processes (deliberately, so `adk web` can see a spec
    approved through the FastAPI app) -- which means, unlike the old
    in-memory dict, it would otherwise persist across separate pytest runs
    and pick up whatever a developer's own local testing already wrote to
    the real default path. Point it at a fresh temp file every test
    session instead."""
    monkeypatch.setattr(settings, "agent_specs_fallback_path", str(tmp_path / "agent_specs_fallback.json"))


@pytest.fixture(autouse=True)
def _isolated_capabilities_mirror(monkeypatch, tmp_path):
    """Same reasoning as `_isolated_agent_specs_fallback` above, for
    `app/registry/store.py`'s JSON mirror of the capability registry: point
    it at a fresh temp file per test instead of the real default path, so
    tests can't see (or pollute) a developer's local `.data/capabilities.json`."""
    monkeypatch.setattr(settings, "capabilities_json_path", str(tmp_path / "capabilities.json"))


@pytest_asyncio.fixture(autouse=True)
async def _fresh_schema():
    async with engine.begin() as conn:
        from app.marketplace import models as _marketplace_models  # noqa: F401
        from app.registry import store as _registry_store  # noqa: F401
        from app.knowledge import store as _knowledge_store  # noqa: F401
        from app.runtime import approvals as _approvals  # noqa: F401

        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def session():
    async with SessionLocal() as s:
        yield s
