"""Persistence for approved `AgentSpec` documents in MongoDB -- one document
per spec, `spec_id` as `_id`. Follows the exact same shape as
`app/runtime/redis_streams.py`'s Redis handling: try the real backend, and if
it's not reachable (no `motor` installed, or nothing listening on
`mongo_url`), fall back to something that keeps the rest of the system
working -- clearly logged, not silent.

That fallback is a small JSON file on disk (`settings.agent_specs_fallback_path`),
not a plain in-process dict, and the difference matters here specifically:
the FastAPI app (where `POST /agent-specs/approve` runs) and `adk web`'s
supervisor (`agents/capability_marketplace/agent.py`, which reads specs back
via `list_specs` to compile them into tools -- see `app/factory/spec_agents.py`)
are two separate processes. A dict scoped to one process's memory would mean
the supervisor could *never* see a spec approved through the UI/API whenever
Mongo isn't running -- not "until a restart", not "until refreshed", just
never, since its own empty dict has nothing in it. A shared file both
processes read and write (on the same machine) fixes that for local/demo use
without requiring a real MongoDB to be running. It isn't a substitute for one:
no file locking, so two writes at the exact same instant can race, and it's
still meant to be replaced by `mongo_url` pointing at a real server for
anything beyond local testing."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import settings
from app.specs.schemas import AgentSpec

logger = logging.getLogger(__name__)

try:
    from motor.motor_asyncio import AsyncIOMotorClient

    _MOTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MOTOR_AVAILABLE = False

_client: "AsyncIOMotorClient | None" = None
_mongo_unavailable = False

COLLECTION = "agent_specs"


def _fallback_path() -> Path:
    return Path(settings.agent_specs_fallback_path).expanduser()


def _read_fallback() -> dict[str, dict]:
    """Re-read from disk on every call rather than caching in memory -- the
    whole point is that another process may have written since we last
    looked."""
    path = _fallback_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("could not read AgentSpec fallback file %s (%s); treating it as empty", path, exc)
        return {}


def _write_fallback(data: dict[str, dict]) -> None:
    path = _fallback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


async def _get_collection():
    global _client, _mongo_unavailable
    if not _MOTOR_AVAILABLE:
        if not _mongo_unavailable:
            logger.warning(
                "motor is not installed; AgentSpecs will only live in %s (pip install .[mongo])", _fallback_path()
            )
            _mongo_unavailable = True
        return None
    if _mongo_unavailable:
        return None
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_url, serverSelectionTimeoutMS=1500)
    try:
        await _client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "MongoDB unavailable (%s); falling back to %s for AgentSpecs", exc, _fallback_path()
        )
        _mongo_unavailable = True
        return None
    return _client[settings.mongo_db_name][COLLECTION]


async def save_spec(spec: AgentSpec) -> AgentSpec:
    """Upsert by `spec_id` -- approving the same plan twice (or bumping
    `version` on a re-approval) replaces the document rather than erroring,
    the same idempotent-by-id stance `registry_store.save`/`marketplace_service.save_plan`
    take with their own SQL `merge()`."""
    doc = spec.model_dump(mode="json")
    collection = await _get_collection()
    if collection is None:
        data = _read_fallback()
        data[spec.spec_id] = doc
        _write_fallback(data)
        return spec
    doc["_id"] = spec.spec_id
    await collection.replace_one({"_id": spec.spec_id}, doc, upsert=True)
    return spec


async def get_spec(spec_id: str) -> AgentSpec | None:
    collection = await _get_collection()
    if collection is None:
        doc = _read_fallback().get(spec_id)
        return AgentSpec.model_validate(doc) if doc else None
    doc = await collection.find_one({"_id": spec_id})
    return AgentSpec.model_validate(doc) if doc else None


async def list_specs(tenant_id: str) -> list[AgentSpec]:
    collection = await _get_collection()
    if collection is None:
        return [AgentSpec.model_validate(d) for d in _read_fallback().values() if d.get("tenant_id") == tenant_id]
    cursor = collection.find({"tenant_id": tenant_id})
    return [AgentSpec.model_validate(doc) async for doc in cursor]
