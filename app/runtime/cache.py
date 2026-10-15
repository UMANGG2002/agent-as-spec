"""Application-level idempotent tool-result cache, keyed by
`(capability_id, args_hash, tenant_id)` -- deliberately separate from ADK's
own `ContextCacheConfig` (app/main.py), per the architecture doc's advanced
patterns section: ADK's cache saves on re-sending model tokens for a reused
system prompt; this cache saves on not calling the tool at all for a
repeated, side-effect-free call (a lookup-style GET, say).

Uses Redis when reachable (shared across processes), an in-memory dict
otherwise -- same fallback shape as runtime/redis_streams.py.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import redis.asyncio as aioredis

from app.config import settings

_redis: aioredis.Redis | None = None
_redis_unavailable = False
_memory: dict[str, tuple[float, Any]] = {}


def cache_key(tenant_id: str, capability_id: str, args: dict[str, Any]) -> str:
    """For repeatable, side-effect-free calls (a lookup-style GET) -- a TTL
    window shared across runs."""
    args_hash = hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"toolcache:{tenant_id}:{capability_id}:{args_hash}"


def idempotency_key(run_id: str, step_id: str, args: dict[str, Any]) -> str:
    """For non-idempotent calls (anything a capability's `ToolSchema` doesn't
    mark `idempotent: true`) -- scoped to one run+step rather than a TTL
    window, so a retry, a Redis-stream redelivery, or resuming a run after a
    long HIL pause replays the same result instead of re-executing a
    side-effecting call. See the architecture doc's "Dynamic policy engine"
    section."""
    args_hash = hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"idempotency:{run_id}:{step_id}:{args_hash}"


async def _get_redis() -> aioredis.Redis | None:
    global _redis, _redis_unavailable
    if _redis_unavailable:
        return None
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await _redis.ping()
    except Exception:  # noqa: BLE001
        _redis_unavailable = True
        return None
    return _redis


async def get(key: str) -> Any | None:
    client = await _get_redis()
    if client is not None:
        raw = await client.get(key)
        return json.loads(raw) if raw else None
    entry = _memory.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        _memory.pop(key, None)
        return None
    return value


async def set(key: str, value: Any, *, ttl_seconds: int = 300) -> None:
    client = await _get_redis()
    if client is not None:
        await client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
        return
    _memory[key] = (time.time() + ttl_seconds, value)
