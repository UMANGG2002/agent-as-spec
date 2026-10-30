"""Tool-result cache and idempotency store, on the shared Redis.

  - `cache_key` -- repeatable, side-effect-free calls (lookup-style GETs):
    a TTL window shared across runs of one tenant.
  - `idempotency_key` -- non-idempotent calls, scoped to one run/invocation
    + step + args: a retry, a work-queue redelivery, or resuming an
    invocation after a crash or a long approval pause returns the first
    execution's result instead of repeating the side effect.
  - `acquire_execution_lock` -- a short NX lock on the idempotency key so
    two workers that both picked up the same (redelivered) job can't run the
    same side-effecting call concurrently; the loser waits for the winner's
    result.

Keys hash-tag the run id, so one run's keys share a Redis Cluster slot.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from typing import Any

import orjson

from app.core import redis as redis_core

_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""


def _args_hash(args: dict[str, Any]) -> str:
    return hashlib.sha256(orjson.dumps(args, option=orjson.OPT_SORT_KEYS, default=str)).hexdigest()[:24]


def cache_key(tenant_id: str, capability_id: str, args: dict[str, Any]) -> str:
    return redis_core.key("toolcache", tenant_id, capability_id, _args_hash(args))


def idempotency_key(run_id: str, step_id: str, args: dict[str, Any]) -> str:
    return redis_core.key("idem", run_id, step_id, _args_hash(args))


async def get(key: str) -> Any | None:
    try:
        client = await redis_core.get_redis()
        raw = await client.get(key)
    except Exception:  # noqa: BLE001
        return None
    return orjson.loads(raw) if raw else None


async def set(key: str, value: Any, *, ttl_seconds: int = 300) -> None:  # noqa: A001 - module API
    try:
        client = await redis_core.get_redis()
        await client.set(key, orjson.dumps(value, default=str), ex=ttl_seconds)
    except Exception:  # noqa: BLE001
        pass


async def acquire_execution_lock(key: str, *, ttl_ms: int = 120_000) -> str | None:
    token = uuid.uuid4().hex
    try:
        client = await redis_core.get_redis()
        ok = await client.set(key + ":lock", token, nx=True, px=ttl_ms)
    except Exception:  # noqa: BLE001 - no Redis: run (single-process dev)
        return token
    return token if ok else None


async def release_execution_lock(key: str | None, token: str | None) -> None:
    if not key or not token:
        return
    try:
        client = await redis_core.get_redis()
        await client.eval(_RELEASE, 1, key + ":lock", token)
    except Exception:  # noqa: BLE001
        pass


async def wait_for_result(key: str, *, timeout: float = 30.0) -> Any | None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        value = await get(key)
        if value is not None:
            return value
        try:
            client = await redis_core.get_redis()
            if not await client.exists(key + ":lock"):
                return await get(key)
        except Exception:  # noqa: BLE001
            return None
        await asyncio.sleep(0.1)
    return None
