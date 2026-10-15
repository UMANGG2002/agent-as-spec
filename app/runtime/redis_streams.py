"""Durable, resumable run events via Redis Streams -- the architecture doc's
"Durable execution & offline resume" section, made concrete.

`XADD` writes every event; a client reads with a consumer group
(`XREADGROUP`) rather than a plain subscribe, which is what makes offline
resume work: on reconnect, the client resumes its own consumer group from
its last-acknowledged id instead of the server tracking "was this client
connected." If Redis isn't reachable (e.g. running the demo without Docker),
this module degrades to an in-process queue per run_id so the rest of the
system still works end to end -- clearly logged, not silent, since that
fallback has none of the durability the real backend provides.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None
_redis_unavailable = False
_fallback_queues: dict[str, asyncio.Queue] = {}

STREAM_PREFIX = "run"
DEFAULT_GROUP = "dispatchers"


def _stream_key(run_id: str) -> str:
    return f"{STREAM_PREFIX}:{run_id}:events"


async def _get_redis() -> aioredis.Redis | None:
    global _redis, _redis_unavailable
    if _redis_unavailable:
        return None
    if _redis is None:
        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await _redis.ping()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis unavailable (%s); falling back to in-process event queues", exc)
        _redis_unavailable = True
        return None
    return _redis


async def publish_event(run_id: str, event: dict[str, Any]) -> None:
    client = await _get_redis()
    payload = json.dumps(event, default=str)
    if client is None:
        _fallback_queues.setdefault(run_id, asyncio.Queue()).put_nowait(payload)
        return
    await client.xadd(_stream_key(run_id), {"data": payload}, maxlen=10_000, approximate=True)


async def ensure_group(run_id: str, group: str = DEFAULT_GROUP) -> None:
    client = await _get_redis()
    if client is None:
        return
    try:
        await client.xgroup_create(_stream_key(run_id), group, id="0", mkstream=True)
    except Exception as exc:  # noqa: BLE001 - BUSYGROUP if it already exists
        if "BUSYGROUP" not in str(exc):
            raise


async def read_events(
    run_id: str, *, consumer: str, group: str = DEFAULT_GROUP, count: int = 20, block_ms: int = 2000
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yields (entry_id, event) pairs. Caller acks with `ack()` once processed
    -- this is what lets a reconnecting client resume only what it hasn't
    seen, instead of replaying the whole run."""
    client = await _get_redis()
    if client is None:
        queue = _fallback_queues.setdefault(run_id, asyncio.Queue())
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=block_ms / 1000)
            except asyncio.TimeoutError:
                return
            yield "in-process", json.loads(payload)
        return

    await ensure_group(run_id, group)
    resp = await client.xreadgroup(group, consumer, {_stream_key(run_id): ">"}, count=count, block=block_ms)
    for _stream, entries in resp or []:
        for entry_id, fields in entries:
            yield entry_id, json.loads(fields["data"])


async def ack(run_id: str, entry_id: str, group: str = DEFAULT_GROUP) -> None:
    if entry_id == "in-process":
        return
    client = await _get_redis()
    if client is not None:
        await client.xack(_stream_key(run_id), group, entry_id)


async def close_stream(run_id: str) -> None:
    """Called once the run's terminal event is written and its
    InvocationRecord is closed -- trims the stream so Redis holds in-flight
    and recently-finished runs, not permanent history (that's Postgres)."""
    client = await _get_redis()
    if client is not None:
        await client.xtrim(_stream_key(run_id), maxlen=0)
    _fallback_queues.pop(run_id, None)
