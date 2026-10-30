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

from typing import Any, AsyncIterator

import orjson

from app.core import redis as redis_core

STREAM_PREFIX = "run"
DEFAULT_GROUP = "dispatchers"


def _stream_key(run_id: str) -> str:
    return redis_core.key(STREAM_PREFIX, run_id, "events")


async def _get_redis():
    return await redis_core.get_redis()


async def publish_event(run_id: str, event: dict[str, Any]) -> None:
    client = await _get_redis()
    payload = orjson.dumps(event, default=str).decode()
    await client.xadd(_stream_key(run_id), {"data": payload}, maxlen=10_000, approximate=True)


async def ensure_group(run_id: str, group: str = DEFAULT_GROUP) -> None:
    client = await _get_redis()
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
    await ensure_group(run_id, group)
    resp = await client.xreadgroup(group, consumer, {_stream_key(run_id): ">"}, count=count, block=block_ms)
    for _stream, entries in resp or []:
        for entry_id, fields in entries:
            yield entry_id, orjson.loads(fields["data"])


async def ack(run_id: str, entry_id: str, group: str = DEFAULT_GROUP) -> None:
    client = await _get_redis()
    await client.xack(_stream_key(run_id), group, entry_id)


async def close_stream(run_id: str) -> None:
    """Called once the run's terminal event is written and its
    InvocationRecord is closed -- trims the stream so Redis holds in-flight
    and recently-finished runs, not permanent history (that's Postgres)."""
    client = await _get_redis()
    await client.xtrim(_stream_key(run_id), maxlen=0)
