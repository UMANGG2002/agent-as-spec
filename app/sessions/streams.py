"""Per-invocation event streams on Redis Streams, readable from any API
replica and resumable after a dropped connection.

Writer (runner worker): every ADK event of an invocation is appended with
XADD to `s:{<session_id>}:inv:<invocation_id>` (capped with MAXLEN ~, and
EXPIREd `stream_retention_seconds` after the last write). The terminal
event is `{"type": "invocation.finished", ...}`.

Reader (API, SSE): the Redis entry id (`<ms>-<seq>`, strictly increasing)
is sent as the SSE `id:` of every event. A client that loses its
connection reconnects with the standard `Last-Event-ID` header (browsers'
EventSource does this automatically) and gets exactly the events after it
-- first replayed with XRANGE, then live -- no duplicates, no gaps, from
whichever API replica the reconnect lands on. No consumer groups: each
reader owns its cursor, so any number of tabs/devices can follow the same
invocation independently.

`StreamHub` multiplexes live tailing: instead of one blocking XREAD (and
one Redis connection) per open SSE connection, each API process runs one
reader loop that XREADs all subscribed streams together (grouped by
cluster slot) and fans entries out to in-process queues. 10k open streams
on a pod cost a handful of Redis connections, not 10k.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import orjson

from app.config import settings
from app.core import redis as redis_core

logger = logging.getLogger(__name__)

TERMINAL = "invocation.finished"


def stream_key(session_id: str, invocation_id: str) -> str:
    return redis_core.key("s", session_id, "inv", invocation_id)


def _id_tuple(entry_id: str) -> tuple[int, int]:
    ms, _, seq = entry_id.partition("-")
    return int(ms), int(seq or 0)


async def publish(session_id: str, invocation_id: str, event: dict[str, Any]) -> str:
    client = await redis_core.get_redis()
    key = stream_key(session_id, invocation_id)
    pipe = client.pipeline(transaction=False)
    pipe.xadd(key, {"d": orjson.dumps(event, default=str).decode()}, maxlen=settings.stream_maxlen, approximate=True)
    pipe.expire(key, settings.stream_retention_seconds)
    entry_id, _ = await pipe.execute()
    return entry_id


async def replay(session_id: str, invocation_id: str, after: str | None) -> list[tuple[str, dict]]:
    client = await redis_core.get_redis()
    start = f"({after}" if after else "-"
    entries = await client.xrange(stream_key(session_id, invocation_id), min=start, max="+", count=10_000)
    return [(eid, orjson.loads(fields["d"])) for eid, fields in entries]


@dataclass
class _Sub:
    key: str
    cursor: str
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=10_000))


class StreamHub:
    def __init__(self, *, block_ms: int = 250, batch: int = 200) -> None:
        self._subs: dict[int, _Sub] = {}
        self._next = 0
        self._task: asyncio.Task | None = None
        self._block_ms = block_ms
        self._batch = batch

    def subscribe(self, key: str, after: str) -> tuple[int, asyncio.Queue]:
        self._next += 1
        self._subs[self._next] = _Sub(key, after or "0-0")
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())
        return self._next, self._subs[self._next].queue

    def unsubscribe(self, sub_id: int) -> None:
        self._subs.pop(sub_id, None)

    def _groups(self) -> list[dict[str, str]]:
        cursors: dict[str, str] = {}
        for sub in self._subs.values():
            cur = cursors.get(sub.key)
            if cur is None or _id_tuple(sub.cursor) < _id_tuple(cur):
                cursors[sub.key] = sub.cursor
        if not settings.redis_cluster:
            return [cursors] if cursors else []
        from redis.crc import key_slot

        by_slot: dict[int, dict[str, str]] = defaultdict(dict)
        for k, c in cursors.items():
            by_slot[key_slot(k.encode())][k] = c
        return list(by_slot.values())

    async def _read_group(self, client, streams: dict[str, str]) -> None:  # noqa: ANN001
        resp = await client.xread(streams, count=self._batch, block=redis_core.block_ms(self._block_ms))
        if not resp:
            await redis_core.idle_wait()
        for key, entries in resp or []:
            for eid, fields in entries:
                event = orjson.loads(fields["d"])
                for sub in list(self._subs.values()):
                    if sub.key == key and _id_tuple(eid) > _id_tuple(sub.cursor):
                        sub.cursor = eid
                        try:
                            sub.queue.put_nowait((eid, event))
                        except asyncio.QueueFull:
                            logger.warning("slow SSE consumer on %s; dropping its live tail", key)

    async def _run(self) -> None:
        client = await redis_core.get_blocking_redis()
        while self._subs:
            groups = self._groups()
            try:
                await asyncio.gather(*(self._read_group(client, g) for g in groups))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the hub alive through Redis blips
                logger.warning("stream hub read failed: %s", exc)
                await asyncio.sleep(0.5)


hub = StreamHub()


async def follow(session_id: str, invocation_id: str, after: str | None, *,
                 heartbeat: float | None = None) -> AsyncIterator[tuple[str | None, dict | None]]:
    """Yield (entry_id, event) from `after` onwards until the terminal event;
    yields (None, None) as a heartbeat when nothing arrived for `heartbeat` s."""
    heartbeat = heartbeat or settings.stream_heartbeat_seconds
    cursor = after
    for eid, event in await replay(session_id, invocation_id, after):
        cursor = eid
        yield eid, event
        if event.get("type") == TERMINAL:
            return
    sub_id, queue = hub.subscribe(stream_key(session_id, invocation_id), cursor or "0-0")
    try:
        while True:
            try:
                eid, event = await asyncio.wait_for(queue.get(), timeout=heartbeat)
            except asyncio.TimeoutError:
                yield None, None
                continue
            yield eid, event
            if event.get("type") == TERMINAL:
                return
    finally:
        hub.unsubscribe(sub_id)
