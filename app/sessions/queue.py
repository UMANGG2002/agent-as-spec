"""The invocation work queue: Redis Streams with consumer groups, sharded.

  - `work_queue_shards` streams `wq:{<n>}` (each its own hash tag, so shards
    spread across Redis Cluster nodes). A job goes to shard
    hash(session_id) % N, so one session's jobs stay in order on one shard.
  - Runner workers share one consumer group per shard (`runners`): each job
    is delivered to exactly one worker; it is ACKed only after the
    invocation reaches a terminal state.
  - At-least-once with crash recovery: a worker that dies leaves its job
    pending; any other worker XAUTOCLAIMs jobs idle longer than
    `runner_visibility_timeout_ms` and resumes them (ADK resumability +
    the tool idempotency cache make the retry safe).
  - Poison jobs: after `MAX_DELIVERIES` the job goes to the dead-letter
    stream `wq:{dlq}` and its invocation is marked failed.
  - Deferral: a job that can't run yet (tenant at its concurrency limit,
    session busy) is re-added with `not_before` and ACKed, so it never
    blocks a worker slot while it waits.
"""

from __future__ import annotations

import time
import zlib
from typing import Any

import orjson

from app.config import settings
from app.core import redis as redis_core

GROUP = "runners"
MAX_DELIVERIES = 3


def shard_for(session_id: str) -> int:
    return zlib.crc32(session_id.encode()) % settings.work_queue_shards


def shard_key(shard: int) -> str:
    return redis_core.key("wq", str(shard))


DLQ = redis_core.key("wq", "dlq")


async def ensure_groups() -> None:
    client = await redis_core.get_redis()
    for shard in range(settings.work_queue_shards):
        try:
            await client.xgroup_create(shard_key(shard), GROUP, id="0", mkstream=True)
        except Exception as exc:  # noqa: BLE001
            if "BUSYGROUP" not in str(exc):
                raise


DELAYED = redis_core.key("wq", "delayed")


async def enqueue(job: dict[str, Any], *, not_before: float | None = None) -> str:
    """Ready now -> its shard stream. Later -> the delayed sorted set
    (scored by due time); `promote_due` moves it onto its shard when due, so
    a deferred job never occupies a worker slot while it waits."""
    client = await redis_core.get_redis()
    if not_before and not_before > time.time():
        await client.zadd(DELAYED, {orjson.dumps(job).decode(): not_before})
        return "delayed"
    return await client.xadd(shard_key(shard_for(job["session_id"])), {"j": orjson.dumps(job).decode()},
                             maxlen=1_000_000, approximate=True)


async def promote_due(*, limit: int = 200) -> int:
    """Move due delayed jobs onto their shards. Safe on many workers: only
    the worker whose ZREM succeeds re-adds a given job."""
    client = await redis_core.get_redis()
    due = await client.zrangebyscore(DELAYED, "-inf", time.time(), start=0, num=limit)
    moved = 0
    for raw in due:
        if await client.zrem(DELAYED, raw):
            job = orjson.loads(raw)
            await client.xadd(shard_key(shard_for(job["session_id"])), {"j": raw}, maxlen=1_000_000,
                              approximate=True)
            moved += 1
    return moved


async def depth(shard: int | None = None) -> int:
    """Jobs waiting or in flight. Acked jobs are XDELed, so XLEN is exactly
    undelivered + pending; plus delayed jobs. This is the autoscaling signal
    for runner workers (scale on depth per worker, not on CPU)."""
    client = await redis_core.get_redis()
    shards = [shard] if shard is not None else range(settings.work_queue_shards)
    total = 0
    for s in shards:
        total += int(await client.xlen(shard_key(s)))
    if shard is None:
        total += int(await client.zcard(DELAYED))
    return total


async def read(consumer: str, *, count: int, block_ms: int) -> list[tuple[int, str, dict]]:
    """Next jobs for this worker across all shards: (shard, entry_id, job).
    Uses the blocking-read pool so a worker parked in XREADGROUP BLOCK never
    holds a connection the rest of the process needs."""
    client = await redis_core.get_blocking_redis()
    out: list[tuple[int, str, dict]] = []
    streams = {shard_key(s): ">" for s in range(settings.work_queue_shards)}
    if settings.redis_cluster:  # cross-slot: read shard by shard
        for s in range(settings.work_queue_shards):
            resp = await client.xreadgroup(GROUP, consumer, {shard_key(s): ">"}, count=count, block=None)
            out += [(s, eid, orjson.loads(f["j"])) for _, entries in resp or [] for eid, f in entries]
            if len(out) >= count:
                break
        return out
    resp = await client.xreadgroup(GROUP, consumer, streams, count=count, block=redis_core.block_ms(block_ms))
    if not resp:
        await redis_core.idle_wait()
    index = {shard_key(s): s for s in range(settings.work_queue_shards)}
    return [(index[k], eid, orjson.loads(f["j"])) for k, entries in resp or [] for eid, f in entries]


async def reclaim(consumer: str, *, count: int = 10) -> list[tuple[int, str, dict, int]]:
    """Jobs abandoned by crashed workers: (shard, entry_id, job, deliveries)."""
    client = await redis_core.get_redis()
    out = []
    for s in range(settings.work_queue_shards):
        key = shard_key(s)
        try:
            _next, entries, *_ = await client.xautoclaim(key, GROUP, consumer,
                                                         min_idle_time=settings.runner_visibility_timeout_ms,
                                                         start_id="0-0", count=count)
        except Exception:  # noqa: BLE001
            continue
        for eid, fields in entries:
            if not fields:
                continue
            pending = await client.xpending_range(key, GROUP, min=eid, max=eid, count=1)
            deliveries = int(pending[0]["times_delivered"]) if pending else 1
            out.append((s, eid, orjson.loads(fields["j"]), deliveries))
    return out


async def ack(shard: int, entry_id: str) -> None:
    client = await redis_core.get_redis()
    pipe = client.pipeline(transaction=False)
    pipe.xack(shard_key(shard), GROUP, entry_id)
    pipe.xdel(shard_key(shard), entry_id)
    await pipe.execute()


async def dead_letter(shard: int, entry_id: str, job: dict, reason: str) -> None:
    client = await redis_core.get_redis()
    await client.xadd(DLQ, {"j": orjson.dumps({**job, "reason": reason, "at": time.time()}).decode()},
                      maxlen=100_000, approximate=True)
    await ack(shard, entry_id)
