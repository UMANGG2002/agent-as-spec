"""Asynchronous memory ingestion. Writing to mem0 means an LLM extraction
call plus embeddings -- far too slow for the request path, and a mem0
outage must never fail a conversation. So a finished invocation is only
*enqueued* here (one XADD); the memory consumer (in each runner process)
reads it, loads just that invocation's events from the session store and
hands them to the memory service.

Consumer group `memory` on the `memq` stream: at-least-once, ACK after a
successful write; failures are retried by reclaim after the visibility
timeout and dropped after a few attempts (memory is best-effort).
In `runner_mode=inline` (local) it runs as a background task instead.
"""

from __future__ import annotations

import asyncio
import logging

import orjson

from app.config import settings
from app.core import redis as redis_core

logger = logging.getLogger(__name__)
STREAM = redis_core.key("memq", "all")
GROUP = "memory"


async def ingest_one(item: dict) -> None:
    from app.memory.service import get_memory_service
    from app.sessions.session_store import adk_user_id, get_session_service

    memory = get_memory_service()
    if memory is None:
        return
    user_id = adk_user_id(item["tenant_id"], item["user_id"])
    session = await get_session_service().get_session(app_name=item["app_name"], user_id=user_id,
                                                      session_id=item["session_id"])
    if session is None:
        return
    events = [e for e in session.events if not item.get("adk_invocation_id")
              or e.invocation_id == item["adk_invocation_id"]]
    await memory.add_events_to_memory(app_name=item["app_name"], user_id=user_id, events=events,
                                      session_id=item["session_id"])


async def enqueue(job: dict, *, adk_invocation_id: str | None) -> None:
    if settings.effective_memory_backend == "off":
        return
    item = {k: job[k] for k in ("tenant_id", "user_id", "session_id", "app_name")}
    item["adk_invocation_id"] = adk_invocation_id
    if settings.runner_mode == "inline":
        async def _bg() -> None:
            try:
                await ingest_one(item)
            except Exception as exc:  # noqa: BLE001
                logger.warning("memory ingestion failed: %s", exc)

        asyncio.get_running_loop().create_task(_bg())
        return
    try:
        client = await redis_core.get_redis()
        await client.xadd(STREAM, {"m": orjson.dumps(item).decode()}, maxlen=500_000, approximate=True)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.warning("could not enqueue memory ingestion: %s", exc)


async def consume_forever(stop: asyncio.Event, *, consumer: str = "mem") -> None:
    client = await redis_core.get_blocking_redis()
    try:
        await client.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except Exception as exc:  # noqa: BLE001
        if "BUSYGROUP" not in str(exc):
            raise
    while not stop.is_set():
        try:
            resp = await client.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=20, block=redis_core.block_ms(1000))
            if not resp:
                await redis_core.idle_wait()
            for _key, entries in resp or []:
                for eid, fields in entries:
                    try:
                        await ingest_one(orjson.loads(fields["m"]))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("memory ingestion failed (dropped): %s", exc)
                    await client.xack(STREAM, GROUP, eid)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("memory consumer error: %s", exc)
            await asyncio.sleep(1)
