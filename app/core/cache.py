"""Two-tier read-through cache: L1 = in-process LRU with per-entry TTL, L2 =
Redis. Built for the hot read paths at high request rates (visible
capability set, grants, specs, policies, tenant limits), where a database
round-trip per request is what falls over first.

Techniques, all in `get_or_load`:
  - Read-through: callers pass a loader; the cache owns hit/miss logic.
  - Generation-based invalidation: every key embeds a namespace generation
    (`gen:{partition}:{ns}`). Invalidating is one INCR -- no SCAN, no
    fan-out DEL -- and old entries simply age out. The generation itself is
    held in L1 for `GEN_TTL` seconds, so a write is visible on other pods
    within that window (and immediately on the writing pod).
  - Single-flight (request coalescing): concurrent misses for the same key
    in one process share one loader call. Across processes, a short Redis
    lock lets one pod load while the others wait briefly for L2 to fill --
    stops a cache stampede when a hot key expires.
  - TTL jitter (+/- `cache_ttl_jitter_ratio`) so keys written together don't
    all expire together.
  - Negative caching: a loader returning None is cached briefly too, so a
    flood of lookups for something that doesn't exist doesn't hit the DB.
  - Fail-open: if Redis errors, the cache degrades to L1 + loader rather
    than failing the request.

Values crossing L2 must be JSON-serialisable; pass `dump`/`load` to store
Pydantic models (`dump=lambda m: m.model_dump(mode="json")`).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable

import orjson

from app.config import settings
from app.core import redis as redis_core

logger = logging.getLogger(__name__)

GEN_TTL = 2.0
_MISSING = object()
_NONE_SENTINEL = {"__none__": True}


class _L1:
    def __init__(self, max_items: int) -> None:
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max = max_items

    def get(self, key: str) -> Any:
        item = self._data.get(key)
        if item is None:
            return _MISSING
        expires, value = item
        if expires < time.monotonic():
            self._data.pop(key, None)
            return _MISSING
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: float) -> None:
        self._data[key] = (time.monotonic() + ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)

    def pop(self, key: str) -> None:
        self._data.pop(key, None)

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


class TwoTierCache:
    def __init__(self, *, max_items: int | None = None) -> None:
        self.l1 = _L1(max_items or settings.cache_l1_max_items)
        self._inflight: dict[str, asyncio.Future] = {}
        self.stats = {"l1_hit": 0, "l2_hit": 0, "miss": 0, "load": 0, "redis_error": 0}

    @staticmethod
    def _jitter(ttl: float) -> float:
        r = settings.cache_ttl_jitter_ratio
        return max(0.05, ttl * (1 + random.uniform(-r, r)))

    async def generation(self, partition: str, ns: str) -> int:
        gkey = redis_core.key("gen", partition, ns)
        cached = self.l1.get(gkey)
        if cached is not _MISSING:
            return cached
        try:
            client = await redis_core.get_redis()
            value = int(await client.get(gkey) or 0)
        except Exception:  # noqa: BLE001
            self.stats["redis_error"] += 1
            value = 0
        self.l1.set(gkey, value, GEN_TTL)
        return value

    async def invalidate(self, partition: str, ns: str) -> None:
        """Make every cached entry in (partition, ns) stale -- O(1)."""
        gkey = redis_core.key("gen", partition, ns)
        try:
            client = await redis_core.get_redis()
            value = int(await client.incr(gkey))
            await client.expire(gkey, 7 * 86400)
        except Exception:  # noqa: BLE001
            self.stats["redis_error"] += 1
            value = int(time.time())
        self.l1.set(gkey, value, GEN_TTL)

    async def _full_key(self, partition: str, ns: str, key: str) -> str:
        gen = await self.generation(partition, ns)
        return redis_core.key("c", partition, ns, f"g{gen}", key)

    async def get_or_load(
        self,
        partition: str,
        ns: str,
        key: str,
        loader: Callable[[], Awaitable[Any]],
        *,
        ttl: float | None = None,
        l1_ttl: float | None = None,
        negative_ttl: float = 5.0,
        dump: Callable[[Any], Any] | None = None,
        load: Callable[[Any], Any] | None = None,
        l2: bool = True,
    ) -> Any:
        ttl = ttl or settings.cache_default_ttl_seconds
        l1_ttl = min(l1_ttl or ttl, ttl)
        full = await self._full_key(partition, ns, key)

        hit = self.l1.get(full)
        if hit is not _MISSING:
            self.stats["l1_hit"] += 1
            return hit

        fut = self._inflight.get(full)
        if fut is not None:  # single-flight within this process
            return await asyncio.shield(fut)
        fut = asyncio.get_running_loop().create_future()
        self._inflight[full] = fut
        try:
            value = await self._load_through_l2(full, loader, ttl, negative_ttl, dump, load, l2)
            self.l1.set(full, value, self._jitter(l1_ttl if value is not None else min(negative_ttl, l1_ttl)))
            fut.set_result(value)
            return value
        except BaseException as exc:
            fut.set_exception(exc)
            fut.exception()  # mark retrieved for waiters-less failures
            raise
        finally:
            self._inflight.pop(full, None)

    async def _load_through_l2(self, full, loader, ttl, negative_ttl, dump, load, l2):  # noqa: ANN001
        client = None
        if l2:
            try:
                client = await redis_core.get_redis()
                raw = await client.get(full)
                if raw is not None:
                    self.stats["l2_hit"] += 1
                    data = orjson.loads(raw)
                    if data == _NONE_SENTINEL:
                        return None
                    return load(data) if load else data
                # Cross-process single-flight: one pod loads, others wait for L2.
                got_lock = await client.set(full + ":lock", "1", nx=True, px=3000)
                if not got_lock:
                    for _ in range(10):
                        await asyncio.sleep(0.03)
                        raw = await client.get(full)
                        if raw is not None:
                            self.stats["l2_hit"] += 1
                            data = orjson.loads(raw)
                            return None if data == _NONE_SENTINEL else (load(data) if load else data)
            except Exception:  # noqa: BLE001 - fail open to the loader
                self.stats["redis_error"] += 1
                client = None

        self.stats["miss"] += 1
        self.stats["load"] += 1
        value = await loader()
        if client is not None:
            try:
                payload = _NONE_SENTINEL if value is None else (dump(value) if dump else value)
                await client.set(
                    full,
                    orjson.dumps(payload, default=str),
                    px=int(1000 * self._jitter(ttl if value is not None else negative_ttl)),
                )
                await client.delete(full + ":lock")
            except Exception:  # noqa: BLE001
                self.stats["redis_error"] += 1
        return value

    def clear_local(self) -> None:
        self.l1.clear()

    async def invalidate_on_commit(self, session, partition: str, ns: str) -> None:  # noqa: ANN001
        """Invalidate now AND again right after the caller's transaction
        commits. The second bump closes the race where a concurrent reader
        reloads pre-commit data into the cache between the first bump and
        the commit (the "double delete" pattern, done with generations)."""
        await self.invalidate(partition, ns)
        sync = getattr(session, "sync_session", session)
        sync.info.setdefault("_cache_invalidations", set()).add((partition, ns))


cache = TwoTierCache()


def _install_commit_hook() -> None:
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    @event.listens_for(Session, "after_commit")
    def _after_commit(session) -> None:  # noqa: ANN001
        pending = session.info.pop("_cache_invalidations", None)
        if not pending:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        for partition, ns in pending:
            loop.create_task(cache.invalidate(partition, ns))

    @event.listens_for(Session, "after_rollback")
    def _after_rollback(session) -> None:  # noqa: ANN001
        session.info.pop("_cache_invalidations", None)


_install_commit_hook()
