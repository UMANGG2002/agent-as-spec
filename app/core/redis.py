"""One Redis client per process, shared by every module (rate limiter, cache
L2, work queue, event streams, leases, idempotency, guardrail counters).

  - `redis_url = "memory://"` (the local/test default) uses fakeredis -- an
    in-process Redis with streams and Lua -- so the whole stack runs with
    zero services. It is NOT shared between processes.
  - A real URL gets a bounded connection pool with timeouts, health checks
    and retry-on-connection-error. `redis_cluster=true` uses RedisCluster.
  - Locally, if a real URL is unreachable we fall back to the in-process
    stand-in with a warning; in staging/production there is no fallback --
    the process fails readiness instead of silently losing durability.

Keys are built with `key()`, which wraps the partition id in a hash tag
(`{t:acme}`), so every key for one tenant (or one session) lands in the same
Redis Cluster slot -- required for multi-key Lua scripts and transactions,
and what lets the keyspace spread evenly across shards by tenant.

Blocking reads (XREAD BLOCK for live streams) use a separate pool so a few
thousand open SSE connections can't starve the request path of connections.
"""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.config import settings

logger = logging.getLogger(__name__)

_client: Any = None
_blocking_client: Any = None
_fake_server: Any = None


def key(kind: str, partition: str, *parts: str) -> str:
    """`key("t", "acme", "ratelimit", "api")` -> `t:{acme}:ratelimit:api`."""
    tail = ":".join(str(p) for p in parts)
    return f"{kind}:{{{partition}}}" + (f":{tail}" if tail else "")


def _fake():
    global _fake_server
    import fakeredis

    if _fake_server is None:
        _fake_server = fakeredis.FakeServer()
    return fakeredis.FakeAsyncRedis(server=_fake_server, decode_responses=True)


def _real(*, blocking: bool):
    common = dict(
        decode_responses=True,
        socket_timeout=None if blocking else settings.redis_socket_timeout_seconds,
        socket_connect_timeout=settings.redis_socket_timeout_seconds,
        health_check_interval=30,
        retry=Retry(ExponentialBackoff(cap=0.5, base=0.02), retries=3),
        retry_on_error=[RedisConnectionError, RedisTimeoutError],
    )
    if settings.redis_cluster:
        from redis.asyncio.cluster import RedisCluster

        return RedisCluster.from_url(settings.redis_url, **common)
    pool = aioredis.BlockingConnectionPool.from_url(
        settings.redis_url,
        max_connections=settings.redis_max_connections,
        timeout=settings.redis_socket_timeout_seconds,
        **common,
    )
    return aioredis.Redis(connection_pool=pool)


async def get_redis():
    """The shared client. First call verifies connectivity."""
    global _client
    if _client is not None:
        return _client
    if settings.redis_url.startswith("memory://"):
        _client = _fake()
        return _client
    client = _real(blocking=False)
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001
        if settings.is_production_like:
            raise
        logger.warning("Redis at %s unreachable (%s); using the in-process stand-in (local only)", settings.redis_url, exc)
        settings.redis_url = "memory://"
        client = _fake()
    _client = client
    return _client


async def get_blocking_redis():
    """Client for long blocking reads (XREAD BLOCK). Separate pool."""
    global _blocking_client
    if _blocking_client is not None:
        return _blocking_client
    await get_redis()  # resolves the memory:// fallback first
    _blocking_client = _fake() if settings.redis_url.startswith("memory://") else _real(blocking=True)
    return _blocking_client


async def ping() -> bool:
    try:
        client = await get_redis()
        return bool(await client.ping())
    except Exception:  # noqa: BLE001
        return False


async def close() -> None:
    global _client, _blocking_client
    for c in (_client, _blocking_client):
        if c is not None:
            try:
                await c.aclose()
            except Exception:  # noqa: BLE001
                pass
    _client = _blocking_client = None


def block_ms(requested: int) -> int | None:
    """BLOCK value for XREAD/XREADGROUP. The in-process stand-in handles
    blocking reads poorly (they can stall other clients in the process), so
    in memory:// mode reads don't block (None) and callers poll briefly
    instead (`idle_wait`). Real Redis gets the requested value."""
    return None if settings.redis_url.startswith("memory://") else requested


async def idle_wait() -> None:
    """Pause after an empty non-blocking read (memory:// mode only)."""
    if settings.redis_url.startswith("memory://"):
        import asyncio

        await asyncio.sleep(0.02)


def reset_for_tests() -> None:
    """Fresh in-process Redis (tests only)."""
    global _client, _blocking_client, _fake_server
    _client = _blocking_client = _fake_server = None
