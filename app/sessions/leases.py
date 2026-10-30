"""Per-session execution lease: at most one runner executes a given
session at a time, across the whole fleet.

ADK appends events to a session as it runs; two workers running the same
session concurrently would interleave or clobber each other's events. The
lease is `SET key holder NX PX ttl`; the holder renews it every ttl/3 while
the invocation runs, and releases it with a compare-and-delete (a Lua
script, so a worker whose lease already expired can't delete someone
else's). A crashed worker's lease simply expires.

Each acquisition also returns a monotonically increasing fencing token
(INCR on a per-session counter). It is recorded on the invocation row, so
stale work by a worker that lost its lease (e.g. after a long GC pause) is
detectable. ADK's DatabaseSessionService additionally rejects an append
from a stale session object (optimistic concurrency on last_update_time).
"""

from __future__ import annotations

from app.config import settings
from app.core import redis as redis_core

_RELEASE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end
return 0
"""
_RENEW = """
if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('PEXPIRE', KEYS[1], ARGV[2]) end
return 0
"""


def _key(session_id: str) -> str:
    return redis_core.key("s", session_id, "lease")


async def acquire(session_id: str, holder: str, *, ttl_ms: int | None = None) -> int | None:
    """Returns a fencing token, or None if another worker holds the lease."""
    ttl_ms = ttl_ms or settings.session_lease_ttl_ms
    client = await redis_core.get_redis()
    if not await client.set(_key(session_id), holder, nx=True, px=ttl_ms):
        return None
    return int(await client.incr(redis_core.key("s", session_id, "fence")))


async def renew(session_id: str, holder: str, *, ttl_ms: int | None = None) -> bool:
    client = await redis_core.get_redis()
    return bool(int(await client.eval(_RENEW, 1, _key(session_id), holder,
                                      ttl_ms or settings.session_lease_ttl_ms)))


async def release(session_id: str, holder: str) -> None:
    client = await redis_core.get_redis()
    await client.eval(_RELEASE, 1, _key(session_id), holder)
