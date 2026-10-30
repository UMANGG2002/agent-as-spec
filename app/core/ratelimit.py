"""Distributed rate limiting, concurrency limits and budgets, all atomic
Lua scripts on Redis -- one round trip per check, correct across any number
of API/runner replicas, and using Redis server time so pod clock skew
can't be gamed.

  - `allow()` -- GCRA (generic cell rate algorithm): a smooth per-key rate
    with a burst allowance, stored as a single timestamp per key (O(1)
    memory, no sliding-window lists). Returns retry-after on denial.
  - `Semaphore` -- a distributed counting semaphore with expiring leases
    (sorted set keyed by expiry). Caps concurrent agent invocations per
    tenant; a crashed holder's lease simply expires, so slots never leak.
  - `consume_budget()` -- per-tenant daily model-token budget (INCRBY on a
    day key that expires by itself).

Fail-open vs fail-closed: if Redis is unreachable, rate checks fail OPEN
(serve the request, log, count it) -- an outage of the limiter should not
become an outage of the product -- while concurrency acquisition fails
CLOSED for new agent invocations (they queue and retry) because those are
the expensive, model-spending path.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from app.core import redis as redis_core

logger = logging.getLogger(__name__)

_GCRA = """
local key = KEYS[1]
local interval = tonumber(ARGV[1])   -- ms per token
local burst = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local tat = tonumber(redis.call('GET', key) or now)
if tat < now then tat = now end
local new_tat = tat + interval * cost
local allow_at = new_tat - interval * burst
if now < allow_at then
  return {0, allow_at - now, 0}
end
redis.call('SET', key, new_tat, 'PX', math.max(1, math.ceil(new_tat - now)))
local remaining = math.floor((now - allow_at) / interval)
return {1, 0, remaining}
"""

_SEM_ACQUIRE = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local lease = tonumber(ARGV[2])
local holder = ARGV[3]
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZSCORE', key, holder) then
  redis.call('ZADD', key, now + lease, holder)
  return 1
end
if redis.call('ZCARD', key) < limit then
  redis.call('ZADD', key, now + lease, holder)
  redis.call('PEXPIRE', key, lease * 2)
  return 1
end
return 0
"""

_SEM_RENEW = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
return redis.call('ZADD', KEYS[1], 'XX', 'CH', now + tonumber(ARGV[1]), ARGV[2])
"""

_BUDGET = """
local used = redis.call('INCRBY', KEYS[1], tonumber(ARGV[1]))
if used == tonumber(ARGV[1]) then redis.call('EXPIRE', KEYS[1], 90000) end
return used
"""


_scripts: dict[tuple[int, str], object] = {}


async def _script(source: str):
    """Registered script (EVALSHA): the script body isn't re-sent on every
    call -- at 100k checks/s that's most of the bytes on the wire. Falls back
    to EVAL transparently if Redis lost its script cache (NOSCRIPT)."""
    client = await redis_core.get_redis()
    key = (id(client), source)
    script = _scripts.get(key)
    if script is None:
        script = _scripts[key] = client.register_script(source)
    return script


@dataclass
class RateDecision:
    allowed: bool
    retry_after_ms: int = 0
    remaining: int = 0


async def allow(bucket: str, partition: str, *, rate_per_second: float, burst: int, cost: int = 1) -> RateDecision:
    if rate_per_second <= 0:
        return RateDecision(True, 0, burst)
    interval_ms = 1000.0 / rate_per_second
    try:
        script = await _script(_GCRA)
        allowed, retry_after, remaining = await script(keys=[redis_core.key("rl", partition, bucket)],
                                                       args=[interval_ms, max(1, burst), cost])
        return RateDecision(bool(int(allowed)), int(retry_after), int(remaining))
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.warning("rate limiter unavailable, failing open: %s", exc)
        return RateDecision(True, 0, 0)


class Semaphore:
    """`async with Semaphore(...)` or explicit acquire/release with a holder id."""

    def __init__(self, name: str, partition: str, *, limit: int, lease_ms: int) -> None:
        self.key = redis_core.key("sem", partition, name)
        self.limit = limit
        self.lease_ms = lease_ms

    async def acquire(self, holder: str | None = None) -> str | None:
        holder = holder or uuid.uuid4().hex
        try:
            script = await _script(_SEM_ACQUIRE)
            ok = await script(keys=[self.key], args=[self.limit, self.lease_ms, holder])
        except Exception as exc:  # noqa: BLE001 - fail closed
            logger.warning("semaphore %s unavailable, failing closed: %s", self.key, exc)
            return None
        return holder if int(ok) == 1 else None

    async def renew(self, holder: str) -> bool:
        try:
            script = await _script(_SEM_RENEW)
            await script(keys=[self.key], args=[self.lease_ms, holder])
            return True
        except Exception:  # noqa: BLE001
            return False

    async def release(self, holder: str) -> None:
        try:
            client = await redis_core.get_redis()
            await client.zrem(self.key, holder)
        except Exception:  # noqa: BLE001
            pass

    async def in_use(self) -> int:
        client = await redis_core.get_redis()
        return int(await client.zcard(self.key))


async def consume_budget(tenant_id: str, tokens: int, *, day: str) -> int:
    """Add `tokens` to today's usage; returns the new total."""
    try:
        script = await _script(_BUDGET)
        return int(await script(keys=[redis_core.key("budget", tenant_id, day)], args=[int(tokens)]))
    except Exception:  # noqa: BLE001
        return 0


async def budget_used(tenant_id: str, *, day: str) -> int:
    try:
        client = await redis_core.get_redis()
        return int(await client.get(redis_core.key("budget", tenant_id, day)) or 0)
    except Exception:  # noqa: BLE001
        return 0
