"""Failure isolation for every outbound dependency (capability endpoints,
the model gateway, mem0): circuit breakers, retries with full jitter,
timeouts and bulkheads.

  - Circuit breaker (per capability endpoint, per process): after
    `failure_threshold` consecutive failures the circuit OPENS and calls
    fail fast for `reset_seconds`, instead of every request waiting out a
    timeout against a dead endpoint (which is how one bad tool drains a
    whole worker pool). Then HALF-OPEN lets one trial call through; success
    closes it, failure re-opens it.
  - Retry with exponential backoff and full jitter, only for errors marked
    retryable, and only for idempotent operations -- the caller decides.
  - Bulkhead: a bounded semaphore per dependency, so one slow capability can
    hold at most N of a worker's concurrent slots; callers beyond that fail
    fast with `BulkheadFull` rather than queueing unboundedly.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypeVar

from app.config import settings

T = TypeVar("T")


class CircuitOpen(Exception):
    def __init__(self, name: str, retry_in: float) -> None:
        super().__init__(f"circuit '{name}' is open; retry in {retry_in:.1f}s")
        self.name = name
        self.retry_in = retry_in


class BulkheadFull(Exception):
    pass


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = field(default_factory=lambda: settings.circuit_breaker_failure_threshold)
    reset_seconds: float = field(default_factory=lambda: settings.circuit_breaker_reset_seconds)
    state: str = "closed"  # closed | open | half_open
    failures: int = 0
    opened_at: float = 0.0
    _trial_inflight: bool = False

    def before(self) -> None:
        if self.state == "open":
            elapsed = time.monotonic() - self.opened_at
            if elapsed < self.reset_seconds:
                raise CircuitOpen(self.name, self.reset_seconds - elapsed)
            self.state = "half_open"
        if self.state == "half_open":
            if self._trial_inflight:
                raise CircuitOpen(self.name, 0.5)
            self._trial_inflight = True

    def success(self) -> None:
        self.state, self.failures, self._trial_inflight = "closed", 0, False

    def failure(self) -> None:
        self._trial_inflight = False
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.state, self.opened_at = "open", time.monotonic()

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        self.before()
        try:
            result = await fn()
        except Exception:
            self.failure()
            raise
        self.success()
        return result


_breakers: dict[str, CircuitBreaker] = {}
_bulkheads: dict[str, asyncio.Semaphore] = {}


def breaker(name: str) -> CircuitBreaker:
    cb = _breakers.get(name)
    if cb is None:
        cb = _breakers[name] = CircuitBreaker(name)
    return cb


def bulkhead(name: str, limit: int | None = None) -> asyncio.Semaphore:
    sem = _bulkheads.get(name)
    if sem is None:
        sem = _bulkheads[name] = asyncio.Semaphore(limit or settings.capability_max_concurrency)
    return sem


async def with_bulkhead(name: str, fn: Callable[[], Awaitable[T]], *, wait_seconds: float = 2.0) -> T:
    sem = bulkhead(name)
    try:
        await asyncio.wait_for(sem.acquire(), timeout=wait_seconds)
    except asyncio.TimeoutError as exc:
        raise BulkheadFull(f"too many concurrent calls to '{name}'") from exc
    try:
        return await fn()
    finally:
        sem.release()


async def retry(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    retry_on: tuple[type[BaseException], ...],
    base: float = 0.1,
    cap: float = 2.0,
) -> T:
    """Exponential backoff with full jitter: sleep U(0, min(cap, base*2^n))."""
    for n in range(attempts + 1):
        try:
            return await fn()
        except retry_on:
            if n >= attempts:
                raise
            await asyncio.sleep(random.uniform(0, min(cap, base * (2**n))))
    raise RuntimeError("unreachable")


def reset_for_tests() -> None:
    _breakers.clear()
    _bulkheads.clear()
