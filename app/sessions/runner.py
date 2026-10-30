"""Runner workers: take invocation jobs off the work queue and run them on
the ADK `Runner`, streaming every event to the invocation's Redis stream.

Per job (`process_job`):
  1. dedupe -- an invocation already terminal is just ACKed;
  2. tenant concurrency slot (distributed semaphore with leases) -- if the
     tenant is at `concurrent_invocations`, the job is deferred (re-enqueued
     with a short jittered delay), never holding a worker slot while waiting;
  3. session lease -- one runner per session at a time, fleet-wide;
  4. load the PINNED spec version and the tenant's current executable
     capabilities (cached), get the compiled `App` (cached);
  5. `Runner.run_async(...)` with the caller's identity as `state_delta`,
     tool confirmations as function responses, an abort signal wired to
     cancellation and to the invocation timeout, and lease/slot renewal
     running alongside;
  6. publish each event (partial token deltas too when requested), record
     tokens, status and timing, emit `invocation.finished` through the
     outbox, enqueue the turn for memory ingestion;
  7. release lease + slot, ACK.

Crash safety: a worker that dies mid-invocation never ACKs; after the
visibility timeout another worker reclaims the job and resumes the ADK
invocation from its last persisted event (resumable App). Side-effecting
tool calls inside it are idempotency-keyed on the ADK invocation id, so
they aren't repeated. After MAX_DELIVERIES the job is dead-lettered.

Graceful shutdown (SIGTERM): stop taking jobs, let in-flight invocations
finish up to a deadline, then exit -- whatever is unfinished is reclaimed
by other workers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import signal
import socket
import time
from datetime import datetime, timezone
from typing import Any

from google.genai import types

from app.config import settings
from app.core import context, ratelimit
from app.core import redis as redis_core
from app.db import session_scope
from app.sessions import leases, queue, streams
from app.sessions.models import InvocationRow
from app.sessions.session_store import adk_user_id, get_session_service

logger = logging.getLogger(__name__)
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"

# Tests inject a scripted model here; None = settings.live_model.
model_override: Any = None
if os.environ.get("SCRIPTED_MODEL_FOR_TESTS") == "1" and not settings.is_production_like:
    # Multi-process smoke tests without a reachable model gateway. Ignored
    # in staging/production.
    from app.testing.scripted_llm import ScriptedToolCallingLlm

    model_override = ScriptedToolCallingLlm(
        transfer_to=[x for x in os.environ.get("SCRIPTED_MODEL_TRANSFER_TO", "").split(",") if x])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _truncate(value: Any, limit: int = 4000) -> Any:
    try:
        import orjson

        raw = orjson.dumps(value, default=str)
    except Exception:  # noqa: BLE001
        raw = str(value).encode()
    if len(raw) <= limit:
        return value
    return {"truncated": True, "preview": raw[:limit].decode(errors="ignore")}


def serialize_event(event) -> dict:  # noqa: ANN001
    """Compact, client-facing view of an ADK event."""
    out: dict[str, Any] = {"type": "event", "id": event.id, "author": event.author,
                           "partial": bool(getattr(event, "partial", False))}
    content = event.content
    if content and content.parts:
        texts = [p.text for p in content.parts if getattr(p, "text", None) and not getattr(p, "thought", False)]
        if texts:
            out["text"] = "".join(texts)
        calls = [p.function_call for p in content.parts if getattr(p, "function_call", None)]
        if calls:
            out["function_calls"] = [{"id": c.id, "name": c.name, "args": _truncate(c.args or {})} for c in calls]
            confirm = [c for c in calls if c.name == "adk_request_confirmation"]
            if confirm:
                args = confirm[0].args or {}
                tc = args.get("toolConfirmation") or args.get("tool_confirmation") or {}
                out["confirmation_request"] = {"function_call_id": confirm[0].id, "hint": tc.get("hint"),
                                               "payload": tc.get("payload")}
        responses = [p.function_response for p in content.parts if getattr(p, "function_response", None)]
        if responses:
            out["function_responses"] = [{"id": r.id, "name": r.name, "response": _truncate(r.response)}
                                         for r in responses]
    actions = getattr(event, "actions", None)
    if actions is not None and getattr(actions, "transfer_to_agent", None):
        out["transfer_to"] = actions.transfer_to_agent
    usage = getattr(event, "usage_metadata", None)
    if usage:
        out["usage"] = {"in": getattr(usage, "prompt_token_count", None),
                        "out": getattr(usage, "candidates_token_count", None)}
    try:
        out["final"] = bool(event.is_final_response())
    except Exception:  # noqa: BLE001
        pass
    return out


def _new_message(job: dict) -> types.Content | None:
    parts: list[types.Part] = []
    for c in job.get("confirmations") or []:
        parts.append(types.Part(function_response=types.FunctionResponse(
            id=c["function_call_id"], name="adk_request_confirmation",
            response={"confirmed": bool(c["confirmed"]), "payload": c.get("payload")})))
    if job.get("text"):
        parts.append(types.Part(text=job["text"]))
    return types.Content(role="user", parts=parts) if parts else None


async def _update(invocation_id: str, tenant_id: str, **values: Any) -> InvocationRow | None:
    async with session_scope(tenant_id=tenant_id) as db:
        row = await db.get(InvocationRow, invocation_id)
        if row is None:
            return None
        for k, v in values.items():
            setattr(row, k, v)
        return row


async def _defer(job: dict, why: str) -> None:
    delay = min(5.0, 0.25 * (2 ** min(int(job.get("deferrals", 0)), 4))) * random.uniform(0.5, 1.5)
    job = {**job, "deferrals": int(job.get("deferrals", 0)) + 1}
    logger.debug("deferring %s (%s) by %.2fs", job["invocation_id"], why, delay)
    if settings.runner_mode == "inline":
        await asyncio.sleep(delay)
        asyncio.get_running_loop().create_task(run_inline(job))
    else:
        await queue.enqueue(job, not_before=time.time() + delay)


async def _keepalive(job: dict, holder: str, sem: ratelimit.Semaphore, abort: asyncio.Event) -> None:
    """Renew lease + slot; turn a cancel request or timeout into abort."""
    from app.sessions.manager import _cancel_key

    client = await redis_core.get_redis()
    deadline = time.monotonic() + settings.invocation_timeout_seconds
    renew_every = settings.session_lease_ttl_ms / 3000
    last_renew = time.monotonic()
    while not abort.is_set():
        await asyncio.sleep(0.5)
        if await client.exists(_cancel_key(job["invocation_id"])):
            job["_cancelled"] = True
            abort.set()
        elif time.monotonic() > deadline:
            job["_timed_out"] = True
            abort.set()
        if time.monotonic() - last_renew >= renew_every:
            last_renew = time.monotonic()
            if not await leases.renew(job["session_id"], holder):
                logger.warning("lost session lease for %s; aborting", job["session_id"])
                job["_lease_lost"] = True
                abort.set()
            await sem.renew(job["invocation_id"])


async def process_job(job: dict, *, resume: bool = False) -> str:
    """Run one invocation to a terminal state. Returns the final status, or
    'deferred' if it was re-queued to run later."""
    from app.registry import store as registry_store
    from app.specs import store as specs_store
    from app.tenancy.service import get_limits

    tenant_id, session_id, invocation_id = job["tenant_id"], job["session_id"], job["invocation_id"]
    with context.bind(tenant_id=tenant_id, user_id=job["user_id"], session_id=session_id,
                      invocation_id=invocation_id):
        async with session_scope(tenant_id=tenant_id) as db:
            row = await db.get(InvocationRow, invocation_id)
        if row is None or row.status in ("succeeded", "failed", "cancelled", "awaiting_input"):
            return row.status if row else "missing"

        limits = await get_limits(tenant_id)
        sem = ratelimit.Semaphore("invocations", tenant_id, limit=limits.concurrent_invocations,
                                  lease_ms=settings.runner_visibility_timeout_ms)
        if await sem.acquire(invocation_id) is None:
            await _defer(job, "tenant at concurrency limit")
            return "deferred"
        holder = f"{WORKER_ID}:{invocation_id}"
        fence = await leases.acquire(session_id, holder)
        if fence is None:
            await sem.release(invocation_id)
            await _defer(job, "session busy")
            return "deferred"

        client = await redis_core.get_redis()
        if row.status == "queued":  # first start: it leaves the tenant's backlog
            await client.decr(redis_core.key("t", tenant_id, "queued"))
        started = time.monotonic()
        from app.telemetry.setup import invocation_ms, invocations_started_total, queue_wait_ms

        queue_wait_ms.record(max(0.0, (time.time() - float(job.get("enqueued_at", time.time()))) * 1000))
        await _update(invocation_id, tenant_id, status="running", started_at=_now(), worker_id=WORKER_ID,
                      attempt=(row.attempt or 0) + 1, fence=fence)
        await streams.publish(session_id, invocation_id, {"type": "invocation.started", "attempt": row.attempt + 1})

        abort = asyncio.Event()
        keepalive = asyncio.get_running_loop().create_task(_keepalive(job, holder, sem, abort))
        status, error, tokens_in, tokens_out, n_events = "succeeded", None, 0, 0, 0
        adk_invocation_id = row.adk_invocation_id
        online_trace, online_calls = None, {}  # set when this run is sampled for online eval
        try:
            from app.memory.service import get_memory_service
            from app.sessions.agent_cache import ROUTER_ID, get_app, get_router_app

            capabilities = await registry_store.visible_capabilities(tenant_id)
            if job["spec_id"] == ROUTER_ID:
                # The router always mounts the tenant's CURRENT approved specs.
                specs = await specs_store.list_specs_cached(tenant_id)
                app = await get_router_app(tenant_id, specs, capabilities, model=model_override)
                max_calls = 100
            else:
                spec = await specs_store.get_spec_version(job["spec_id"], int(job["spec_version"]),
                                                          tenant_id=tenant_id)
                if spec is None:
                    raise RuntimeError("the agent spec version this session runs no longer exists")
                app = await get_app(spec, capabilities, model=model_override)
                max_calls = spec.runtime.max_llm_calls or 50
            from google.adk.agents.run_config import RunConfig, StreamingMode
            from google.adk.runners import Runner

            runner = Runner(app=app, session_service=get_session_service(), memory_service=get_memory_service())
            run_config = RunConfig(streaming_mode=StreamingMode.SSE if job.get("stream_tokens") else StreamingMode.NONE,
                                   max_llm_calls=max(1, max_calls))
            kwargs: dict[str, Any] = {
                "user_id": adk_user_id(tenant_id, job["user_id"]), "session_id": session_id,
                "state_delta": {"__identity__": job["identity"]}, "run_config": run_config, "abort_signal": abort,
            }
            if resume and adk_invocation_id:
                kwargs["invocation_id"] = adk_invocation_id
            else:
                kwargs["new_message"] = _new_message(job)
            awaiting = False
            if online_trace is None and not resume:
                from app.evals import online as _online

                if _online.should_sample():
                    from app.evals.schemas import Trace as _Trace

                    online_trace, online_calls = _Trace(), {}
            async for event in runner.run_async(**kwargs):
                n_events += 1
                if adk_invocation_id is None and getattr(event, "invocation_id", None):
                    adk_invocation_id = event.invocation_id
                    await _update(invocation_id, tenant_id, adk_invocation_id=adk_invocation_id)
                payload = serialize_event(event)
                if "confirmation_request" in payload:
                    awaiting = True
                if online_trace is not None:
                    from app.evals.runner import _absorb as _absorb_event

                    _absorb_event(online_trace, payload, online_calls)
                usage = payload.get("usage") or {}
                if not payload["partial"]:
                    tokens_in += int(usage.get("in") or 0)
                    tokens_out += int(usage.get("out") or 0)
                await streams.publish(session_id, invocation_id, payload)
            if job.get("_cancelled"):
                status = "cancelled"
            elif job.get("_timed_out"):
                status, error = "failed", "invocation timed out"
            elif job.get("_lease_lost"):
                status, error = "failed", "worker lost its session lease"
            elif awaiting:
                status = "awaiting_input"
        except Exception as exc:  # noqa: BLE001 - reported to the client, never crashes the worker
            logger.exception("invocation %s failed", invocation_id)
            status, error = "failed", f"{type(exc).__name__}: {exc}"[:1000]
        finally:
            abort.set()
            keepalive.cancel()
            await leases.release(session_id, holder)
            await sem.release(invocation_id)

        async with session_scope(tenant_id=tenant_id) as db:
            inv = await db.get(InvocationRow, invocation_id)
            if inv is not None:
                inv.status, inv.error, inv.finished_at = status, error, _now()
                inv.tokens_in, inv.tokens_out, inv.events = tokens_in, tokens_out, n_events
            from app.core.outbox import emit

            emit(db, tenant_id=tenant_id, topic="invocation.finished",
                 payload={"invocation_id": invocation_id, "session_id": session_id, "status": status,
                          "tokens_in": tokens_in, "tokens_out": tokens_out, "spec_id": job["spec_id"]})
        await streams.publish(session_id, invocation_id, {"type": streams.TERMINAL, "status": status,
                                                          "error": error, "tokens": {"in": tokens_in, "out": tokens_out}})
        invocations_started_total.add(1, {"status": status})
        invocation_ms.record((time.monotonic() - started) * 1000, {"status": status})
        if online_trace is not None and status in ("succeeded", "awaiting_input"):
            from app.evals import online as _online

            _online.schedule_score(job, online_trace, status)
        if status in ("succeeded", "awaiting_input"):
            from app.memory import ingest

            await ingest.enqueue(job, adk_invocation_id=adk_invocation_id)
        return status


async def run_inline(job: dict) -> str:
    """runner_mode=inline (local dev): run in the API process."""
    try:
        return await process_job(job)
    except Exception:  # noqa: BLE001
        logger.exception("inline invocation crashed")
        return "failed"


class Worker:
    """The runner process main loop: N concurrent invocation slots."""

    def __init__(self, *, concurrency: int | None = None) -> None:
        self.concurrency = concurrency or settings.runner_concurrency
        self.slots = asyncio.Semaphore(self.concurrency)
        self.stopping = asyncio.Event()
        self.inflight: set[asyncio.Task] = set()

    async def _handle(self, shard: int, entry_id: str, job: dict, *, deliveries: int = 1, resume: bool = False) -> None:
        try:
            if deliveries > queue.MAX_DELIVERIES:
                await queue.dead_letter(shard, entry_id, job, "max deliveries exceeded")
                await _update(job["invocation_id"], job["tenant_id"], status="failed",
                              error="could not be processed after repeated attempts", finished_at=_now())
                await streams.publish(job["session_id"], job["invocation_id"],
                                      {"type": streams.TERMINAL, "status": "failed", "error": "dead-lettered"})
                return
            await process_job(job, resume=resume)
            await queue.ack(shard, entry_id)
        except Exception:  # noqa: BLE001 - leave unacked: it will be reclaimed
            logger.exception("job %s crashed; leaving it for reclaim", entry_id)
        finally:
            self.slots.release()

    async def run(self) -> None:
        await queue.ensure_groups()
        last_reclaim = 0.0
        while not self.stopping.is_set():
            free = self.slots._value  # noqa: SLF001 - free invocation slots
            if free <= 0:
                await asyncio.sleep(0.05)
                continue
            batch: list[tuple[int, str, dict, int, bool]] = []
            if time.monotonic() - last_reclaim > 5:
                last_reclaim = time.monotonic()
                await queue.promote_due()
                batch += [(s, e, j, d, True) for s, e, j, d in await queue.reclaim(WORKER_ID, count=free)]
            if not batch:
                batch += [(s, e, j, 1, False) for s, e, j in
                          await queue.read(WORKER_ID, count=free, block_ms=1000)]
            if not batch and settings.redis_cluster:
                await asyncio.sleep(0.2)
            for shard, entry_id, job, deliveries, resume in batch:
                await self.slots.acquire()
                task = asyncio.get_running_loop().create_task(
                    self._handle(shard, entry_id, job, deliveries=deliveries, resume=resume))
                self.inflight.add(task)
                task.add_done_callback(self.inflight.discard)

    async def shutdown(self, *, deadline_seconds: float = 25.0) -> None:
        self.stopping.set()
        if self.inflight:
            await asyncio.wait(self.inflight, timeout=deadline_seconds)


async def main() -> None:
    from app.core.logging import configure_logging
    from app.db import init_db
    from app.telemetry.setup import configure_telemetry

    configure_logging(settings.log_level, json=settings.log_json)
    configure_telemetry()
    await init_db()
    worker = Worker()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker.stopping.set)
    runner_task = loop.create_task(worker.run())

    async def _promoter() -> None:
        while not worker.stopping.is_set():
            await queue.promote_due()
            await asyncio.sleep(0.2)

    promoter_task = loop.create_task(_promoter())
    from app.memory import ingest

    memory_task = loop.create_task(ingest.consume_forever(worker.stopping))
    await worker.stopping.wait()
    await worker.shutdown()
    for task in (runner_task, memory_task, promoter_task):
        task.cancel()
    await redis_core.close()


if __name__ == "__main__":
    asyncio.run(main())
