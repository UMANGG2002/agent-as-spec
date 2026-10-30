"""SessionManager: what the API calls. It never runs an agent itself (in
`runner_mode=queue`); it authorizes, admits, records and enqueues, and
returns immediately with an invocation id and a stream URL.

Per message:
  1. authorize: the session must belong to the caller's tenant AND user;
  2. validate: message size, confirmation shape;
  3. idempotency: an `Idempotency-Key` header maps to one invocation for 24h,
     so a client that retries a POST after a dropped connection gets the
     same invocation back instead of running the agent twice;
  4. admission control: the tenant's invocation rate (GCRA) and its queued
     backlog (`max_queue_depth`) -- over either, 429 with Retry-After,
     before anything is enqueued;
  5. record the invocation (queued), publish `invocation.queued` to its
     stream, and enqueue the job (or start it in-process when
     `runner_mode=inline`, the local default).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.auth.models import IdentityContext
from app.config import settings
from app.core import ratelimit
from app.core import redis as redis_core
from app.core.ids import new_id
from app.db import session_scope
from app.sessions import queue, streams
from app.sessions.models import ChatSessionRow, InvocationRow
from app.sessions.session_store import adk_user_id, get_session_service
from app.specs import store as specs_store

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "awaiting_input"}


class NotFound(Exception):
    pass


class Rejected(Exception):
    def __init__(self, status: int, detail: str, retry_after: int | None = None) -> None:
        super().__init__(detail)
        self.status, self.detail, self.retry_after = status, detail, retry_after


@dataclass
class Confirmation:
    function_call_id: str
    confirmed: bool
    payload: dict | None = None


def _iso(dt: datetime | None) -> str | None:
    """UTC ISO-8601 with offset (SQLite hands back naive datetimes)."""
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def _queued_counter(tenant_id: str) -> str:
    return redis_core.key("t", tenant_id, "queued")


def _identity_state(identity: IdentityContext) -> dict:
    return {"tenant_id": identity.tenant_id, "brand_id": identity.brand_id, "user_id": identity.user_id,
            "scopes": list(identity.scopes), "act_chain": list(identity.act_chain)}


async def create_session(identity: IdentityContext, spec_id: str) -> dict:
    """`spec_id` is one approved spec -- or `spec_test_agent`, the router that
    has every approved spec of the tenant mounted as a sub-agent."""
    from app.sessions.agent_cache import ROUTER_ID

    if spec_id == ROUTER_ID:
        session_id = new_id("ses")
        await get_session_service().create_session(
            app_name=ROUTER_ID, user_id=adk_user_id(identity.tenant_id, identity.user_id),
            session_id=session_id, state={"policy_id": "default"},
        )
        async with session_scope(tenant_id=identity.tenant_id) as db:
            db.add(ChatSessionRow(session_id=session_id, tenant_id=identity.tenant_id, user_id=identity.user_id,
                                  app_name=ROUTER_ID, spec_id=ROUTER_ID, spec_version=0))
        return {"session_id": session_id, "spec_id": ROUTER_ID, "spec_version": 0}

    spec = await specs_store.get_spec_cached(spec_id, tenant_id=identity.tenant_id)
    if spec is None or spec.tenant_id != identity.tenant_id:
        raise NotFound("agent spec not found")
    from app.sessions.agent_cache import app_name

    session_id = new_id("ses")
    await get_session_service().create_session(
        app_name=app_name(spec.spec_id), user_id=adk_user_id(identity.tenant_id, identity.user_id),
        session_id=session_id, state={"policy_id": spec.guardrail_policy_id},
    )
    async with session_scope(tenant_id=identity.tenant_id) as db:
        db.add(ChatSessionRow(session_id=session_id, tenant_id=identity.tenant_id, user_id=identity.user_id,
                              app_name=app_name(spec.spec_id), spec_id=spec.spec_id, spec_version=spec.version))
    return {"session_id": session_id, "spec_id": spec.spec_id, "spec_version": spec.version}


async def _owned_session(identity: IdentityContext, session_id: str) -> ChatSessionRow:
    async with session_scope(tenant_id=identity.tenant_id) as db:
        row = await db.get(ChatSessionRow, session_id)
    if row is None or row.tenant_id != identity.tenant_id or row.user_id != identity.user_id:
        raise NotFound("session not found")
    return row


async def list_sessions(identity: IdentityContext, *, limit: int = 50) -> list[dict]:
    async with session_scope(tenant_id=identity.tenant_id) as db:
        rows = (await db.execute(
            select(ChatSessionRow)
            .where(ChatSessionRow.tenant_id == identity.tenant_id, ChatSessionRow.user_id == identity.user_id)
            .order_by(ChatSessionRow.created_at.desc()).limit(min(limit, 200))
        )).scalars().all()
    return [{"session_id": r.session_id, "spec_id": r.spec_id, "spec_version": r.spec_version,
             "created_at": _iso(r.created_at), "last_activity_at": _iso(r.last_activity_at)} for r in rows]


async def _admit(identity: IdentityContext) -> None:
    from app.tenancy.service import get_limits

    limits = await get_limits(identity.tenant_id)
    if limits.status != "active":
        raise Rejected(403, f"tenant is {limits.status}")
    decision = await ratelimit.allow("invocations", identity.tenant_id,
                                     rate_per_second=limits.invocations_per_minute / 60.0,
                                     burst=max(1, limits.invocations_per_minute // 6))
    if not decision.allowed:
        raise Rejected(429, "invocation rate limit exceeded", retry_after=max(1, decision.retry_after_ms // 1000 + 1))
    try:
        client = await redis_core.get_redis()
        backlog = int(await client.get(_queued_counter(identity.tenant_id)) or 0)
    except Exception:  # noqa: BLE001
        backlog = 0
    if backlog >= limits.max_queue_depth:
        raise Rejected(429, "too many queued invocations for this workspace", retry_after=2)


async def send_message(
    identity: IdentityContext,
    session_id: str,
    *,
    text: str | None,
    confirmations: list[Confirmation] | None = None,
    stream_tokens: bool = False,
    idempotency_key: str | None = None,
) -> dict:
    row = await _owned_session(identity, session_id)
    confirmations = confirmations or []
    if not text and not confirmations:
        raise Rejected(400, "a message needs text or at least one confirmation")
    if text and len(text) > settings.max_message_chars:
        raise Rejected(413, f"message longer than {settings.max_message_chars} characters")

    client = await redis_core.get_redis()
    idem_redis_key = None
    if idempotency_key:
        idem_redis_key = redis_core.key("t", identity.tenant_id, "idem-msg", session_id, idempotency_key[:128])
        existing = await client.get(idem_redis_key)
        if existing:
            return await get_invocation(identity, session_id, existing)

    await _admit(identity)

    invocation_id = new_id("inv")
    if idem_redis_key and not await client.set(idem_redis_key, invocation_id, nx=True, ex=86_400):
        return await get_invocation(identity, session_id, await client.get(idem_redis_key))

    async with session_scope(tenant_id=identity.tenant_id) as db:
        db.add(InvocationRow(invocation_id=invocation_id, session_id=session_id, tenant_id=identity.tenant_id,
                             user_id=identity.user_id, status="queued"))
        chat = await db.get(ChatSessionRow, session_id)
        if chat is not None:
            chat.last_activity_at = datetime.now(timezone.utc)

    job: dict[str, Any] = {
        "invocation_id": invocation_id, "session_id": session_id, "tenant_id": identity.tenant_id,
        "user_id": identity.user_id, "spec_id": row.spec_id, "spec_version": row.spec_version,
        "app_name": row.app_name, "identity": _identity_state(identity), "text": text,
        "confirmations": [c.__dict__ for c in confirmations], "stream_tokens": stream_tokens,
        "enqueued_at": datetime.now(timezone.utc).timestamp(),
    }
    await streams.publish(session_id, invocation_id, {"type": "invocation.queued", "invocation_id": invocation_id})
    await client.incr(_queued_counter(identity.tenant_id))
    if settings.runner_mode == "inline":
        from app.sessions.runner import run_inline

        asyncio.get_running_loop().create_task(run_inline(job))
    else:
        await queue.enqueue(job)
    return {"invocation_id": invocation_id, "session_id": session_id, "status": "queued",
            "stream_url": f"/v1/sessions/{session_id}/invocations/{invocation_id}/stream"}


async def get_invocation(identity: IdentityContext, session_id: str, invocation_id: str) -> dict:
    await _owned_session(identity, session_id)
    async with session_scope(tenant_id=identity.tenant_id) as db:
        inv = await db.get(InvocationRow, invocation_id)
    if inv is None or inv.session_id != session_id:
        raise NotFound("invocation not found")
    return {"invocation_id": inv.invocation_id, "session_id": inv.session_id, "status": inv.status,
            "attempt": inv.attempt, "tokens_in": inv.tokens_in, "tokens_out": inv.tokens_out, "events": inv.events,
            "error": inv.error, "enqueued_at": _iso(inv.enqueued_at),
            "started_at": _iso(inv.started_at), "finished_at": _iso(inv.finished_at),
            "stream_url": f"/v1/sessions/{session_id}/invocations/{invocation_id}/stream"}


def _cancel_key(invocation_id: str) -> str:
    return redis_core.key("inv", invocation_id, "cancel")


async def cancel(identity: IdentityContext, session_id: str, invocation_id: str) -> dict:
    await get_invocation(identity, session_id, invocation_id)
    client = await redis_core.get_redis()
    await client.set(_cancel_key(invocation_id), "1", ex=3600)
    return {"invocation_id": invocation_id, "cancel_requested": True}


async def history(identity: IdentityContext, session_id: str, *, limit: int = 100) -> list[dict]:
    row = await _owned_session(identity, session_id)
    from google.adk.sessions.base_session_service import GetSessionConfig

    from app.sessions.runner import serialize_event

    session = await get_session_service().get_session(
        app_name=row.app_name, user_id=adk_user_id(identity.tenant_id, identity.user_id), session_id=session_id,
        config=GetSessionConfig(num_recent_events=min(limit, 500)))
    return [serialize_event(e) for e in (session.events if session else [])]
