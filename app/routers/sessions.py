"""v1 conversational API: talk to an approved spec agent.

    POST /v1/sessions                                         {spec_id} -> {session_id}
    GET  /v1/sessions                                         my sessions
    POST /v1/sessions/{sid}/messages                          {text?, confirmations?, stream_tokens?}
                                                              (+ Idempotency-Key header) -> 202 {invocation_id, stream_url}
    GET  /v1/sessions/{sid}/invocations/{iid}                 status, tokens, timings
    GET  /v1/sessions/{sid}/invocations/{iid}/stream          SSE, resumable (Last-Event-ID or ?after=)
    POST /v1/sessions/{sid}/invocations/{iid}/cancel
    GET  /v1/sessions/{sid}/events                            recent history (reload after a refresh)

The SSE stream: every event carries `id:` (its Redis stream id); on a
dropped connection the client (EventSource does this by itself) reconnects
with `Last-Event-ID` and receives exactly what it missed, from any API
replica. `: ping` comments every `stream_heartbeat_seconds` keep proxies
and load balancers from closing an idle stream; `retry:` tells the client
how soon to reconnect. The stream ends after `invocation.finished`.
"""

from __future__ import annotations

import orjson
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth.deps import get_identity
from app.auth.models import IdentityContext
from app.config import settings
from app.sessions import manager, streams

router = APIRouter(prefix="/v1/sessions", tags=["sessions"])
agents_router = APIRouter(prefix="/v1", tags=["sessions"])


@agents_router.get("/agents")
async def chat_targets(identity: IdentityContext = Depends(get_identity)) -> list[dict]:
    """What a chat session can talk to: the `spec_test_agent` router (every
    approved spec mounted as a sub-agent) and each approved spec on its own."""
    from app.sessions.agent_cache import ROUTER_ID
    from app.specs import store as specs_store

    specs = await specs_store.list_specs_cached(identity.tenant_id)
    out = [{"id": ROUTER_ID, "name": "spec_test_agent", "kind": "router",
            "description": f"Routes to any of your {len(specs)} approved spec(s)"}]
    out += [{"id": s.spec_id, "name": s.name, "kind": "spec", "version": s.version,
             "pattern": s.runtime.pattern.value, "description": s.goal} for s in specs]
    return out


class CreateSessionBody(BaseModel):
    spec_id: str = Field(min_length=1, max_length=64)


class ConfirmationBody(BaseModel):
    function_call_id: str = Field(min_length=1, max_length=128)
    confirmed: bool
    payload: dict | None = None


class MessageBody(BaseModel):
    text: str | None = Field(default=None, max_length=200_000)
    confirmations: list[ConfirmationBody] = Field(default_factory=list, max_length=20)
    stream_tokens: bool = False


def _http(exc: Exception) -> HTTPException:
    if isinstance(exc, manager.NotFound):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, manager.Rejected):
        headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after else None
        return HTTPException(status_code=exc.status, detail=exc.detail, headers=headers)
    raise exc


@router.post("", status_code=201)
async def create_session(body: CreateSessionBody, identity: IdentityContext = Depends(get_identity)) -> dict:
    try:
        return await manager.create_session(identity, body.spec_id)
    except (manager.NotFound, manager.Rejected) as exc:
        raise _http(exc) from exc


@router.get("")
async def list_sessions(limit: int = Query(50, ge=1, le=200), identity: IdentityContext = Depends(get_identity)):
    return await manager.list_sessions(identity, limit=limit)


@router.post("/{session_id}/messages", status_code=202)
async def send_message(
    session_id: str,
    body: MessageBody,
    identity: IdentityContext = Depends(get_identity),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> dict:
    try:
        return await manager.send_message(
            identity, session_id, text=body.text,
            confirmations=[manager.Confirmation(**c.model_dump()) for c in body.confirmations],
            stream_tokens=body.stream_tokens, idempotency_key=idempotency_key,
        )
    except (manager.NotFound, manager.Rejected) as exc:
        raise _http(exc) from exc


@router.get("/{session_id}/invocations/{invocation_id}")
async def get_invocation(session_id: str, invocation_id: str, identity: IdentityContext = Depends(get_identity)):
    try:
        return await manager.get_invocation(identity, session_id, invocation_id)
    except manager.NotFound as exc:
        raise _http(exc) from exc


@router.post("/{session_id}/invocations/{invocation_id}/cancel")
async def cancel_invocation(session_id: str, invocation_id: str, identity: IdentityContext = Depends(get_identity)):
    try:
        return await manager.cancel(identity, session_id, invocation_id)
    except manager.NotFound as exc:
        raise _http(exc) from exc


@router.get("/{session_id}/events")
async def session_events(session_id: str, limit: int = Query(100, ge=1, le=500),
                         identity: IdentityContext = Depends(get_identity)):
    try:
        return await manager.history(identity, session_id, limit=limit)
    except manager.NotFound as exc:
        raise _http(exc) from exc


@router.get("/{session_id}/invocations/{invocation_id}/stream")
async def stream_invocation(
    request: Request,
    session_id: str,
    invocation_id: str,
    after: str | None = Query(default=None, pattern=r"^\d+-\d+$"),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    identity: IdentityContext = Depends(get_identity),
):
    try:
        await manager.get_invocation(identity, session_id, invocation_id)
    except manager.NotFound as exc:
        raise _http(exc) from exc
    cursor = last_event_id if (last_event_id and last_event_id.replace("-", "").isdigit()) else after

    async def _sse():
        yield b"retry: 2000\n\n"
        async for entry_id, event in streams.follow(session_id, invocation_id, cursor,
                                                    heartbeat=settings.stream_heartbeat_seconds):
            if await request.is_disconnected():
                return
            if entry_id is None:
                yield b": ping\n\n"
                continue
            data = orjson.dumps(event, default=str)
            yield b"id: " + entry_id.encode() + b"\nevent: " + event.get("type", "event").encode() + \
                b"\ndata: " + data + b"\n\n"

    return StreamingResponse(_sse(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"})
