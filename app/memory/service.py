"""Long-term memory for agents, as an ADK `BaseMemoryService`.

  - `Mem0MemoryService` (staging/production): self-hosted mem0 OSS with its
    vectors in our own Postgres via pgvector, the LLM/embedder pointed at
    the same model gateway as the agents. Memory never leaves our infra.
  - `MockMemoryService` (local/test, `MEMORY_BACKEND=mock`): deterministic,
    in-process, keyword-ranked -- same interface, no model calls, so agent
    behaviour that depends on memory is testable offline.

Isolation: the ADK `user_id` every runner uses is `"<tenant>:<user>"`, and
mem0 is always queried with that as `user_id` and the app (spec) as
`agent_id` -- one tenant's memories are never in another tenant's result
set, and one user's are never in another user's.

Latency and failure: memory must never be what makes a turn slow or fail.
Searches run under a hard timeout (`memory_search_timeout_seconds`) behind
a circuit breaker and return nothing on timeout/error; writes never happen
on the request path at all -- the runner enqueues the finished invocation
and the memory worker ingests it (app/memory/ingest.py).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Mapping, Sequence

from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types

from app.config import settings
from app.core.resilience import CircuitOpen, breaker

logger = logging.getLogger(__name__)


def _event_texts(events: Sequence[Any]) -> list[dict[str, str]]:
    """User/model text turns from ADK events, as chat messages."""
    out = []
    for event in events:
        content = getattr(event, "content", None)
        if not content or not content.parts or getattr(event, "partial", False):
            continue
        text = " ".join(p.text for p in content.parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        if not text.strip():
            continue
        role = "user" if (content.role == "user" or event.author == "user") else "assistant"
        out.append({"role": role, "content": text.strip()[:4000]})
    return out


def _entry(text: str, *, author: str = "memory", ts: str | None = None, meta: dict | None = None) -> MemoryEntry:
    return MemoryEntry(content=types.Content(role="user", parts=[types.Part(text=text)]), author=author,
                       timestamp=ts, custom_metadata=meta or {})


class MockMemoryService(BaseMemoryService):
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], list[dict]] = defaultdict(list)

    async def add_session_to_memory(self, session) -> None:  # noqa: ANN001
        await self.add_events_to_memory(app_name=session.app_name, user_id=session.user_id, events=session.events,
                                        session_id=session.id)

    async def add_events_to_memory(self, *, app_name: str, user_id: str, events: Sequence[Any],
                                   session_id: str | None = None,
                                   custom_metadata: Mapping[str, object] | None = None) -> None:
        bucket = self._items[(app_name, user_id)]
        seen = {i["text"] for i in bucket}
        for msg in _event_texts(events):
            if msg["role"] == "user" and msg["content"] not in seen:
                bucket.append({"text": msg["content"], "ts": time.time(), "session_id": session_id})
                seen.add(msg["content"])

    async def add_memory(self, *, app_name: str, user_id: str, memories: Sequence[MemoryEntry],
                         custom_metadata: Mapping[str, object] | None = None) -> None:
        for m in memories:
            text = " ".join(p.text for p in (m.content.parts or []) if p.text)
            self._items[(app_name, user_id)].append({"text": text, "ts": time.time(), "session_id": None})

    async def search_memory(self, *, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        terms = {t for t in query.lower().split() if len(t) > 2}
        scored = []
        for item in self._items.get((app_name, user_id), []):
            hits = sum(1 for t in terms if t in item["text"].lower())
            if hits:
                scored.append((hits, item))
        scored.sort(key=lambda x: (-x[0], -x[1]["ts"]))
        return SearchMemoryResponse(memories=[_entry(i["text"]) for _, i in scored[: settings.memory_top_k]])


def mem0_config() -> dict:
    dsn = settings.mem0_pgvector_dsn or settings.database_url.replace("postgresql+asyncpg", "postgresql")
    model = settings.mem0_llm_model or settings.llm_model_name
    cfg: dict[str, Any] = {
        "vector_store": {
            "provider": "pgvector",
            "config": {
                "connection_string": dsn,
                "collection_name": settings.mem0_collection,
                "embedding_model_dims": settings.mem0_embedding_dims,
                "hnsw": True,
                "maxconn": 20,
            },
        },
        "version": "v1.1",
    }
    if settings.llm_base_url and settings.llm_api_key:
        cfg["llm"] = {"provider": "openai", "config": {"model": model, "api_key": settings.llm_api_key,
                                                         "openai_base_url": settings.llm_base_url}}
        cfg["embedder"] = {"provider": "openai", "config": {"model": settings.mem0_embedding_model,
                                                             "api_key": settings.llm_api_key,
                                                             "openai_base_url": settings.llm_base_url,
                                                             "embedding_dims": settings.mem0_embedding_dims}}
    return cfg


class Mem0MemoryService(BaseMemoryService):
    def __init__(self, memory: Any = None, *, infer: bool = True) -> None:
        self._memory = memory
        self._infer = infer
        self._lock = asyncio.Lock()

    async def _mem(self):
        if self._memory is None:
            async with self._lock:
                if self._memory is None:
                    import os

                    # mem0 OSS phones home usage telemetry by default; memory
                    # content must not leave our infrastructure.
                    os.environ.setdefault("MEM0_TELEMETRY", "False")
                    from mem0 import AsyncMemory

                    # Constructing it opens DB pools (blocking I/O): off the event loop.
                    self._memory = await asyncio.to_thread(AsyncMemory.from_config, mem0_config())
        return self._memory

    async def add_session_to_memory(self, session) -> None:  # noqa: ANN001
        await self.add_events_to_memory(app_name=session.app_name, user_id=session.user_id, events=session.events,
                                        session_id=session.id)

    async def add_events_to_memory(self, *, app_name: str, user_id: str, events: Sequence[Any],
                                   session_id: str | None = None,
                                   custom_metadata: Mapping[str, object] | None = None) -> None:
        messages = _event_texts(events)
        if not messages:
            return
        mem = await self._mem()
        await mem.add(messages, user_id=user_id, agent_id=app_name, run_id=None,
                      metadata={"session_id": session_id, **dict(custom_metadata or {})}, infer=self._infer)

    async def add_memory(self, *, app_name: str, user_id: str, memories: Sequence[MemoryEntry],
                         custom_metadata: Mapping[str, object] | None = None) -> None:
        mem = await self._mem()
        for m in memories:
            text = " ".join(p.text for p in (m.content.parts or []) if p.text)
            await mem.add(text, user_id=user_id, agent_id=app_name, metadata=dict(custom_metadata or {}),
                          infer=self._infer)

    async def search_memory(self, *, app_name: str, user_id: str, query: str) -> SearchMemoryResponse:
        cb = breaker("memory:mem0")
        try:
            async def _search():
                mem = await self._mem()
                return await asyncio.wait_for(
                    mem.search(query, top_k=settings.memory_top_k, filters={"user_id": user_id, "agent_id": app_name}),
                    timeout=settings.memory_search_timeout_seconds,
                )

            raw = await cb.call(_search)
        except (CircuitOpen, asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001 - never fail the turn
            logger.warning("memory search skipped: %s", exc)
            return SearchMemoryResponse(memories=[])
        results = raw.get("results", raw) if isinstance(raw, dict) else raw
        return SearchMemoryResponse(memories=[
            _entry(r.get("memory", ""), ts=str(r.get("created_at") or ""), meta={"score": r.get("score")})
            for r in (results or []) if r.get("memory")
        ])


_service: BaseMemoryService | None = None


def get_memory_service() -> BaseMemoryService | None:
    global _service
    backend = settings.effective_memory_backend
    if backend == "off":
        return None
    if _service is None:
        _service = Mem0MemoryService() if backend == "mem0" else MockMemoryService()
    return _service


def reset_for_tests() -> None:
    global _service
    _service = None
