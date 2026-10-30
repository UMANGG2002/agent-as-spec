"""Per-request / per-invocation context carried in contextvars, so logs,
spans, DB sessions (row-level security) and rate limiters can read the
current tenant without every function threading it through by hand.

Contextvars are copied into tasks created with `asyncio.create_task`, so a
background invocation started from a request keeps its tenant -- and a
worker sets them explicitly per job (`bind`).
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
invocation_id_var: ContextVar[str | None] = ContextVar("invocation_id", default=None)


def new_request_id() -> str:
    return uuid.uuid4().hex


def current_tenant() -> str | None:
    return tenant_id_var.get()


@contextmanager
def bind(**values: str | None) -> Iterator[None]:
    """Temporarily set context values (tenant_id=..., user_id=..., ...)."""
    tokens = []
    mapping = {
        "request_id": request_id_var,
        "tenant_id": tenant_id_var,
        "user_id": user_id_var,
        "session_id": session_id_var,
        "invocation_id": invocation_id_var,
    }
    for key, value in values.items():
        tokens.append((mapping[key], mapping[key].set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def snapshot() -> dict[str, str]:
    out = {
        "request_id": request_id_var.get(),
        "tenant_id": tenant_id_var.get(),
        "user_id": user_id_var.get(),
        "session_id": session_id_var.get(),
        "invocation_id": invocation_id_var.get(),
    }
    return {k: v for k, v in out.items() if v}
