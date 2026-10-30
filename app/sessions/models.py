"""Index rows for sessions and invocations. ADK's own session store
(`DatabaseSessionService`) holds the conversation events; these tables hold
what the platform needs to authorize, list, meter and observe them without
reading event history: who owns a session, which spec version it runs,
and each invocation's lifecycle (queued -> running -> succeeded / failed /
cancelled / awaiting_input) with timings and token usage.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ChatSessionRow(Base):
    __tablename__ = "chat_sessions"
    __table_args__ = (Index("ix_chat_sessions_tenant_user", "tenant_id", "user_id", "created_at"),)

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64))
    user_id: Mapped[str] = mapped_column(String(128))
    app_name: Mapped[str] = mapped_column(String(128))
    spec_id: Mapped[str] = mapped_column(String(64), index=True)
    spec_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class InvocationRow(Base):
    __tablename__ = "invocations"
    __table_args__ = (
        Index("ix_invocations_session", "session_id", "enqueued_at"),
        Index("ix_invocations_tenant_status", "tenant_id", "status"),
    )

    invocation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64))
    tenant_id: Mapped[str] = mapped_column(String(64))
    user_id: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fence: Mapped[int] = mapped_column(Integer, default=0)
    # ADK's own invocation id (captured from the first event), used to
    # resume this invocation after a worker crash.
    adk_invocation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enqueued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    events: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
