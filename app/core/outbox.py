"""Transactional outbox: domain events are written to `outbox_events` in
the SAME database transaction as the change that caused them, and a relay
process publishes them to Redis Streams afterwards.

This removes the dual-write problem ("row committed but event lost", or
"event sent but transaction rolled back") without distributed
transactions: if the transaction commits, the event will be published at
least once; if it rolls back, the event never existed. Consumers
(notifications, metering, analytics, cache invalidation on other services)
dedupe on the event id.

The relay claims batches with `FOR UPDATE SKIP LOCKED`, so any number of
relay replicas can run concurrently without double-claiming. On Postgres
the table is range-partitioned by month (migrations/sql/0003_partitions.sql)
so old events are dropped by detaching a partition, not by DELETE.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import orjson
from sqlalchemy import JSON, DateTime, String, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core import redis as redis_core
from app.core.ids import new_id
from app.db import Base

logger = logging.getLogger(__name__)
_JSON = JSON().with_variant(JSONB(), "postgresql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class OutboxEventRow(Base):
    __tablename__ = "outbox_events"
    __table_args__ = {"info": {"partitioned": True}}

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, default=_now)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    topic: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(_JSON)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)


def emit(session: AsyncSession, *, tenant_id: str, topic: str, payload: dict[str, Any]) -> str:
    """Stage an event inside the caller's transaction. Returns its id."""
    event_id = new_id("evt")
    session.add(OutboxEventRow(id=event_id, tenant_id=tenant_id, topic=topic, payload=payload, created_at=_now()))
    return event_id


def topic_stream(topic: str) -> str:
    return f"events:{topic}"


async def relay_once(session: AsyncSession, *, batch: int = 500) -> int:
    """Publish up to `batch` pending events. Safe to run on many replicas."""
    stmt = (
        select(OutboxEventRow)
        .where(OutboxEventRow.published_at.is_(None))
        .order_by(OutboxEventRow.created_at)
        .limit(batch)
    )
    if session.bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    rows = (await session.execute(stmt)).scalars().all()
    if not rows:
        return 0
    client = await redis_core.get_redis()
    pipe = client.pipeline(transaction=False)
    for row in rows:
        body = {"id": row.id, "tenant_id": row.tenant_id, "topic": row.topic, "payload": row.payload,
                "created_at": row.created_at.isoformat()}
        pipe.xadd(topic_stream(row.topic), {"data": orjson.dumps(body, default=str).decode()},
                  maxlen=100_000, approximate=True)
    await pipe.execute()
    now = _now()
    await session.execute(
        update(OutboxEventRow)
        .where(OutboxEventRow.id.in_([r.id for r in rows]))
        .values(published_at=now)
        .execution_options(synchronize_session=False)
    )
    return len(rows)
