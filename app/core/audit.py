"""Append-only audit log of security-relevant actions: capability
registration and publishing, access grants and decisions, spec approvals,
tool-call approvals, policy changes, token issuance.

Written in the same transaction as the action itself, so an action can't
happen without its audit row. On Postgres the table is partitioned by month
and the application role is granted INSERT/SELECT only (no UPDATE/DELETE --
see migrations/sql/0003_partitions.sql), making it tamper-evident from the
application's side.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.core import context
from app.core.ids import new_id
from app.core.outbox import emit
from app.db import Base

_JSON = JSON().with_variant(JSONB(), "postgresql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AuditLogRow(Base):
    __tablename__ = "audit_log"
    __table_args__ = {"info": {"partitioned": True}}

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, default=_now)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    actor: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(64), index=True)
    resource_type: Mapped[str] = mapped_column(String(32))
    resource_id: Mapped[str] = mapped_column(String(128))
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    details: Mapped[dict] = mapped_column(_JSON, default=dict)


def record(
    session: AsyncSession,
    *,
    tenant_id: str,
    actor: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict[str, Any] | None = None,
    publish: bool = True,
) -> None:
    session.add(
        AuditLogRow(
            id=new_id("aud"), created_at=_now(), tenant_id=tenant_id, actor=actor, action=action,
            resource_type=resource_type, resource_id=resource_id,
            request_id=context.request_id_var.get(), details=details or {},
        )
    )
    if publish:
        emit(session, tenant_id=tenant_id, topic="audit",
             payload={"action": action, "resource_type": resource_type, "resource_id": resource_id,
                      "actor": actor, **(details or {})})
