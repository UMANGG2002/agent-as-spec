"""SQLAlchemy tables for the marketplace mechanics + the core data model
table in the architecture doc. IDs are plain strings (no cross-table FK
constraints) deliberately -- this is a minimal build on SQLite by default;
add FKs when you move to Postgres and want them enforced."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class CapabilityGrantRow(Base):
    __tablename__ = "capability_grants"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _id("grant"))
    capability_id: Mapped[str] = mapped_column(String, index=True)
    grantor_tenant_id: Mapped[str] = mapped_column(String, index=True)
    grantee_tenant_id: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|approved|denied|revoked
    scope_limits: Mapped[list] = mapped_column(JSON, default=list)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class InvocationRecordRow(Base):
    __tablename__ = "invocation_records"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, index=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    capability_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    grant_id: Mapped[str | None] = mapped_column(String, nullable=True)
    plan_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="running")  # running|succeeded|failed|blocked|pending_approval
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_estimate: Mapped[float] = mapped_column(Float, default=0.0)
    trace_id: Mapped[str | None] = mapped_column(String, nullable=True)


class PlanRecordRow(Base):
    __tablename__ = "execution_plans"

    plan_id: Mapped[str] = mapped_column(String, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String, index=True)
    goal: Mapped[str] = mapped_column(String)
    data: Mapped[dict] = mapped_column(JSON)  # ExecutionPlan.model_dump()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AgentDefinitionRow(Base):
    """A compiled, reusable agent -- optionally itself registered back into
    the capability registry as a new marketplace-listable capability, per the
    architecture doc's "the output of agent creation is itself an entry the
    registry can serve back to the planner next time."""

    __tablename__ = "agent_definitions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _id("agentdef"))
    plan_id: Mapped[str] = mapped_column(String, index=True)
    tenant_id: Mapped[str] = mapped_column(String, index=True)
    capability_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class NotificationPreferenceRow(Base):
    __tablename__ = "notification_preferences"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: _id("notifpref"))
    tenant_id: Mapped[str] = mapped_column(String, index=True)
    event_type: Mapped[str] = mapped_column(String)  # run_completed|run_failed|approval_needed|grant_requested|...
    channel: Mapped[str] = mapped_column(String)  # composio connected-account handle / channel id
    vault_entry_id: Mapped[str | None] = mapped_column(String, nullable=True)
