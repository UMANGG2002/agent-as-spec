"""Tenants and their service tier. A tier sets every per-tenant limit the
platform enforces (request rate, burst, concurrent agent invocations,
daily model-token budget); a row can override individual limits for one
tenant (an enterprise contract, a temporary raise during a launch).

Suspending a tenant (`status = "suspended"`) blocks its API traffic at the
edge within one cache TTL.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TenantRow(Base):
    __tablename__ = "tenants"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), default="")
    tier: Mapped[str] = mapped_column(String(32), default="standard")
    status: Mapped[str] = mapped_column(String(16), default="active")  # active | suspended
    limits_override: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


@dataclass(frozen=True)
class TenantLimits:
    tier: str
    requests_per_second: float
    burst: int
    concurrent_invocations: int
    invocations_per_minute: int
    daily_token_budget: int  # model tokens (in + out) per UTC day; 0 = unlimited
    max_queue_depth: int
    status: str = "active"

    def to_dict(self) -> dict:
        return asdict(self)


TIERS: dict[str, TenantLimits] = {
    "free": TenantLimits("free", 5, 10, 2, 20, 200_000, 20),
    "standard": TenantLimits("standard", 50, 100, 20, 300, 5_000_000, 200),
    "enterprise": TenantLimits("enterprise", 2_000, 4_000, 1_000, 20_000, 0, 10_000),
    # Internal/test traffic: effectively unlimited.
    "internal": TenantLimits("internal", 100_000, 200_000, 100_000, 1_000_000, 0, 1_000_000),
}
