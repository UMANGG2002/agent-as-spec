"""The ADK session service shared by API and runners: ADK's
`DatabaseSessionService` on Postgres (SQLite locally), optionally on its own
database (`SESSION_DATABASE_URL`) since event history is the most
write-heavy data in the system.

`BoundedSessionService` caps how many events a turn loads
(`session_max_events_loaded`): a months-long conversation would otherwise
load its entire history on every message. Context compaction keeps the
summary of older turns inside that recent window.
"""

from __future__ import annotations

from typing import Any

from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.database_session_service import DatabaseSessionService

from app.config import settings

_service: Any = None


class BoundedSessionService(DatabaseSessionService):
    async def get_session(self, *, app_name: str, user_id: str, session_id: str,
                          config: GetSessionConfig | None = None):
        if config is None:
            config = GetSessionConfig(num_recent_events=settings.session_max_events_loaded)
        return await super().get_session(app_name=app_name, user_id=user_id, session_id=session_id, config=config)


def _engine_kwargs(url: str) -> dict:
    if not url.startswith("postgresql"):
        return {}
    return {"pool_size": settings.db_pool_size, "max_overflow": settings.db_max_overflow, "pool_pre_ping": True,
            "pool_recycle": settings.db_pool_recycle_seconds}


def get_session_service():
    global _service
    if _service is None:
        url = settings.session_database_url or settings.database_url
        _service = BoundedSessionService(db_url=url, **_engine_kwargs(url))
    return _service


def adk_user_id(tenant_id: str, user_id: str) -> str:
    """Tenant-qualified user id: ADK scopes sessions and memory by user_id,
    so qualifying it isolates tenants even if two use the same user name."""
    return f"{tenant_id}:{user_id}"


def reset_for_tests() -> None:
    global _service
    _service = None
