"""Lifecycle notifications via Composio -- architecture doc's "Notifications"
section. Deliberately out of the execution hot path: callers just call
`dispatch(...)`, which looks up the tenant's preference and fires a Composio
action; a Composio outage degrades notifications, never execution (the
caller never awaits anything that can block a run).

Composio's Python SDK surface is evolving quickly, so the actual
`composio` import is isolated to `_send_via_composio` and guarded: without
`COMPOSIO_API_KEY` configured (or the package installed), every event is
logged instead of sent, which keeps the rest of the system runnable and
testable without a live Composio account.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.marketplace.models import NotificationPreferenceRow

logger = logging.getLogger(__name__)

EventType = str  # "run_completed" | "run_failed" | "approval_needed" | "grant_requested" | "grant_approved"


class NotificationDispatcher:
    async def dispatch(self, session: AsyncSession, *, tenant_id: str, event_type: EventType, payload: dict[str, Any]) -> None:
        stmt = select(NotificationPreferenceRow).where(
            NotificationPreferenceRow.tenant_id == tenant_id, NotificationPreferenceRow.event_type == event_type
        )
        prefs = (await session.execute(stmt)).scalars().all()
        if not prefs:
            logger.info("no notification preference for tenant=%s event=%s; skipping", tenant_id, event_type)
            return
        for pref in prefs:
            await self._send(tenant_id=tenant_id, channel=pref.channel, event_type=event_type, payload=payload)

    async def _send(self, *, tenant_id: str, channel: str, event_type: str, payload: dict[str, Any]) -> None:
        if not settings.composio_api_key:
            logger.info("[notify:dry-run] tenant=%s channel=%s event=%s payload=%s", tenant_id, channel, event_type, payload)
            return
        try:
            await self._send_via_composio(tenant_id=tenant_id, channel=channel, event_type=event_type, payload=payload)
        except Exception as exc:  # noqa: BLE001 - never let a notification failure propagate into the run
            logger.warning("composio dispatch failed (tenant=%s, event=%s): %s", tenant_id, event_type, exc)

    async def _send_via_composio(self, *, tenant_id: str, channel: str, event_type: str, payload: dict[str, Any]) -> None:
        from composio import Composio  # deferred: only imported when actually sending

        client = Composio(api_key=settings.composio_api_key)
        session = client.create(user_id=tenant_id)
        message = _format_message(event_type, payload)
        # `channel` is the tenant's own connected-account handle (BYOK --
        # stored the same way any other credential is, see auth/vault.py).
        # The exact tool name depends on which channel the tenant connected
        # (Slack/email/SMS/...); this is the one seam to fill in against your
        # Composio project's actual toolkit names.
        tools = session.tools(toolkits=[channel])
        logger.info("composio tools resolved for channel=%s: %s", channel, [t for t in tools])
        # session.execute(...) / the resolved tool call goes here once the
        # tenant's specific toolkit + action name is known.
        _ = message


def _format_message(event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "run_completed":
        return f"Run {payload.get('run_id')} completed."
    if event_type == "run_failed":
        return f"Run {payload.get('run_id')} failed: {payload.get('error')}"
    if event_type == "approval_needed":
        return f"Run {payload.get('run_id')} step {payload.get('step_id')} needs your approval."
    if event_type == "grant_requested":
        return f"{payload.get('grantee_tenant_id')} requested access to capability {payload.get('capability_id')}."
    return f"{event_type}: {payload}"


dispatcher = NotificationDispatcher()
