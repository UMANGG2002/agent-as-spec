"""Human-in-the-loop approvals. One pathway for every pause, per the
architecture doc: a `human_approval` plan step, a tenant granting a
marketplace request, or a guardrail escalating a destructive tool call all
resolve through this same `request_approval` / `resolve_approval` pair.

Mirrors ADK's `LongRunningFunctionTool` pattern (emit a call, suspend, resume
on an external event) without requiring a live ADK agent tree for the
deterministic executor path: `request_approval` genuinely suspends the
awaiting coroutine via an `asyncio.Future` rather than polling, and
`resolve_approval` is what an approval API endpoint calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    TIMED_OUT = "timed_out"


@dataclass
class ApprovalRequest:
    run_id: str
    step_id: str
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    resolved_by: str | None = None
    note: str | None = None
    _future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future(), repr=False)


class ApprovalStore:
    def __init__(self) -> None:
        self._pending: dict[tuple[str, str], ApprovalRequest] = {}

    def key(self, run_id: str, step_id: str) -> tuple[str, str]:
        return (run_id, step_id)

    async def request_approval(self, run_id: str, step_id: str, reason: str, *, timeout_seconds: float = 3600) -> ApprovalStatus:
        request = ApprovalRequest(run_id=run_id, step_id=step_id, reason=reason)
        self._pending[self.key(run_id, step_id)] = request
        try:
            await asyncio.wait_for(request._future, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            request.status = ApprovalStatus.TIMED_OUT
        return request.status

    def resolve_approval(self, run_id: str, step_id: str, *, approved: bool, resolved_by: str, note: str | None = None) -> ApprovalRequest:
        request = self._pending.get(self.key(run_id, step_id))
        if request is None:
            raise KeyError(f"no pending approval for run={run_id} step={step_id}")
        request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        request.resolved_by = resolved_by
        request.note = note
        if not request._future.done():
            request._future.set_result(request.status)
        return request

    def list_pending(self, run_id: str | None = None) -> list[ApprovalRequest]:
        values = self._pending.values()
        if run_id:
            values = [r for r in values if r.run_id == run_id]
        return [r for r in values if r.status == ApprovalStatus.PENDING]


approval_store = ApprovalStore()
