"""Golden datasets: the reviewed cases a platform change must keep passing.

  - stored as an `EvalSet` (DB, per tenant) and exchanged as JSONL (one case
    per line) so they live in git next to the code and are reviewed in PRs;
  - `promote_session` turns a real production conversation into a draft
    golden case (PII redacted, expectations snapshotted from what the agent
    actually did) -- a person reviews and tightens it before it gates anything.
"""

from __future__ import annotations

import json
import re

from app.evals import store
from app.evals.runner import _absorb, _derive_outcome
from app.evals.schemas import EvalCase, EvalSet, ExpectedToolCall, Expectations, Target, ToolCallRecord, Trace

_REDACTIONS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"(?i)\b(bearer|api[_-]?key|token|secret|password)\b\s*[:=]?\s*[\w\-.~+/]{8,}"), r"\1 [redacted]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[card]"),
    (re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d"), "[phone]"),
]


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


# -- JSONL -----------------------------------------------------------------------------


def dump_jsonl(eval_set: EvalSet) -> str:
    return "\n".join(c.model_dump_json(exclude_defaults=True) for c in eval_set.cases) + "\n"


def load_jsonl(text: str, *, tenant_id: str, set_id: str, name: str | None = None, kind: str = "golden") -> EvalSet:
    cases: list[EvalCase] = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(EvalCase.model_validate(json.loads(line)))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"line {n}: {exc}") from exc
    ids = [c.case_id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate case_id(s): {sorted(dupes)}")
    return EvalSet(set_id=set_id, tenant_id=tenant_id, name=name or set_id, kind=kind, cases=cases)  # type: ignore[arg-type]


async def add_cases(tenant_id: str, set_id: str, cases: list[EvalCase], *, name: str | None = None) -> EvalSet:
    """Upsert cases (by case_id) into a set, creating the set if needed."""
    existing = await store.get_set(tenant_id, set_id) or EvalSet(set_id=set_id, tenant_id=tenant_id,
                                                                  name=name or set_id)
    by_id = {c.case_id: c for c in existing.cases}
    for c in cases:
        by_id[c.case_id] = c
    existing.cases = list(by_id.values())
    return await store.save_set(existing)


# -- promote a real conversation -------------------------------------------------------


async def promote_session(tenant_id: str, session_id: str, *, case_id: str | None = None,
                          set_id: str | None = None) -> EvalCase:
    """Draft a golden case from a stored chat session: user messages (redacted),
    the tool path and outcome the agent took, and its final answer as the
    reference. Needs human review: the snapshot encodes what happened, not
    necessarily what should have."""
    from sqlalchemy import select

    from app.db import session_scope
    from app.sessions.models import ChatSessionRow
    from app.sessions.runner import serialize_event
    from app.sessions.session_store import get_session_service

    async with session_scope(tenant_id=tenant_id) as db:
        row = (await db.execute(select(ChatSessionRow).where(
            ChatSessionRow.session_id == session_id, ChatSessionRow.tenant_id == tenant_id))).scalars().first()
    if row is None:
        raise LookupError("session not found")
    session = await get_session_service().get_session(app_name=row.app_name, user_id=f"{tenant_id}:{row.user_id}",
                                                      session_id=session_id)
    if session is None:
        raise LookupError("session history not found")

    messages: list[str] = []
    trace = Trace()
    by_id: dict[str, ToolCallRecord] = {}
    confirmation = "approve"
    awaiting = False
    for event in session.events:
        if event.author == "user":
            for part in (event.content.parts if event.content else []) or []:
                if part.text:
                    messages.append(redact(part.text))
                fr = part.function_response
                if fr is not None and fr.name == "adk_request_confirmation":
                    confirmation = "approve" if (fr.response or {}).get("confirmed") else "reject"
            continue
        payload = serialize_event(event)
        if "confirmation_request" in payload:
            awaiting = True
        elif payload.get("function_responses") or payload.get("text"):
            awaiting = awaiting and not payload.get("final")
        _absorb(trace, payload, by_id)
    trace.outcome = _derive_outcome(trace, awaiting)  # type: ignore[assignment]

    target = Target(kind="router") if row.spec_id == "spec_test_agent" else Target(
        kind="spec", spec_id=row.spec_id, spec_version=row.spec_version)
    exp = Expectations(
        tool_calls=[ExpectedToolCall(name=c.step_id or c.name, status=c.status if c.status in
                                     ("approved", "denied_by_policy", "denied", "rejected") else None)
                    for c in trace.tool_calls],
        outcome=trace.outcome if trace.outcome != "error" else "any", transfers=list(trace.transfers),
        reference_answer=redact(trace.final_text) or None)
    case = EvalCase(case_id=case_id or f"promoted_{session_id[-10:]}", name=f"promoted from {session_id}",
                    target=target, messages=messages, confirmation=confirmation,  # type: ignore[arg-type]
                    expectations=exp, rubrics=["task_completion", "groundedness"], tags=["promoted"],
                    source="promoted", notes="Draft: review the expectations before relying on this case.")
    if set_id:
        await add_cases(tenant_id, set_id, [case])
    return case
