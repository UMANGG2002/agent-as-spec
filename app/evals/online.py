"""Online evaluation: score a sample of REAL finished invocations.

Offline eval says "the agent passes the cases we thought of"; this says how
the live traffic is doing, and surfaces the weakest real conversations as
candidates to promote into the golden set (`golden.promote_session`).

Sampling is by `EVAL_ONLINE_SAMPLE_RATE` (0 = off). The trace is captured
from the events the runner already streams, so scoring costs no extra
agent run -- only the judge calls, in a background task that can never delay
or fail the user's invocation. What is stored is redacted: message and answer
text with PII removed, tool names/status/ok only (no arguments, no outputs).
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from typing import Any

from app.config import settings
from app.evals import golden, judges
from app.evals.schemas import EvalCase, MetricResult, Trace
from app.evals.store import EvalSampleRow, list_samples, save_sample

logger = logging.getLogger(__name__)
judge_override: Any = None
_tasks: set[asyncio.Task] = set()
ONLINE_RUBRICS = ["task_completion", "groundedness", "policy_compliance"]


def should_sample() -> bool:
    rate = settings.eval_online_sample_rate
    return rate > 0 and random.random() < rate


def generic_checks(trace: Trace, status: str) -> list[MetricResult]:
    failed_tools = [c for c in trace.tool_calls if c.ok is False and c.status not in
                    ("awaiting_approval", "denied_by_policy", "denied", "rejected")]
    return [
        MetricResult(name="run_ok", score=1.0 if status in ("succeeded", "awaiting_input") else 0.0,
                     passed=status in ("succeeded", "awaiting_input"), detail=f"status {status}"),
        MetricResult(name="has_answer", score=1.0 if trace.final_text.strip() or status == "awaiting_input" else 0.0,
                     passed=bool(trace.final_text.strip()) or status == "awaiting_input", detail="final answer present"),
        MetricResult(name="tools_ok", score=1.0 if not failed_tools else 0.0, passed=not failed_tools,
                     detail=f"tool errors: {[c.name for c in failed_tools]}" if failed_tools else "no tool errors"),
    ]


async def score_invocation(job: dict, trace: Trace, status: str) -> EvalSampleRow:
    case = EvalCase(case_id=job["invocation_id"], messages=[job.get("text") or ""], rubrics=ONLINE_RUBRICS,
                    source="online")
    judge = judge_override or judges.get_judge()
    metrics = generic_checks(trace, status) + await judges.run_judges(judge, case, trace)
    score = sum(m.score for m in metrics) / len(metrics)
    from app.telemetry.setup import eval_online_score

    eval_online_score.record(score, {"spec_id": job["spec_id"]})
    stored_trace = trace.model_copy(update={
        "final_text": golden.redact(trace.final_text)[:2000],
        "tool_calls": [c.model_copy(update={"args": {}, "output": None, "error": None}) for c in trace.tool_calls]})
    row = EvalSampleRow(
        sample_id=f"evs_{uuid.uuid4().hex[:12]}", tenant_id=job["tenant_id"], spec_id=job["spec_id"],
        session_id=job["session_id"], invocation_id=job["invocation_id"], score=round(score, 4),
        data={"judge": getattr(judge, "name", "?"), "status": status,
              "message": golden.redact(job.get("text") or "")[:1000],
              "metrics": [m.model_dump(mode="json") for m in metrics], "trace": stored_trace.model_dump(mode="json")})
    await save_sample(row)
    return row


def schedule_score(job: dict, trace: Trace, status: str) -> None:
    async def _run() -> None:
        try:
            await score_invocation(job, trace, status)
        except Exception:  # noqa: BLE001
            logger.exception("online eval of %s failed", job.get("invocation_id"))

    task = asyncio.get_running_loop().create_task(_run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def summary(tenant_id: str, spec_id: str | None = None, *, limit: int = 200, worst: int = 5) -> dict:
    rows = await list_samples(tenant_id, spec_id, limit)
    scored = [r for r in rows if r.score is not None]
    return {
        "n": len(rows), "mean_score": round(sum(r.score for r in scored) / len(scored), 4) if scored else None,
        "judge": rows[0].data.get("judge") if rows else None,
        "worst": [{"sample_id": r.sample_id, "session_id": r.session_id, "invocation_id": r.invocation_id,
                   "spec_id": r.spec_id, "score": r.score, "message": r.data.get("message"),
                   "failed": [m["name"] for m in r.data.get("metrics", []) if not m["passed"]]}
                  for r in sorted(scored, key=lambda r: r.score)[:worst]]}
