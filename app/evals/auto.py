"""Auto-eval: evaluation that runs without anyone remembering to.

  1. On spec approval (`EVAL_ON_APPROVAL=true`): when a spec version is
     approved, run (a) every eval set of the tenant that has a case for this
     spec, and (b) a freshly generated synthetic suite for it, each compared
     with its baseline. The result is recorded (eval run + audit entry +
     metrics) and a failed gate is logged at WARNING and visible at
     GET /v1/evals/specs/{spec_id}/status. Approval is NOT blocked: the spec
     is already saved by then, so this is a tripwire to act on (roll back,
     fix the plan), not a pre-merge gate. The pre-merge gate is
     `scripts/eval.py --gate` in CI.
  2. On live traffic (`EVAL_ONLINE_SAMPLE_RATE`): see `online.py`.

Needs a model: the live one when configured; tests inject `model_override`.
Without either, auto-eval skips rather than reporting meaningless failures.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import settings
from app.evals import generator, judges, service, store
from app.evals.schemas import EvalRunReport
from app.specs.schemas import AgentSpec

logger = logging.getLogger(__name__)

model_override: Any = None  # tests; None = settings.live_model when configured
judge_override: Any = None
_tasks: set[asyncio.Task] = set()


def _model() -> Any:
    if model_override is not None:
        return model_override
    return settings.live_model if settings.planner_is_live else None


async def on_spec_approved(spec: AgentSpec) -> list[EvalRunReport]:
    model = _model()
    if model is None:
        logger.info("auto-eval skipped for %s: no model configured", spec.spec_id)
        return []
    tenant = spec.tenant_id
    identity = service.eval_identity(tenant)
    judge = judge_override or judges.get_judge()
    reports: list[EvalRunReport] = []

    sets = []
    auto_id = f"auto_{spec.spec_id}"
    for row in await store.list_sets(tenant):
        if row["set_id"] == auto_id:  # regenerated below for the new version
            continue
        s = await store.get_set(tenant, row["set_id"])
        if s and any(c.target.spec_id == spec.spec_id or c.target.kind == "router" for c in s.cases):
            sets.append(s)
    synthetic = generator.generate_set(tenant, [spec], set_id=auto_id, include_router=False)
    await store.save_set(synthetic)
    sets.append(synthetic)

    for s in sets:
        report = await service.run_set(s, identity, model=model, judge=judge, trigger="spec_approval")
        reports.append(report)
        await _record(spec, report)
    return reports


async def _record(spec: AgentSpec, report: EvalRunReport) -> None:
    from app.core import audit
    from app.db import session_scope
    from app.telemetry.setup import eval_pass_rate, eval_runs_total

    gate_ok = report.regression.ok if report.regression else True
    eval_runs_total.add(1, {"trigger": report.trigger, "gate": "pass" if gate_ok else "fail"})
    eval_pass_rate.record(report.summary.pass_rate, {"trigger": report.trigger})
    async with session_scope(tenant_id=spec.tenant_id) as db:
        audit.record(db, tenant_id=spec.tenant_id, actor="eval", action="eval.run", resource_type="agent_spec",
                     resource_id=spec.spec_id,
                     details={"run_id": report.run_id, "set_id": report.set_id, "version": spec.version,
                              "pass_rate": report.summary.pass_rate, "gate_ok": gate_ok})
    if not gate_ok:
        logger.warning("eval gate FAILED for spec %s v%s (set %s): %s", spec.spec_id, spec.version, report.set_id,
                       "; ".join(report.regression.reasons) if report.regression else "")


def schedule_on_spec_approved(spec: AgentSpec) -> asyncio.Task | None:
    """Fire and forget: approval never waits on, or fails because of, an eval."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    async def _run() -> None:
        try:
            await on_spec_approved(spec)
        except Exception:  # noqa: BLE001
            logger.exception("auto-eval for spec %s crashed", spec.spec_id)

    task = loop.create_task(_run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return task


async def spec_status(tenant_id: str, spec_id: str) -> dict:
    """Latest auto-eval result for a spec: run, gate, and which sets ran."""
    out = []
    for row in await store.list_runs(tenant_id, limit=200):
        if row["trigger"] == "spec_approval":
            run = await store.get_run(tenant_id, row["run_id"], with_results=False)
            if run and (spec_id in run.spec_versions or run.set_id == f"auto_{spec_id}"):
                out.append({"run_id": run.run_id, "set_id": run.set_id, "spec_versions": run.spec_versions,
                            "pass_rate": run.summary.pass_rate, "gate_ok": run.regression.ok if run.regression else None,
                            "reasons": run.regression.reasons if run.regression else [], "created_at": row["created_at"]})
    return {"spec_id": spec_id, "runs": out[:10], "latest_gate_ok": out[0]["gate_ok"] if out else None}
