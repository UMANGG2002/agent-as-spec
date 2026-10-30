"""Orchestration: run an eval set, score, summarise, compare, persist."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from collections import defaultdict
from typing import Any

from app.auth.models import IdentityContext
from app.config import settings
from app.evals import judges, metrics, regression, runner, store
from app.evals.schemas import CaseResult, EvalCase, EvalRunReport, EvalSet, RunSummary, Trace


EVAL_USER = "eval-bot"


def eval_identity(tenant_id: str, user_id: str = EVAL_USER, scopes: tuple[str, ...] = ("*",)) -> IdentityContext:
    """Evals always run as their own synthetic user (never the caller's), so the
    approval rows they create are attributable and can be cleaned up without
    touching a real person's pending approvals. `scopes` bounds what the
    agents can do: callers pass their own, so an eval never exceeds its caller."""
    return IdentityContext(tenant_id=tenant_id, brand_id="default", user_id=user_id, scopes=scopes,
                           act_chain=("eval",))


async def _cleanup_approvals(identity: IdentityContext, since: datetime) -> None:
    """Delete the approval rows this run's simulated users created (pending
    pauses, consumed approvals), so eval traffic never lingers in the inbox."""
    from sqlalchemy import delete

    from app.db import session_scope
    from app.runtime.approvals import ApprovalRow

    async with session_scope(tenant_id=identity.tenant_id) as db:
        await db.execute(delete(ApprovalRow).where(
            ApprovalRow.tenant_id == identity.tenant_id, ApprovalRow.requested_by == identity.user_id,
            ApprovalRow.requested_at >= since))


def score_case(case: EvalCase, trace: Trace, judge_results: list) -> CaseResult:
    hard = metrics.run_checks(case, trace)
    all_metrics = hard + judge_results
    hard_score = sum(m.score for m in hard) / len(hard) if hard else 1.0
    judge_score = (sum(m.score for m in judge_results) / len(judge_results)) if judge_results else None
    mean = sum(m.score for m in all_metrics) / len(all_metrics) if all_metrics else 1.0
    return CaseResult(case_id=case.case_id, passed=all(m.passed for m in hard), score=round(mean, 4),
                      hard_score=round(hard_score, 4),
                      judge_score=round(judge_score, 4) if judge_score is not None else None,
                      metrics=all_metrics, trace=trace, tags=case.tags, error=trace.error)


async def eval_case(case: EvalCase, identity: IdentityContext, *, model: Any = None, judge: judges.Judge | None = None,
                    capabilities: dict | None = None) -> CaseResult:
    trace = await runner.run_case(case, identity, model=model, capabilities=capabilities)
    jr = await judges.run_judges(judge, case, trace) if (judge is not None and case.rubrics) else []
    return score_case(case, trace, jr)


def summarise(results: list[CaseResult]) -> RunSummary:
    n = len(results)
    if not n:
        return RunSummary()
    by_metric: dict[str, list[float]] = defaultdict(list)
    by_tag: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for m in r.metrics:
            by_metric[m.name].append(m.score)
        for t in r.tags:
            by_tag[t].append(1.0 if r.passed else 0.0)
    js = [r.judge_score for r in results if r.judge_score is not None]
    return RunSummary(
        n_cases=n, n_passed=sum(r.passed for r in results), pass_rate=round(sum(r.passed for r in results) / n, 4),
        mean_score=round(sum(r.score for r in results) / n, 4),
        mean_judge_score=round(sum(js) / len(js), 4) if js else None,
        mean_tokens=round(sum(r.trace.tokens_in + r.trace.tokens_out for r in results) / n, 1),
        mean_latency_ms=round(sum(r.trace.latency_ms for r in results) / n, 1),
        by_metric={k: round(sum(v) / len(v), 4) for k, v in sorted(by_metric.items())},
        by_tag={k: round(sum(v) / len(v), 4) for k, v in sorted(by_tag.items())})


async def run_set(eval_set: EvalSet, identity: IdentityContext, *, model: Any = None,
                  judge: judges.Judge | None = None, trigger: str = "manual", persist: bool = True,
                  compare_to_baseline: bool = True, concurrency: int | None = None,
                  tags: list[str] | None = None) -> EvalRunReport:
    """Run every case (bounded concurrency), score, summarise, and compare with
    the set's baseline run. `model` is the AGENT model (scripted in CI, live
    otherwise); `judge` scores the rubrics (None skips them)."""
    from app.registry import store as registry_store
    from app.specs import store as specs_store

    cases = [c for c in eval_set.cases if not tags or set(tags) & set(c.tags)]
    capabilities = await registry_store.visible_capabilities(identity.tenant_id)
    sem = asyncio.Semaphore(concurrency or settings.eval_concurrency)
    started_at = datetime.now(timezone.utc)

    async def one(case: EvalCase) -> CaseResult:
        async with sem:
            return await eval_case(case, identity, model=model, judge=judge, capabilities=capabilities)

    # The approval gate matches approvals by the exact call (tenant, spec, step,
    # args), so two cases of one spec running at once could consume each
    # other's approval. Cases of the same spec therefore run one after another;
    # different specs run in parallel. Router cases can reach any spec, so
    # they run after every spec group has finished.
    groups: dict[str, list[tuple[int, EvalCase]]] = defaultdict(list)
    for i, c in enumerate(cases):
        groups["__router__" if c.target.kind == "router" else (c.target.spec_id or "")].append((i, c))
    done: dict[int, CaseResult] = {}

    async def run_group(items: list[tuple[int, EvalCase]]) -> None:
        for i, c in items:
            done[i] = await one(c)

    await asyncio.gather(*(run_group(v) for k, v in groups.items() if k != "__router__"))
    await run_group(groups.get("__router__", []))
    results = [done[i] for i in range(len(cases))]
    await _cleanup_approvals(identity, started_at)
    versions: dict[str, int] = {}
    for sid in {c.target.spec_id for c in cases if c.target.kind == "spec" and c.target.spec_id}:
        spec = await specs_store.get_spec(sid, tenant_id=identity.tenant_id)
        if spec:
            versions[sid] = spec.version
    report = EvalRunReport(
        run_id=f"evr_{uuid.uuid4().hex[:12]}", set_id=eval_set.set_id, tenant_id=identity.tenant_id, trigger=trigger,
        model=str(getattr(model, "model", model) or "live"), judge=getattr(judge, "name", "none") if judge else "none",
        spec_versions=versions, summary=summarise(results), results=results)
    if compare_to_baseline:
        baseline = await store.get_baseline(identity.tenant_id, eval_set.set_id)
        report.regression = regression.compare(report, baseline)
    if persist:
        await store.save_run(report)
    return report


def format_report(report: EvalRunReport) -> str:
    s = report.summary
    lines = [f"run {report.run_id}  set {report.set_id}  trigger {report.trigger}  model {report.model}  judge {report.judge}",
             f"  cases {s.n_passed}/{s.n_cases} passed ({s.pass_rate:.0%})  mean score {s.mean_score:.3f}"
             + (f"  judge {s.mean_judge_score:.3f}" if s.mean_judge_score is not None else "")
             + f"  tokens/case {s.mean_tokens:.0f}  latency/case {s.mean_latency_ms:.0f}ms"]
    for name, val in s.by_metric.items():
        lines.append(f"    {name:<28} {val:.3f}")
    for r in report.results:
        if not r.passed:
            bad = "; ".join(f"{m.name}: {m.detail}" for m in r.metrics if m.hard and not m.passed)
            lines.append(f"  FAIL {r.case_id}: {bad}")
    if report.regression:
        g = report.regression
        lines.append(f"  gate: {'PASS' if g.ok else 'FAIL'}" + (f" (baseline {g.baseline_run_id})" if g.baseline_run_id else " (no baseline)"))
        lines += [f"    - {x}" for x in g.reasons]
        if g.flipped_to_pass:
            lines.append(f"    fixed since baseline: {sorted(g.flipped_to_pass)}")
    return "\n".join(lines)
