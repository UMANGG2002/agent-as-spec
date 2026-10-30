"""Evaluate the planner (intent -> `ExecutionPlan`) on its own.

Spec agents are only as good as the plan they were approved from, so the
planner gets its own cases. All checks are deterministic and use the same
validators approval uses, so "valid here" means "approvable":

  - non_empty        a plan with at least one step
  - capabilities     every bound capability exists and is visible to the tenant
                     (a hallucinated or cross-tenant id fails)
  - schema_valid     each step's tool exists and its arguments satisfy the
                     tool's input schema (the approval-time check)
  - dag_valid        dependencies resolve and are acyclic
  - recall           every capability in `must_use` is in the plan
  - forbidden        nothing in `must_not_use` is in the plan
  - step_budget      no more than `max_steps` steps (fewest-steps preference)

`generate_planner_cases` builds one retrieval case per registered capability
from its own description, so the planner's capability discovery is checked
against the live registry.
"""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field

from app.auth.models import IdentityContext
from app.evals.schemas import CaseResult, EvalRunReport, MetricResult, Trace
from app.evals.service import summarise
from app.planner.schemas import ExecutionPlan, StepKind


class PlannerCase(BaseModel):
    case_id: str
    intent: str
    must_use: list[str] = Field(default_factory=list, description="capability ids or names that must appear")
    must_not_use: list[str] = Field(default_factory=list)
    max_steps: int | None = None
    tags: list[str] = Field(default_factory=list)


def _m(name: str, ok: bool, detail: str, score: float | None = None) -> MetricResult:
    return MetricResult(name=name, score=(1.0 if ok else 0.0) if score is None else score, passed=ok, detail=detail)


def check_plan(case: PlannerCase, plan: ExecutionPlan, visible: dict) -> list[MetricResult]:
    from app.factory import agent_patterns
    from app.specs import service as specs_service
    from app.specs.schemas import AgentRuntimeConfig

    out = [_m("non_empty", bool(plan.steps), f"{len(plan.steps)} step(s)")]
    if not plan.steps:
        return out
    bound = [s.capability_id for s in plan.steps if s.kind != StepKind.HUMAN_APPROVAL and s.capability_id]
    unknown = [c for c in bound if c not in visible]
    out.append(_m("capabilities", not unknown, f"unknown or invisible: {unknown}" if unknown else "all capabilities exist"))

    snapshots = {c: visible[c] for c in bound if c in visible}
    try:
        specs_service._validate_step_calls(plan, snapshots)
        out.append(_m("schema_valid", True, "tools and arguments valid"))
    except Exception as exc:  # noqa: BLE001
        out.append(_m("schema_valid", False, str(exc)[:300]))
    try:
        plan.step_order()
        agent_patterns.validate_runtime_config(plan.steps, AgentRuntimeConfig())
        out.append(_m("dag_valid", True, "dependencies resolve"))
    except Exception as exc:  # noqa: BLE001
        out.append(_m("dag_valid", False, str(exc)[:300]))

    names = {c: visible[c].name for c in bound if c in visible}

    def used(ref: str) -> bool:
        return any(ref == cid or ref.lower() == names.get(cid, "").lower() for cid in bound)

    if case.must_use:
        hit = [r for r in case.must_use if used(r)]
        out.append(_m("recall", len(hit) == len(case.must_use), f"{len(hit)}/{len(case.must_use)} required capabilities used",
                      score=len(hit) / len(case.must_use)))
    if case.must_not_use:
        bad = [r for r in case.must_not_use if used(r)]
        out.append(_m("forbidden", not bad, f"used forbidden: {bad}" if bad else "none used"))
    if case.max_steps is not None:
        out.append(_m("step_budget", len(plan.steps) <= case.max_steps, f"{len(plan.steps)} steps (max {case.max_steps})"))
    return out


async def run_planner_cases(identity: IdentityContext, cases: list[PlannerCase], *, set_id: str = "planner",
                            trigger: str = "manual") -> EvalRunReport:
    from app.db import session_scope
    from app.planner.planner_agent import create_plan
    from app.registry import store as registry_store

    visible = await registry_store.visible_capabilities(identity.tenant_id)
    results: list[CaseResult] = []
    for case in cases:
        trace = Trace()
        started = time.monotonic()
        try:
            async with session_scope(tenant_id=identity.tenant_id) as db:
                plan = await create_plan(db, identity, case.intent)
            trace.final_text = f"{plan.goal} -> " + ", ".join(f"{s.step_id}:{s.capability_id}" for s in plan.steps)
            ms = check_plan(case, plan, visible)
        except Exception as exc:  # noqa: BLE001
            trace.error = f"{type(exc).__name__}: {exc}"[:300]
            ms = [_m("no_error", False, trace.error)]
        trace.latency_ms = int((time.monotonic() - started) * 1000)
        results.append(CaseResult(
            case_id=case.case_id, passed=all(m.passed for m in ms), score=round(sum(m.score for m in ms) / len(ms), 4),
            hard_score=round(sum(m.score for m in ms) / len(ms), 4), metrics=ms, trace=trace, tags=case.tags,
            error=trace.error))
    return EvalRunReport(run_id=f"evr_{uuid.uuid4().hex[:12]}", set_id=set_id, tenant_id=identity.tenant_id,
                         trigger=trigger, model="planner", judge="none", summary=summarise(results), results=results)


def generate_planner_cases(visible: dict, *, tenant_id: str, limit: int = 50) -> list[PlannerCase]:
    """One retrieval case per capability the tenant owns, phrased from its own
    description: the planner should find and bind it."""
    out = []
    for cap in sorted(visible.values(), key=lambda c: c.id):
        if cap.tenant_id != tenant_id or not cap.tools:
            continue
        what = (cap.description or cap.name).rstrip(".")
        out.append(PlannerCase(case_id=f"plan_{cap.id}", intent=f"I need to: {what}", must_use=[cap.id], max_steps=4,
                               tags=["synthetic", "retrieval"]))
        if len(out) >= limit:
            break
    return out
