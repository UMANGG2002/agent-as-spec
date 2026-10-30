"""Persistence for eval sets, runs and per-case results (tenant-scoped, RLS)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base, session_scope
from app.evals.schemas import CaseResult, EvalRunReport, EvalSet, Regression, RunSummary

_JSON = JSON().with_variant(JSONB(), "postgresql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class EvalSetRow(Base):
    __tablename__ = "eval_sets"

    set_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(16), default="golden")
    version: Mapped[int] = mapped_column(Integer, default=1)
    data: Mapped[dict] = mapped_column(_JSON)  # the full EvalSet (cases included)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class EvalRunRow(Base):
    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    set_id: Mapped[str] = mapped_column(String(64), index=True)
    trigger: Mapped[str] = mapped_column(String(24), default="manual")
    model: Mapped[str] = mapped_column(String(200), default="")
    judge: Mapped[str] = mapped_column(String(200), default="")
    status: Mapped[str] = mapped_column(String(16), default="completed")
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    pass_rate: Mapped[float] = mapped_column(Float, default=0.0)
    mean_score: Mapped[float] = mapped_column(Float, default=0.0)
    spec_versions: Mapped[dict] = mapped_column(_JSON, default=dict)
    summary: Mapped[dict] = mapped_column(_JSON, default=dict)
    regression: Mapped[dict | None] = mapped_column(_JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class EvalCaseResultRow(Base):
    __tablename__ = "eval_case_results"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    data: Mapped[dict] = mapped_column(_JSON)  # the full CaseResult (trace + metrics)


class EvalSampleRow(Base):
    """A real production invocation scored after the fact (online eval)."""

    __tablename__ = "eval_samples"

    sample_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    spec_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64))
    invocation_id: Mapped[str] = mapped_column(String(64))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    data: Mapped[dict] = mapped_column(_JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


# -- sets ------------------------------------------------------------------------------


async def save_set(eval_set: EvalSet) -> EvalSet:
    async with session_scope(tenant_id=eval_set.tenant_id) as db:
        row = await db.get(EvalSetRow, eval_set.set_id)
        if row is None:
            db.add(EvalSetRow(set_id=eval_set.set_id, tenant_id=eval_set.tenant_id, name=eval_set.name,
                              kind=eval_set.kind, version=eval_set.version, data=eval_set.model_dump(mode="json")))
        else:
            eval_set.version = row.version + 1
            row.name, row.kind, row.version = eval_set.name, eval_set.kind, eval_set.version
            row.data, row.updated_at = eval_set.model_dump(mode="json"), _now()
    return eval_set


async def get_set(tenant_id: str, set_id: str) -> EvalSet | None:
    async with session_scope(tenant_id=tenant_id) as db:
        row = await db.get(EvalSetRow, set_id)
    if row is None or row.tenant_id != tenant_id:
        return None
    return EvalSet.model_validate(row.data)


async def list_sets(tenant_id: str) -> list[dict]:
    async with session_scope(tenant_id=tenant_id) as db:
        rows = (await db.execute(select(EvalSetRow).where(EvalSetRow.tenant_id == tenant_id)
                                 .order_by(EvalSetRow.updated_at.desc()))).scalars().all()
    return [{"set_id": r.set_id, "name": r.name, "kind": r.kind, "version": r.version,
             "n_cases": len((r.data or {}).get("cases", []))} for r in rows]


async def delete_set(tenant_id: str, set_id: str) -> bool:
    async with session_scope(tenant_id=tenant_id) as db:
        row = await db.get(EvalSetRow, set_id)
        if row is None or row.tenant_id != tenant_id:
            return False
        await db.delete(row)
    return True


# -- runs ------------------------------------------------------------------------------


async def save_run(report: EvalRunReport) -> None:
    async with session_scope(tenant_id=report.tenant_id) as db:
        db.add(EvalRunRow(
            run_id=report.run_id, tenant_id=report.tenant_id, set_id=report.set_id, trigger=report.trigger,
            model=report.model, judge=report.judge, status=report.status, pass_rate=report.summary.pass_rate,
            mean_score=report.summary.mean_score, spec_versions=report.spec_versions,
            summary=report.summary.model_dump(mode="json"),
            regression=report.regression.model_dump(mode="json") if report.regression else None))
        for r in report.results:
            db.add(EvalCaseResultRow(run_id=report.run_id, case_id=r.case_id, tenant_id=report.tenant_id,
                                     passed=r.passed, score=r.score, data=r.model_dump(mode="json")))


def _report(row: EvalRunRow, results: list[CaseResult]) -> EvalRunReport:
    return EvalRunReport(
        run_id=row.run_id, set_id=row.set_id, tenant_id=row.tenant_id, trigger=row.trigger, model=row.model,
        judge=row.judge, spec_versions=row.spec_versions or {}, status=row.status,
        summary=RunSummary.model_validate(row.summary or {}), results=results,
        regression=Regression.model_validate(row.regression) if row.regression else None)


async def get_run(tenant_id: str, run_id: str, *, with_results: bool = True) -> EvalRunReport | None:
    async with session_scope(tenant_id=tenant_id) as db:
        row = await db.get(EvalRunRow, run_id)
        if row is None or row.tenant_id != tenant_id:
            return None
        results: list[CaseResult] = []
        if with_results:
            rows = (await db.execute(select(EvalCaseResultRow).where(EvalCaseResultRow.run_id == run_id)
                                     .order_by(EvalCaseResultRow.case_id))).scalars().all()
            results = [CaseResult.model_validate(r.data) for r in rows]
    return _report(row, results)


async def list_runs(tenant_id: str, set_id: str | None = None, limit: int = 30) -> list[dict]:
    stmt = select(EvalRunRow).where(EvalRunRow.tenant_id == tenant_id)
    if set_id:
        stmt = stmt.where(EvalRunRow.set_id == set_id)
    async with session_scope(tenant_id=tenant_id) as db:
        rows = (await db.execute(stmt.order_by(EvalRunRow.created_at.desc()).limit(min(limit, 200)))).scalars().all()
    return [{"run_id": r.run_id, "set_id": r.set_id, "trigger": r.trigger, "model": r.model, "judge": r.judge,
             "status": r.status, "is_baseline": r.is_baseline, "pass_rate": r.pass_rate,
             "mean_score": r.mean_score, "created_at": r.created_at.isoformat(),
             "regression_ok": (r.regression or {}).get("ok") if r.regression else None} for r in rows]


async def get_baseline(tenant_id: str, set_id: str) -> EvalRunReport | None:
    async with session_scope(tenant_id=tenant_id) as db:
        row = (await db.execute(select(EvalRunRow).where(
            EvalRunRow.tenant_id == tenant_id, EvalRunRow.set_id == set_id, EvalRunRow.is_baseline.is_(True))
            .order_by(EvalRunRow.created_at.desc()).limit(1))).scalars().first()
    return await get_run(tenant_id, row.run_id) if row else None


async def set_baseline(tenant_id: str, run_id: str) -> bool:
    async with session_scope(tenant_id=tenant_id) as db:
        row = await db.get(EvalRunRow, run_id)
        if row is None or row.tenant_id != tenant_id:
            return False
        await db.execute(update(EvalRunRow).where(EvalRunRow.tenant_id == tenant_id, EvalRunRow.set_id == row.set_id)
                         .values(is_baseline=False))
        row.is_baseline = True
    return True


# -- online samples --------------------------------------------------------------------


async def save_sample(row: EvalSampleRow) -> None:
    async with session_scope(tenant_id=row.tenant_id) as db:
        db.add(row)


async def list_samples(tenant_id: str, spec_id: str | None = None, limit: int = 100) -> list[EvalSampleRow]:
    stmt = select(EvalSampleRow).where(EvalSampleRow.tenant_id == tenant_id)
    if spec_id:
        stmt = stmt.where(EvalSampleRow.spec_id == spec_id)
    async with session_scope(tenant_id=tenant_id) as db:
        return list((await db.execute(stmt.order_by(EvalSampleRow.created_at.desc()).limit(limit))).scalars().all())
