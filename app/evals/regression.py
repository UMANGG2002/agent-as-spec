"""Compare a run with its baseline and decide whether the gate passes.

Hard gates (deterministic): any case that passed in the baseline and fails
now; the pass rate dropping more than `eval_gate_max_pass_rate_drop`.
Soft gate (judge): the mean judge score dropping more than
`eval_gate_max_judge_drop`. A new failing case also fails the gate; a case
missing from the current run is reported but does not fail it.
"""

from __future__ import annotations

from app.config import settings
from app.evals.schemas import Delta, EvalRunReport, Regression


def compare(current: EvalRunReport, baseline: EvalRunReport | None, *,
            max_pass_rate_drop: float | None = None, max_judge_drop: float | None = None) -> Regression:
    max_pass_rate_drop = settings.eval_gate_max_pass_rate_drop if max_pass_rate_drop is None else max_pass_rate_drop
    max_judge_drop = settings.eval_gate_max_judge_drop if max_judge_drop is None else max_judge_drop
    reg = Regression(baseline_run_id=baseline.run_id if baseline else None)
    cur = {r.case_id: r for r in current.results}

    if baseline is None:
        # No baseline yet: the gate is the absolute hard checks.
        failing = [cid for cid, r in cur.items() if not r.passed]
        reg.new_cases = sorted(cur)
        if failing:
            reg.ok = False
            reg.reasons.append(f"no baseline yet and {len(failing)} case(s) fail: {sorted(failing)}")
        return reg

    base = {r.case_id: r for r in baseline.results}
    for cid, r in cur.items():
        b = base.get(cid)
        if b is None:
            reg.new_cases.append(cid)
            if not r.passed:
                reg.reasons.append(f"new case {cid} fails")
        elif b.passed and not r.passed:
            reg.flipped_to_fail.append(cid)
        elif not b.passed and r.passed:
            reg.flipped_to_pass.append(cid)
    reg.missing_cases = sorted(set(base) - set(cur))

    if reg.flipped_to_fail:
        reg.reasons.append(f"{len(reg.flipped_to_fail)} case(s) regressed pass -> fail: {sorted(reg.flipped_to_fail)}")

    def delta(name: str, b: float, c: float) -> Delta:
        return Delta(metric=name, baseline=round(b, 4), current=round(c, 4), delta=round(c - b, 4))

    reg.deltas.append(delta("pass_rate", baseline.summary.pass_rate, current.summary.pass_rate))
    reg.deltas.append(delta("mean_score", baseline.summary.mean_score, current.summary.mean_score))
    for name in sorted(set(baseline.summary.by_metric) | set(current.summary.by_metric)):
        reg.deltas.append(delta(name, baseline.summary.by_metric.get(name, 0.0), current.summary.by_metric.get(name, 0.0)))

    if baseline.summary.pass_rate - current.summary.pass_rate > max_pass_rate_drop + 1e-9:
        reg.reasons.append(f"pass rate fell {baseline.summary.pass_rate:.2%} -> {current.summary.pass_rate:.2%}")
    bj, cj = baseline.summary.mean_judge_score, current.summary.mean_judge_score
    if bj is not None and cj is not None and bj - cj > max_judge_drop + 1e-9:
        reg.reasons.append(f"mean judge score fell {bj:.3f} -> {cj:.3f} (tolerance {max_judge_drop})")
    if any(not cur[c].passed for c in reg.new_cases if c in cur):
        pass  # already in reasons
    reg.ok = not reg.reasons
    return reg
