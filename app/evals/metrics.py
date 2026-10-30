"""Deterministic checks over a `Trace`. Each returns a `MetricResult`; a check
whose expectation is empty is skipped (returns None), so a case only fails on
what it actually asserted. These are the hard gates of the eval."""

from __future__ import annotations

import json
import re

from app.evals.schemas import EvalCase, ExpectedToolCall, MetricResult, ToolCallRecord, Trace


def _norm(name: str) -> str:
    return name[5:] if name.startswith("step_") else name


def _matches(expected: ExpectedToolCall, call: ToolCallRecord) -> bool:
    names = {_norm(call.name)}
    if call.step_id:
        names.add(_norm(call.step_id))
    if _norm(expected.name) not in names:
        return False
    return expected.agent is None or expected.agent == call.agent


def _detail_ok(expected: ExpectedToolCall, call: ToolCallRecord) -> str | None:
    """None when the matched call satisfies the expected call's detail checks."""
    if expected.status is not None and call.status != expected.status:
        return f"{expected.name}: status {call.status!r} != {expected.status!r}"
    if expected.ok is not None and call.ok is not expected.ok:
        return f"{expected.name}: ok {call.ok!r} != {expected.ok!r}"
    if expected.output_contains:
        blob = json.dumps(call.output, default=str, sort_keys=True) if call.output is not None else ""
        missing = [s for s in expected.output_contains if s.lower() not in blob.lower()]
        if missing:
            return f"{expected.name}: output lacks {missing}"
    for key, want in expected.args_subset.items():
        if call.args.get(key) != want:
            return f"{expected.name}: arg {key}={call.args.get(key)!r} != {want!r}"
    return None


def tool_trajectory(case: EvalCase, trace: Trace) -> MetricResult | None:
    exp = case.expectations
    if not exp.tool_calls:
        return None
    calls = trace.tool_calls
    problems: list[str] = []
    matched = 0
    if exp.trajectory_mode == "exact":
        if len(calls) != len(exp.tool_calls):
            problems.append(f"expected {len(exp.tool_calls)} calls, saw {len(calls)}: "
                            f"{[c.name for c in calls]}")
        for e, c in zip(exp.tool_calls, calls):
            if _matches(e, c):
                bad = _detail_ok(e, c)
                matched += 0 if bad else 1
                if bad:
                    problems.append(bad)
            else:
                problems.append(f"expected {e.name}, saw {c.name}")
    elif exp.trajectory_mode == "in_order":
        pos = 0
        for e in exp.tool_calls:
            found = next((i for i in range(pos, len(calls)) if _matches(e, calls[i])), None)
            if found is None:
                problems.append(f"missing (in order): {e.name}")
                continue
            bad = _detail_ok(e, calls[found])
            if bad:
                problems.append(bad)
            else:
                matched += 1
            pos = found + 1
    else:  # any_order
        used: set[int] = set()
        for e in exp.tool_calls:
            found = next((i for i, c in enumerate(calls) if i not in used and _matches(e, c)), None)
            if found is None:
                problems.append(f"missing: {e.name}")
                continue
            used.add(found)
            bad = _detail_ok(e, calls[found])
            if bad:
                problems.append(bad)
            else:
                matched += 1
    score = matched / len(exp.tool_calls)
    return MetricResult(name="tool_trajectory", score=score, passed=not problems,
                        detail="; ".join(problems) or f"{matched}/{len(exp.tool_calls)} expected calls ok")


def forbidden_tools(case: EvalCase, trace: Trace) -> MetricResult | None:
    if not case.expectations.forbidden_tools:
        return None
    bad = [c.name for c in trace.tool_calls
           if any(_norm(f) in {_norm(c.name), _norm(c.step_id or "")} for f in case.expectations.forbidden_tools)
           and c.ok is not False and c.status not in ("denied", "denied_by_policy", "rejected", "awaiting_approval")]
    return MetricResult(name="forbidden_tools", score=0.0 if bad else 1.0, passed=not bad,
                        detail=f"executed forbidden tool(s): {bad}" if bad else "none executed")


def outcome(case: EvalCase, trace: Trace) -> MetricResult | None:
    want = case.expectations.outcome
    if want == "any":
        return None
    ok = trace.outcome == want
    return MetricResult(name="policy_outcome", score=1.0 if ok else 0.0, passed=ok,
                        detail=f"outcome {trace.outcome!r}, expected {want!r}" + (f" ({trace.error})" if trace.error else ""))


def transfers(case: EvalCase, trace: Trace) -> MetricResult | None:
    want = case.expectations.transfers
    if not want:
        return None
    it = iter(trace.transfers)
    ok = all(any(w == got for got in it) for w in want)  # in order, extras allowed
    return MetricResult(name="routing", score=1.0 if ok else 0.0, passed=ok,
                        detail=f"transfers {trace.transfers}, expected {want}")


def answer(case: EvalCase, trace: Trace) -> MetricResult | None:
    exp = case.expectations
    if not (exp.answer_contains or exp.answer_not_contains or exp.answer_regex):
        return None
    text = trace.final_text
    low = text.lower()
    problems = [f"missing {s!r}" for s in exp.answer_contains if s.lower() not in low]
    problems += [f"contains forbidden {s!r}" for s in exp.answer_not_contains if s.lower() in low]
    if exp.answer_regex and not re.search(exp.answer_regex, text, re.S):
        problems.append(f"does not match /{exp.answer_regex}/")
    total = len(exp.answer_contains) + len(exp.answer_not_contains) + (1 if exp.answer_regex else 0)
    return MetricResult(name="answer_checks", score=(total - len(problems)) / total, passed=not problems,
                        detail="; ".join(problems) or "all answer checks ok")


def budget(case: EvalCase, trace: Trace) -> MetricResult | None:
    exp = case.expectations
    problems = []
    if exp.max_tool_calls is not None and len(trace.tool_calls) > exp.max_tool_calls:
        problems.append(f"{len(trace.tool_calls)} tool calls > {exp.max_tool_calls}")
    if exp.max_tokens is not None and trace.tokens_in + trace.tokens_out > exp.max_tokens:
        problems.append(f"{trace.tokens_in + trace.tokens_out} tokens > {exp.max_tokens}")
    if exp.max_latency_ms is not None and trace.latency_ms > exp.max_latency_ms:
        problems.append(f"{trace.latency_ms}ms > {exp.max_latency_ms}ms")
    if exp.max_tool_calls is None and exp.max_tokens is None and exp.max_latency_ms is None:
        return None
    return MetricResult(name="budget", score=0.0 if problems else 1.0, passed=not problems,
                        detail="; ".join(problems) or "within budget")


def no_error(case: EvalCase, trace: Trace) -> MetricResult:
    return MetricResult(name="no_error", score=0.0 if trace.error else 1.0, passed=not trace.error,
                        detail=trace.error or "ran without error")


CHECKS = (no_error, outcome, tool_trajectory, forbidden_tools, transfers, answer, budget)


def run_checks(case: EvalCase, trace: Trace) -> list[MetricResult]:
    return [r for check in CHECKS if (r := check(case, trace)) is not None]
