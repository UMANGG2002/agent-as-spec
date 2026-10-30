"""Typed contract for evaluation cases, results and runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Outcome = Literal["completed", "awaiting_approval", "blocked", "any"]
TrajectoryMode = Literal["exact", "in_order", "any_order"]
Confirmation = Literal["approve", "reject", "leave_pending"]
Source = Literal["golden", "synthetic", "promoted", "online"]


class Target(BaseModel):
    kind: Literal["spec", "router"] = "spec"
    spec_id: str | None = None  # kind=spec
    spec_version: int | None = None  # None = the current head


class ExpectedToolCall(BaseModel):
    """One tool call the agent should make. `name` matches the call's name,
    its step id, or `step_<id>`; `agent` (optional) pins which agent made
    it, which is what makes an end-to-end case assert 'the router handed
    this to the right sub-agent AND that sub-agent called the tool'."""

    name: str
    agent: str | None = None
    status: str | None = None  # e.g. "approved", "awaiting_approval", "denied_by_policy"
    ok: bool | None = None
    output_contains: list[str] = Field(default_factory=list)
    args_subset: dict[str, Any] = Field(default_factory=dict)


class Expectations(BaseModel):
    """Everything checked in code. Leave a field empty to skip that check."""

    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    trajectory_mode: TrajectoryMode = "in_order"
    forbidden_tools: list[str] = Field(default_factory=list)
    outcome: Outcome = "any"
    transfers: list[str] = Field(default_factory=list, description="router sub-agents the run must hand off to, in order")
    answer_contains: list[str] = Field(default_factory=list)
    answer_not_contains: list[str] = Field(default_factory=list)
    answer_regex: str | None = None
    reference_answer: str | None = Field(default=None, description="used by the judge, never string-matched")
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    max_latency_ms: int | None = None


class EvalCase(BaseModel):
    case_id: str
    name: str = ""
    target: Target = Field(default_factory=Target)
    messages: list[str] = Field(description="user turns, in order")
    confirmation: Confirmation = "approve"  # what the simulated approver does when a tool asks
    expectations: Expectations = Field(default_factory=Expectations)
    rubrics: list[str] = Field(default_factory=list, description="rubric ids for the LLM judge; empty = none")
    tags: list[str] = Field(default_factory=list)
    source: Source = "golden"
    notes: str = ""


class EvalSet(BaseModel):
    set_id: str
    tenant_id: str
    name: str
    description: str = ""
    kind: Literal["golden", "synthetic", "mixed"] = "golden"
    cases: list[EvalCase] = Field(default_factory=list)
    version: int = 1


class MetricResult(BaseModel):
    name: str
    score: float  # 0..1
    passed: bool
    hard: bool = True  # hard = deterministic gate; soft = judge trend
    detail: str = ""


class ToolCallRecord(BaseModel):
    name: str
    agent: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    step_id: str | None = None
    status: str | None = None
    ok: bool | None = None
    output: Any = None
    error: str | None = None


class Trace(BaseModel):
    """What the agent actually did for one case."""

    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    transfers: list[str] = Field(default_factory=list)
    final_text: str = ""
    answers_by_turn: list[str] = Field(default_factory=list)
    outcome: Literal["completed", "awaiting_approval", "blocked", "error"] = "completed"
    confirmations_requested: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    latency_ms: int = 0
    n_events: int = 0
    error: str | None = None


class CaseResult(BaseModel):
    case_id: str
    passed: bool  # all hard metrics passed
    score: float  # mean of all metric scores (hard + soft)
    hard_score: float
    judge_score: float | None = None
    metrics: list[MetricResult] = Field(default_factory=list)
    trace: Trace = Field(default_factory=Trace)
    tags: list[str] = Field(default_factory=list)
    error: str | None = None


class RunSummary(BaseModel):
    n_cases: int = 0
    n_passed: int = 0
    pass_rate: float = 0.0
    mean_score: float = 0.0
    mean_judge_score: float | None = None
    mean_tokens: float = 0.0
    mean_latency_ms: float = 0.0
    by_metric: dict[str, float] = Field(default_factory=dict)  # mean score per metric
    by_tag: dict[str, float] = Field(default_factory=dict)  # pass rate per tag


class Delta(BaseModel):
    metric: str
    baseline: float
    current: float
    delta: float


class Regression(BaseModel):
    baseline_run_id: str | None = None
    ok: bool = True
    reasons: list[str] = Field(default_factory=list)
    flipped_to_fail: list[str] = Field(default_factory=list)
    flipped_to_pass: list[str] = Field(default_factory=list)
    new_cases: list[str] = Field(default_factory=list)
    missing_cases: list[str] = Field(default_factory=list)
    deltas: list[Delta] = Field(default_factory=list)


class EvalRunReport(BaseModel):
    run_id: str
    set_id: str
    tenant_id: str
    trigger: str = "manual"  # manual | ci | spec_approval | schedule
    model: str = ""
    judge: str = ""
    spec_versions: dict[str, int] = Field(default_factory=dict)
    summary: RunSummary = Field(default_factory=RunSummary)
    results: list[CaseResult] = Field(default_factory=list)
    regression: Regression | None = None
    status: str = "completed"
