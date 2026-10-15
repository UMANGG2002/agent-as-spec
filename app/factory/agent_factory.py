"""Compiles an `ExecutionPlan` into something runnable, per the architecture
doc's "Agent Factory" section. Two execution backends behind one
`CompiledAgent.run()`:

- `DeterministicExecutor` (always available): walks the plan's dependency
  layers (`ExecutionPlan.step_order()`), running each layer's steps
  concurrently and layers in sequence, calling `tool_wrapper.call_capability`
  directly. This is the default -- every step already has a bound
  capability_id, so there's nothing for a per-step LLM call to decide;
  reasoning already happened once, in the planner.
- An ADK agent tree (`SequentialAgent` of `ParallelAgent`s of `LlmAgent`s,
  one per layer/step) is what you'd reach for instead if a step should be
  allowed to *improvise* within its instruction (retry with different
  arguments, ask a clarifying question) rather than just execute
  deterministically. `compile_plan_to_adk_agent` builds that tree and is the
  extension point the architecture doc's Phase 3+ assumes; wiring it fully
  (ADK Runner + session service + the GuardrailPlugin's state contract) is
  left as the clearly-bounded extension point this scaffold's README points
  at, since it needs a live model to be worth exercising.

The important invariant either backend preserves: every tool call -- no
matter which backend runs it -- passes through `tool_wrapper.call_capability`,
so auth delegation, guardrails, and telemetry are identical either way.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator

from app.factory.tool_wrapper import ToolCallContext, ToolCallResult, call_capability
from app.guardrails.policy import resolve_policy
from app.planner.schemas import ExecutionPlan, PlanStep, StepKind
from app.registry.schemas import Capability
from app.runtime import redis_streams
from app.runtime.hil import ApprovalStatus, approval_store


@dataclass
class StepEvent:
    step_id: str
    kind: str  # "step.started" | "step.finished" | "step.blocked" | "step.approval_needed"
    capability_id: str | None = None
    result: ToolCallResult | None = None


@dataclass
class CompiledAgent:
    plan: ExecutionPlan
    capabilities: dict[str, Capability]
    ctx: ToolCallContext

    async def run(self) -> AsyncIterator[StepEvent]:
        async for event in DeterministicExecutor(self.plan, self.capabilities, self.ctx).run():
            yield event


def compile_plan(plan: ExecutionPlan, capabilities: dict[str, Capability], ctx: ToolCallContext) -> CompiledAgent:
    missing = [s.capability_id for s in plan.steps if s.kind == StepKind.TOOL_CALL and s.capability_id not in capabilities]
    if missing:
        raise ValueError(f"plan references unknown capabilities: {missing}")
    return CompiledAgent(plan=plan, capabilities=capabilities, ctx=ctx)


class DeterministicExecutor:
    def __init__(self, plan: ExecutionPlan, capabilities: dict[str, Capability], ctx: ToolCallContext) -> None:
        self.plan = plan
        self.capabilities = capabilities
        self.ctx = ctx

    async def run(self) -> AsyncIterator[StepEvent]:
        for layer in self.plan.step_order():
            results = await asyncio.gather(*(self._run_step(step) for step in layer))
            for step, event in zip(layer, results):
                yield event

    async def _run_step(self, step: PlanStep) -> StepEvent:
        if step.kind == StepKind.HUMAN_APPROVAL:
            # Publish the pause immediately (not just on this method's return)
            # so a streaming client sees "waiting for approval" right away --
            # genuinely suspends here via approval_store's Future, the same
            # shape as ADK's LongRunningFunctionTool pattern. A real ADK tree
            # would use that tool directly instead of this inline shortcut.
            await redis_streams.publish_event(
                self.ctx.run_id, {"step_id": step.step_id, "kind": "step.approval_needed"}
            )
            status = await approval_store.request_approval(
                self.ctx.run_id, step.step_id, reason=step.instruction
            )
            kind = "step.finished" if status == ApprovalStatus.APPROVED else "step.blocked"
            return StepEvent(step_id=step.step_id, kind=kind)

        capability = self.capabilities[step.capability_id]
        # Resolved fresh per call, not just looked up by the plan's static
        # guardrail_policy_id -- this is the hook adaptive escalation runs
        # through (architecture doc's "Dynamic policy engine" section).
        policy = resolve_policy(
            tenant_id=self.ctx.identity.tenant_id,
            brand_id=self.ctx.identity.brand_id,
            capability_id=capability.id,
            base_policy_id=step.guardrail_policy_id,
        )
        ctx = ToolCallContext(
            identity=self.ctx.identity, run_id=self.ctx.run_id, run_token=self.ctx.run_token, policy=policy
        )
        tool_name = capability.tools[0].name if capability.tools else capability.name
        result = await call_capability(
            capability, tool_name, {"instruction": step.instruction}, ctx, step_id=step.step_id
        )
        kind = "step.finished" if result.ok else ("step.blocked" if result.blocked_by else "step.finished")
        return StepEvent(step_id=step.step_id, kind=kind, capability_id=capability.id, result=result)
