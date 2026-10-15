"""The typed contract between the Planner Agent and the Agent Factory --
`ExecutionPlan` is exactly what the brief asked for: "a planner-agent-generated
typed execution flow with toolset and agent-ready instruction from which I can
create a new agent." See the architecture doc's "Dynamic agent creation"
section."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepKind(str, Enum):
    TOOL_CALL = "tool_call"
    SUB_AGENT_CALL = "sub_agent_call"
    HUMAN_APPROVAL = "human_approval"


class PlanStep(BaseModel):
    step_id: str
    kind: StepKind
    capability_id: str = Field(description="Registry capability this step binds to; empty for human_approval")
    instruction: str = Field(description="The system instruction the compiled agent runs under for this step")
    # Which of the capability's tools this step calls. None keeps the
    # original behavior (the capability's first tool) -- fine for
    # single-tool capabilities, but a multi-tool MCP server needs this to
    # reach anything past its first tool.
    tool_name: str | None = Field(default=None, description="Tool on the capability to call; defaults to its first tool")
    # The exact arguments the step calls that tool with -- approved along
    # with the rest of the plan, never chosen by the model at run time.
    # Empty keeps the original behavior of sending {"instruction": ...},
    # which free-text tools accept; typed tools (e.g. add(a, b)) need these.
    arguments: dict[str, Any] = Field(default_factory=dict, description="Fixed arguments for the tool call")
    required_scopes: list[str] = Field(default_factory=list)
    guardrail_policy_id: str = "default"
    depends_on: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    plan_id: str
    goal: str = Field(description="The restated user intent this plan accomplishes")
    steps: list[PlanStep]

    def step_order(self) -> list[list[PlanStep]]:
        """Group steps into dependency-respecting layers: layer 0 has no
        unmet deps, layer 1 depends only on layer 0, etc. Layers map directly
        onto ParallelAgent (within a layer) / SequentialAgent (across layers)
        in the Agent Factory."""
        remaining = {s.step_id: s for s in self.steps}
        done: set[str] = set()
        layers: list[list[PlanStep]] = []
        while remaining:
            layer = [s for s in remaining.values() if all(d in done for d in s.depends_on)]
            if not layer:
                raise ValueError(f"circular or unresolved dependency among steps: {list(remaining)}")
            layers.append(layer)
            for s in layer:
                done.add(s.step_id)
                del remaining[s.step_id]
        return layers
