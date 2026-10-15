"""Approved spec + runtime config -> the actual ADK agent tree.

`AgentSpec.runtime.pattern` (see `app/specs/schemas.py`'s `AgentPattern`)
picks one of these shapes; `runtime.callbacks` are attached to every
`LlmAgent` in whichever tree gets built (agent-level hooks to the root
only). The steps themselves never change shape: every pattern is built from
the same per-step tools `spec_agents._step_tool_function` produces -- one
fixed, no-argument tool per approved step, so no pattern can let a model
improvise *what* a step does, only (depending on the pattern) *when*.

  single        LlmAgent with every step tool. The model orders the calls
                from the instruction. The original, default shape.
  sequential    SequentialAgent: one LlmAgent per step, each holding only
                its own step's tool, run in `depends_on` (topological) order.
                Ordering is enforced by ADK, not left to the model.
  parallel      ParallelAgent: the same per-step agents, run concurrently.
                Rejected at approval if any step has `depends_on` -- running
                dependent steps at the same time would silently violate the
                plan a human approved.
  loop          LoopAgent over the sequential chain; the last step agent
                also gets ADK's `exit_loop` tool and is told to call it once
                the result is acceptable. Hard-capped at
                `runtime.max_iterations` either way.
  hierarchical  A coordinator LlmAgent whose tools are one specialist
                sub-agent per step, each wrapped in `AgentTool`. Deliberately
                AgentTool rather than `sub_agents=` + transfer_to_agent: a
                transfer hands control away and only comes back if the
                specialist transfers it back, which makes "run every approved
                step and report on all of them" depend on every specialist
                cooperating. With AgentTool the coordinator stays in charge
                and gets each specialist's result back as a tool response.
  react         `single`, plus ADK's `PlanReActPlanner` -- an explicit
                plan / act / reason / final-answer protocol. Safe here only
                because spec agents never set `output_schema`; the planner +
                output_schema + tools combination is exactly what caused the
                live planner's infinite loop (app/config.py's _GatewayLiteLlm
                docstring).
  plan_execute  SequentialAgent of two LlmAgents: a tool-less planner that
                writes an execution plan into session state, then an
                executor holding every step tool whose instruction includes
                that plan. Plan-then-act without depending on a
                Gemini-only thinking planner, so it works through the
                LiteLLM gateway too.

Every instruction is passed as an `InstructionProvider` (`_LiteralInstruction`
below), never a raw string. ADK runs `{name}`-style session-state
substitution over raw-string instructions and raises if a referenced key is
missing -- and a model-polished instruction (spec_agents.polish_spec_instruction)
can contain braces nobody intended as template variables. A provider
bypasses that substitution entirely (see `LlmAgent.canonical_instruction`).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

from app.factory import callback_registry
from app.planner.schemas import PlanStep
from app.specs.schemas import AgentPattern, AgentRuntimeConfig, CallbackHook


class _LiteralInstruction:
    """An `InstructionProvider` that always returns the same text -- so ADK
    skips state templating on it -- while still behaving like that text for
    `in` checks and `str()`, which keeps it readable in logs and tests."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __call__(self, ctx: Any) -> str:  # noqa: ARG002 - ADK's InstructionProvider signature
        return self.text

    def __contains__(self, item: str) -> bool:
        return item in self.text

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"_LiteralInstruction({self.text[:60]!r}...)"


class _PlanAwareInstruction(_LiteralInstruction):
    """Executor instruction for `plan_execute`: the fixed text plus whatever
    the planner sub-agent wrote to `state[plan_state_key]` -- read directly
    from state rather than through `{key}` templating, for the same reason
    as `_LiteralInstruction`."""

    def __init__(self, text: str, plan_state_key: str) -> None:
        super().__init__(text)
        self.plan_state_key = plan_state_key

    def __call__(self, ctx: Any) -> str:
        plan = ""
        try:
            plan = ctx.state.get(self.plan_state_key, "") or ""
        except Exception:  # noqa: BLE001 - a missing/unreadable plan just means "no extra guidance"
            plan = ""
        if not plan:
            return self.text
        return f"{self.text}\n\nExecution plan written by the planning stage -- follow it:\n{plan}"


def topological_steps(steps: list[PlanStep]) -> list[PlanStep]:
    """Steps in an order that respects every `depends_on` (stable: ties keep
    plan order). Raises ValueError on an unknown dependency or a cycle --
    both mean the plan can't be run in order, and a sequential/loop
    compile would otherwise silently run steps out of dependency order."""
    by_id = {s.step_id: s for s in steps}
    for step in steps:
        for dep in step.depends_on:
            if dep not in by_id:
                raise ValueError(f"step '{step.step_id}' depends on unknown step '{dep}'")
    indegree = {s.step_id: len(set(s.depends_on)) for s in steps}
    dependents: dict[str, list[str]] = {s.step_id: [] for s in steps}
    for step in steps:
        for dep in set(step.depends_on):
            dependents[dep].append(step.step_id)

    ready = deque(s.step_id for s in steps if indegree[s.step_id] == 0)
    ordered: list[PlanStep] = []
    while ready:
        step_id = ready.popleft()
        ordered.append(by_id[step_id])
        for nxt in dependents[step_id]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if len(ordered) != len(steps):
        cyclic = sorted(sid for sid, deg in indegree.items() if deg > 0)
        raise ValueError(f"steps {cyclic} form a dependency cycle")
    return ordered


def validate_runtime_config(steps: list[PlanStep], runtime: AgentRuntimeConfig) -> None:
    """Approval-time check that this plan can actually be compiled under
    this runtime config. Raises ValueError with a message meant for the
    human approving it."""
    if not steps:
        raise ValueError("a spec needs at least one step to compile into an agent")
    topological_steps(steps)  # unknown deps / cycles are invalid for every pattern
    if runtime.pattern == AgentPattern.PARALLEL:
        dependent = [s.step_id for s in steps if s.depends_on]
        if dependent:
            raise ValueError(
                f"pattern 'parallel' runs every step at once, but steps {dependent} declare depends_on -- "
                "use 'sequential' (or remove the dependencies) instead"
            )
    callback_registry.validate_callback_configs(runtime.callbacks)


def _step_instruction(step: PlanStep, goal: str, capability_label: str) -> str:
    return (
        f"You handle exactly one step of an approved plan whose overall goal is: {goal}\n\n"
        f"Your step is '{step.step_id}' (capability: {capability_label}): {step.instruction}\n\n"
        "Call your one tool exactly once. Then report plainly what happened -- including any "
        "error, guardrail block, or pending approval it returned -- rather than just saying \"done\". "
        "Do not attempt any other step."
    )


def build_agent_tree(
    *,
    root_name: str,
    description: str,
    goal: str,
    instruction: str,
    steps: list[PlanStep],
    step_tools: dict[str, Callable],
    capability_labels: dict[str, str],
    runtime: AgentRuntimeConfig,
    model: Any,
):
    """Build the agent tree for one compiled spec. `step_tools` maps
    step_id -> that step's fixed tool function; `capability_labels` maps
    step_id -> a human-readable capability name for per-step instructions."""
    from google.adk.agents import LlmAgent, LoopAgent, ParallelAgent, SequentialAgent
    from google.adk.tools import FunctionTool
    from google.adk.tools.agent_tool import AgentTool

    callbacks = callback_registry.build_callbacks(runtime.callbacks, max_llm_calls=runtime.max_llm_calls)

    def _llm_callbacks() -> dict[str, Any]:
        return {
            "before_model_callback": list(callbacks[CallbackHook.BEFORE_MODEL]) or None,
            "after_model_callback": list(callbacks[CallbackHook.AFTER_MODEL]) or None,
            "before_tool_callback": list(callbacks[CallbackHook.BEFORE_TOOL]) or None,
            "after_tool_callback": list(callbacks[CallbackHook.AFTER_TOOL]) or None,
        }

    def _agent_callbacks() -> dict[str, Any]:
        return {
            "before_agent_callback": list(callbacks[CallbackHook.BEFORE_AGENT]) or None,
            "after_agent_callback": list(callbacks[CallbackHook.AFTER_AGENT]) or None,
        }

    def _step_agent(step: PlanStep, *, extra_tools: list | None = None, extra_instruction: str = ""):
        text = _step_instruction(step, goal, capability_labels.get(step.step_id, step.capability_id))
        if extra_instruction:
            text += "\n\n" + extra_instruction
        return LlmAgent(
            name=f"{root_name}__{_safe(step.step_id)}",
            model=model,
            description=f"Runs approved step '{step.step_id}': {step.instruction}",
            instruction=_LiteralInstruction(text),
            tools=[FunctionTool(step_tools[step.step_id])] + list(extra_tools or []),
            **_llm_callbacks(),
        )

    all_tools = [FunctionTool(step_tools[s.step_id]) for s in steps]
    pattern = runtime.pattern

    if pattern in (AgentPattern.SINGLE, AgentPattern.REACT):
        planner = None
        if pattern == AgentPattern.REACT:
            from google.adk.planners import PlanReActPlanner

            planner = PlanReActPlanner()
        return LlmAgent(
            name=root_name,
            model=model,
            description=description,
            instruction=_LiteralInstruction(instruction),
            tools=all_tools,
            planner=planner,
            **_llm_callbacks(),
            **_agent_callbacks(),
        )

    ordered = topological_steps(steps)

    if pattern == AgentPattern.SEQUENTIAL:
        return SequentialAgent(
            name=root_name,
            description=description,
            sub_agents=[_step_agent(s) for s in ordered],
            **_agent_callbacks(),
        )

    if pattern == AgentPattern.PARALLEL:
        return ParallelAgent(
            name=root_name,
            description=description,
            sub_agents=[_step_agent(s) for s in steps],
            **_agent_callbacks(),
        )

    if pattern == AgentPattern.LOOP:
        from google.adk.tools import exit_loop

        sub_agents = [_step_agent(s) for s in ordered[:-1]]
        sub_agents.append(
            _step_agent(
                ordered[-1],
                extra_tools=[FunctionTool(exit_loop)],
                extra_instruction=(
                    "You are the last step of a loop. After calling your step's tool, call exit_loop "
                    "if the overall result is acceptable; otherwise just report, and the loop will "
                    f"run again (at most {runtime.max_iterations} times in total)."
                ),
            )
        )
        return LoopAgent(
            name=root_name,
            description=description,
            max_iterations=runtime.max_iterations,
            sub_agents=sub_agents,
            **_agent_callbacks(),
        )

    if pattern == AgentPattern.HIERARCHICAL:
        specialists = [AgentTool(agent=_step_agent(s)) for s in ordered]
        coordinator_text = (
            f"{instruction}\n\nYou are the coordinator. Each step is handled by its own specialist, "
            "available to you as a tool named after it. Delegate each step to its specialist exactly "
            "once, in an order that respects the dependencies above, passing it a short request "
            "describing what to do. Then report every specialist's result."
        )
        return LlmAgent(
            name=root_name,
            model=model,
            description=description,
            instruction=_LiteralInstruction(coordinator_text),
            tools=specialists,
            **_llm_callbacks(),
            **_agent_callbacks(),
        )

    if pattern == AgentPattern.PLAN_EXECUTE:
        plan_key = f"{root_name}_execution_plan"
        planner_text = (
            f"{instruction}\n\nYou are the PLANNING stage only: you have no tools and must not try to "
            "run anything. Write a short, numbered execution plan for the steps above -- the order to "
            "run them in, what to check after each, and what to do if one fails or is blocked. Output "
            "only the plan."
        )
        planner_agent = LlmAgent(
            name=f"{root_name}__planner",
            model=model,
            description="Writes the execution plan for this approved spec.",
            instruction=_LiteralInstruction(planner_text),
            output_key=plan_key,
            **_llm_callbacks(),
        )
        executor_agent = LlmAgent(
            name=f"{root_name}__executor",
            model=model,
            description="Executes the approved steps following the plan.",
            instruction=_PlanAwareInstruction(instruction, plan_key),
            tools=all_tools,
            **_llm_callbacks(),
        )
        return SequentialAgent(
            name=root_name,
            description=description,
            sub_agents=[planner_agent, executor_agent],
            **_agent_callbacks(),
        )

    raise ValueError(f"unsupported agent pattern: {pattern}")


def _safe(text: str) -> str:
    from app.factory.spec_agents import _safe_identifier

    return _safe_identifier(text, fallback="step")


def describe_patterns() -> dict[str, str]:
    """One line per pattern -- for the supervisor's
    `describe_agent_runtime_options` tool."""
    return {
        AgentPattern.SINGLE.value: "One agent holding every step's tool; the model orders the calls.",
        AgentPattern.SEQUENTIAL.value: "One sub-agent per step, run strictly in depends_on order.",
        AgentPattern.PARALLEL.value: "One sub-agent per step, all at once (plan must have no depends_on).",
        AgentPattern.LOOP.value: "The sequential chain repeated until exit_loop or max_iterations.",
        AgentPattern.HIERARCHICAL.value: "A coordinator delegating each step to its own specialist sub-agent.",
        AgentPattern.REACT.value: "One agent with an explicit plan/act/reason/answer (ReAct) protocol.",
        AgentPattern.PLAN_EXECUTE.value: "A planning stage writes a plan, then an executor carries it out.",
    }
