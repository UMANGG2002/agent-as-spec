"""Every approved AgentSpec, compiled and mounted as a real ADK sub-agent
under one router `LlmAgent` -- used by `agents/spec_test_agent/agent.py`.

Sub-agents, not AgentTools (contrast the live supervisor in
`agents/capability_marketplace/agent.py`, which wraps each spec in an
`AgentTool`): the router hands the conversation to a spec with ADK's
`transfer_to_agent`, the spec runs exactly as compiled -- its own pattern,
its own callbacks, its own events in the session -- and then control comes
back to the router for the next message. That's the whole point of a test
harness: you're watching the spec itself run, not a summary of it returned
through a tool call.

Two ADK behaviors this relies on, both checked against ADK 2.12's source:

  - `BaseAgent.find_agent` / `find_sub_agent` walk `sub_agents` live on
    every lookup -- there's no cached agent index -- so `mount_sub_agents`
    can swap the router's sub-agents in place (setting each new child's
    `parent_agent`), which is what lets `reload_specs` pick up specs
    approved after startup without restarting `adk web`.
  - Which agent answers the next message (`_agent_router.find_agent_to_run`):
    the last agent that replied keeps the conversation only if it, and
    every ancestor, may transfer to its parent. `isolate` sets
    `disallow_transfer_to_parent` (and `_to_peers`) on each mounted spec, so
    after a spec runs, the next message goes back to the router -- and an
    approved spec can never transfer itself into a *different* spec it
    wasn't approved to be. Workflow-pattern specs (sequential / parallel /
    loop / plan_execute) need nothing extra: their step agents' parent is a
    workflow agent, which can't be transferred to, so they get no transfer
    tool at all.
"""

from __future__ import annotations

from typing import Any

from app.auth.models import IdentityContext
from app.factory import spec_agents
from app.planner.schemas import StepKind
from app.registry.schemas import Capability
from app.specs.schemas import AgentSpec

ROUTER_NAME = "spec_test_agent"


def filter_specs(specs: list[AgentSpec], only: set[str] | None) -> list[AgentSpec]:
    """Keep specs whose name or spec_id is in `only` (all of them if `only`
    is empty/None), sorted by name for a stable sub-agent order."""
    if only:
        specs = [s for s in specs if s.spec_id in only or s.name in only]
    return sorted(specs, key=lambda s: (s.name, s.spec_id))


def capabilities_for(spec: AgentSpec, visible: dict[str, Capability]) -> dict[str, Capability]:
    """Live-resolved capabilities a spec's steps need -- never the spec's
    own snapshots, so a capability revoked since approval stays uncallable
    (the step reports it instead)."""
    needed = {s.capability_id for s in spec.steps if s.kind != StepKind.HUMAN_APPROVAL}
    return {cid: visible[cid] for cid in needed if cid in visible}


def isolate(agent: Any) -> None:
    """A mounted spec may not transfer to the router or to another spec:
    it runs what was approved, then the next message returns to the router."""
    if hasattr(agent, "disallow_transfer_to_parent"):
        agent.disallow_transfer_to_parent = True
        agent.disallow_transfer_to_peers = True


def compile_sub_agents(
    specs: list[AgentSpec],
    visible: dict[str, Capability],
    identity: IdentityContext,
    *,
    model: Any = None,
) -> tuple[list, dict[str, dict], dict[str, str]]:
    """Compile each spec (with its own pattern and callbacks) and isolate
    it. One spec that fails to compile is skipped with its reason; it never
    stops the others from loading. Returns (agents, loaded, skipped), where
    `loaded` maps spec_id -> a short description of what's mounted."""
    agents: list = []
    loaded: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for spec in specs:
        try:
            agent = spec_agents.compile_spec_to_agent(spec, capabilities_for(spec, visible), identity, model=model)
        except Exception as exc:  # noqa: BLE001
            skipped[spec.spec_id] = f"{spec.name}: {exc}"
            continue
        isolate(agent)
        agents.append(agent)
        loaded[spec.spec_id] = {
            "agent_name": agent.name,
            "spec_name": spec.name,
            "version": spec.version,
            "pattern": spec.runtime.pattern.value,
            "goal": spec.goal,
            "steps": [s.step_id for s in spec.steps],
            "callbacks": [f"{c.hook.value}:{c.kind}" for c in spec.runtime.callbacks],
        }
    return agents, loaded, skipped


def mount_sub_agents(router: Any, agents: list) -> None:
    """Replace the router's sub-agents in place. Children get their
    `parent_agent` set the way ADK's own constructor does it, so
    `find_agent`, transfer targets and next-turn routing all see the new
    tree immediately."""
    for agent in agents:
        agent.parent_agent = router
    router.sub_agents[:] = agents


class RouterInstruction:
    """The router's instruction, rebuilt from the current mounted set on
    every model call (an InstructionProvider, so it also bypasses ADK's
    `{var}` state templating) -- after `reload_specs` it describes the new
    specs without rebuilding the router."""

    def __init__(self, state: dict) -> None:
        self.state = state

    def __call__(self, ctx: Any = None) -> str:  # noqa: ARG002
        loaded: dict = self.state.get("loaded", {})
        skipped: dict = self.state.get("skipped", {})
        if loaded:
            lines = "\n".join(
                f"- {info['agent_name']}: \"{info['spec_name']}\" v{info['version']} ({info['pattern']}) -- {info['goal']}"
                for info in loaded.values()
            )
        else:
            lines = "(none loaded -- approve a plan first, then call reload_specs)"
        text = (
            "You are the test router for approved agent specs. Each approved spec is mounted as "
            "one of your sub-agents, compiled exactly as approved (its own execution pattern and "
            "callbacks). When the person asks for something one spec does -- or names a spec -- "
            "transfer to that sub-agent with transfer_to_agent; it runs its approved steps and the "
            "conversation comes back to you afterwards. Don't try to do a spec's work yourself; "
            "you have no tools for it.\n\n"
            f"Mounted specs:\n{lines}\n\n"
            "Call list_loaded_specs for each spec's steps and callbacks, and reload_specs if the "
            "person just approved or changed a spec (picks it up without a restart)."
        )
        if skipped:
            text += "\n\nSpecs that failed to load:\n" + "\n".join(f"- {reason}" for reason in skipped.values())
        return text

    def __contains__(self, item: str) -> bool:
        return item in self()

    def __str__(self) -> str:
        return self()
