"""The Planner Agent: NL intent -> typed `ExecutionPlan`.

Two implementations behind one `create_plan` entrypoint, matching the
architecture doc's goals section ("runs fully local... mocked LLM calls"):

- `_LivePlanner` is the real thing: an ADK `LlmAgent` with
  `output_schema=ExecutionPlan` and a bound `search_capabilities` tool,
  relying on ADK's own output_schema+tools support (not an explicit
  `planner=`) to call the tool then emit a schema-conformant final answer --
  see `app/config.py`'s `_GatewayLiteLlm` for why adding `PlanReActPlanner`
  on top caused an infinite loop of LLM calls against the custom gateway.
  Used when `settings.planner_is_live`.
- `_OfflinePlanner` is a deterministic keyword-matcher over the same registry
  search used for tests, the demo script, and any environment without a
  configured model key -- it produces a real, schema-valid `ExecutionPlan`,
  just without an LLM doing the reasoning.

Both paths only ever *see* capabilities visible to the caller's tenant --
search is called with the identity context already applied (see
registry/store.py's `search`).
"""

from __future__ import annotations

import json
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.models import IdentityContext
from app.config import settings
from app.marketplace import service as marketplace_service
from app.planner.schemas import ExecutionPlan, PlanStep, StepKind
from app.registry import store as registry_store
from app.registry.schemas import CapabilitySearchResult

PLANNER_SYSTEM_INSTRUCTION = """You are the planning layer of a multi-tenant \
capability marketplace. Given a user's natural-language goal and the list of \
capabilities available to them (already filtered to what they're allowed to \
use), produce an ExecutionPlan: a goal restatement and an ordered list of \
steps, each bound to exactly one capability_id you found via search_capabilities. \
Never invent a capability_id. Prefer the fewest steps that accomplish the goal. \
Use `depends_on` to express ordering: a step that needs another step's output \
must list that step's step_id in depends_on. Each step's `instruction` is the \
system prompt the compiled agent will run under for that step -- be specific \
about what it should do with its one bound tool."""


async def create_plan(session: AsyncSession, identity: IdentityContext, intent: str) -> ExecutionPlan:
    if settings.planner_is_live:
        try:
            return await _LivePlanner(session, identity).plan(intent)
        except Exception:
            # A live-planner failure (quota, malformed output, model outage)
            # degrades to the deterministic planner rather than failing the
            # request outright -- see guardrails' before_model_callback for
            # where a *rejected* intent is handled instead.
            pass
    return await _OfflinePlanner(session, identity).plan(intent)


class _OfflinePlanner:
    def __init__(self, session: AsyncSession, identity: IdentityContext) -> None:
        self.session = session
        self.identity = identity

    async def plan(self, intent: str) -> ExecutionPlan:
        granted = await marketplace_service.granted_capability_ids(self.session, self.identity.tenant_id)
        results: list[CapabilitySearchResult] = await registry_store.search(
            self.session, self.identity.tenant_id, intent, granted_capability_ids=granted, limit=5
        )
        if not results:
            return ExecutionPlan(plan_id=f"plan_{uuid.uuid4().hex[:12]}", goal=intent, steps=[])

        steps: list[PlanStep] = []
        previous_id: str | None = None
        for result in results:
            cap = result.capability
            step_id = f"step_{uuid.uuid4().hex[:8]}"
            steps.append(
                PlanStep(
                    step_id=step_id,
                    kind=StepKind.TOOL_CALL,
                    capability_id=cap.id,
                    instruction=f"Use the '{cap.name}' capability to accomplish: {intent}",
                    required_scopes=cap.required_scopes,
                    depends_on=[previous_id] if previous_id else [],
                )
            )
            previous_id = step_id

        return ExecutionPlan(plan_id=f"plan_{uuid.uuid4().hex[:12]}", goal=intent, steps=steps)


class _LivePlanner:
    def __init__(self, session: AsyncSession, identity: IdentityContext) -> None:
        self.session = session
        self.identity = identity

    async def plan(self, intent: str) -> ExecutionPlan:
        from google.adk.agents import LlmAgent
        from google.adk.runners import InMemoryRunner
        from google.adk.tools import FunctionTool
        from google.genai import types

        async def search_capabilities(query: str) -> list[dict]:
            """Search the capabilities this tenant can use (owned + granted)."""
            granted = await marketplace_service.granted_capability_ids(self.session, self.identity.tenant_id)
            results = await registry_store.search(
                self.session, self.identity.tenant_id, query, granted_capability_ids=granted, limit=8
            )
            return [
                {
                    "capability_id": r.capability.id,
                    "name": r.capability.name,
                    "description": r.capability.description,
                    "tools": [t.name for t in r.capability.tools],
                    "required_scopes": r.capability.required_scopes,
                }
                for r in results
            ]

        agent = LlmAgent(
            name="planner_agent",
            model=settings.live_model,
            instruction=PLANNER_SYSTEM_INSTRUCTION,
            tools=[FunctionTool(search_capabilities)],
            output_schema=ExecutionPlan,
            output_key="execution_plan",
        )
        runner = InMemoryRunner(agent=agent, app_name="capability-marketplace")
        session_id = f"plan-session-{uuid.uuid4().hex[:8]}"
        await runner.session_service.create_session(
            app_name="capability-marketplace", user_id=self.identity.user_id, session_id=session_id
        )

        final_text: str | None = None
        async for event in runner.run_async(
            user_id=self.identity.user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=intent)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = event.content.parts[0].text

        if not final_text:
            raise RuntimeError("planner produced no final response")
        return ExecutionPlan.model_validate(json.loads(final_text))
