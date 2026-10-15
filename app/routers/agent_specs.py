"""The "chat with the agent, get the plan approved, save it as a spec" part
of the configuration plane. Two steps, matching the architecture doc's
reviewable-plan stance:

  POST /plans (existing)      -- intent -> typed ExecutionPlan, nothing runs
  POST /agent-specs/approve   -- a human signs off on that exact plan -> it
                                  graduates into a named, versioned AgentSpec,
                                  persisted in MongoDB (see app/specs/store.py)

Approving is deliberately a *separate* call from planning: the UI shows the
plan first (which capabilities it bound to, each step's instruction) so the
approval is of what was actually produced, not blind. Re-approving the same
plan (or a plan with the same `name`) bumps the version instead of creating
a parallel spec_id, since an AgentSpec is meant to be the one
current-and-versioned artifact for a given agent, not a pile of one-offs.

What "approve" does NOT do, by itself: make this spec runnable as a live
sub-agent. This endpoint only persists the document -- it has no way to
reach into the `adk web` supervisor's memory, since that's a separate
process (see `README.md`'s "two different clients" note). The supervisor
(`agents/capability_marketplace/agent.py`'s `root_agent`) is what turns an
approved spec into a callable sub-agent, by loading it itself -- once at its
own startup, and again whenever its `refresh_agent_spec_tools` tool is
called. See `app/factory/spec_agents.py` for the compile step.

What approving DOES capture, beyond the plan's bare `capability_id` strings:
a full snapshot of each referenced capability's structured schema (its
tools, kind, endpoint, everything in `Capability` -- see
`app/registry/schemas.py`), taken at this exact moment, into
`AgentSpec.capability_snapshots`. Without this, the saved spec would be
unreadable on its own -- just an id pointing at a row that lives in a
different store and can change or vanish later.

It also bakes the compiled agent's full instruction text (goal + a line per
step, dependencies spelled out) into `AgentSpec.instruction`, built once here
against that same snapshot (see `app/factory/spec_agents.py`'s
`build_spec_instruction`). `compile_spec_to_agent` just passes it straight
through rather than re-stitching it every time a process compiles this spec
-- so the saved document is the literal prompt the agent runs with, not just
the ingredients to reconstruct one.

The actual versioning/snapshotting/instruction-building logic lives in
`app/specs/service.py`'s `approve_plan`, shared with the supervisor's own
`approve_last_plan` tool (`agents/capability_marketplace/agent.py`) -- so a
plan can be approved either from here, over HTTP, or directly from a chat
with `adk web`'s single supervisor agent, without the two paths drifting
apart.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.auth.deps import get_identity
from app.auth.models import IdentityContext
from app.db import SessionLocal
from app.planner.schemas import ExecutionPlan
from app.specs import service as specs_service
from app.specs import store as specs_store
from app.specs.schemas import AgentRuntimeConfig, AgentSpec

router = APIRouter(prefix="/agent-specs", tags=["agent-specs"])


async def _db() -> AsyncSession:
    async with SessionLocal() as session:
        yield session


class ApproveRequest(BaseModel):
    plan: ExecutionPlan
    name: str
    guardrail_policy_id: str = "default"
    # Optional free text from whoever is approving -- tone, edge cases,
    # error-handling preferences, anything else they want the compiled
    # agent's instruction to reflect. Folded into the instruction-polishing
    # pass in app/specs/service.py, not stored separately -- see that
    # module's docstring.
    author_notes: str = ""
    # Agent pattern + configured callbacks -- e.g.
    #   {"pattern": "sequential",
    #    "callbacks": [{"hook": "before_tool", "kind": "validate_args_schema"},
    #                  {"hook": "before_model", "kind": "block_keywords",
    #                   "params": {"keywords": ["wire transfer"]}}]}
    # See GET /agent-specs/runtime-options for everything available.
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)


@router.get("/runtime-options")
async def runtime_options() -> dict:
    """Every agent pattern and every configurable callback kind per hook --
    what an ApproveRequest's `runtime` may name."""
    from app.factory import agent_patterns, callback_registry
    from app.specs.schemas import POLICY_FACTS

    return {
        "patterns": agent_patterns.describe_patterns(),
        "callbacks": callback_registry.catalog(),
        "tool_policy": {
            "facts": sorted(POLICY_FACTS) + ["args.<name>"],
            "ops": ["eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "contains", "matches", "exists"],
            "effects": ["allow", "require_approval", "deny"],
            "confidence_methods": ["combined", "signals", "self_reported"],
            "approval_channels": ["chat", "inbox"],
        },
    }


@router.post("/approve", response_model=AgentSpec)
async def approve_plan(
    body: ApproveRequest,
    identity: IdentityContext = Depends(get_identity),
    session: AsyncSession = Depends(_db),
) -> AgentSpec:
    try:
        return await specs_service.approve_plan(
            session,
            identity,
            name=body.name,
            plan=body.plan,
            guardrail_policy_id=body.guardrail_policy_id,
            author_notes=body.author_notes,
            runtime=body.runtime,
        )
    except ValueError as exc:
        # An invalid pattern/callback configuration for this plan -- the
        # request was well-formed, but can't be approved as-is.
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("", response_model=list[AgentSpec])
async def list_agent_specs(identity: IdentityContext = Depends(get_identity)) -> list[AgentSpec]:
    return await specs_store.list_specs(identity.tenant_id)


@router.get("/{spec_id}", response_model=AgentSpec)
async def get_agent_spec(spec_id: str, identity: IdentityContext = Depends(get_identity)) -> AgentSpec:
    spec = await specs_store.get_spec(spec_id)
    if spec is None or spec.tenant_id != identity.tenant_id:
        raise HTTPException(status_code=404, detail="agent spec not found")
    return spec
