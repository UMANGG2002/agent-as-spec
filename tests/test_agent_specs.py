"""AgentSpec persistence -- exercises the real `app/specs/store.py` path,
which tries MongoDB first and falls back to an in-process dict when nothing
is listening on `mongo_url` (exactly like `runtime/redis_streams.py`'s own
fallback), so this test passes the same way whether or not a real Mongo is
running wherever it's executed."""

from __future__ import annotations

import pytest

from app.auth.models import IdentityContext
from app.planner.schemas import ExecutionPlan, PlanStep, StepKind
from app.registry import store as registry_store
from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility
from app.routers.agent_specs import ApproveRequest, approve_plan
from app.specs import store as specs_store
from app.specs.schemas import AgentSpec


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_test123",
        goal="ingest content using my ingestion tool",
        steps=[
            PlanStep(
                step_id="step_1",
                kind=StepKind.TOOL_CALL,
                capability_id="cap_demo",
                instruction="Use the ingestion tool to ingest content",
            )
        ],
    )


@pytest.mark.asyncio
async def test_save_and_get_spec_roundtrips():
    plan = _plan()
    spec = AgentSpec(
        spec_id="spec_abc123",
        tenant_id="acme",
        version=1,
        name="ingestion-agent",
        goal=plan.goal,
        source_plan_id=plan.plan_id,
        steps=plan.steps,
        approved_by="demo-user",
    )
    saved = await specs_store.save_spec(spec)
    assert saved.spec_id == "spec_abc123"

    fetched = await specs_store.get_spec("spec_abc123", tenant_id="acme")
    assert fetched is not None
    assert fetched.name == "ingestion-agent"
    assert fetched.steps[0].capability_id == "cap_demo"
    assert fetched.tenant_id == "acme"


@pytest.mark.asyncio
async def test_list_specs_is_scoped_per_tenant():
    plan = _plan()
    for tenant, name in [("acme", "acme-agent"), ("globex", "globex-agent")]:
        await specs_store.save_spec(
            AgentSpec(
                spec_id=f"spec_{tenant}",
                tenant_id=tenant,
                name=name,
                goal=plan.goal,
                source_plan_id=plan.plan_id,
                steps=plan.steps,
                approved_by="demo-user",
            )
        )

    acme_specs = await specs_store.list_specs("acme")
    assert {s.name for s in acme_specs} >= {"acme-agent"}
    assert all(s.tenant_id == "acme" for s in acme_specs)


@pytest.mark.asyncio
async def test_reapproving_same_spec_id_bumps_version_in_place():
    plan = _plan()
    base = dict(
        spec_id="spec_versioned",
        tenant_id="acme",
        name="versioned-agent",
        goal=plan.goal,
        source_plan_id=plan.plan_id,
        steps=plan.steps,
        approved_by="demo-user",
    )
    await specs_store.save_spec(AgentSpec(version=1, **base))
    await specs_store.save_spec(AgentSpec(version=2, **base))

    fetched = await specs_store.get_spec("spec_versioned", tenant_id="acme")
    assert fetched.version == 2  # replaced in place, not duplicated


@pytest.mark.asyncio
async def test_approve_plan_snapshots_the_bound_capabilities(session):
    """`POST /agent-specs/approve` (here called directly as `approve_plan`,
    the same way other tests call service functions directly rather than
    going through HTTP) must do more than copy `capability_id` strings onto
    the spec -- it should resolve each one against the live registry and
    embed the full `Capability` into `AgentSpec.capability_snapshots`, so the
    saved spec is self-describing without cross-referencing the registry DB."""
    cap = Capability(
        id="cap_ingest",
        tenant_id="acme",
        kind=CapabilityKind.OPENAPI,
        name="ingestion-tool",
        description="ingest content",
        tools=[ToolSchema(name="run", description="run it")],
        visibility=Visibility.PRIVATE,
        endpoint_ref=EndpointRef(openapi_base_url="http://example.test"),
    )
    await registry_store.save(session, cap)
    await session.commit()

    plan = ExecutionPlan(
        plan_id="plan_snap1",
        goal="ingest content using my ingestion tool",
        steps=[
            PlanStep(
                step_id="step_1",
                kind=StepKind.TOOL_CALL,
                capability_id="cap_ingest",
                instruction="Use the ingestion tool to ingest content",
            ),
            PlanStep(
                step_id="step_2",
                kind=StepKind.HUMAN_APPROVAL,
                capability_id="",
                instruction="Have a human sign off before sending",
            ),
        ],
    )
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))

    spec = await approve_plan(
        ApproveRequest(plan=plan, name="ingestion-agent-snap"),
        identity=identity,
        session=session,
    )

    assert set(spec.capability_snapshots.keys()) == {"cap_ingest"}
    snapshot = spec.capability_snapshots["cap_ingest"]
    assert snapshot.name == "ingestion-tool"
    assert snapshot.tools[0].name == "run"

    # The compiled agent's full instruction text is baked in at approval
    # time too, not left for compile_spec_to_agent to re-stitch later --
    # every step gets a line, including the human_approval one (it just has
    # no capability to label, since build_spec_instruction lists all steps
    # unconditionally, the same as the pre-snapshot inline version did).
    assert "ingest content using my ingestion tool" in spec.instruction
    assert "step_1: ingestion-tool -- Use the ingestion tool to ingest content" in spec.instruction
    assert "step_2" in spec.instruction
    assert "Have a human sign off before sending" in spec.instruction

    # Re-fetching from the spec store (not just the in-memory return value)
    # confirms the snapshot and instruction actually round-trip through
    # persistence.
    fetched = await specs_store.get_spec(spec.spec_id, tenant_id=spec.tenant_id)
    assert fetched is not None
    assert fetched.capability_snapshots["cap_ingest"].name == "ingestion-tool"
    assert fetched.instruction == spec.instruction


@pytest.mark.asyncio
async def test_approve_plan_handles_missing_capability_gracefully(session):
    """A step can reference a capability_id that no longer resolves (deleted,
    or never registered under this call's session) -- approval shouldn't
    crash, it should just omit that id from the snapshot dict."""
    plan = ExecutionPlan(
        plan_id="plan_snap2",
        goal="use a capability that doesn't exist",
        steps=[
            PlanStep(
                step_id="step_1",
                kind=StepKind.TOOL_CALL,
                capability_id="cap_does_not_exist",
                instruction="try anyway",
            )
        ],
    )
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))

    spec = await approve_plan(
        ApproveRequest(plan=plan, name="missing-cap-agent"),
        identity=identity,
        session=session,
    )

    assert spec.capability_snapshots == {}
