from __future__ import annotations

import pytest

from app.auth.models import IdentityContext
from app.planner.planner_agent import create_plan
from app.planner.schemas import StepKind
from app.registry import store as registry_store
from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility


@pytest.mark.asyncio
async def test_offline_planner_binds_only_visible_capabilities(session):
    video_gen = Capability(
        id="cap_video_gen",
        tenant_id="acme",
        kind=CapabilityKind.MCP,
        name="video-gen",
        description="generates videos from a text prompt",
        tools=[ToolSchema(name="generate_video")],
        visibility=Visibility.PRIVATE,
        endpoint_ref=EndpointRef(stdio_command=["video-gen-mcp"]),
    )
    ingestion = Capability(
        id="cap_ingest",
        tenant_id="acme",
        kind=CapabilityKind.OPENAPI,
        name="ingestion-tool",
        description="ingests generated content into the content pipeline",
        tools=[ToolSchema(name="ingest")],
        visibility=Visibility.PRIVATE,
        endpoint_ref=EndpointRef(openapi_base_url="http://ingest.test"),
    )
    await registry_store.save(session, video_gen)
    await registry_store.save(session, ingestion)
    await session.commit()

    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))
    plan = await create_plan(session, identity, "use my video-gen MCP with my ingestion tool to create one capability")

    assert plan.goal
    assert {s.capability_id for s in plan.steps} <= {"cap_video_gen", "cap_ingest"}
    assert all(s.kind == StepKind.TOOL_CALL for s in plan.steps)
    # step_order() must not raise -- validates the dependency graph is acyclic
    layers = plan.step_order()
    assert sum(len(layer) for layer in layers) == len(plan.steps)


@pytest.mark.asyncio
async def test_offline_planner_empty_result_when_nothing_matches(session):
    identity = IdentityContext(tenant_id="acme", brand_id="default", user_id="u1", scopes=("*",))
    plan = await create_plan(session, identity, "do something nobody registered a capability for")
    assert plan.steps == []
