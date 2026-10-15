from __future__ import annotations

import json

import pytest

from app.config import settings
from app.marketplace import service as marketplace_service
from app.registry import store as registry_store
from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility


def _capability(tenant_id: str, name: str, visibility: Visibility = Visibility.PRIVATE) -> Capability:
    return Capability(
        id=f"cap_{name}",
        tenant_id=tenant_id,
        kind=CapabilityKind.OPENAPI,
        name=name,
        description=f"{name} description with video generation keywords",
        tools=[ToolSchema(name="run", description="run it")],
        visibility=visibility,
        endpoint_ref=EndpointRef(openapi_base_url="http://example.test"),
    )


@pytest.mark.asyncio
async def test_search_only_returns_own_and_granted_capabilities(session):
    mine = _capability("acme", "video-gen", Visibility.PRIVATE)
    others_private = _capability("globex", "secret-tool", Visibility.PRIVATE)
    others_marketplace = _capability("globex", "ingestion-tool", Visibility.MARKETPLACE)

    for cap in (mine, others_private, others_marketplace):
        await registry_store.save(session, cap)
    await session.commit()

    # Not granted yet: acme should see its own + nothing from globex at all --
    # `search` is the authorization-scoped, planner/execution-facing function.
    results = await registry_store.search(session, "acme", "video")
    assert {r.capability.id for r in results} == {"cap_video-gen"}

    results = await registry_store.search(session, "acme", "ingestion")
    assert results == []  # marketplace-visible, but acme has no grant yet

    results = await registry_store.search(session, "acme", "secret")
    assert results == []

    # `discover_marketplace` is the separate, wider function for browsing
    # what's requestable -- it must surface globex's marketplace capability
    # even with no grant (how else would acme know to request one?), and
    # must never surface globex's still-private one.
    discovered = await registry_store.discover_marketplace(session, "ingestion")
    assert {r.capability.id for r in discovered} == {"cap_ingestion-tool"}
    assert all(r.capability.id != "cap_secret-tool" for r in await registry_store.discover_marketplace(session, "secret"))


@pytest.mark.asyncio
async def test_grant_workflow_approve_then_visible(session):
    cap = _capability("globex", "shared-kb", Visibility.MARKETPLACE)
    await registry_store.save(session, cap)
    await session.commit()

    granted_before = await marketplace_service.granted_capability_ids(session, "acme")
    assert granted_before == set()

    grant = await marketplace_service.request_grant(
        session, capability_id=cap.id, grantor_tenant_id="globex", grantee_tenant_id="acme"
    )
    await session.commit()
    assert grant.status == "pending"

    await marketplace_service.decide_grant(session, grant.id, approve=True)
    await session.commit()

    granted_after = await marketplace_service.granted_capability_ids(session, "acme")
    assert granted_after == {cap.id}


@pytest.mark.asyncio
async def test_save_mirrors_capability_to_readable_json_file(session):
    """`registry_store.save` writes the SQL row AND a plain JSON mirror
    (`settings.capabilities_json_path`) -- this is what lets someone read the
    registry's full structured schema off disk directly, the same way
    `app/specs/store.py` does for AgentSpecs. The DB row stays the one thing
    ever read back by the app; this just checks the mirror file itself ends
    up holding the full capability, keyed by id."""
    cap = _capability("acme", "mirrored-tool", Visibility.PRIVATE)
    await registry_store.save(session, cap)
    await session.commit()

    mirror_path = registry_store._mirror_path()
    assert mirror_path.exists()
    data = json.loads(mirror_path.read_text())
    assert data[cap.id]["name"] == "mirrored-tool"
    assert data[cap.id]["tools"][0]["name"] == "run"

    # A second capability merges in rather than clobbering the first.
    other = _capability("acme", "second-tool", Visibility.PRIVATE)
    await registry_store.save(session, other)
    await session.commit()
    data = json.loads(mirror_path.read_text())
    assert set(data.keys()) == {cap.id, other.id}
