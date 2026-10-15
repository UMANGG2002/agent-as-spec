"""`spec_test_agent` -- a second `adk web` app (next to
`capability_marketplace` in the dropdown) holding EVERY approved AgentSpec
for the demo tenant, each compiled exactly as approved (its own pattern and
callbacks) and mounted as a real ADK sub-agent under one router.

Ask for something a spec does, or name it, and the router transfers the
conversation to that spec; it runs its approved steps, and the next message
comes back to the router. See `app/factory/spec_router.py` for how the tree
is built and why (sub-agents rather than AgentTools; why each spec is
isolated from transferring elsewhere; why in-place reload is safe).

Two router tools:
  list_loaded_specs  -- what's mounted, with pattern, steps and callbacks,
                        plus any spec that failed to compile and why
  reload_specs       -- recompile from the store and remount, so a spec
                        approved after `adk web` started shows up without a
                        restart

Filter which specs get mounted (optional; default is all of them):

  TEST_SPECS="order-desk,spec_ab12cd34ef56"   names and/or spec_ids, comma-separated
  TEST_SPEC_NAME=... / TEST_SPEC_ID=...       older single-spec forms, still honored

    TEST_SPECS=order-desk uv run adk web agents

Loading happens once at import (ADK's agent loader needs `root_agent` to be
a real agent object, not something awaitable), on a dedicated thread with
its own event loop -- `adk web` imports this module from inside its running
loop, where `asyncio.run` would refuse. That thread uses its *own*
short-lived database engine and Mongo client and closes both before
returning: async DB connections are tied to the event loop that opened
them, so sharing the app-wide ones from a loop that's about to close would
leave them broken for every later request in this process -- including the
supervisor's, which shares them.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path

# `adk web`/`adk run` only add `agents/` to sys.path, never the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from app.auth.models import IdentityContext
from app.config import settings
from app.factory import spec_router
from app.marketplace import service as marketplace_service
from app.registry import store as registry_store
from app.specs import store as specs_store

TEST_IDENTITY = IdentityContext(tenant_id="acme", brand_id="acme-brand", user_id="spec-test-agent", scopes=("*",))

# Mutable state the router instruction reads on every model call.
_state: dict = {"loaded": {}, "skipped": {}}

# Tests swap in a scripted model; None means settings.live_model.
_model_override = None


def _only() -> set[str]:
    raw = os.environ.get("TEST_SPECS", "")
    names = {x.strip() for x in raw.split(",") if x.strip()}
    for legacy in ("TEST_SPEC_ID", "TEST_SPEC_NAME"):
        if os.environ.get(legacy):
            names.add(os.environ[legacy].strip())
    return names


async def _build(session):  # noqa: ANN001
    """Compile every (filtered) approved spec against this tenant's
    currently usable capabilities."""
    granted = await marketplace_service.granted_capability_ids(session, TEST_IDENTITY.tenant_id)
    visible = {
        c.id: c
        for c in await registry_store.list_for_tenant(session, TEST_IDENTITY.tenant_id, granted_capability_ids=granted)
    }
    specs = spec_router.filter_specs(await specs_store.list_specs(TEST_IDENTITY.tenant_id), _only())
    return spec_router.compile_sub_agents(specs, visible, TEST_IDENTITY, model=_model_override)


async def _import_time_build():
    """`_build`, with its own engine and Mongo client, both closed before
    this thread's event loop goes away (see the module docstring)."""
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

    from app.db import Base

    saved_mongo_client = specs_store._client
    specs_store._client = None
    engine = create_async_engine(settings.database_url)
    try:
        from app.knowledge import store as _knowledge_store  # noqa: F401 - register tables
        from app.marketplace import models as _marketplace_models  # noqa: F401
        from app.registry import store as _registry_store  # noqa: F401
        from app.runtime import approvals as _approvals  # noqa: F401

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)() as session:
            return await _build(session)
    finally:
        await engine.dispose()
        ours = specs_store._client
        if ours is not None and ours is not saved_mongo_client:
            ours.close()
        specs_store._client = saved_mongo_client


def _run_sync(coro):
    result: dict = {}

    def _target() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            result["error"] = exc

    thread = threading.Thread(target=_target, name="spec-test-agent-load")
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


async def list_loaded_specs() -> dict:
    """List every approved spec mounted as a sub-agent of this router --
    its sub-agent name (what to transfer to), version, execution pattern,
    steps and callbacks -- plus any spec that failed to compile, and why."""
    return {"loaded": list(_state["loaded"].values()), "skipped": dict(_state["skipped"])}


async def reload_specs() -> dict:
    """Recompile every approved spec from the store and remount them as
    this router's sub-agents -- use after approving or changing a spec, so
    it's available without restarting adk web."""
    from app.db import session_scope

    async with session_scope() as session:
        agents, loaded, skipped = await _build(session)
    spec_router.mount_sub_agents(root_agent, agents)
    _state["loaded"], _state["skipped"] = loaded, skipped
    return {"mounted": [info["agent_name"] for info in loaded.values()], "skipped": skipped}


try:
    _agents, _state["loaded"], _state["skipped"] = _run_sync(_import_time_build())
except Exception as exc:  # noqa: BLE001 - a broken store must not crash adk web's app discovery
    _agents = []
    _state["skipped"] = {"(load)": f"could not load specs at startup: {exc}"}

root_agent = LlmAgent(
    name=spec_router.ROUTER_NAME,
    model=settings.live_model,
    description="Routes to every approved agent spec, each mounted as a sub-agent.",
    instruction=spec_router.RouterInstruction(_state),
    tools=[FunctionTool(list_loaded_specs), FunctionTool(reload_specs)],
    sub_agents=_agents,
)
