"""FastAPI entrypoint. Wires telemetry, the database, and every router --
this is the "Phase 0-3 as working code" the architecture doc's repository
layout section describes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.telemetry.setup import configure_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_telemetry()
    await init_db()
    yield


app = FastAPI(
    title="Capability Marketplace",
    description="Minimal multi-tenant capability marketplace runtime on Google ADK",
    version="0.1.0",
    lifespan=lifespan,
)

from app.routers import agent_specs, approvals, auth, capabilities, knowledge, plans, runs  # noqa: E402
from app.marketplace import routes as marketplace_routes  # noqa: E402

app.include_router(auth.router)
app.include_router(capabilities.router)
app.include_router(plans.router)
app.include_router(runs.router)
app.include_router(marketplace_routes.router)
app.include_router(knowledge.router)
app.include_router(agent_specs.router)
app.include_router(approvals.router)

# The minimal configuration-plane UI (app/static/index.html): knowledge base,
# tool registration/search, and the chat -> plan -> approve -> AgentSpec
# flow, all just a thin client over the API routes above. `html=True` serves
# index.html at /ui/ itself, not only files under it.
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
