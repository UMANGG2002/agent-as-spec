"""FastAPI entrypoint (the `api` process role).

Stateless by design -- every replica is interchangeable behind the load
balancer: no in-process session state, no sticky sessions. Agent execution
happens in runner workers (app/sessions/runner.py) when RUNNER_MODE=queue;
event streams, rate limits, leases and caches live in Redis; everything
durable lives in Postgres.

Probes: `/live` (process is up), `/ready` (Postgres + Redis reachable --
take the pod out of rotation otherwise), `/health` (kept for compatibility).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.logging import configure_logging
from app.core.middleware import EdgeMiddleware
from app.db import init_db
from app.telemetry.setup import configure_telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level, json=settings.log_json)
    configure_telemetry()
    await init_db()
    from app.core import redis as redis_core

    await redis_core.get_redis()
    if settings.runner_mode == "queue":
        from app.sessions import queue

        await queue.ensure_groups()
    yield
    from app.core import http as http_core
    from app.registry.adapters import mcp_pool

    await mcp_pool.close_all()
    await http_core.close()
    await redis_core.close()


app = FastAPI(
    title="Agent-as-Spec",
    description="Multi-tenant capability marketplace and agent runtime on Google ADK",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

if settings.cors_allow_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_allow_origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["Authorization",
                       "Content-Type", "Idempotency-Key", "Last-Event-ID", "X-Request-ID"])
app.add_middleware(EdgeMiddleware)

from app.routers import agent_specs, approvals, auth, capabilities, evals, knowledge, plans, runs, sessions  # noqa: E402
from app.marketplace import routes as marketplace_routes  # noqa: E402

app.include_router(auth.router)
app.include_router(capabilities.router)
app.include_router(plans.router)
app.include_router(runs.router)
app.include_router(marketplace_routes.router)
app.include_router(knowledge.router)
app.include_router(agent_specs.router)
app.include_router(approvals.router)
app.include_router(evals.router)
app.include_router(sessions.router)
app.include_router(sessions.agents_router)

app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


@app.get("/health")
@app.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    from app import db
    from app.core import redis as redis_core

    checks = {"database": await db.ping(), "redis": await redis_core.ping()}
    ok = all(checks.values())
    return JSONResponse({"status": "ready" if ok else "not_ready", **checks}, status_code=200 if ok else 503)
