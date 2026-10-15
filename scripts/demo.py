"""End-to-end demo, no running server or external services required: issues
a dev token, registers an OpenAPI-derived capability, asks the planner for a
plan in plain English, executes it, and prints the streamed events. Runs the
FastAPI app in-process via an ASGI transport.

    python3 scripts/demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run directly: python3 scripts/demo.py

from httpx import ASGITransport, AsyncClient

import app.main as m
from app.factory import tool_wrapper
from app.runtime import redis_streams

DEMO_OPENAPI_SPEC = {
    "info": {"description": "a toy content-ingestion API"},
    "paths": {
        "/ingest": {
            "post": {
                "operationId": "ingest_content",
                "summary": "Ingest generated content into the pipeline",
            }
        }
    },
}


async def _fake_dispatch(capability, tool_name, args, *, bearer_token):
    """Stands in for actually calling the registered base_url -- this demo
    is about the orchestration (auth delegation, guardrails, telemetry,
    streaming), which all run for real below; it shouldn't also depend on a
    real external API being reachable from wherever this script runs."""
    return {"status": "ingested", "tool": tool_name, "args": args}


async def main() -> None:
    tool_wrapper._dispatch = _fake_dispatch
    transport = ASGITransport(app=m.app)
    async with AsyncClient(transport=transport, base_url="http://demo") as client, m.lifespan(m.app):
        print("1. Issuing a dev token for tenant 'acme'...")
        resp = await client.post(
            "/auth/dev-token", json={"tenant_id": "acme", "user_id": "demo-user", "scopes": ["*"]}
        )
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        print("2. Registering an OpenAPI-derived capability ('ingestion-tool')...")
        resp = await client.post(
            "/capabilities/openapi",
            json={
                "name": "ingestion-tool",
                "spec": DEMO_OPENAPI_SPEC,
                "base_url": "https://httpbin.org",
                "visibility": "private",
            },
            headers=headers,
        )
        capability = resp.json()
        print(f"   -> registered {capability['id']} with tools: {[t['name'] for t in capability['tools']]}")

        print("3. Asking the planner to turn an intent into a typed ExecutionPlan...")
        resp = await client.post(
            "/plans", json={"intent": "ingest content using my ingestion tool"}, headers=headers
        )
        plan = resp.json()
        print(f"   -> plan {plan['plan_id']} with {len(plan['steps'])} step(s):")
        print(json.dumps(plan, indent=2))

        print("4. Starting a run from that plan...")
        resp = await client.post("/runs", json={"plan": plan}, headers=headers)
        run_id = resp.json()["run_id"]
        print(f"   -> run {run_id} started, streaming events:")

        await asyncio.sleep(1.5)  # let the background execution task finish
        async for entry_id, event in redis_streams.read_events(run_id, consumer="demo-consumer", block_ms=500):
            print(f"   [{event.get('kind')}] {event}")
            await redis_streams.ack(run_id, entry_id)

        print("\nDone. Re-run with a Redis server running (`docker run -p 6379:6379 redis`)")
        print("to see the same flow backed by real durable streams instead of the in-process fallback.")


if __name__ == "__main__":
    asyncio.run(main())
