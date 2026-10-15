"""Run the test worker in the foreground: 22 MCP servers and 20 A2A agents
(see app/testing/worker_catalog.py), each at its own URL on one port.

    uv run python scripts/run_worker.py                     # serve + register all 42, publicly
    uv run python scripts/run_worker.py --port 9200
    uv run python scripts/run_worker.py --visibility private  # register for the owner tenant only
    uv run python scripts/run_worker.py --no-register         # just serve

Open http://127.0.0.1:9100/ for the manifest: every MCP server's /sse URL
with its tools, input schemas and a worked example, and every A2A agent's
card URL and skills. Point any MCP client at an /sse URL, or any A2A client
at an agent card.

On startup it registers every server and agent in this project's capability
registry (the DATABASE_URL in your .env -- the same one the FastAPI app and
the `adk web` supervisor use) -- publicly by default: marketplace
visibility, so every tenant can discover them and request a grant, while
the owner (`--tenant`) can use them straight away. It goes through the real
onboarding adapters, with stable ids
-- `cap_worker_mcp_<server>` / `cap_worker_a2a_<agent>` -- so re-running it
after a restart (new port included) updates those rows rather than adding
duplicates. After that, the supervisor's list_capabilities / make_plan see
all 42, and approved specs can bind steps to any of them. Keep this worker
running while those specs run: the registry stores *where* each tool lives,
the worker is what answers there.

`--host 0.0.0.0` exposes it beyond localhost; there is no auth on the
worker itself, so only do that on a network you trust.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run directly: python scripts/run_worker.py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument(
        "--no-register", dest="register", action="store_false",
        help="only serve; don't register the servers/agents in the capability registry",
    )
    parser.add_argument(
        "--visibility", choices=["marketplace", "org", "private"], default="marketplace",
        help="registry visibility for every registered capability (default: marketplace = public)",
    )
    parser.add_argument("--tenant", default="acme", help="owning tenant for registered capabilities (default: acme, the adk web demo tenant)")
    parser.add_argument(
        "--public-url",
        default=None,
        help="base URL to register, if clients reach this worker at a different address than --host/--port",
    )
    args = parser.parse_args()

    import httpx
    import uvicorn

    from app.testing import worker

    base_url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    server = uvicorn.Server(uvicorn.Config(worker.build_app(), host=args.host, port=args.port, log_level="info"))
    thread = threading.Thread(target=server.run, name="worker", daemon=True)
    thread.start()
    worker._wait_healthy(base_url)

    manifest = httpx.get(f"{base_url}/").json()
    counts = manifest["counts"]
    print(f"\nworker up at {base_url}  (manifest: {base_url}/)")
    print(
        f"  {counts['mcp_servers']} MCP servers / {counts['mcp_tools']} tools, "
        f"{counts['a2a_agents']} A2A agents / {counts['a2a_skills']} skills\n"
    )
    print("MCP servers (SSE):")
    for s in manifest["mcp_servers"]:
        print(f"  {s['sse_url']:<62} {', '.join(t['name'] for t in s['tools'])}")
    print("\nA2A agents (agent cards):")
    for a in manifest["a2a_agents"]:
        print(f"  {a['agent_card_url']:<80} {', '.join(sk['id'] for sk in a['skills'])}")

    if args.register:
        registered_url = (args.public_url or base_url).rstrip("/")

        async def _register() -> int:
            from app.db import init_db, session_scope
            from app.registry.schemas import Visibility

            await init_db()
            async with session_scope() as session:
                saved = await worker.register_all(
                    session, args.tenant, registered_url, visibility=Visibility(args.visibility)
                )
                return len(saved)

        count = asyncio.run(_register())
        print(
            f"\nregistered {count} capabilities ({args.visibility}) owned by tenant '{args.tenant}' "
            f"at {registered_url}"
        )

    print("\nCtrl+C to stop.")
    try:
        thread.join()
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
