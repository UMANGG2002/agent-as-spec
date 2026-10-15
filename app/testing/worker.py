"""The test worker: one process, one port, serving every MCP server and A2A
agent in `app/testing/worker_catalog.py` at its own URL -- real protocols,
real network round trips -- so compiled multi-tool agents can be exercised
end to end against many diverse, independently addressable tools.

URL layout (all relative to the worker's base URL, e.g. http://127.0.0.1:9100):

  GET  /                                        manifest: every server and agent,
                                                with its URLs, tools/skills,
                                                input schemas and worked examples
  GET  /healthz                                 liveness
  GET  /mcp/{server}/sse                        MCP over SSE (one FastMCP server
  POST /mcp/{server}/messages/                  per catalog entry) -- point ADK's
                                                SseConnectionParams / any MCP
                                                client at the /sse URL
  GET  /a2a/{agent}/.well-known/agent-card.json the agent's A2A card
  POST /a2a/{agent}                             A2A JSON-RPC `message/send`

The MCP servers are native FastMCP servers built from the catalog's typed
Python functions (FastMCP derives each tool's JSON input schema from the
function signature). They share one Starlette app via `Mount`, which works
because the parent app's lifespan enters every child app's lifespan --
Starlette doesn't run a mounted app's lifespan on its own.

The A2A agents are deterministic text-in/text-out handlers behind a minimal,
spec-shaped A2A surface: a card (name, description, url, version, skills,
capabilities, input/output modes, preferred JSONRPC transport) and a
`message/send` method returning an A2A `Message` (kind "message", role
"agent", text parts). That's exactly what
`app/registry/adapters/agent_card_adapter.py` speaks. They're deterministic
stand-ins on purpose -- a regression suite can't assert on a live model's
output -- but a real ADK agent served with ADK's own A2A support is
addressed the same way.

`register_all` puts every server and agent into the capability registry --
publicly (marketplace visibility) by default -- through the project's real
onboarding paths (`mcp_adapter.introspect_and_register`
over the network for MCP, `agent_card_adapter.fetch_agent_card` for A2A),
with stable ids (`cap_worker_mcp_<server>`, `cap_worker_a2a_<agent>`) so
re-registering after a restart updates the same rows -- new port and all --
instead of piling up duplicates.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from app.testing.worker_catalog import A2A_AGENTS, MCP_SERVERS, A2AAgentDef

A2A_PROTOCOL_VERSION = "0.3.0"


def mcp_capability_id(server_name: str) -> str:
    return f"cap_worker_mcp_{server_name.replace('-', '_')}"


def a2a_capability_id(agent_name: str) -> str:
    return f"cap_worker_a2a_{agent_name.replace('-', '_')}"


def _build_mcp_servers() -> dict[str, Any]:
    from fastmcp import FastMCP

    servers = {}
    for server_def in MCP_SERVERS:
        server = FastMCP(server_def.name, instructions=server_def.description)
        for tool_def in server_def.tools:
            server.tool(tool_def.fn)
        servers[server_def.name] = server
    return servers


def _agent_card(agent: A2AAgentDef, base_url: str) -> dict:
    return {
        "protocolVersion": A2A_PROTOCOL_VERSION,
        "name": agent.name,
        "description": agent.description,
        "url": f"{base_url}/a2a/{agent.name}",
        "preferredTransport": "JSONRPC",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "tags": [agent.name],
                "examples": [skill.example_input],
            }
            for skill in agent.skills
        ],
    }


def _rpc_error(rpc_id: Any, code: int, message: str, status: int = 200) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}, status_code=status)


def build_app() -> Starlette:
    mcp_servers = _build_mcp_servers()
    mcp_apps = {name: server.http_app(transport="sse") for name, server in mcp_servers.items()}
    agents = {agent.name: agent for agent in A2A_AGENTS}

    def _base_url(request: Request) -> str:
        return str(request.base_url).rstrip("/")

    async def manifest(request: Request) -> JSONResponse:
        base = _base_url(request)
        servers_out = []
        for server_def in MCP_SERVERS:
            listed = {t.name: t for t in await mcp_servers[server_def.name].list_tools()}
            servers_out.append(
                {
                    "name": server_def.name,
                    "description": server_def.description,
                    "capability_id": mcp_capability_id(server_def.name),
                    "sse_url": f"{base}/mcp/{server_def.name}/sse",
                    "tools": [
                        {
                            "name": tool_def.name,
                            "description": (tool_def.fn.__doc__ or "").strip(),
                            "input_schema": getattr(listed.get(tool_def.name), "parameters", None),
                            "example": {"arguments": tool_def.example_args, "expected": tool_def.expected},
                        }
                        for tool_def in server_def.tools
                    ],
                }
            )
        agents_out = [
            {
                "name": agent.name,
                "description": agent.description,
                "capability_id": a2a_capability_id(agent.name),
                "agent_card_url": f"{base}/a2a/{agent.name}/.well-known/agent-card.json",
                "rpc_url": f"{base}/a2a/{agent.name}",
                "skills": [
                    {
                        "id": skill.id,
                        "description": skill.description,
                        "example": {"input": skill.example_input, "expected": skill.expected},
                    }
                    for skill in agent.skills
                ],
            }
            for agent in A2A_AGENTS
        ]
        return JSONResponse(
            {
                "worker": "capability-marketplace-test-worker",
                "base_url": base,
                "counts": {
                    "mcp_servers": len(servers_out),
                    "mcp_tools": sum(len(s["tools"]) for s in servers_out),
                    "a2a_agents": len(agents_out),
                    "a2a_skills": sum(len(a["skills"]) for a in agents_out),
                },
                "mcp_servers": servers_out,
                "a2a_agents": agents_out,
            }
        )

    async def healthz(request: Request) -> JSONResponse:  # noqa: ARG001
        return JSONResponse({"ok": True})

    async def agent_card(request: Request) -> JSONResponse:
        agent = agents.get(request.path_params["agent"])
        if agent is None:
            return JSONResponse({"error": "unknown agent"}, status_code=404)
        return JSONResponse(_agent_card(agent, _base_url(request)))

    async def agent_rpc(request: Request) -> JSONResponse:
        agent = agents.get(request.path_params["agent"])
        if agent is None:
            return JSONResponse({"error": "unknown agent"}, status_code=404)
        try:
            rpc = await request.json()
        except ValueError:
            return _rpc_error(None, -32700, "parse error", status=400)
        rpc_id = rpc.get("id")
        if rpc.get("method") != "message/send":
            return _rpc_error(rpc_id, -32601, f"method not found: {rpc.get('method')!r} (only message/send is supported)")
        message = (rpc.get("params") or {}).get("message") or {}
        text = " ".join(p.get("text", "") for p in message.get("parts", []) if p.get("kind") == "text")
        skill_id = (message.get("metadata") or {}).get("skill_id")
        if skill_id is None and len(agent.skills) == 1:
            skill_id = agent.skills[0].id
        skill = next((s for s in agent.skills if s.id == skill_id), None)
        if skill is None:
            available = [s.id for s in agent.skills]
            return _rpc_error(rpc_id, -32602, f"unknown skill {skill_id!r} for {agent.name}; available: {available}")
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "kind": "message",
                    "messageId": uuid.uuid4().hex,
                    "role": "agent",
                    "parts": [{"kind": "text", "text": skill.handler(text)}],
                    "metadata": {"skill_id": skill.id, "agent": agent.name},
                },
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):  # noqa: ARG001
        async with contextlib.AsyncExitStack() as stack:
            for sub_app in mcp_apps.values():
                await stack.enter_async_context(sub_app.router.lifespan_context(sub_app))
            yield

    routes = [
        Route("/", manifest, methods=["GET"]),
        Route("/healthz", healthz, methods=["GET"]),
        Route("/a2a/{agent}/.well-known/agent-card.json", agent_card, methods=["GET"]),
        Route("/a2a/{agent}", agent_rpc, methods=["POST"]),
    ] + [Mount(f"/mcp/{name}", app=sub_app) for name, sub_app in mcp_apps.items()]
    return Starlette(routes=routes, lifespan=lifespan)


# -- running it ----------------------------------------------------------------

_thread: threading.Thread | None = None
_base_url_running: str | None = None


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def _wait_healthy(base_url: str, timeout: float = 15.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/healthz", timeout=1.0).status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"worker at {base_url} did not become healthy within {timeout}s ({last_error})")


def ensure_running(host: str = "127.0.0.1", port: int = 0) -> str:
    """Start the worker in a background thread (once per process) and return
    its base URL, after it actually answers /healthz -- polled rather than a
    fixed sleep, since starting 22 MCP servers' lifespans takes a moment.
    Used by the regression tests; `scripts/run_worker.py` runs it in the
    foreground instead."""
    global _thread, _base_url_running
    if _thread is not None:
        return _base_url_running  # type: ignore[return-value]

    import uvicorn

    port = port or _free_port(host)
    base_url = f"http://{host}:{port}"
    app = build_app()
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    _thread = threading.Thread(target=server.run, name="capability-marketplace-worker", daemon=True)
    _thread.start()
    _wait_healthy(base_url)
    _base_url_running = base_url
    return base_url


# -- registering it ---------------------------------------------------------------


async def register_all(session, tenant_id: str, base_url: str, *, visibility=None) -> list:  # noqa: ANN001
    """Register every MCP server and A2A agent a running worker serves as a
    capability, via the same onboarding adapters a real capability goes
    through (`mcp_adapter.introspect_and_register` lists each server's tools
    and input schemas over the wire; `agent_card_adapter` fetches each
    card). Returns the saved capabilities. The caller commits.

    Driven by the worker's own manifest (`GET {base_url}/`), not this
    module's catalog, so it registers exactly what *that* worker serves --
    including a worker on another machine running a different catalog.

    Public by default: `Visibility.MARKETPLACE`, so every tenant can find
    them through marketplace discovery (`registry_store.discover_marketplace`,
    `GET /marketplace/discover`) and request a grant -- not just the owning
    tenant. The owner (`tenant_id`) can plan against them immediately; any
    other tenant still needs an approved grant before a plan may bind to
    one, same as any marketplace capability (see `list_for_tenant`).
    Pass `visibility=Visibility.PRIVATE` to keep them to the owner."""
    import httpx

    from app.registry import store as registry_store
    from app.registry.adapters import agent_card_adapter, mcp_adapter
    from app.registry.schemas import Visibility

    visibility = visibility or Visibility.MARKETPLACE
    base_url = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(f"{base_url}/")
        resp.raise_for_status()
        manifest = resp.json()
    if manifest.get("worker") != "capability-marketplace-test-worker":
        raise ValueError(f"{base_url} does not look like a capability-marketplace test worker (no worker manifest at /)")

    saved = []
    for server in manifest["mcp_servers"]:
        capability = await mcp_adapter.introspect_and_register(
            tenant_id=tenant_id,
            name=server["name"],
            description=server["description"],
            sse_url=f"{base_url}/mcp/{server['name']}/sse",
            visibility=visibility,
        )
        capability.id = mcp_capability_id(server["name"])
        await registry_store.save(session, capability)
        saved.append(capability)
    for agent in manifest["a2a_agents"]:
        card = await agent_card_adapter.fetch_agent_card(f"{base_url}/a2a/{agent['name']}")
        capability = agent_card_adapter.capability_from_agent_card(card, tenant_id=tenant_id, visibility=visibility)
        capability.id = a2a_capability_id(agent["name"])
        await registry_store.save(session, capability)
        saved.append(capability)
    return saved
