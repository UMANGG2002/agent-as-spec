"""A tiny local MCP server, generated from a toy OpenAPI spec via FastMCP
(`app/registry/adapters/openapi_adapter.py`'s `build_mcp_app`) and run as a
background ASGI app -- so the `CapabilityKind.MCP` dispatch branch in
`app/factory/tool_wrapper.py` (via ADK's `McpToolset` + `SseConnectionParams`)
can be exercised against a *genuine* MCP server speaking the real wire
protocol over SSE, not a stub.

This server is a thin protocol translator in front of `echo_server`'s plain
REST API -- exactly the "OpenAPI -> MCP" shape `build_mcp_app`'s own
docstring describes: an MCP client (an `McpToolset`-backed agent, or any
other MCP client) speaks MCP to this server, which forwards each tool call
over plain HTTP to `echo_server`, same backend the OPENAPI-kind demo
capability already points at directly. Two tools are exposed
(`summarize_text`, `translate_text`) so a multi-tool regression test has more
than one MCP tool to pick from, not just a single trivial one.

Mirrors `echo_server.py`'s shape -- idempotent `ensure_running()`, started in
a background thread on first call, ephemeral port -- but runs a Starlette
ASGI app through uvicorn instead of `http.server`, since FastMCP's server is
an ASGI app, not a raw socket handler."""

from __future__ import annotations

import threading
import time

import uuid

from app.registry.adapters import openapi_adapter
from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility
from app.testing import echo_server

TOOL_NAMES = ["summarize_text", "translate_text"]

_TOOL_SPEC = {
    "openapi": "3.0.0",
    "info": {
        "title": "demo-multi-tool-api",
        "version": "1.0.0",
        "description": "a toy multi-tool API, exposed over MCP for end-to-end testing",
    },
    "paths": {
        "/summarize_text": {
            "post": {
                "operationId": "summarize_text",
                "summary": "Summarize a piece of text",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"instruction": {"type": "string"}},
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "ok"}},
            }
        },
        "/translate_text": {
            "post": {
                "operationId": "translate_text",
                "summary": "Translate a piece of text",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {"instruction": {"type": "string"}},
                            }
                        }
                    }
                },
                "responses": {"200": {"description": "ok"}},
            }
        },
    },
}

_thread: threading.Thread | None = None
_port: int | None = None
_SSE_PATH = "/sse"  # the path FastMCP's http_app(transport="sse") mounts its SSE endpoint at


def ensure_running() -> int:
    """Idempotent, like `echo_server.ensure_running()`: starts the MCP
    server in a background thread on first call (pointed at a freshly
    started `echo_server` as its real backend) and returns its port; every
    later call just returns that same port without starting a second
    server."""
    global _thread, _port
    if _thread is not None:
        return _port  # type: ignore[return-value]

    echo_port = echo_server.ensure_running()
    mcp_app = openapi_adapter.build_mcp_app(
        _TOOL_SPEC, name="demo-multi-tool-mcp", base_url=f"http://127.0.0.1:{echo_port}"
    )
    asgi_app = mcp_app.http_app(transport="sse")

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    _port = port

    def _serve() -> None:
        import uvicorn

        config = uvicorn.Config(asgi_app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
        server = uvicorn.Server(config)
        server.run()

    _thread = threading.Thread(target=_serve, name="capability-marketplace-mcp-demo", daemon=True)
    _thread.start()

    # uvicorn.Server.run() binds and starts serving asynchronously inside the
    # thread's own event loop; give it a moment before any caller tries to
    # connect. A fixed short sleep rather than a readiness probe -- simple
    # and good enough for a local demo/test server, not worth the extra
    # moving parts of polling the port.
    time.sleep(0.5)
    return port


def sse_url() -> str:
    """The URL a `SseConnectionParams`-backed `McpToolset` should connect to
    -- only meaningful after `ensure_running()`."""
    if _port is None:
        raise RuntimeError("mcp_demo_server.ensure_running() hasn't been called yet")
    return f"http://127.0.0.1:{_port}{_SSE_PATH}"


def capability_for(tenant_id: str, *, visibility: Visibility = Visibility.PRIVATE) -> Capability:
    """A registry `Capability` pointing at this server -- call
    `ensure_running()` first. One `ToolSchema` per name in `TOOL_NAMES`,
    matching exactly what the generated MCP server actually exposes (see the
    smoke test in this module's own development history: tool names come
    straight from each operation's `operationId`, not something separately
    declared)."""
    return Capability(
        id=f"cap_{uuid.uuid4().hex[:12]}",
        tenant_id=tenant_id,
        kind=CapabilityKind.MCP,
        name="demo-multi-tool-mcp",
        description="a toy multi-tool API, exposed over MCP for end-to-end testing",
        tools=[ToolSchema(name=name, description=f"{name.replace('_', ' ')} via MCP") for name in TOOL_NAMES],
        visibility=visibility,
        endpoint_ref=EndpointRef(sse_url=sse_url()),
    )
