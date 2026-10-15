"""Capability <-> MCP server, both directions: introspecting an existing MCP
server into a registry entry, and building the live ADK toolset back out of
one at execution time.

Requires the `adk` extra (`pip install capability-marketplace[adk]`). Kept
import-guarded so the rest of the registry works (search, CRUD, the OpenAPI
adapter) even in an environment that hasn't installed google-adk yet.
"""

from __future__ import annotations

import uuid
from typing import Literal

from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility

McpTransport = Literal["streamable_http", "sse"]

try:
    from google.adk.tools.mcp_tool import (
        McpToolset,
        SseConnectionParams,
        StdioConnectionParams,
        StreamableHTTPConnectionParams,
    )
    from mcp import StdioServerParameters

    _ADK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when the adk extra is missing
    _ADK_AVAILABLE = False


def _require_adk() -> None:
    if not _ADK_AVAILABLE:
        raise RuntimeError(
            "google-adk is not installed. Run `pip install capability-marketplace[adk]` "
            "to connect to live MCP servers; registry CRUD and search work without it."
        )


async def introspect_and_register(
    *,
    tenant_id: str,
    name: str,
    description: str,
    stdio_command: list[str] | None = None,
    sse_url: str | None = None,
    required_scopes: list[str] | None = None,
    visibility: Visibility = Visibility.PRIVATE,
) -> Capability:
    """Connect to an MCP server once at onboarding time, list its tools, and
    cache their schemas into a Capability -- so later registry search never
    needs a live round-trip to the remote server (see the architecture doc's
    registry section)."""
    _require_adk()
    transport: McpTransport | None = None
    if stdio_command:
        toolset = _build_toolset(stdio_command=stdio_command, sse_url=None)
        tools = await _list_tools(toolset)
    elif sse_url:
        tools, transport = await _introspect_remote(sse_url)
    else:
        raise ValueError("either stdio_command or sse_url is required")

    return Capability(
        id=f"cap_{uuid.uuid4().hex[:12]}",
        tenant_id=tenant_id,
        kind=CapabilityKind.MCP,
        name=name,
        description=description,
        tools=tools,
        required_scopes=required_scopes or [],
        visibility=visibility,
        endpoint_ref=EndpointRef(
            stdio_command=stdio_command,
            sse_url=sse_url,
            mcp_transport=transport,
        ),
    )


def build_toolset(capability: Capability):
    """What the Agent Factory binds onto a compiled ADK agent at execution
    time -- a live `McpToolset` pointed at the capability's endpoint."""
    _require_adk()
    if capability.kind != CapabilityKind.MCP:
        raise ValueError(f"not an MCP capability: {capability.id}")
    return _build_toolset(
        stdio_command=capability.endpoint_ref.stdio_command,
        sse_url=capability.endpoint_ref.sse_url,
        mcp_transport=capability.endpoint_ref.mcp_transport,
    )


async def _introspect_remote(url: str) -> tuple[list[ToolSchema], McpTransport]:
    """Prefer Streamable HTTP (current MCP default); fall back to SSE."""
    errors: list[str] = []
    for transport in ("streamable_http", "sse"):
        toolset = _build_toolset(sse_url=url, mcp_transport=transport)
        try:
            return await _list_tools(toolset), transport
        except Exception as exc:  # noqa: BLE001 - probe both transports, then surface
            errors.append(f"{transport}: {exc}")
    raise ConnectionError(
        "Could not connect to MCP server over Streamable HTTP or SSE. " + "; ".join(errors)
    )


from typing import Any

async def _list_tools(toolset) -> list[ToolSchema]:
    try:
        adk_tools = await toolset.get_tools()
        result: list[ToolSchema] = []

        for t in adk_tools:
            raw_tool = getattr(t, "_mcp_tool", None)

            schema: Any = None
            if raw_tool is not None:
                schema = getattr(raw_tool, "inputSchema", None)
                if schema is None:
                    schema = getattr(raw_tool, "input_schema", None)

            # Fallbacks for ADK wrappers that expose the schema
            # directly rather than retaining the underlying MCP object.
            if schema is None:
                schema = getattr(t, "input_schema", None)

            if schema is None:
                schema = getattr(t, "inputSchema", None)

            if hasattr(schema, "model_dump"):
                schema = schema.model_dump(by_alias=True)

            if schema is None:
                schema = {
                    "type": "object",
                    "properties": {},
                }

            result.append(
                ToolSchema(
                    name=t.name,
                    description=t.description or "",
                    input_schema=schema,
                )
            )

        return result
    finally:
        await toolset.close()

def _build_toolset(
    *,
    stdio_command: list[str] | None = None,
    sse_url: str | None = None,
    mcp_transport: str | None = None,
):
    if sse_url:
        if mcp_transport == "sse":
            return McpToolset(connection_params=SseConnectionParams(url=sse_url))
        return McpToolset(connection_params=StreamableHTTPConnectionParams(url=sse_url))
    if stdio_command:
        return McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(command=stdio_command[0], args=stdio_command[1:]),
            )
        )
    raise ValueError("either stdio_command or sse_url is required")
