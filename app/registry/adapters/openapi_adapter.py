"""Capability <-> OpenAPI spec. Two separate things happen here, matching the
architecture doc's "OpenAPI -> MCP" section:

1. `capability_from_openapi` parses the spec into a registry `Capability`
   (one `ToolSchema` per operation) -- pure parsing, no extra dependency.
2. `build_mcp_app` turns the same spec into an actual, externally-reachable
   MCP server (via FastMCP) that other MCP clients -- not just this agent
   factory -- can call. This is the "create MCP out of that" half of the
   brief. It needs the `openapi-mcp` extra; the registry entry and internal
   execution (`build_toolset`, via ADK's own `OpenAPIToolset`) work without it.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility

try:
    from google.adk.tools.openapi_tool import OpenAPIToolset

    _ADK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ADK_AVAILABLE = False

try:
    from fastmcp import FastMCP

    _FASTMCP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTMCP_AVAILABLE = False


def capability_from_openapi(
    spec: dict[str, Any],
    *,
    tenant_id: str,
    name: str,
    base_url: str,
    auth_header_name: str = "Authorization",
    vault_entry_id: str | None = None,
    required_scopes: list[str] | None = None,
    visibility: Visibility = Visibility.PRIVATE,
) -> Capability:
    tools = []
    for path, operations in spec.get("paths", {}).items():
        for method, op in operations.items():
            if method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                continue
            operation_id = op.get("operationId") or f"{method}_{path}".replace("/", "_").strip("_")
            params_schema = _params_to_json_schema(op.get("parameters", []), op.get("requestBody"))
            tools.append(
                ToolSchema(
                    name=operation_id,
                    description=op.get("summary") or op.get("description") or f"{method.upper()} {path}",
                    input_schema=params_schema,
                )
            )

    return Capability(
        id=f"cap_{uuid.uuid4().hex[:12]}",
        tenant_id=tenant_id,
        kind=CapabilityKind.OPENAPI,
        name=name,
        description=spec.get("info", {}).get("description", ""),
        tools=tools,
        required_scopes=required_scopes or [],
        visibility=visibility,
        endpoint_ref=EndpointRef(openapi_base_url=base_url),
        auth_header_name=auth_header_name,
        vault_entry_id=vault_entry_id,
        pricing_hint=None,
    )


def build_toolset(capability: Capability, spec: dict[str, Any], *, bearer_token: str | None = None):
    """Internal execution path: ADK calls the API directly via
    `OpenAPIToolset`, no separate MCP process required."""
    if not _ADK_AVAILABLE:
        raise RuntimeError("google-adk is not installed; run `pip install capability-marketplace[adk]`")

    client = httpx.AsyncClient(base_url=capability.endpoint_ref.openapi_base_url or "")
    if bearer_token:
        client.headers[capability.auth_header_name or "Authorization"] = f"Bearer {bearer_token}"
    return OpenAPIToolset(spec_dict=spec, tool_filter=[t.name for t in capability.tools])


def build_mcp_app(spec: dict[str, Any], *, name: str, base_url: str, bearer_token: str | None = None):
    """External-facing path: an actual MCP server generated from the spec,
    mountable as its own ASGI app (e.g. under
    `/mcp/{capability_id}` in app/main.py) so other tenants' agents -- or any
    MCP client -- can reach this capability directly over the protocol.

    `base_url` is the real backend FastMCP forwards each generated tool call
    to (the same REST API `capability_from_openapi` would point an OPENAPI
    capability's `openapi_base_url` at) -- without it, the httpx client
    FastMCP calls through has nothing to resolve the spec's relative paths
    against, and every tool call fails outright. See
    `app/testing/mcp_demo_server.py` for this wired up against a real
    backend end to end."""
    if not _FASTMCP_AVAILABLE:
        raise RuntimeError("fastmcp is not installed; run `pip install capability-marketplace[openapi-mcp]`")

    headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
    client = httpx.AsyncClient(base_url=base_url, headers=headers)
    return FastMCP.from_openapi(openapi_spec=spec, client=client, name=name)


def _params_to_json_schema(parameters: list[dict[str, Any]], request_body: dict[str, Any] | None) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param in parameters:
        properties[param["name"]] = param.get("schema", {"type": "string"})
        if param.get("required"):
            required.append(param["name"])
    if request_body:
        content = request_body.get("content", {})
        json_schema = content.get("application/json", {}).get("schema")
        if json_schema:
            properties["body"] = json_schema
    return {"type": "object", "properties": properties, "required": required}
