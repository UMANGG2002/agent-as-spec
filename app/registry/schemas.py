"""The unified `Capability` record -- see the architecture doc's "Capability
registry" section. MCP servers, A2A agent cards, and OpenAPI-derived tools are
all normalized into this one shape so the planner never has to branch on
source kind."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class CapabilityKind(str, Enum):
    MCP = "mcp"
    AGENT_CARD = "agent_card"
    OPENAPI = "openapi"


class Visibility(str, Enum):
    PRIVATE = "private"
    ORG = "org"
    MARKETPLACE = "marketplace"


class ToolSchema(BaseModel):
    """One callable unit inside a capability (an MCP tool, an A2A skill, an
    OpenAPI operation)."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    idempotent: bool = Field(
        default=False,
        description="True for safe-to-retry calls (a lookup-style GET). False means the tool wrapper "
        "enforces an idempotency key before dispatch -- see guardrails/policy.py's ApprovalRule note "
        "and runtime/cache.py's idempotency_key().",
    )


class EndpointRef(BaseModel):
    """How the runtime actually reaches the capability. Exactly one of these
    is populated, matching `kind`."""

    stdio_command: list[str] | None = None
    sse_url: str | None = None
    mcp_transport: Literal["streamable_http", "sse"] | None = None
    agent_card_url: str | None = None
    openapi_base_url: str | None = None


class Capability(BaseModel):
    id: str
    tenant_id: str
    kind: CapabilityKind
    name: str
    description: str = ""
    tools: list[ToolSchema] = Field(default_factory=list)
    required_scopes: list[str] = Field(default_factory=list)
    visibility: Visibility = Visibility.PRIVATE
    endpoint_ref: EndpointRef
    pricing_hint: str | None = None
    vault_entry_id: str | None = None
    auth_header_name: str | None = None  # where a BYOK secret gets injected, for OpenAPI-derived tools

    def visible_to(self, tenant_id: str, granted_capability_ids: set[str]) -> bool:
        if self.tenant_id == tenant_id:
            return True
        if self.visibility == Visibility.MARKETPLACE and self.id in granted_capability_ids:
            return True
        return False


class CapabilitySearchResult(BaseModel):
    capability: Capability
    score: float
    matched_via: Literal["metadata", "keyword"]
