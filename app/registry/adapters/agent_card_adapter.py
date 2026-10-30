"""Capability <-> A2A agent card. A card published by one tenant's compiled
agent (ADK's `to_a2a()` serves one at `/.well-known/agent-card.json` for any
ADK agent) becomes a searchable capability for any other tenant it's granted
to -- this is how a specialized agent, not just a tool, becomes a marketplace
capability. See the architecture doc's registry section.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.registry.schemas import Capability, CapabilityKind, EndpointRef, ToolSchema, Visibility

AGENT_CARD_WELL_KNOWN_PATH = "/.well-known/agent-card.json"


async def fetch_agent_card(base_url: str) -> dict[str, Any]:
    from app.core import http as http_core

    url = base_url.rstrip("/") + AGENT_CARD_WELL_KNOWN_PATH
    await http_core.check_url(url)
    resp = await http_core.client().get(url, timeout=10)
    resp.raise_for_status()
    if len(resp.content) > 1_000_000:
        raise ValueError("agent card too large")
    return resp.json()


def capability_from_agent_card(
    card: dict[str, Any],
    *,
    tenant_id: str,
    required_scopes: list[str] | None = None,
    visibility: Visibility = Visibility.PRIVATE,
) -> Capability:
    tools = [
        ToolSchema(
            name=skill.get("id", skill.get("name", "skill")),
            description=skill.get("description", ""),
        )
        for skill in card.get("skills", [])
    ]
    return Capability(
        id=f"cap_{uuid.uuid4().hex[:12]}",
        tenant_id=tenant_id,
        kind=CapabilityKind.AGENT_CARD,
        name=card.get("name", "remote-agent"),
        description=card.get("description", ""),
        tools=tools,
        required_scopes=required_scopes or [],
        visibility=visibility,
        endpoint_ref=EndpointRef(agent_card_url=card.get("url")),
    )


async def invoke_remote_skill(
    capability: Capability, *, skill_id: str, input_text: str, bearer_token: str | None = None
) -> str:
    """Minimal A2A message send. Production code should use the `a2a` SDK's
    client for full protocol conformance (streaming, task polling,
    push-notification callbacks); this is the illustrative, dependency-free
    path for a minimal build -- a single JSON-RPC-shaped `message/send` call.
    """
    if not capability.endpoint_ref.agent_card_url:
        raise ValueError("capability has no agent_card_url")

    headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
    payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": input_text}],
                "metadata": {"skill_id": skill_id},
            }
        },
    }
    from app.core import http as http_core

    await http_core.check_url(capability.endpoint_ref.agent_card_url)
    resp = await http_core.client().post(capability.endpoint_ref.agent_card_url, json=payload, headers=headers)
    resp.raise_for_status()
    return _extract_text(resp.json())


def _extract_text(rpc_response: dict[str, Any]) -> str:
    result = rpc_response.get("result", {})
    parts = result.get("parts") or result.get("message", {}).get("parts", [])
    return " ".join(p.get("text", "") for p in parts if p.get("kind") == "text")
