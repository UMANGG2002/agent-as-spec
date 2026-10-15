"""A tiny local A2A-style agent, serving its own agent card
(`/.well-known/agent-card.json`, per `app/registry/adapters/agent_card_adapter.py`'s
`AGENT_CARD_WELL_KNOWN_PATH`) plus a minimal `message/send` JSON-RPC endpoint
-- so the `CapabilityKind.AGENT_CARD` dispatch branch in
`app/factory/tool_wrapper.py` (`agent_card_adapter.invoke_remote_skill`) can
be exercised against a genuine remote "agent", not a stub, exactly the way
one tenant's compiled agent would publish a card for another tenant to call
(see `agent_card_adapter.py`'s module docstring).

Mirrors `echo_server.py`'s shape -- idempotent `ensure_running()`, background
thread, ephemeral port, `http.server`-based (no ASGI needed here, unlike
`mcp_demo_server.py`) -- and exposes one skill, `draft_reply`, that just
echoes back what it was asked to do, tagged as coming from this toy remote
agent, so a caller can tell a multi-tool regression test's step actually
reached this server and not some other one."""

from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app.registry.adapters import agent_card_adapter
from app.registry.adapters.agent_card_adapter import AGENT_CARD_WELL_KNOWN_PATH
from app.registry.schemas import Capability, Visibility

SKILL_ID = "draft_reply"
AGENT_NAME = "demo-remote-agent"

_server: ThreadingHTTPServer | None = None
_port: int | None = None


def _agent_card(base_url: str) -> dict:
    return {
        "name": AGENT_NAME,
        "description": "a toy remote agent for end-to-end testing, reached over A2A",
        "url": base_url,
        "skills": [
            {
                "id": SKILL_ID,
                "name": "Draft a reply",
                "description": "Draft a short reply to whatever input text it's given",
            }
        ],
    }


class _AgentCardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == AGENT_CARD_WELL_KNOWN_PATH:
            base_url = f"http://{self.headers.get('Host', '127.0.0.1')}"
            self._write_json(200, _agent_card(base_url))
            return
        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            rpc_request = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._write_json(400, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            return

        params = rpc_request.get("params", {})
        message = params.get("message", {})
        skill_id = message.get("metadata", {}).get("skill_id", SKILL_ID)
        parts = message.get("parts", [])
        input_text = " ".join(p.get("text", "") for p in parts if p.get("kind") == "text")

        reply_text = f"[{AGENT_NAME}/{skill_id}] drafted a reply to: {input_text}"
        response = {
            "jsonrpc": "2.0",
            "id": rpc_request.get("id", uuid.uuid4().hex),
            "result": {"message": {"role": "agent", "parts": [{"kind": "text", "text": reply_text}]}},
        }
        self._write_json(200, response)


def ensure_running() -> int:
    """Idempotent, like `echo_server.ensure_running()`: starts the agent
    card server in a background thread on first call and returns its port;
    every later call just returns that same port without starting a second
    server."""
    global _server, _port
    if _server is not None:
        return _port  # type: ignore[return-value]
    _server = ThreadingHTTPServer(("127.0.0.1", 0), _AgentCardHandler)
    _port = _server.server_address[1]
    thread = threading.Thread(target=_server.serve_forever, name="capability-marketplace-agent-card-demo", daemon=True)
    thread.start()
    return _port


def base_url() -> str:
    if _port is None:
        raise RuntimeError("agent_card_server.ensure_running() hasn't been called yet")
    return f"http://127.0.0.1:{_port}"


async def capability_for(tenant_id: str, *, visibility: Visibility = Visibility.PRIVATE) -> Capability:
    """A registry `Capability` for this server -- call `ensure_running()`
    first. Fetches the card over a real HTTP round trip
    (`agent_card_adapter.fetch_agent_card`) rather than building one
    in-process from this module's own `_agent_card()`, so this exercises the
    same "discover a remote agent by its published card" path a real
    marketplace onboarding flow would use."""
    card = await agent_card_adapter.fetch_agent_card(base_url())
    return agent_card_adapter.capability_from_agent_card(card, tenant_id=tenant_id, visibility=visibility)
