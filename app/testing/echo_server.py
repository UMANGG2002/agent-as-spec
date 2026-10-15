"""A tiny local echo HTTP server used as the backing `base_url` for the
auto-registered demo capability (see `agents/capability_marketplace/agent.py`
and `scripts/run_agent.py`). It exists so the adk-web / runner-script testing
path exercises the *real* OPENAPI dispatch branch in
`app/factory/tool_wrapper.py` -- a genuine HTTP round trip, with the
delegated bearer token actually sent in the Authorization header -- without
depending on external network access or a second process to run.

Not used by the FastAPI app itself (`scripts/demo.py` fakes dispatch instead,
for a fully offline demo); this is specifically for the interactive
`adk web` / `adk run` path, where a human is actually watching the result.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from app.registry.adapters import openapi_adapter
from app.registry.schemas import Capability, Visibility

_server: ThreadingHTTPServer | None = None
_port: int | None = None

_DEMO_SPEC = {
    "info": {"description": "a toy content-ingestion API for local testing"},
    "paths": {
        "/ingest_content": {
            "post": {
                "operationId": "ingest_content",
                "summary": "Ingest generated content into the pipeline",
            }
        }
    },
}


class _EchoHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002 - silence per-request access logging
        pass

    def do_POST(self) -> None:  # noqa: N802 - http.server's own naming convention
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            body = {"raw": raw.decode("utf-8", errors="replace")}

        response = {
            "status": "ingested",
            "path": self.path,
            "received": body,
            "authorization_header_present": "Authorization" in self.headers,
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def ensure_running() -> int:
    """Idempotent: starts the echo server in a background thread on first
    call and returns its port; every later call just returns that same port
    without starting a second server."""
    global _server, _port
    if _server is not None:
        return _port  # type: ignore[return-value]
    _server = ThreadingHTTPServer(("127.0.0.1", 0), _EchoHandler)
    _port = _server.server_address[1]
    thread = threading.Thread(target=_server.serve_forever, name="capability-marketplace-echo", daemon=True)
    thread.start()
    return _port


def capability_for(tenant_id: str, *, visibility: Visibility = Visibility.PRIVATE) -> Capability:
    """An OPENAPI-kind registry `Capability` pointing straight at this
    server -- call `ensure_running()` first. Same toy spec
    `agents/capability_marketplace/agent.py`'s `register_demo_capability`
    builds inline for the supervisor's own demo flow; kept as a second copy
    here rather than refactored into one shared helper, since that function
    is already in active use and this one exists purely for
    `tests/test_e2e_multi_tool_regression.py` to build a same-shaped OPENAPI
    capability without depending on the supervisor module."""
    if _port is None:
        raise RuntimeError("echo_server.ensure_running() hasn't been called yet")
    return openapi_adapter.capability_from_openapi(
        _DEMO_SPEC,
        tenant_id=tenant_id,
        name="demo-ingestion-tool",
        base_url=f"http://127.0.0.1:{_port}",
        visibility=visibility,
    )
