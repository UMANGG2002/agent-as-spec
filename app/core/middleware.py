"""Edge middleware, written as one plain ASGI middleware rather than
Starlette's `BaseHTTPMiddleware` (which wraps every response body, adds
per-request overhead, and interferes with streaming responses and
contextvars -- all of which matter for SSE at high request rates).

Per request, in order:
  - request id: honour an incoming `X-Request-ID` (from the load balancer)
    or mint one; echo it back; bind it into the logging/tracing context.
  - load shedding: if this process already has `load_shed_max_inflight`
    requests in flight, answer 503 + Retry-After immediately instead of
    queueing -- under overload, fast rejection keeps latency bounded for the
    requests that are admitted and lets the load balancer retry elsewhere.
    Health probes and long-lived streams are exempt from the count.
  - body size limit: reject oversized bodies (413) from Content-Length, and
    stop reading if a chunked body runs past the limit.
  - security headers on every response (HSTS in production).
  - RED metrics: request count, errors and latency by route template.
"""

from __future__ import annotations

import time

from app.config import settings
from app.core import context

_EXEMPT_PREFIXES = ("/health", "/ready", "/live", "/metrics")
_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]
_UI_CSP = (b"content-security-policy",
           b"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
           b"img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")


class EdgeMiddleware:
    def __init__(self, app) -> None:  # noqa: ANN001
        self.app = app
        self.inflight = 0

    async def __call__(self, scope, receive, send):  # noqa: ANN001
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path: str = scope.get("path", "")
        headers = dict(scope.get("headers") or [])
        rid = headers.get(b"x-request-id", b"").decode()[:64] or context.new_request_id()
        token = context.request_id_var.set(rid)
        streaming = path.endswith("/stream") or headers.get(b"accept", b"") == b"text/event-stream"
        counted = not path.startswith(_EXEMPT_PREFIXES) and not streaming

        if counted and self.inflight >= settings.load_shed_max_inflight:
            context.request_id_var.reset(token)
            return await _reply(send, 503, b'{"detail":"overloaded, retry"}', rid, extra=[(b"retry-after", b"1")])

        length = headers.get(b"content-length")
        if length and length.isdigit() and int(length) > settings.max_request_body_bytes:
            context.request_id_var.reset(token)
            return await _reply(send, 413, b'{"detail":"request body too large"}', rid)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > settings.max_request_body_bytes:
                    raise _BodyTooLarge()
            return message

        status_holder = {"status": 500}
        start = time.perf_counter()

        async def send_wrapper(message):  # noqa: ANN001
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                hdrs = list(message.get("headers") or [])
                hdrs.append((b"x-request-id", rid.encode()))
                hdrs.extend(_SECURITY_HEADERS)
                if path.startswith("/ui"):
                    hdrs.append(_UI_CSP)
                if settings.is_production_like:
                    hdrs.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                message = {**message, "headers": hdrs}
            await send(message)

        if counted:
            self.inflight += 1
        try:
            await self.app(scope, limited_receive, send_wrapper)
        except _BodyTooLarge:
            await _reply(send, 413, b'{"detail":"request body too large"}', rid)
        finally:
            if counted:
                self.inflight -= 1
            _record(scope, status_holder["status"], (time.perf_counter() - start) * 1000)
            context.request_id_var.reset(token)


class _BodyTooLarge(Exception):
    pass


async def _reply(send, status: int, body: bytes, rid: str, extra=None):  # noqa: ANN001
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"x-request-id", rid.encode()),
                            *_SECURITY_HEADERS, *(extra or [])]})
    await send({"type": "http.response.body", "body": body})


def _record(scope, status: int, ms: float) -> None:  # noqa: ANN001
    from app.telemetry.setup import http_requests_total, http_request_ms

    route = scope.get("route")
    template = getattr(route, "path", None) or "unmatched"
    attrs = {"route": template, "method": scope.get("method", ""), "status_class": f"{status // 100}xx"}
    http_requests_total.add(1, attrs)
    http_request_ms.record(ms, attrs)
