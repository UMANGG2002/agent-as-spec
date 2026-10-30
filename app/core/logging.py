"""Structured logging. In staging/production every line is one JSON object
carrying the request/tenant/session/invocation ids and the active trace id,
so logs join to traces and can be filtered per tenant. Locally it stays a
readable single line.

Secrets never reach a log line by construction: the formatter scrubs a few
well-known shapes (bearer tokens, API keys) as a last line of defence.
"""

from __future__ import annotations

import logging
import re
import sys
import time

import orjson

from app.core import context

_SCRUB = [
    (re.compile(r"(?i)(bearer\s+)[a-z0-9._\-]+"), r"\1[redacted]"),
    (re.compile(r"(?i)(api[_-]?key[\"'=: ]+)[^\s\"',]+"), r"\1[redacted]"),
    (re.compile(r"eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-]{10,}"), "[redacted-jwt]"),
]


def scrub(text: str) -> str:
    for pattern, repl in _SCRUB:
        text = pattern.sub(repl, text)
    return text


def _trace_ids() -> dict[str, str]:
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return {"trace_id": format(ctx.trace_id, "032x"), "span_id": format(ctx.span_id, "016x")}
    except Exception:  # noqa: BLE001
        pass
    return {}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": scrub(record.getMessage()),
            **context.snapshot(),
            **_trace_ids(),
        }
        for key in ("event", "duration_ms", "status", "route", "capability_id", "spec_id"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = scrub(self.formatException(record.exc_info))
        return orjson.dumps(payload, default=str).decode()


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ctx = context.snapshot()
        tag = " ".join(f"{k}={v}" for k, v in ctx.items() if k in ("tenant_id", "session_id"))
        base = super().format(record)
        return scrub(f"{base} {tag}".rstrip())


_configured = False


def configure_logging(level: str = "INFO", *, json: bool = False) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json else TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "LiteLLM", "litellm", "asyncio", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
