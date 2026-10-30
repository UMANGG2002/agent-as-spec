"""Pooled MCP connections.

Before: every tool call built an `McpToolset`, connected, listed tools,
called one, and disconnected -- a full connection + `tools/list` round trip
per call, which dominates latency and melts MCP servers under load.

Now: one long-lived toolset per capability endpoint per process, with its
tool list cached. Calls reuse it; a connection error evicts the entry and
the call is retried once on a fresh connection; entries idle longer than
`mcp_pool_idle_seconds` are closed by a reaper. Tenant-registered stdio MCP
servers are refused in staging/production (they would run arbitrary
commands on our hosts); only network (SSE/HTTP) MCP is allowed there.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any

from app.config import settings
from app.registry.schemas import Capability

logger = logging.getLogger(__name__)


class _Entry:
    def __init__(self, toolset: Any) -> None:
        self.toolset = toolset
        self.tools: dict[str, Any] = {}
        self.last_used = time.monotonic()
        self.lock = asyncio.Lock()


_pool: dict[str, _Entry] = {}
_reaper: asyncio.Task | None = None


def _pool_key(capability: Capability) -> str:
    ref = capability.endpoint_ref
    raw = f"{capability.id}|{ref.sse_url}|{ref.stdio_command}"
    # Connections belong to an event loop; key by loop so a process with
    # several loops (tests, adk web's loader thread) never shares one.
    return f"{id(asyncio.get_running_loop())}:{hashlib.sha1(raw.encode()).hexdigest()}"


async def _get(capability: Capability) -> _Entry:
    from app.registry.adapters import mcp_adapter

    if capability.endpoint_ref.stdio_command and settings.is_production_like:
        raise PermissionError("stdio MCP servers are disabled in this environment")
    if capability.endpoint_ref.sse_url:
        from app.core.http import check_url

        await check_url(capability.endpoint_ref.sse_url)
    k = _pool_key(capability)
    entry = _pool.get(k)
    if entry is None:
        entry = _pool[k] = _Entry(mcp_adapter.build_toolset(capability))
        _ensure_reaper()
    async with entry.lock:
        if not entry.tools:
            entry.tools = {t.name: t for t in await entry.toolset.get_tools()}
    entry.last_used = time.monotonic()
    return entry


async def _evict(capability: Capability) -> None:
    entry = _pool.pop(_pool_key(capability), None)
    if entry is not None:
        try:
            await entry.toolset.close()
        except Exception:  # noqa: BLE001
            pass


async def call_tool(capability: Capability, tool_name: str, args: dict[str, Any]) -> Any:
    for attempt in (1, 2):
        try:
            entry = await _get(capability)
            tool = entry.tools.get(tool_name)
            if tool is None:
                raise ValueError(f"tool '{tool_name}' not found on capability {capability.id}")
            return await tool.run_async(args=args, tool_context=None)
        except (ConnectionError, OSError, asyncio.TimeoutError) as exc:
            await _evict(capability)
            if attempt == 2:
                raise
            logger.info("MCP connection to %s failed (%s); reconnecting", capability.id, exc)
        except Exception as exc:  # noqa: BLE001 - anyio/httpx wrap transport errors in their own types
            if "connect" in type(exc).__name__.lower() or "closed" in str(exc).lower():
                await _evict(capability)
                if attempt == 1:
                    continue
            raise
    raise RuntimeError("unreachable")


def _ensure_reaper() -> None:
    global _reaper
    if _reaper is None or _reaper.done():
        try:
            _reaper = asyncio.get_running_loop().create_task(_reap())
        except RuntimeError:
            pass


async def _reap() -> None:
    while _pool:
        await asyncio.sleep(min(60.0, settings.mcp_pool_idle_seconds))
        cutoff = time.monotonic() - settings.mcp_pool_idle_seconds
        for k, entry in list(_pool.items()):
            if entry.last_used < cutoff and not entry.lock.locked():
                _pool.pop(k, None)
                try:
                    await entry.toolset.close()
                except Exception:  # noqa: BLE001
                    pass


async def close_all() -> None:
    for k in list(_pool):
        entry = _pool.pop(k)
        try:
            await entry.toolset.close()
        except Exception:  # noqa: BLE001
            pass
