"""Outbound HTTP: one pooled client per process, plus the SSRF guard.

Pooling: creating an `httpx.AsyncClient` per call (what the adapters used
to do) costs a TCP + TLS handshake every time and can exhaust ephemeral
ports under load. One shared client keeps connections alive (HTTP/2 where
the server offers it, so many concurrent calls share one connection).

SSRF guard: capability endpoints are tenant-supplied URLs that our servers
then call. Without a check, a tenant could register `http://169.254.169.254/`
(cloud metadata credentials) or an internal service and have the platform
fetch it for them. `check_url` resolves the host and refuses loopback,
private, link-local, multicast, reserved and unspecified addresses unless
the host is explicitly allowlisted (or the environment is local/test). It
runs at registration AND before every call (DNS can change). Production
should additionally route egress through a proxy that enforces the same
rule at the network layer, which also closes DNS-rebinding races.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

import httpx

from app.config import settings

_client: httpx.AsyncClient | None = None


class UnsafeURL(ValueError):
    pass


def client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(settings.tool_call_timeout_seconds, connect=5.0),
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200, keepalive_expiry=60),
            follow_redirects=False,  # a redirect could bounce us to an internal address
        )
    return _client


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _is_blocked(ip: ipaddress._BaseAddress) -> bool:
    return (
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
        or ip.is_reserved or ip.is_unspecified or (ip.version == 6 and ip.ipv4_mapped is not None
                                                   and _is_blocked(ip.ipv4_mapped))
    )


async def check_url(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise UnsafeURL(f"scheme '{parts.scheme}' not allowed")
    host = parts.hostname
    if not host:
        raise UnsafeURL("URL has no host")
    if host in settings.outbound_allowed_hosts or settings.outbound_allow_private_networks:
        return
    if settings.is_production_like and parts.scheme != "https":
        raise UnsafeURL("plain http is not allowed for capability endpoints in production")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, parts.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURL(f"cannot resolve {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if _is_blocked(ip):
            raise UnsafeURL(f"{host} resolves to a non-public address ({ip})")
