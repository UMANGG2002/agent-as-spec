"""Compiled-agent cache and the `App` every invocation runs.

Compiling a spec (building the ADK agent tree, its step tools, callbacks)
is pure CPU work that depends only on the spec version and the capability
set -- not on who is calling -- so it is done once per process and reused
by every user and session of the tenant:

    key = (spec_id, version, fingerprint of the capabilities its steps use, model id)

The caller's identity is NOT part of the key: step tools read it from
session state at call time (`__identity__`, written by the runner from the
authenticated request), see `spec_agents._resolve_identity`. A capability
being revoked or changed changes the fingerprint, so the next invocation
compiles fresh; LRU eviction bounds memory.

The `App` adds, for every agent in the tree:
  - plugins: guardrails, long-term memory, usage/budget;
  - context compaction (`EventsCompactionConfig`): when the last prompt
    reached `context_compaction_token_threshold` tokens, older events are
    summarised by an LLM (a cheaper model if `COMPACTION_MODEL_NAME` is set)
    and only the last `context_compaction_retain_events` stay verbatim; also
    every `context_compaction_interval` invocations as a sliding window;
  - resumability, so an invocation interrupted by a crash or paused for a
    tool confirmation resumes from its last event.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from typing import Any

import orjson

from app.auth.models import IdentityContext
from app.config import settings
from app.registry.schemas import Capability
from app.specs.schemas import AgentSpec

_cache: OrderedDict[tuple, Any] = OrderedDict()
_locks: dict[tuple, asyncio.Lock] = {}
stats = {"hit": 0, "miss": 0}


def _fingerprint(spec: AgentSpec, capabilities: dict[str, Capability]) -> str:
    used = sorted({s.capability_id for s in spec.steps if s.capability_id})
    payload = [capabilities[c].model_dump(mode="json") if c in capabilities else None for c in used]
    return hashlib.sha256(orjson.dumps([used, payload], option=orjson.OPT_SORT_KEYS)).hexdigest()[:20]


def compaction_config():
    if not settings.context_compaction_enabled:
        return None
    from google.adk.apps._configs import EventsCompactionConfig
    from google.adk.models.base_llm import BaseLlm

    kwargs: dict[str, Any] = {
        "token_threshold": settings.context_compaction_token_threshold,
        "event_retention_size": settings.context_compaction_retain_events,
    }
    if settings.context_compaction_interval:
        kwargs["compaction_interval"] = settings.context_compaction_interval
        kwargs["overlap_size"] = settings.context_compaction_overlap
    model = settings.compaction_model
    if isinstance(model, BaseLlm):
        from google.adk.apps.llm_event_summarizer import LlmEventSummarizer

        kwargs["summarizer"] = LlmEventSummarizer(llm=model)
    return EventsCompactionConfig(**kwargs)


def build_app(spec: AgentSpec, capabilities: dict[str, Capability], *, model: Any = None):
    from google.adk.apps import App
    from google.adk.apps._configs import ResumabilityConfig

    from app.factory.spec_agents import compile_spec_to_agent
    from app.guardrails.plugin import build_guardrail_plugin
    from app.sessions.plugins import MemoryPlugin, UsagePlugin

    unbound = IdentityContext(tenant_id=spec.tenant_id, brand_id="default", user_id="__unbound__", scopes=())
    root = compile_spec_to_agent(spec, capabilities, unbound, model=model)
    return App(
        name=app_name(spec.spec_id),
        root_agent=root,
        plugins=[build_guardrail_plugin(), MemoryPlugin(), UsagePlugin()],
        events_compaction_config=compaction_config(),
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


def app_name(spec_id: str) -> str:
    return spec_id if spec_id[:1].isalpha() else f"s_{spec_id}"


async def get_app(spec: AgentSpec, capabilities: dict[str, Capability], *, model: Any = None):
    key = (spec.spec_id, spec.version, _fingerprint(spec, capabilities), id(model) if model is not None else "live")
    app = _cache.get(key)
    if app is not None:
        _cache.move_to_end(key)
        stats["hit"] += 1
        return app
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:  # single-flight compile
        app = _cache.get(key)
        if app is None:
            stats["miss"] += 1
            app = build_app(spec, capabilities, model=model)
            _cache[key] = app
            while len(_cache) > settings.compiled_agent_cache_size:
                old, _ = _cache.popitem(last=False)
                _locks.pop(old, None)
    return app


ROUTER_ID = "spec_test_agent"  # the chat target that routes to every approved spec


def build_router_app(tenant_id: str, specs: list[AgentSpec], capabilities: dict[str, Capability], *,
                     model: Any = None):
    """`spec_test_agent` under the production runtime: a router LlmAgent with
    every approved spec of the tenant mounted as an isolated sub-agent (the
    same tree `agents/spec_test_agent` builds for adk web), wrapped in the
    same App as single-spec sessions -- plugins, compaction, resumability."""
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.apps._configs import ResumabilityConfig
    from google.adk.tools import FunctionTool

    from app.factory import spec_router
    from app.guardrails.plugin import build_guardrail_plugin
    from app.sessions.plugins import MemoryPlugin, UsagePlugin

    unbound = IdentityContext(tenant_id=tenant_id, brand_id="default", user_id="__unbound__", scopes=())
    agents, loaded, skipped = spec_router.compile_sub_agents(
        spec_router.filter_specs(specs, None), capabilities, unbound, model=model)
    state = {"loaded": loaded, "skipped": skipped}

    async def list_loaded_specs() -> dict:
        """List every approved spec mounted as a sub-agent -- its sub-agent
        name (what to transfer to), version, pattern, steps and callbacks --
        plus any spec that failed to compile, and why."""
        return {"loaded": list(state["loaded"].values()), "skipped": dict(state["skipped"])}

    from app.config import settings as _settings

    root = LlmAgent(
        name=spec_router.ROUTER_NAME,
        model=model if model is not None else _settings.live_model,
        description="Routes to every approved agent spec, each mounted as a sub-agent.",
        instruction=spec_router.RouterInstruction(state, auto_reload=True),
        tools=[FunctionTool(list_loaded_specs)],
        sub_agents=agents,
    )
    return App(
        name=ROUTER_ID,
        root_agent=root,
        plugins=[build_guardrail_plugin(), MemoryPlugin(), UsagePlugin()],
        events_compaction_config=compaction_config(),
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


async def get_router_app(tenant_id: str, specs: list[AgentSpec], capabilities: dict[str, Capability], *,
                         model: Any = None):
    """Cached per tenant, keyed by every mounted spec's (id, version, capability
    fingerprint): approving or changing a spec, or a grant changing, gives the
    next message a freshly compiled tree."""
    parts = sorted((s.spec_id, s.version, _fingerprint(s, capabilities)) for s in specs)
    digest = hashlib.sha256(orjson.dumps(parts)).hexdigest()[:20]
    key = ("router", tenant_id, digest, id(model) if model is not None else "live")
    app = _cache.get(key)
    if app is not None:
        _cache.move_to_end(key)
        stats["hit"] += 1
        return app
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        app = _cache.get(key)
        if app is None:
            stats["miss"] += 1
            app = build_router_app(tenant_id, specs, capabilities, model=model)
            _cache[key] = app
            while len(_cache) > settings.compiled_agent_cache_size:
                old, _ = _cache.popitem(last=False)
                _locks.pop(old, None)
    return app


def clear() -> None:
    _cache.clear()
    _locks.clear()
