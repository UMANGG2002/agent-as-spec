"""Configured callbacks -> real ADK callback functions.

An `AgentSpec`'s `runtime.callbacks` (see `app/specs/schemas.py`) is a list
of `{hook, kind, params}` entries. This module owns the mapping from `kind`
to an implementation, per hook, so every guardrail / validation / logging
behavior an agent gets is chosen by configuration rather than hardcoded into
the compile step. Two responsibilities, kept deliberately separate:

  - `validate_callback_configs` -- called at *approval* time
    (`app/specs/service.py`). Rejects an unknown kind, a kind attached to a
    hook it doesn't support, or bad params, while a human is still looking
    at the spec -- instead of letting a broken spec through to be discovered
    later by some other process (the `adk web` supervisor) when it tries to
    compile it.
  - `build_callbacks` -- called at *compile* time
    (`app/factory/agent_patterns.py`). Turns validated configs into the
    actual callables, grouped by hook, ready to hand to `LlmAgent(...)`.

Built-in kinds (hook -> kind: params):

  before_model -> guardrail_policy:  {policy_id="default"}  reuse the
                  project's GuardrailEngine.before_model (blocked-keyword
                  check on the latest user text) under a named policy
               -> block_keywords:    {keywords: [str]}  case-insensitive
                  denylist on the latest user text
               -> max_llm_calls:     {limit: int}  stop the agent after N
                  model calls in one invocation (also added automatically
                  from `runtime.max_llm_calls`)
  after_model  -> redact_pii:        {policy_id="default"}  GuardrailEngine
                  redaction over every text part of the model's response
               -> redact_patterns:   {patterns: {regex: replacement}}
  before_tool  -> validate_args_schema: {}  validate the model-supplied args
                  against the tool's own declared JSON schema before running
                  it; a mismatch becomes the tool's result (the model sees
                  the validation error and can retry) instead of a call
               -> tool_allowlist:    {tools: [str]}
               -> tool_denylist:     {tools: [str]}
               -> guardrail_policy:  {policy_id="default"}  the policy's
                  denied_tool_name_patterns
  after_tool   -> redact_pii:        {policy_id="default"}  recursively
                  redact every string in the tool's result
               -> truncate_output:   {max_chars: int}
  before_agent / after_agent -> log_event: {level="info"}

  any hook     -> custom: {path: "pkg.module:factory", params: {...}}  --
                  `factory(**params)` must return the callback. Disabled
                  unless settings.allow_custom_callbacks, and restricted to
                  settings.custom_callback_module_prefixes: naming an import
                  path in config is arbitrary code execution chosen by
                  whoever approves a spec.

ADK calls these with keyword arguments whose names matter
(`callback_context`/`llm_request`/`llm_response` for model hooks,
`tool`/`args`/`tool_context`/`tool_response` for tool hooks,
`callback_context` for agent hooks -- see
`google/adk/flows/llm_flows/tools/_caller.py` and `core/_finalizer.py`), so
every factory below returns a function with exactly those parameter names.
Returning None means "carry on"; returning a value short-circuits that step
-- an `LlmResponse` skips the model call, a dict replaces the tool's result.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections import OrderedDict
from typing import Any, Callable

from app.guardrails import policy as policy_mod
from app.guardrails.plugin import guardrail_engine
from app.specs.schemas import CallbackConfig, CallbackHook

logger = logging.getLogger(__name__)

Factory = Callable[..., Callable]


def _text_response(text: str):
    from google.adk.models import LlmResponse
    from google.genai import types

    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _last_user_text(llm_request) -> str:  # noqa: ANN001
    for content in reversed(llm_request.contents or []):
        if content.role == "user" and content.parts:
            texts = [p.text for p in content.parts if getattr(p, "text", None)]
            if texts:
                return " ".join(texts)
    return ""


def _input_text(callback_context, llm_request) -> str:  # noqa: ANN001
    """The text an input guardrail must check: the person's own message for
    this invocation (`callback_context.user_content`) *and* the latest
    user-role text in the request. Checking only the latter is bypassable:
    when an agent is reached by `transfer_to_agent`, ADK inserts the
    previous agent's activity into its conversation as user-role "For
    context: ..." text, which then sits after -- and hides -- what the person
    actually typed."""
    texts = []
    user_content = getattr(callback_context, "user_content", None)
    if user_content is not None and getattr(user_content, "parts", None):
        texts.extend(p.text for p in user_content.parts if getattr(p, "text", None))
    texts.append(_last_user_text(llm_request))
    return " ".join(t for t in texts if t)


# -- before_model ------------------------------------------------------------


def _before_model_guardrail_policy(*, policy_id: str = "default") -> Callable:
    def before_model_guardrail_policy(*, callback_context, llm_request):  # noqa: ANN001
        policy = policy_mod.get_policy(policy_id)
        decision = guardrail_engine.before_model(text=_input_text(callback_context, llm_request), policy=policy)
        if decision.blocked:
            return _text_response(f"Request blocked by guardrail policy '{policy_id}': {decision.reason}")
        return None

    return before_model_guardrail_policy


def _before_model_block_keywords(*, keywords: list[str]) -> Callable:
    lowered = [k.lower() for k in keywords if k]

    def before_model_block_keywords(*, callback_context, llm_request):  # noqa: ANN001
        text = _input_text(callback_context, llm_request).lower()
        hit = next((k for k in lowered if k in text), None)
        if hit:
            return _text_response(f"Request blocked: input contains the denied phrase '{hit}'.")
        return None

    return before_model_block_keywords


def _before_model_max_llm_calls(*, limit: int) -> Callable:
    """Counts model calls per invocation (keyed by invocation_id, kept in a
    small bounded map rather than session state, so it never leaks into
    persisted state and never depends on how a given ADK version scopes
    `temp:` keys). On call `limit + 1` it returns a final text response
    instead of calling the model -- which is a final response, so ADK's run
    loop ends there."""
    counts: OrderedDict[str, int] = OrderedDict()

    def before_model_max_llm_calls(*, callback_context, llm_request):  # noqa: ANN001
        key = getattr(callback_context, "invocation_id", "") or "default"
        counts[key] = counts.get(key, 0) + 1
        counts.move_to_end(key)
        while len(counts) > 1024:
            counts.popitem(last=False)
        if counts[key] > limit:
            return _text_response(
                f"Stopped: reached the configured limit of {limit} model calls for this run "
                "without a final answer. Reporting what was completed so far."
            )
        return None

    return before_model_max_llm_calls


# -- after_model -------------------------------------------------------------


def _redact_response_text(llm_response, transform: Callable[[str], str]):  # noqa: ANN001
    if not llm_response.content or not llm_response.content.parts:
        return None
    changed = False
    for part in llm_response.content.parts:
        if getattr(part, "text", None):
            new_text = transform(part.text)
            if new_text != part.text:
                part.text = new_text
                changed = True
    return llm_response if changed else None


def _after_model_redact_pii(*, policy_id: str = "default") -> Callable:
    def after_model_redact_pii(*, callback_context, llm_response):  # noqa: ANN001
        policy = policy_mod.get_policy(policy_id)
        return _redact_response_text(llm_response, lambda t: policy_mod.redact(t, policy))

    return after_model_redact_pii


def _compile_patterns(patterns: dict[str, str]) -> list[tuple[re.Pattern, str]]:
    return [(re.compile(pattern), replacement) for pattern, replacement in patterns.items()]


def _after_model_redact_patterns(*, patterns: dict[str, str]) -> Callable:
    compiled = _compile_patterns(patterns)

    def _apply(text: str) -> str:
        for regex, replacement in compiled:
            text = regex.sub(replacement, text)
        return text

    def after_model_redact_patterns(*, callback_context, llm_response):  # noqa: ANN001
        return _redact_response_text(llm_response, _apply)

    return after_model_redact_patterns


# -- before_tool -------------------------------------------------------------


def _tool_args_schema(tool) -> dict | None:  # noqa: ANN001
    try:
        declaration = tool._get_declaration()
    except Exception:  # noqa: BLE001
        return None
    if declaration is None:
        return None
    if declaration.parameters_json_schema is not None:
        return declaration.parameters_json_schema
    if declaration.parameters is not None:
        return declaration.parameters.model_dump(mode="json", exclude_none=True)
    return None


def _before_tool_validate_args_schema() -> Callable:
    """Validates the model-supplied args against the exact schema the tool
    declared to the model in the first place. A mismatch is returned as the
    tool's *result* (ADK skips the real call when a before_tool callback
    returns a dict), so the model sees a precise validation error and can
    correct itself -- rather than the tool crashing on malformed input, or
    worse, acting on it."""
    import jsonschema

    def before_tool_validate_args_schema(*, tool, args, tool_context):  # noqa: ANN001
        schema = _tool_args_schema(tool)
        if not schema:
            return None
        try:
            jsonschema.validate(instance=args or {}, schema=schema)
        except jsonschema.ValidationError as exc:
            path = "/".join(str(p) for p in exc.absolute_path) or "(root)"
            return {"error": f"invalid arguments for tool '{tool.name}' at {path}: {exc.message}", "ok": False}
        return None

    return before_tool_validate_args_schema


def _before_tool_allowlist(*, tools: list[str]) -> Callable:
    allowed = set(tools)

    def before_tool_allowlist(*, tool, args, tool_context):  # noqa: ANN001
        if tool.name not in allowed:
            return {"error": f"tool '{tool.name}' is not in this agent's configured allowlist", "ok": False}
        return None

    return before_tool_allowlist


def _before_tool_denylist(*, tools: list[str]) -> Callable:
    denied = set(tools)

    def before_tool_denylist(*, tool, args, tool_context):  # noqa: ANN001
        if tool.name in denied:
            return {"error": f"tool '{tool.name}' is denied by this agent's configuration", "ok": False}
        return None

    return before_tool_denylist


def _before_tool_guardrail_policy(*, policy_id: str = "default") -> Callable:
    def before_tool_guardrail_policy(*, tool, args, tool_context):  # noqa: ANN001
        policy = policy_mod.get_policy(policy_id)
        if policy_mod.tool_name_denied(tool.name, policy):
            guardrail_engine._record_block("before_tool", policy.id)
            return {"error": f"tool '{tool.name}' is denylisted by policy {policy.id}", "ok": False}
        return None

    return before_tool_guardrail_policy


# -- after_tool --------------------------------------------------------------


def _map_strings(value: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return transform(value)
    if isinstance(value, dict):
        return {k: _map_strings(v, transform) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_strings(v, transform) for v in value]
    return value


def _after_tool_redact_pii(*, policy_id: str = "default") -> Callable:
    def after_tool_redact_pii(*, tool, args, tool_context, tool_response):  # noqa: ANN001
        policy = policy_mod.get_policy(policy_id)
        redacted = _map_strings(tool_response, lambda t: policy_mod.redact(t, policy))
        return redacted if redacted != tool_response else None

    return after_tool_redact_pii


def _after_tool_truncate_output(*, max_chars: int) -> Callable:
    def _truncate(text: str) -> str:
        return text if len(text) <= max_chars else text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"

    def after_tool_truncate_output(*, tool, args, tool_context, tool_response):  # noqa: ANN001
        truncated = _map_strings(tool_response, _truncate)
        return truncated if truncated != tool_response else None

    return after_tool_truncate_output


# -- before_agent / after_agent ---------------------------------------------


def _agent_log_event(hook: str) -> Factory:
    def factory(*, level: str = "info") -> Callable:
        log = getattr(logger, level.lower(), logger.info)

        def agent_log_event(*, callback_context):  # noqa: ANN001
            log(
                "%s agent=%s invocation=%s",
                hook,
                getattr(callback_context, "agent_name", "?"),
                getattr(callback_context, "invocation_id", "?"),
            )
            return None

        return agent_log_event

    return factory


_REGISTRY: dict[CallbackHook, dict[str, Factory]] = {
    CallbackHook.BEFORE_MODEL: {
        "guardrail_policy": _before_model_guardrail_policy,
        "block_keywords": _before_model_block_keywords,
        "max_llm_calls": _before_model_max_llm_calls,
    },
    CallbackHook.AFTER_MODEL: {
        "redact_pii": _after_model_redact_pii,
        "redact_patterns": _after_model_redact_patterns,
    },
    CallbackHook.BEFORE_TOOL: {
        "validate_args_schema": _before_tool_validate_args_schema,
        "tool_allowlist": _before_tool_allowlist,
        "tool_denylist": _before_tool_denylist,
        "guardrail_policy": _before_tool_guardrail_policy,
    },
    CallbackHook.AFTER_TOOL: {
        "redact_pii": _after_tool_redact_pii,
        "truncate_output": _after_tool_truncate_output,
    },
    CallbackHook.BEFORE_AGENT: {"log_event": _agent_log_event("before_agent")},
    CallbackHook.AFTER_AGENT: {"log_event": _agent_log_event("after_agent")},
}


def catalog() -> dict[str, list[str]]:
    """Every built-in kind per hook, plus `custom` -- what a spec's
    `runtime.callbacks` may name. Exposed to the supervisor's
    `describe_agent_runtime_options` tool so it can tell a person what's
    configurable instead of guessing."""
    out = {hook.value: sorted(kinds) for hook, kinds in _REGISTRY.items()}
    for hook in out:
        out[hook].append("custom")
    return out


def register(hook: CallbackHook, kind: str, factory: Factory) -> None:
    """Add a new built-in kind at process start (e.g. from an app module's
    import side effect) -- the in-code, reviewed way to extend the registry,
    as opposed to `custom` naming an import path from config."""
    _REGISTRY.setdefault(hook, {})[kind] = factory


def _resolve_custom(params: dict[str, Any]) -> Callable:
    from app.config import settings

    if not settings.allow_custom_callbacks:
        raise ValueError("custom callbacks are disabled (set ALLOW_CUSTOM_CALLBACKS=true to enable)")
    path = params.get("path", "")
    if ":" not in path:
        raise ValueError(f"custom callback path must look like 'package.module:factory', got {path!r}")
    module_name, attr = path.split(":", 1)
    if not any(module_name.startswith(prefix) or module_name + "." == prefix for prefix in settings.custom_callback_module_prefixes):
        raise ValueError(
            f"custom callback module {module_name!r} is outside the allowed prefixes "
            f"{settings.custom_callback_module_prefixes}"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise ValueError(f"{path!r} is not a callable factory")
    return factory(**params.get("params", {}))


def _build_one(config: CallbackConfig) -> Callable:
    if config.kind == "custom":
        return _resolve_custom(config.params)
    factory = _REGISTRY.get(config.hook, {}).get(config.kind)
    if factory is None:
        available = sorted(_REGISTRY.get(config.hook, {})) + ["custom"]
        raise ValueError(f"unknown callback kind {config.kind!r} for hook {config.hook.value!r}; available: {available}")
    try:
        return factory(**config.params)
    except TypeError as exc:
        raise ValueError(f"bad params for callback {config.hook.value}/{config.kind}: {exc}") from exc


def validate_callback_configs(configs: list[CallbackConfig]) -> None:
    """Build every configured callback once and throw the result away --
    the cheapest way to prove each one *can* be built (known kind, right
    hook, params the factory accepts, custom import allowed and resolvable)
    while a human is still approving the spec. Raises ValueError naming the
    first bad entry."""
    for index, config in enumerate(configs):
        try:
            _build_one(config)
        except ValueError as exc:
            raise ValueError(f"runtime.callbacks[{index}]: {exc}") from exc


def build_callbacks(configs: list[CallbackConfig], *, max_llm_calls: int | None) -> dict[CallbackHook, list[Callable]]:
    """Group built callbacks by hook, in config order. The `max_llm_calls`
    safety net goes *first* among before_model callbacks, so a runaway agent
    is stopped even if a later configured callback would have let the call
    through."""
    grouped: dict[CallbackHook, list[Callable]] = {hook: [] for hook in CallbackHook}
    if max_llm_calls is not None:
        grouped[CallbackHook.BEFORE_MODEL].append(_before_model_max_llm_calls(limit=max_llm_calls))
    for config in configs:
        grouped[config.hook].append(_build_one(config))
    return grouped
