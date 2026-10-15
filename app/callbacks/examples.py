"""Example custom callback factories -- the shape to copy when the built-in
kinds in app/factory/callback_registry.py aren't enough.

A factory takes its config `params` as keyword arguments and returns the
callback. The callback's parameter names must match what ADK passes for its
hook (callback_context/llm_request, callback_context/llm_response,
tool/args/tool_context, tool/args/tool_context/tool_response, or
callback_context). Return None to carry on; return a value to short-circuit
(an LlmResponse skips/replaces the model call, a dict becomes the tool's
result).

Enable with ALLOW_CUSTOM_CALLBACKS=true, then reference e.g.
  {"hook": "after_model", "kind": "custom",
   "params": {"path": "app.callbacks.examples:append_disclaimer",
              "params": {"text": "Not financial advice."}}}
"""

from __future__ import annotations

from typing import Callable


def append_disclaimer(*, text: str) -> Callable:
    """after_model: append a fixed disclaimer to every final text answer."""

    def after_model_append_disclaimer(*, callback_context, llm_response):  # noqa: ANN001
        content = llm_response.content
        if not content or not content.parts:
            return None
        if any(getattr(p, "function_call", None) for p in content.parts):
            return None  # only decorate answers, not tool calls
        last = content.parts[-1]
        if getattr(last, "text", None) and not last.text.endswith(text):
            last.text = f"{last.text}\n\n{text}"
            return llm_response
        return None

    return after_model_append_disclaimer


def require_args(*, required: dict[str, list[str]]) -> Callable:
    """before_tool: per-tool required argument names, e.g.
    {"price_quote": ["sku", "quantity"]} -- a lighter-weight alternative to
    validate_args_schema when a tool's declared schema is too loose."""

    def before_tool_require_args(*, tool, args, tool_context):  # noqa: ANN001
        missing = [name for name in required.get(tool.name, []) if name not in (args or {})]
        if missing:
            return {"ok": False, "error": f"tool '{tool.name}' called without required args {missing}"}
        return None

    return before_tool_require_args
