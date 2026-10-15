"""A deterministic stand-in for a real model, for running compiled agent
trees end to end -- every pattern, every callback -- without a live LLM.

It's a genuine ADK `BaseLlm`, so it goes through the real run loop:
callbacks fire, planners process its output, tools execute (and, against
the test worker, really call MCP servers and A2A agents), workflow agents
sequence/parallelize/loop over it, AgentTool spins up nested runs with it.
Only the *decision* is scripted: on each call it looks at which tools this
agent has (`llm_request.tools_dict`) and which of them already have a
response in the conversation, and

  - calls the next tool that hasn't run yet (declaration order), filling
    any required string argument -- an AgentTool's `request` -- with a
    short instruction;
  - then `exit_loop`, if this agent has it (the last agent of a LOOP
    pattern), so the loop ends after one clean pass;
  - then returns a final text answer naming what ran -- wrapped in
    PlanReActPlanner's /*FINAL_ANSWER*/ tag when the system instruction
    shows that planner is in use, the way a model following that protocol
    would.

`confidence` is what it reports to confidence-gated tools (default 0.9).

`loop_forever=True` instead calls a tool on every turn and never answers:
the runaway-model case, for proving the `max_llm_calls` safety net stops it.

`transfer_to=[...]` scripts a router over ADK `sub_agents`: on its Nth turn
(N = transfers it has already made in this conversation) it calls
`transfer_to_agent` with the Nth name, before anything else. Without it,
`transfer_to_agent` is never called -- it isn't a step to run, and calling
it with an invented agent name would just be an error.
"""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

FINAL_ANSWER_TAG = "/*FINAL_ANSWER*/"

# Zero usage, reported explicitly -- ADK's telemetry warns on every response
# that has no usage metadata at all.
_NO_USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=0, candidates_token_count=0, total_token_count=0
)


class ScriptedToolCallingLlm(BaseLlm):
    model: str = "scripted-tool-caller"
    loop_forever: bool = False
    transfer_to: list[str] = []
    # What it passes as `confidence` to confidence-gated step tools.
    confidence: float = 0.9

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"scripted-.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        tools = dict(llm_request.tools_dict or {})
        responded = [
            part.function_response.name
            for content in llm_request.contents or []
            for part in content.parts or []
            if part.function_response is not None
        ]

        if "transfer_to_agent" in tools:
            done = responded.count("transfer_to_agent")
            if done < len(self.transfer_to):
                yield self._call(tools, "transfer_to_agent", args={"agent_name": self.transfer_to[done]})
                return
            tools.pop("transfer_to_agent")

        if self.loop_forever and tools:
            yield self._call(tools, next(n for n in tools if n != "exit_loop") if len(tools) > 1 else next(iter(tools)))
            return

        pending = [name for name in tools if name not in responded and name != "exit_loop"]
        if pending:
            yield self._call(tools, pending[0])
            return
        if "exit_loop" in tools and "exit_loop" not in responded:
            yield self._call(tools, "exit_loop")
            return

        ran = [name for name in responded if name != "exit_loop"]
        text = f"Completed: {', '.join(ran) if ran else 'no tools to call'}."
        if FINAL_ANSWER_TAG in self._system_instruction(llm_request):
            text = f"{FINAL_ANSWER_TAG}\n{text}"
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]), usage_metadata=_NO_USAGE)

    @staticmethod
    def _system_instruction(llm_request: LlmRequest) -> str:
        instruction = getattr(llm_request.config, "system_instruction", None) if llm_request.config else None
        if instruction is None:
            return ""
        if isinstance(instruction, str):
            return instruction
        parts = getattr(instruction, "parts", None) or []
        return " ".join(getattr(p, "text", "") or "" for p in parts)

    def _call(self, tools: dict, name: str, args: dict | None = None) -> LlmResponse:
        if args is not None:
            return LlmResponse(
                content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]),
                usage_metadata=_NO_USAGE,
            )
        args = {}
        try:
            declaration = tools[name]._get_declaration()
            schema = (declaration.parameters_json_schema if declaration else None) or {}
        except Exception:  # noqa: BLE001
            schema = {}
        for prop in schema.get("required", []):
            prop_type = (schema.get("properties", {}).get(prop) or {}).get("type")
            if prop == "confidence" and prop_type in ("number", "integer"):
                args[prop] = self.confidence
            elif prop_type == "string":
                args[prop] = "Run your approved step and report the result."
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]),
            usage_metadata=_NO_USAGE,
        )
