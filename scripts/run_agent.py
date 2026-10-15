"""Programmatic runner for the adk-web agent (`agents/capability_marketplace`),
using ADK's own `InMemoryRunner` -- a quick way to sanity-check the agent
before poking at it interactively through `adk web`:

    python3 scripts/run_agent.py

Needs a live model configured -- either GOOGLE_API_KEY, or LLM_API_KEY +
LLM_BASE_URL (+ LLM_MODEL_NAME) pointed at an OpenAI-compatible gateway via
LiteLLM, see app/config.py's `live_model`. Unlike scripts/demo.py (which
exercises the HTTP API and works fully offline via the deterministic
planner), there's no offline mode for *this* path: turning the person's
plain-English message into a tool call is itself an LLM call, made by the
orchestrator agent, not just the planner it talks to.

Runs three fixed turns -- what's available, make a plan, run it -- printing
every tool call, tool result, and the model's final reply, so you can see
the whole round trip (registry -> planner -> OAuth delegation -> guardrails
-> real HTTP dispatch -> OpenTelemetry spans) without opening a browser.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run directly: python3 scripts/run_agent.py

from google.adk.runners import InMemoryRunner
from google.genai import types

from app.config import settings

TURNS = [
    "What capabilities do I have available right now?",
    "Register the demo capability, then make a plan to ingest some content using it.",
    "Looks good -- run it, and tell me what happened at each step.",
]


async def main() -> None:
    from agents.capability_marketplace.agent import root_agent

    runner = InMemoryRunner(agent=root_agent, app_name="capability-marketplace")
    session = await runner.session_service.create_session(
        app_name="capability-marketplace", user_id="run-agent-script"
    )

    for turn_num, message in enumerate(TURNS, start=1):
        print(f"\n{'=' * 70}\nTurn {turn_num}: {message}\n{'=' * 70}")
        async for event in runner.run_async(
            user_id="run-agent-script",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            for call in event.get_function_calls():
                print(f"  -> tool call: {call.name}({call.args})")
            for resp in event.get_function_responses():
                print(f"  <- tool result: {resp.name} => {resp.response}")
            if event.is_final_response() and event.content and event.content.parts:
                text = event.content.parts[0].text
                if text:
                    print(f"\nagent: {text}")


if __name__ == "__main__":
    if not settings.planner_is_live:
        print(
            "No live model configured -- the orchestrator agent needs one to hold this "
            "conversation. Set GOOGLE_API_KEY, or LLM_API_KEY + LLM_BASE_URL, in .env "
            "(or export them), then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)
    asyncio.run(main())
