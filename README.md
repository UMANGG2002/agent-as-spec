# Agent-as-Spec — Capability Marketplace on Google ADK

A minimal, runnable implementation of the architecture proposed for a
multi-tenant capability marketplace: clients register MCP servers, A2A agent
cards, or OpenAPI specs (auto-converted to MCP-callable tools) as
**capabilities**, can request and be granted access to each other's
capabilities, and get a **planner agent** that turns a natural-language
request into a typed `ExecutionPlan` the system compiles into a running,
auth-delegated, guardrailed, traced agent.

See the companion architecture doc for the full design rationale. This
README covers running what's actually implemented here.

> **Production architecture:** the platform has been rebuilt for horizontal
> scale and multi-tenancy (Postgres with row-level security, Redis Cluster,
> sharded work queue + runner workers, resumable SSE streams, mem0 memory,
> context compaction, rate limits, quotas, circuit breakers, transactional
> outbox, key rotation). Everything that changed, why, the design patterns
> and the capacity math are in **[`docs/PRODUCTION_ARCHITECTURE.md`](docs/PRODUCTION_ARCHITECTURE.md)**.
> Local development is unchanged: `ENVIRONMENT=local` needs no services.

## Production deployment (summary)

```bash
python scripts/gen_keys.py > .env.production   # RS256 signing key + vault key
docker compose up --build                       # Postgres+pgvector, Redis, migrate, api, 2 runners, relay
```

Four process roles from one image: `uvicorn app.main:app` (API),
`python -m app.sessions.runner` (runner workers), `python -m app.relay_main`
(outbox relay), `python -m app.core.migrate` (once per deploy, owner DB role).
With `ENVIRONMENT=staging|production` the process refuses to start on any
dev default (SQLite, in-process Redis, HS256, no vault key, inline runners).

Talking to an agent -- in the UI's **Chat** tab, or over the v1 API. Pick
`spec_test_agent` to reach every approved spec through one router (it
transfers to the right sub-agent), or one spec on its own:

```bash
GET  /v1/agents                       chat targets: spec_test_agent + every approved spec
POST /v1/sessions                     {"spec_id": "spec_test_agent" | "spec_..."}  -> {session_id}
POST /v1/sessions/{sid}/messages      {"text": "..."}  + Idempotency-Key -> 202 {invocation_id, stream_url}
GET  {stream_url}                     SSE; reconnect with Last-Event-ID to resume exactly where it stopped
POST /v1/sessions/{sid}/messages      {"confirmations": [{"function_call_id": "...", "confirmed": true}]}
POST /v1/sessions/{sid}/invocations/{iid}/cancel
```

## What's real vs. stubbed

Everything below is working code, not a sketch:

- **Capability registry** -- MCP (`McpToolset`), OpenAPI (parsed + optionally
  turned into a real MCP server via FastMCP), and A2A agent cards, unified
  into one `Capability` schema with hybrid keyword search.
- **Marketplace mechanics** -- publish, cross-tenant request/grant, BYOK
  vault, per-capability invocation ledger. `/marketplace/discover` browses
  what's requestable; `/capabilities/search` is the grant-gated function the
  planner actually uses -- these are deliberately different functions (see
  `registry/store.py`'s docstrings for why conflating them is a security bug
  waiting to happen).
- **Planner -> typed plan -> compiled agent** -- an offline, deterministic
  planner (keyword search -> `ExecutionPlan`) always works; a real ADK
  `LlmAgent` planner (`PlanReActPlanner` + `output_schema=ExecutionPlan`)
  runs when a live model is configured -- `GOOGLE_API_KEY`, or
  `LLM_API_KEY`/`LLM_BASE_URL` pointed at an OpenAI-compatible gateway via
  LiteLLM (see `app/config.py`'s `live_model`).
- **OAuth delegation chain** -- `platform -> tenant -> run -> tool-call`,
  each hop a short-lived JWT with an `act` claim chain.
- **Guardrails** -- the six ADK callback hook points, a data-driven
  `GuardrailPolicy` (denylist, PII redaction), dynamic HIL via
  `ApprovalRule`, and adaptive escalation off repeated blocks.
- **Idempotency** -- a `(run_id, step_id, args)`-keyed result cache so a
  retry or a Redis-stream redelivery never re-executes a non-idempotent call.
- **OpenTelemetry** -- one `TracerProvider`/`MeterProvider`, spans on every
  auth/guardrail/tool-call boundary, console exporter by default.
- **Durable execution + offline resume** -- Redis Streams with consumer
  groups; falls back to an in-process queue (clearly logged) if Redis isn't
  running, so the demo works with zero infrastructure.
- **Human-in-the-loop** -- `runtime/hil.py`'s suspend/resume, reached either
  by an explicit `human_approval` plan step or by a dynamic `ApprovalRule`.
  Compiled spec agents go further: a per-spec tool policy (approval before
  chosen steps, a confidence threshold, allow/deny/approve rules) enforced
  before every tool call, with durable approvals decided in chat or an inbox
  -- see "Human-in-the-loop and policy enforcement" below.

What's intentionally a thin extension point rather than fully built out:

- The **real ADK agent tree** (`SequentialAgent`/`ParallelAgent` of
  `LlmAgent`s with per-step improvisation) -- the default execution path is
  the `DeterministicExecutor`, since a compiled plan's steps are already
  bound to specific capabilities; `factory/agent_factory.py`'s docstring
  says where the ADK-tree path plugs in.
- **Composio notifications** -- the dispatcher, preference table, and event
  wiring are real; the actual `composio` SDK call is a one-line fill-in once
  you know your tenants' connected toolkit names (it logs a dry-run
  message until then).
- **A2A protocol conformance** -- `agent_card_adapter.invoke_remote_skill`
  sends a minimal JSON-RPC `message/send`; swap in the real `a2a` SDK client
  for streaming/task-polling/push-notification conformance.

## Quick start: real Postgres, Redis and mem0, no image build

```bash
uv sync --all-extras
chmod +x scripts/dev.sh
./scripts/dev.sh all          # pulls Postgres + Redis, migrates, runs api + runner + relay with uvicorn
```

Then open http://127.0.0.1:8000/ui/. Details, settings and troubleshooting: [docs/LOCAL_DEV.md](docs/LOCAL_DEV.md).

## Evaluating agents

Automated evaluation of individual agents, the router end to end and the planner: deterministic checks, an LLM judge, golden datasets, regression gates and auto-eval. What it proves and what it cannot: [docs/EVALUATION.md](docs/EVALUATION.md). Run it with `python scripts/eval.py --help`.

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[adk,openapi-mcp,dev]"
cp .env.example .env   # safe local defaults; no external services required
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API. Without Redis
running, durable streaming falls back to an in-process queue (logged, not
silent) -- fine for trying things out; `docker run -p 6379:6379 redis` for
the real thing. Without a live model configured (`GOOGLE_API_KEY`, or
`LLM_API_KEY`/`LLM_BASE_URL` for an OpenAI-compatible gateway via LiteLLM),
the planner runs its deterministic offline mode -- real plans, just not
LLM-reasoned ones.

### A five-minute walkthrough

```bash
# 1. Get a dev token (stands in for a real login/OAuth front door)
TOKEN=$(curl -s -X POST localhost:8000/auth/dev-token \
  -H 'content-type: application/json' \
  -d '{"tenant_id":"acme","user_id":"u1","scopes":["*"]}' | jq -r .access_token)

# 2. Register two capabilities (an MCP tool + an OpenAPI-derived one)
curl -s -X POST localhost:8000/capabilities/mcp -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"name":"video-gen","description":"generates short videos from a prompt","stdio_command":["npx","some-video-gen-mcp-server"]}'

# 3. Ask the planner for a plan in plain English
curl -s -X POST localhost:8000/plans -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"intent":"use my video-gen MCP and my ingestion tool to create one capability"}'

# 4. Run the plan and watch it stream
curl -s -X POST localhost:8000/runs -H "Authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"plan": <the plan JSON from step 3>}'
curl -N localhost:8000/runs/<run_id>/events -H "Authorization: Bearer $TOKEN"
```

`scripts/demo.py` runs this whole flow programmatically, end to end, without
needing `curl` or a running server (it talks to the FastAPI app in-process).

## Configuration plane: a minimal UI

`app/static/index.html`, served at `/ui` once the server is running, is a
small (no build step, vanilla JS) front end over the configuration-plane
API: a knowledge-base tab (add/search free-text documents a tenant's agents
get grounded against), a tools tab (register an MCP/OpenAPI/A2A capability,
search what's already registered), and an agent-builder tab (describe what
you want in plain English, review the typed plan the planner produced,
approve it, and it's saved as a versioned `AgentSpec` in Postgres (SQLite
locally) -- re-approving the same name bumps the version, and every version
stays readable in `agent_spec_versions`).

```bash
pip install -e ".[adk,openapi-mcp,dev]"
uvicorn app.main:app --reload
```

Open `http://localhost:8000/ui`, hit **Connect** (tenant `acme` by default --
mints a dev token the same way the curl walkthrough above does), and work
through the three tabs. Nothing else needs to run locally: specs live in
the same SQL database as everything else (MongoDB is no longer used), and
Redis defaults to an in-process stand-in. The UI and the `adk web` agent from the section below are two
different clients of the exact same backend and data -- see
`ARCHITECTURE_MODULES.md` for the full module map of who calls what.

## Interactive testing with `adk web`

Besides the HTTP API above, there's a second way in: a conversational
orchestrator agent at `agents/capability_marketplace/` that sits in front of
the same registry/planner/factory/guardrail stack, so you can chat your way
through registering a capability, getting a plan, and running it, instead of
scripting `curl` calls.

```bash
pip install -e ".[adk]"
export GOOGLE_API_KEY=...   # or LLM_API_KEY + LLM_BASE_URL -- see note below
adk web agents
```

Open the URL it prints, pick `capability_marketplace` from the app
dropdown, and try:

```
What capabilities do I have available right now?
Register the demo capability, then make a plan to ingest some content using it.
Looks good -- run it, and tell me what happened at each step.
```

The agent exposes four tools -- `list_capabilities`, `register_demo_capability`,
`make_plan`, `run_last_plan` -- and the real run path underneath: OAuth
delegation, guardrails, idempotency, and OpenTelemetry spans all execute for
real, exactly as they do over the HTTP API, against the *same* SQLite
registry, so a capability you registered via `curl`/`scripts/demo.py` shows
up here too (and vice versa). `register_demo_capability` sets up a small
capability backed by a local echo server (`app/testing/echo_server.py`) so
there's something to plan against with zero setup; it's idempotent, and it
self-heals the demo capability's URL to the current process's port each time
(the echo server is per-process and ephemeral, while the registry row
persists in SQLite across restarts).

**This needs a live model configured, unlike everything else in this
README.** Set either `GOOGLE_API_KEY` (Google's native Gemini API), or
`LLM_API_KEY` + `LLM_BASE_URL` (+ `LLM_MODEL_NAME`) to point the agent at
any OpenAI-compatible endpoint via LiteLLM -- a self-hosted gateway, a
multi-provider proxy, etc. (needs `pip install -e ".[litellm]"`; see
`app/config.py`'s `live_model`). Holding a chat conversation -- deciding
*which* tool to call from your English message -- is itself an LLM call
made by the orchestrator agent, so there's no offline mode for this path
the way there is for the planner (`settings.planner_is_live` still governs
whether *that inner planner call* is live or deterministic once a tool is
invoked).

`scripts/run_agent.py` runs the same agent programmatically via ADK's
`InMemoryRunner` -- a fixed three-turn conversation, printing every tool
call/result and the model's replies -- useful as a quick sanity check before
opening the browser, or in CI once you have a model key available:

```bash
python3 scripts/run_agent.py
```

### Approved specs become sub-agents, for real

Approving a plan (`POST /agent-specs/approve`, or the chat -> plan -> approve
flow in the configuration-plane UI) doesn't just save a database row: the
orchestrator above loads every approved `AgentSpec` and compiles it into a
real ADK agent it can call as a tool, named after the spec -- see
`app/factory/spec_agents.py`. Ask it "what agents have been approved" (calls
`list_agent_specs`), and if you approved something new in this same running
conversation, ask it to pick up the new one (calls
`refresh_agent_spec_tools`) -- new approvals don't appear automatically
mid-conversation, since approving happens in the FastAPI process and `adk
web` is a separate one; it has to load specs itself, not get pushed to.

Each compiled spec-agent gets one fixed, non-negotiable tool per plan step
(the capability, tool, and instruction a human already approved -- the model
decides only *when* to call each one, never *what* it does), and every step
still runs through the exact same guardrails/OAuth-delegation/idempotency/
telemetry path as the HTTP API's deterministic executor. It deliberately
does *not* share the output_schema+tools+planner combination used by the
live planner above -- see `app/config.py`'s `_GatewayLiteLlm` docstring for
why that combination is the thing to avoid.

### The compiled agent's instruction gets a real polish pass, not just a template

`build_spec_instruction` (`app/factory/spec_agents.py`) builds a mechanical
draft -- goal + a bare "- step_id: capability -- instruction" line per step.
Before that's persisted as `AgentSpec.instruction`, `polish_spec_instruction`
sends it to the configured live model with a system prompt asking for the
quality bar a senior engineer hand-authoring an agent's system prompt would
aim for: connected, readable prose instead of a bullet dump, covering what
the agent is (and isn't) trusted to do. Pass `author_notes` to
`approve_last_plan` (or `author_notes` in the `POST /agent-specs/approve`
body) -- tone, edge cases, error-handling preferences, anything you want the
compiled agent actually told -- and the model weaves it in directly, rather
than it just getting appended as an afterthought.

Three safety nets, since this text is the only thing telling the compiled
agent what to do at runtime: no live model configured -> returns the
mechanical draft unchanged; the rewrite call fails for any reason (timeout,
malformed response, quota) -> falls back to the mechanical draft; the
rewrite silently drops a `step_id` the plan actually has -> treated as a bad
rewrite and discarded, same as an exception. An approval should never fail,
or silently lose a step the runtime depends on, because this one polish step
had a bad day. `tests/test_spec_agents.py` exercises all three fallback
paths directly (no live model needed to test them).

### Testing the whole loop against one supervisor agent, no second process

There's exactly one supervisor (`root_agent` in `agents/capability_marketplace/agent.py`)
and every compiled spec-agent is attached to *that same* agent's `tools` --
never a separate supervisor per spec. You don't need the FastAPI app running
at all to test an agent spec end to end through `adk web`; the whole loop
happens in one conversation with this one agent:

```bash
uv run adk web agents
```

In the chat: ask it to register a demo capability (or describe your own),
then describe a goal so it calls `make_plan`, then ask it to **approve that
as `<some-name>`** (calls `approve_last_plan` -- the same versioning,
capability-snapshotting and instruction-baking `POST /agent-specs/approve`
does over HTTP, just called in-process against this agent's own DB session
instead of a separate request to a separate process -- see
`app/specs/service.py`), then ask it to **pick up the new one** (calls
`refresh_agent_spec_tools`), then just ask it to run it. `list_active_tools`
at any point shows exactly what's callable right now, parameter schemas
included. `tests/test_supervisor_agent.py` exercises this same loop directly
against the tool functions, without a live model.

### Every approved spec as a sub-agent: `spec_test_agent`

`adk web agents` lists every subfolder under `agents/` as its own app.
Alongside the `capability_marketplace` supervisor there's
`spec_test_agent`: one router with **every** approved spec mounted under it
as a real ADK sub-agent, each compiled exactly as approved (its own pattern
and callbacks) -- see `app/factory/spec_router.py`.

```bash
uv run adk web agents                          # pick "spec_test_agent"
TEST_SPECS=order-desk,support-desk uv run adk web agents   # optional: only these (names or spec_ids)
```

Ask for something a spec does, or name it: the router hands the
conversation over with `transfer_to_agent`, the spec runs its steps (you see
its own events, not a summary), and your next message goes back to the
router -- which can then hand off to a different spec. Each mounted spec is
barred from transferring anywhere itself, so it can't wander into another
spec it wasn't approved to be. `list_loaded_specs` shows what's mounted
(pattern, steps, callbacks) and anything that failed to compile and why;
`reload_specs` picks up specs approved after startup without a restart.

The difference from the supervisor: the supervisor calls specs as *tools*
(`AgentTool`) and stays in charge, which is what lets it attach specs
mid-conversation; this router *transfers* to them, which is the better way
to watch a spec behave as itself.

### Building an agent spec in the UI: search, discover, compose, approve

`uv run uvicorn app.main:app` then open http://127.0.0.1:8000/ui/ and
Connect. The **Agent Builder** tab is the whole flow on one page:

1. **Search & discover** -- keyword search over capability names,
   descriptions and every tool's name/description, filtered by kind (MCP /
   A2A / OpenAPI), in two scopes: *Usable by me* (owned + granted) or
   *Public marketplace* (everything any tenant published). Each result
   shows its tools with their argument types (required ones starred),
   tools matching your query highlighted, and whether you can use it:
   *yours*, *access granted*, *access requested*, or *needs access* with a
   **Request access** button. The same card has **Register all worker MCP +
   A2A capabilities** -- point it at a running `scripts/run_worker.py` and
   all 42 are published to the marketplace under your tenant.
2. **Access requests** -- approve or deny other tenants' requests for your
   capabilities; see the status of yours.
3. **Plan the agent** -- **+ Add step** on any usable tool appends a step
   with an argument skeleton built from that tool's schema; or draft steps
   from a description with the planner. Every step is editable: tool
   (any tool on the capability), arguments (JSON), instruction (for A2A
   skills, the message sent), what it waits for, order.
4. **Runtime, policy & approvals** -- four sub-tabs: execution pattern
   (pattern, loop iterations, model-call cap); **tool policy & approvals**
   (see the next section); callback presets (argument validation, input
   guardrail, keyword block, PII redaction, truncation, logging) plus any
   extra callbacks as JSON; notes for the agent's instruction.
5. **Approve & save** -- a bar pinned to the bottom of the page summarises
   what you're about to save (steps, pattern, callbacks, approval gates,
   rules) and holds the name + **Approve & save**. Approval validates
   everything server-side -- the tool exists, the arguments satisfy its
   schema, the pattern fits the dependencies, the callbacks are known, every
   policy rule targets a real step and names a real fact -- and shows the
   reason if it refuses. Saved specs (with their policy badges) can be
   loaded back as a new version, policy included.

The page is a set of small vanilla-JS components (Discover, Grants, Plan,
Policy, Runtime, ActionBar, Approvals, Specs, Register, Knowledge) sharing
one state object, rendered through an `html` tagged template that escapes
every interpolated value unless explicitly wrapped in `raw()` -- discovery
shows text written by other tenants (a remote agent card's description is
whatever its owner put there). Buttons show a spinner while their request is
in flight, results come back as toasts, search is debounced, and it follows
the OS light/dark setting and works down to phone width.

### Human-in-the-loop and policy enforcement before every tool call

Each spec carries a `runtime.tool_policy` (`app/specs/schemas.py`'s
`ToolPolicyConfig`), evaluated by `app/factory/tool_policy.py` inside every
compiled step tool, *before* the capability is called:

```json
"tool_policy": {
  "require_approval_before": ["charge_card"],          // step ids, or "*"
  "confidence_threshold": 0.7,                          // null = off
  "confidence_tools": ["*"],
  "confidence_method": "combined",                      // self_reported | signals | combined
  "rules": [
    {"name": "big_orders", "tools": ["place_order"], "effect": "require_approval",
     "when": [{"fact": "args.quantity", "op": "gt", "value": 100}], "reason": "large order"},
    {"name": "retry_storm", "tools": ["*"], "effect": "deny",
     "when": [{"fact": "attempt", "op": "gte", "value": 3}]}
  ],
  "default_effect": "allow",                            // "deny" turns rules into an allowlist
  "approval_channel": "chat",                           // or "inbox" (separation of duties)
  "approval_ttl_seconds": 86400
}
```

- **Facts** a rule can test: `step_id, tool, tool_name, capability_id,
  capability_kind, confidence, self_confidence, signal_confidence, attempt,
  failed_steps, calls_in_run, hour_utc, weekday, tenant_id, user_id` and
  `args.<path>`. Ops: `eq ne gt gte lt lte in not_in contains matches exists`.
  Effects combine most-restrictive-first: deny > require_approval > allow.
  A fact that can't be compared (missing arg, wrong type) makes the
  condition false, never an error. Platform `ApprovalRule`s from the
  guardrail engine feed into the same decision.
- **Confidence**: *self_reported* gives the gated step tool a required
  `confidence` (0-1) + `rationale` argument the model fills in;
  *signals* derives it from the run so far (-0.5 per unmet dependency,
  -0.15 per other failed step, -0.2 per earlier failure of this step, -0.1
  per guardrail block); *combined* takes the lower. The approver sees the
  number and its breakdown.
- **Deny** returns `status: "denied_by_policy"` with the rule names and
  reasons -- the tool never runs. **Require approval** files a durable
  request in the `tool_approvals` table (`app/runtime/approvals.py`) keyed
  by a fingerprint of tenant/spec/step/tool/arguments, and:
  - in `adk web` (spec mounted directly or as a sub-agent of
    `spec_test_agent`), ADK's native tool confirmation pauses the turn and
    shows Approve/Reject; confirming decides the request and the same call
    resumes. With `approval_channel: "inbox"`, the chat confirmation can't
    approve -- an approver decides it in the inbox and the step is run again.
  - under the supervisor (spec wrapped as an `AgentTool`), ADK doesn't
    surface the inner pause, so the tool returns `awaiting_approval` with the
    approval id; decide it with the supervisor's `decide_approval` tool or the
    inbox, then run the spec again.
- An approval is **single-use**: it lets that exact call (same arguments)
  run once, then it's consumed. Requests expire after the TTL. A
  `human_approval` step in a plan uses the same store.
- **Inbox API** (also the **Approvals** tab in the UI, with a pending-count
  badge): `GET /approvals?status=pending`, `GET /approvals/{id}`,
  `POST /approvals/{id}/decide {"approve": true, "note": "..."}`. Tenant
  scoped; the first decision is final (409 after).
- `GET /agent-specs/runtime-options` lists the facts, ops, effects,
  confidence methods and channels; the supervisor's
  `describe_agent_runtime_options` does the same in chat, and
  `approve_last_plan(..., tool_policy={...})` accepts the config.

### Agent patterns and callbacks are configuration, not code

Every approved spec carries a `runtime` (`AgentRuntimeConfig` in
`app/specs/schemas.py`) that decides how it compiles and what runs around
each model and tool call. Set it when approving -- `runtime` in the
`POST /agent-specs/approve` body, or `pattern` / `callbacks` /
`max_iterations` on the supervisor's `approve_last_plan` -- and every process
that loads the spec compiles the same tree. `GET /agent-specs/runtime-options`
(or the supervisor's `describe_agent_runtime_options` tool) lists everything.

```json
{"pattern": "sequential",
 "max_llm_calls": 25,
 "callbacks": [
   {"hook": "before_model", "kind": "guardrail_policy", "params": {"policy_id": "default"}},
   {"hook": "before_model", "kind": "block_keywords",   "params": {"keywords": ["wire transfer"]}},
   {"hook": "before_tool",  "kind": "validate_args_schema"},
   {"hook": "after_tool",   "kind": "redact_pii"},
   {"hook": "after_model",  "kind": "redact_patterns",  "params": {"patterns": {"ACME-\\d+": "[account]"}}}
 ]}
```

**Patterns** (`app/factory/agent_patterns.py`): `single` (default -- one
agent, every step tool), `sequential` (one sub-agent per step, run in
`depends_on` order, enforced by ADK rather than left to the model),
`parallel` (all steps at once; rejected if any step has `depends_on`),
`loop` (the sequential chain until the last step calls `exit_loop`, capped
at `max_iterations` -- note each iteration re-runs every step's tool for
real), `hierarchical` (a coordinator delegating to one specialist per step
via `AgentTool`, so it stays in control and gets every result back),
`react` (single + `PlanReActPlanner`), `plan_execute` (a tool-less planner
writes a plan into state, an executor follows it).

**Callbacks** (`app/factory/callback_registry.py`), at every ADK hook:
`before_model` -- `guardrail_policy`, `block_keywords`, `max_llm_calls`;
`after_model` -- `redact_pii`, `redact_patterns`; `before_tool` --
`validate_args_schema` (validates the model's arguments against the tool's
own declared JSON schema; a mismatch is returned to the model as the tool
result so it can correct itself), `tool_allowlist`, `tool_denylist`,
`guardrail_policy`; `after_tool` -- `redact_pii`, `truncate_output`;
`before_agent` / `after_agent` -- `log_event`. Anything else: a `custom`
callback naming a factory by import path (see `app/callbacks/examples.py`),
off unless `ALLOW_CUSTOM_CALLBACKS=true` and restricted to
`CUSTOM_CALLBACK_MODULE_PREFIXES` -- naming code in config is code execution
chosen by whoever approves a spec. In-code extensions go through
`callback_registry.register(...)` instead.

Every compiled agent also gets a `max_llm_calls` safety net (default 25 per
run, shared across one agent tree; `null` disables it): ADK's run loop has
no cap of its own, which is exactly how the live planner once looped
forever. A model that keeps calling tools without answering is stopped with
a final "Stopped: ..." response instead.

Invalid configurations -- parallel over dependent steps, a dependency
cycle, an unknown callback kind, a kind on a hook it doesn't support, bad
params, a disallowed custom callback -- are rejected **at approval**, with a
reason and nothing saved, not discovered later when the supervisor tries to
compile. If a saved spec stops compiling anyway (say custom callbacks were
switched off afterwards), `refresh_agent_spec_tools` skips just that spec
and reports why under `skipped`; every other spec still loads.

`tests/test_agent_patterns.py` runs **every pattern end to end** through
ADK's real run loop, with a deterministic scripted model
(`app/testing/scripted_llm.py`) choosing the tool calls and every step
really calling the test worker's MCP servers and A2A agents -- plus each
callback, a guardrail blocking a run before any tool fires, and the
`max_llm_calls` net stopping a model that never answers.

### The test worker: 22 MCP servers + 20 A2A agents at real URLs

For end-to-end and regression testing of compiled multi-tool agents, one
process serves every MCP server and A2A agent in
`app/testing/worker_catalog.py`, each at its own URL on one port:

```bash
uv run python scripts/run_worker.py                    # http://127.0.0.1:9100
uv run python scripts/run_worker.py --register         # ...and add all 42 to the registry (tenant acme)
```

- `GET /` -- manifest: every server's `/sse` URL with its tools, JSON input
  schemas and a worked example; every agent's card URL and skills.
- `/mcp/<server>/sse` -- a native FastMCP server over SSE (math, stats, JSON,
  dates, units, hashing, base64, regex, CSV, URLs, geo, finance, lists,
  color, validation, inventory, weather, sentiment, translation,
  summarization, ...). Typed arguments, JSON-object results.
- `/a2a/<agent>/.well-known/agent-card.json` + `POST /a2a/<agent>` -- an A2A
  card and JSON-RPC `message/send` (ticket triage/routing, sentiment,
  summarizer, translator, code review, FAQ, scheduler, invoice, legal
  clauses, SEO, email drafter, QA, risk scoring, recommender, data cleaner,
  titles, keywords, PII compliance, pricing, onboarding). Text in, text out.

Everything is deterministic (no clock, randomness or network), and every
tool and skill carries an example input and its exact expected output, so
the catalog doubles as a regression suite: `tests/test_worker_regression.py`
calls every one of them over its real protocol through this project's real
client code, registers them all through the real onboarding adapters, and
compiles + runs an agent whose six steps span four MCP servers and two A2A
agents. Add an entry to the catalog and it's covered automatically.

`--register` uses stable ids (`cap_worker_mcp_<server>`,
`cap_worker_a2a_<agent>`), so re-running after a restart updates the same
rows (new port included). Keep the worker running while specs bound to it
run -- the registry records where each tool lives; the worker answers there.

Plan steps can now target any tool on a multi-tool capability with fixed,
approved arguments -- `PlanStep.tool_name` and `PlanStep.arguments`. Leave
them out and a step behaves as before (first tool, `{"instruction": ...}`),
which suits free-text tools and every A2A skill.

The MCP client side needs the `mcp` package (now pulled in by the `litellm`
extra) and the worker's servers need `fastmcp` (the `openapi-mcp` extra):

```bash
uv sync --extra litellm --extra dev --extra openapi-mcp   # add --extra memory for mem0
```

### Readable on disk locally

In the local environment every registered capability is also mirrored to
**`.data/capabilities.json`** (`app/registry/store.py`) so the registry can
be read without a DB client. Approved specs now live in the `agent_specs`
table (head) and `agent_spec_versions` (every version, immutable) -- each
spec carries `capability_snapshots`, a full copy of every capability its
steps were bound to at approval time, so a spec is self-describing.
Execution never trusts the snapshot for authorization: capabilities are
re-resolved against current grants on every run.

## Tests

```bash
pytest -q                                        # SQLite + in-process Redis, no services

# Same suite on real infrastructure (RLS enforced: connect as a NON-superuser role)
DATABASE_MIGRATION_URL=postgresql+asyncpg://postgres@localhost:5432/agentspec \
DATABASE_URL=postgresql+asyncpg://agentspec_app:app@localhost:5432/agentspec \
REDIS_URL=redis://localhost:6379/0 pytest -q
```

209 tests. Beyond the original coverage: platform primitives
(`test_platform_core.py`), the session runtime end to end incl. resumable
streams, crash reclaim, confirmations, memory and compaction
(`test_sessions_runtime.py`), and the data platform -- outbox, audit, RLS,
partitions, migrations, mem0 on pgvector (`test_data_platform.py`).

Also covers registry/marketplace grant visibility, the offline planner, guardrail
policy (denylist/redaction/dynamic-HIL/adaptive-escalation), the OAuth
delegation chain (including a regression test for a real scoping bug this
build caught and fixed -- see `tests/test_delegation.py`), and idempotency.

## Repository layout

See the architecture doc's "Repository layout & phased build plan" section
for the original Phase 0-3 layout, and **`ARCHITECTURE_MODULES.md`** for a
module-by-module map (with a diagram) of the full repo as it stands now,
config plane and adk-web agent included.
