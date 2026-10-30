# Evaluating agents

This platform runs many agents that clients configure themselves (a spec made from a plan, tools from the registry, their own policy), plus a router that sends a request to the right one. Evaluation therefore has to work on agents we did not write and whose correct behaviour we do not know. This document says what is checked, how, and what it cannot tell you.

Code: `app/evals/`. API: `/v1/evals/...`. CLI: `scripts/eval.py`. Tests: `tests/test_evals.py`.

## What this proves, and what it does not

| Question | Can the framework answer it? | How |
|---|---|---|
| Does the platform enforce guardrails, approvals and policy on this agent? | Yes, for any config | Generated cases: an injection is blocked before any tool runs, an approval pause and a rejection execute nothing, nothing forbidden executes |
| Does the agent do what its own spec says? | Yes | Generated happy-path case: every tool step runs, the run completes, within budget |
| Is the answer supported by what the tools returned? | Yes, as a score | LLM judge, rubric `groundedness`, with the tool outputs as evidence. No reference answer needed |
| Did a spec edit, a model switch or a platform change alter the agent's behaviour? | Yes | Baseline comparison and the regression gate |
| Is the agent's business logic correct? | **No, only with the client's data** | Needs expected results that only the client knows. They come from cases the client writes or promotes from real sessions and reviews |
| Is the judge itself accurate on a live model? | **Not validated yet** | The judge is a model with its own noise; treat its scores as trends |

The generated cases check that the platform and the agent agree with the spec. They cannot tell whether the spec is the right one. That is why client-owned golden cases exist.

## The three layers

1. **Deterministic checks (hard gates).** Exact, cheap, never flaky. Tool trajectory (exact, in order, or any order; per-call status, output and agent), forbidden tools, policy outcome (completed, awaiting approval, blocked), router handoffs, answer contains / must not contain / regex, tool-call, token and latency budgets, and no error. A case passes when all of its hard checks pass.
2. **LLM-as-judge (soft scores).** Rubrics: `task_completion`, `groundedness`, `tool_use`, `policy_compliance`, `helpfulness`, each 1 to 5 normalised to 0..1. The judge sees the user messages, every tool call with its real output, and the final answer, and is told to treat all of it as data. Temperature 0, optional median of an odd number of samples, optional separate judge model (`EVAL_JUDGE_MODEL_NAME`). Without a model key the offline `HeuristicJudge` runs instead (number overlap with tool outputs, outcome checks); the report names the judge used. Judge scores feed trends and a tolerance gate, never the hard pass/fail.
3. **Regression.** Each run is compared with the set's baseline run: a case that passed and now fails, a pass-rate drop (`EVAL_GATE_MAX_PASS_RATE_DROP`, default 0), or a mean judge-score drop past `EVAL_GATE_MAX_JUDGE_DROP` (default 0.05) fails the gate. With no baseline yet, the gate is "every hard check passes".

## Individual and end to end

- **Individual (one spec agent):** target `{"kind": "spec", "spec_id": ...}`; the compiled agent is run exactly as the production runner runs it (same plugins, step tools, policy and approval gate), in-process with an in-memory session.
- **End to end (router + sub-agent + tools):** target `{"kind": "router"}`; `expectations.transfers` asserts the handoff to the right sub-agent and `expectations.tool_calls[].agent` pins which agent made each call.
- **Planner:** `planner_eval` runs intent to plan and checks the plan with the same validators approval uses: capabilities exist and are visible to the tenant (a hallucinated or cross-tenant id fails), tool and arguments satisfy the tool's input schema, dependencies are acyclic, required capabilities are present, no forbidden ones, step count within budget. One retrieval case per registered capability is generated from its own description.

## Where cases come from

- **Generated** (`generator.py`) from every approved spec: happy path, approval pause, approval rejection (only when the spec has approval rules), guardrail injection, and an adversarial "ignore the injected instruction" case judged by rubric; plus router cases. Tagged `synthetic`. Regenerated for each new spec version.
- **Client-owned golden sets** (`golden.py`): written through `PUT /v1/evals/sets/{id}` or imported as JSONL (one case per line, so they live in git and are reviewed in pull requests).
- **Promoted from real sessions:** `POST /v1/evals/sets/{id}/promote` turns a stored conversation into a draft case: messages with PII and secrets redacted, the tool path and outcome the agent actually took as the expectation, and its final answer as the reference. It is a draft: it records what happened, not necessarily what should have. A person reviews it before it gates anything.
- **Sampled live traffic** (`online.py`): with `EVAL_ONLINE_SAMPLE_RATE` above 0, that share of finished runs is scored in a background task from the events the runner already streams (no extra agent run). The stored copy is redacted: tool names and status only, no arguments or outputs. `GET /v1/evals/online` returns the mean score and the weakest conversations to promote.

## Auto-eval

- **On spec approval** (`EVAL_ON_APPROVAL=true`): runs the tenant's sets that have a case for the approved spec, plus a freshly generated suite for it, each compared with its baseline. Recorded as an eval run, an audit entry and the `eval_runs_total` / `eval_pass_rate` metrics; a failed gate is logged at WARNING and shown at `GET /v1/evals/specs/{spec_id}/status`. It does **not** block the approval (the spec is already saved); it is a tripwire.
- **In CI:** `scripts/eval.py run ... --gate` exits 1 when the gate fails. This is the pre-merge gate.
- **Online:** see above.

## CLI

```bash
# deterministic run with the scripted model: no API key, no flakiness
python scripts/eval.py generate --tenant acme --set synthetic
python scripts/eval.py run --tenant acme --set synthetic --model scripted --no-judge --gate --make-baseline

# a JSONL golden file kept in git, live model, judge on
python scripts/eval.py run --tenant acme --file evals/quotes.jsonl --gate

python scripts/eval.py planner --tenant acme
python scripts/eval.py runs --tenant acme
python scripts/eval.py export --tenant acme --set synthetic > cases.jsonl
```

Router cases with the scripted model need `--transfer-to <sub-agent names>`; with the live model the router decides.

## Case format (JSONL, one per line)

```json
{"case_id": "quote-sku3", "target": {"kind": "spec", "spec_id": "spec_ab12"},
 "messages": ["Quote 4 x SKU-3"], "confirmation": "approve",
 "expectations": {"tool_calls": [{"name": "quote", "ok": true, "output_contains": ["SKU-3"]}],
                  "trajectory_mode": "in_order", "outcome": "completed",
                  "answer_contains": ["total"], "max_tool_calls": 3},
 "rubrics": ["task_completion", "groundedness"], "tags": ["smoke"]}
```

`confirmation` is what the simulated approver does when a tool asks: `approve`, `reject`, or `leave_pending` (stop at the pause and report `awaiting_approval`). It answers through the chat channel; a spec whose approvals are inbox-only stays `awaiting_approval`.

## Things to know before running it

- **Evals call real tools.** An eval runs the agent for real, with the spec's fixed tool arguments, against the tenant's real capabilities. A side-effecting tool will actually run. Point evals at a sandbox tenant or read-only tools, or keep side-effecting steps behind an approval rule and use `reject` / `leave_pending` cases for them. A dry-run mode with recorded tool responses is not built.
- **Isolation.** Runs happen in the caller's tenant with the caller's scopes, as the synthetic user `eval-bot`, under row-level security. The approval rows an eval creates are deleted at the end of the run and never touch a real person's. Cases of one spec run one after another because approvals match on the exact call (tenant, spec, step, arguments); different specs run in parallel and router cases run last.
- **Cost.** Eval runs consume the tenant's model tokens and daily budget; judge calls cost extra. The API runs at most 50 cases inline; use the CLI for more.
- **Not built yet:** thumbs up/down labels on real sessions feeding the golden set; LLM-generated test inputs from a spec; repeated runs to measure variance; a pre-approval gate that evaluates a candidate spec before it is saved; dry-run tool mocking.
- **Not validated:** judge accuracy against human labels, and any eval on a live model at scale. All 31 framework tests use the scripted model and the test worker, on SQLite and on Postgres 16 with Redis 7.
