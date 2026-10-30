# Agent-as-Spec — Pitch Script

Use the version that fits the time you have. Words in *italics* are stage directions.

---

## Taglines

- **Agents as specs.** Discover tools, compose, approve, govern.
- Configure AI agents. Don't code them.
- Every agent reviewed. Every action governed.

**One-liner:** Agent-as-Spec turns AI agents into reviewable, versioned configurations, built from a shared marketplace of tools and governed by policy and human approval before every action.

---

## 30-second elevator pitch

> Every company wants AI agents that *do* things, like file tickets, move money or update records. Two things stop them. The tools agents need are scattered across teams with no way to share them, and once an agent is live, nobody controls what it actually does.
>
> Agent-as-Spec fixes both. Teams publish their tools to a shared marketplace. Anyone can compose those tools into an agent, and a person approves the result as a versioned **spec**, so no code is needed. Then every tool call the agent makes goes through policy first: rules, confidence checks, and human approval where it matters.
>
> It's the layer that makes agents safe enough to put in production.

---

## 2-minute pitch (marketing / demo day)

**The hook (15s)**
> Building an AI agent demo takes an afternoon. Getting it into production takes months, and most never get there. That's not because the model is bad. It's because nobody can answer two questions: *what is this agent allowed to use*, and *who's accountable when it acts*?

**The problem (25s)**
> In most companies, the tools an agent needs, like MCP servers, partner agents and internal APIs, are spread across teams with no catalog and no access process. Agents themselves are code, so every change needs an engineer and nobody reviews the plan. And when the agent runs, it calls real systems with nothing between it and the action.

**The solution (45s)**
> Agent-as-Spec is built on Google's Agent Development Kit and has three parts.
>
> **One: a marketplace.** Teams register MCP servers, A2A agents and REST APIs in one catalog. Other teams search it and request access, and owners approve.
>
> **Two: agents as specs.** You describe the agent you want or pick tools directly. The platform drafts a plan, you edit it, and a person approves it. That approval produces a versioned spec: the tools, the instructions, how the agent reasons, and its rules. The spec *is* the agent. It's compiled into a live agent automatically, so changing it is an edit, not a deploy.
>
> **Three: governance on every call.** Before any tool runs, the policy engine checks it. It can block it, let it through, or pause for a human, based on the arguments, the time, how many retries there have been, or how confident the agent is. Approvers see exactly why the call paused and decide it from chat or an inbox.

**Why now / close (20s)**
> Standards like MCP and A2A mean tools and agents can finally talk to each other. What's missing is the layer that decides *which* ones should, and *when*. That's Agent-as-Spec. Agents you can review before they run, and trust while they do.

---

## Interview version (~3 minutes, first person)

**Context**
> I wanted to solve the gap between agent demos and agents in production. The blockers I kept seeing weren't model quality. They were tool sprawl across teams, agents hard-coded by engineers, and no control over what an agent does at runtime.

**What I built**
> I built Agent-as-Spec, a multi-tenant platform on Google ADK, in nine modules. A **capability registry** unifies MCP servers, A2A agents and OpenAPI services under one schema. A **marketplace** adds cross-tenant access requests, bring-your-own-keys and a usage log. An **agent builder** turns a plain-English request into a typed plan; a person approves it into a versioned spec that snapshots every tool's schema. A **factory** compiles specs into live ADK agents across seven patterns: single, sequential, parallel, loop, hierarchical, ReAct and plan-and-execute. A **governance layer** sits in front of every tool call. The supporting modules are runtime, identity, knowledge and observability.

**Technical decisions I'd highlight**
- **The spec is the source of truth.** Approving a plan freezes a snapshot of each tool's schema and the final instructions into the spec. It can be read and run on its own, even if the registry changes later.
- **One gate for every tool call.** Guardrails, delegated auth, idempotency, telemetry and policy all go through a single wrapper, so nothing bypasses them.
- **Policy as data.** Rules are declarative: conditions over facts like `args.amount`, `attempt` or `confidence`, with 11 operators. The most restrictive effect wins: deny, then require approval, then allow. Rules are validated when a spec is approved, not when it runs.
- **Confidence-based human-in-the-loop.** Confidence combines the model's own estimate with signals from the run, such as unmet dependencies, earlier failures and guardrail blocks, and uses the lower of the two. When a call pauses, the approver sees exactly why.
- **Durable, single-use approvals.** An approval is tied to the exact call and its arguments, can only be used once, and expires. It works through ADK's native pause/resume in chat, or through a separate inbox when the requester shouldn't approve their own call.
- **Security by default.** Every tool call gets a short-lived delegated token, discovery is kept separate from what you're allowed to use, and all text from other tenants is escaped in the UI.

**How I validated it**
> I wrote a test worker that serves 42 real capabilities: 22 MCP servers and 20 A2A agents. I also wrote a deterministic scripted model, so the whole flow is tested without a live LLM: 158 automated tests, plus end-to-end browser tests of the UI.

**A problem I solved**
> One example: after an agent handed off to a sub-agent, input guardrails stopped working. ADK replaces the last user message with its own "for context" note during a handoff, so keyword checks never saw what the person actually typed. I traced it, made the guardrails also check the original user content, and added a regression test.

**What's next**
> Validating on a live model at scale, approver roles for separation of duties, semantic search over knowledge, and live notifications.

---

## 5-minute demo flow

1. **Marketplace.** *Register the test worker.* "42 capabilities published in one click: MCP servers and A2A agents."
2. **Discover.** *Search "math".* "Everything I can use, with argument types. Another tenant would see **Request access** instead."
3. **Compose.** *Add two tools and a human sign-off step.* "The plan fills in argument templates from each tool's schema."
4. **Policy.** *Open Tool policy: require approval before `add`, set a confidence threshold of 0.7, add a rule that denies after 3 attempts, set Inbox only.* "This is the governance, as configuration."
5. **Approve.** *Name it and Approve & save.* "Validated and versioned. It's now a live agent in `adk web`."
6. **Run and pause.** *Run it in chat.* "The agent stops before `add` and explains why."
7. **Inbox.** *Open Approvals and approve.* "The approver sees the reasons, the confidence breakdown and the exact call. The approval is good for that one call only."
8. **Iterate.** *Saved Specs → Edit as new version.* "Change the policy and save v2. No deploy."

---

## Likely questions

**How is this different from LangChain or plain ADK?**
Those are frameworks for *writing* agents. This is the platform layer on top: a shared catalog, access control between teams, agents as reviewable configuration, and policy on every call. It uses ADK underneath.

**Why not just tell the prompt "ask before risky actions"?**
A prompt is a suggestion. The model can ignore it, and a prompt injection can override it. Here the check runs in code before the tool call, outside the model's control.

**Can an agent run without anyone approving it?**
Only if its policy allows that. An approval-before step or a deny rule is enforced every time, and inbox-only mode stops the person chatting from approving their own call.

**What happens if a tool changes after approval?**
The spec keeps a snapshot of the schema it was approved against. Re-approving creates a new version, so changes are explicit.

**Does it lock me into one model?**
No. It runs on Gemini or on any OpenAI-compatible gateway through LiteLLM.

**Is it production-ready?**
The architecture and governance are built and tested. The next steps before production are live-model validation at scale, approver roles and a real org model.
