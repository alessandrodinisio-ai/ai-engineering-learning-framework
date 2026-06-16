# A2A — Agent-to-Agent Protocol

> Google announced A2A in April 2025; by April 2026 the spec lives at https://a2a-protocol.org/latest/specification/ with 150+ organizations backing it. A2A is the lateral complement to MCP (Lesson 13): MCP is vertical (agent ↔ tool), A2A is peer-to-peer (agent ↔ agent). It defines Agent Cards (discovery), Tasks with artifacts (text, structured data, video), an opaque task lifecycle, and authentication. Production systems increasingly pair MCP and A2A together. Google Cloud rolled A2A support into Vertex AI Agent Builder throughout 2025-2026.

**Type:** Learn + Build
**Languages:** Python (standard library, `http.server`, `json`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 min

## The Problem

Your agent needs to call another agent on a different system. How? You could expose an HTTP endpoint, define a custom JSON schema, and pray the other side speaks it. Every agent pair becomes a custom integration.

A2A is the universal wire protocol for that call. Standard discovery, standard task model, standard transport, standard artifacts. Like HTTP+REST, but with agents as first-class citizens.

## The Concept

### Four Building Blocks

**Agent Card.** A JSON document at `/.well-known/agent.json` describing the agent: name, skills, endpoints, supported modalities, authentication requirements. Discovery is reading this card.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task.** The unit of work. An async, stateful object with a lifecycle: `submitted → working → completed / failed / canceled`. Client sends a task, polls or subscribes for updates.

**Artifact.** The typed result of a task. Text, structured JSON, image, video, audio. Artifacts are typed so different modalities are first-class.

**Opaque lifecycle.** A2A doesn't prescribe *how* the remote agent solves the task. The client sees state transitions and artifacts; the implementation is free to use any framework.

### MCP/A2A Division of Labor

- **MCP** (Lesson 13): agent ↔ tool. An agent reads/writes tools via JSON-RPC. Stateless by default.
- **A2A**: agent ↔ agent. Peer protocol; both sides are agents with their own reasoning.

Production multi-agent systems use both. An A2A peer calls MCP tools on its own side. The division keeps both concerns clean.

### Discovery Flow

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

Or streaming: subscribe to SSE at `/tasks/{id}/events` for push updates.

### Authentication

A2A supports three common patterns:

- **Bearer token** — OAuth2 or opaque tokens.
- **mTLS** — mutual TLS; organizations prove identity to each other.
- **Signed requests** — HMAC over payload.

Authentication is declared in the Agent Card; the client discovers and complies.

### 150+ Organizations by April 2026

Enterprise adoption drove A2A's scale. The key: A2A became how enterprise agent systems cross trust boundaries. Google Cloud shipped Vertex AI Agent Builder A2A support; Microsoft Agent Framework supports it; most major frameworks (LangGraph, CrewAI, AutoGen) ship A2A adapters.

### Where A2A Wins

- **Cross-organization calls.** Company A's agent calls Company B's agent. Without A2A, every pair is a custom contract.
- **Heterogeneous frameworks.** LangGraph agent calls CrewAI agent calls custom Python agent. A2A normalizes.
- **Typed artifacts.** Video results, structured JSON, audio — all first-class.
- **Long-running tasks.** Opaque lifecycle + polling makes multi-hour tasks straightforward.

### Where A2A Struggles

- **Latency-sensitive micro-calls.** A2A's lifecycle is async. Sub-millisecond agent-to-agent doesn't fit; use direct RPC.
- **Tightly-coupled in-process agents.** If two agents run in the same Python process, A2A's HTTP round-trip is overkill.
- **Small teams.** Spec overhead is real; purely internal agents may not need the ceremony.

### A2A vs ACP, ANP, NLIP

Several related specs appeared between 2024-2026:

- **ACP** (IBM/Linux Foundation) — A2A's predecessor, narrower scope.
- **ANP** (Agent Network Protocol) — peer-discovery-heavy, decentralization-first.
- **NLIP** (Ecma Natural Language Interaction Protocol, standardized December 2025) — natural-language content types.

As of April 2026, A2A is the most widely adopted peer protocol. Comparison in arXiv:2505.02279 (Liu et al., "A Survey of Agent Interoperability Protocols").

## Build It

`code/main.py` implements a minimal A2A server and client using `http.server` and JSON. The server:

- Exposes `/.well-known/agent.json`,
- Accepts `POST /tasks`,
- Manages task state,
- Returns artifacts at `GET /tasks/{id}`.

The client:

- Fetches the Agent Card,
- Submits a task,
- Polls until completion,
- Reads artifacts.

Run:

```
python3 code/main.py
```

The script starts the server in a background thread and runs the client against it. You see the full flow: discovery, submission, polling, artifact.

## Use It

`outputs/skill-a2a-integrator.md` designs an A2A integration: Agent Card contents, task schema, auth choice, streaming vs polling.

## Ship It

Checklist:

- **Pin the spec version.** A2A is still evolving; Agent Cards should declare protocol version.
- **Idempotent task creation.** Duplicate submissions (network retries) should produce only one task.
- **Artifact schema.** Declare what shape the agent returns; consumers should validate.
- **Rate limiting + auth.** A2A is internet-facing; apply standard web security.
- **Dead-letter for failed tasks.** Monitor patterns over time to surface recurring failure types.

## Exercises

1. Run `code/main.py`. Confirm the client discovers the server and receives the correct artifact.
2. Add a second skill to the server (e.g., "summarize"). Update the Agent Card. Write a client that picks the skill based on the task type.
3. Implement an SSE streaming endpoint: `/tasks/{id}/events` that emits state changes. What does the client need to do differently?
4. Read the A2A specification (https://a2a-protocol.org/latest/specification/). Identify three things the spec mandates that this demo doesn't implement.
5. Compare A2A (Agent Card discovery) vs MCP (server-side capability listing via `listTools`). What's the tradeoff between self-describing agents and capability probing?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| A2A | "Agent to agent" | Peer protocol for agents to call other agents across systems. Google 2025. |
| Agent Card | "Agent's business card" | JSON at `/.well-known/agent.json` describing skills, endpoints, auth. |
| Task | "Unit of work" | Async stateful object with a lifecycle; produces artifacts on completion. |
| Artifact | "The result" | Typed output: text, structured JSON, image, video, audio. First-class media. |
| Opaque lifecycle | "How it's solved is the agent's business" | Client sees state transitions; server is free to use any framework/tools. |
| Discovery | "Find that agent" | `GET /.well-known/agent.json` returns the card. |
| MCP vs A2A | "Tools vs peers" | MCP: vertical agent ↔ tool. A2A: horizontal agent ↔ agent. |
| ACP / ANP / NLIP | "Sibling protocols" | Adjacent specs; A2A is the most adopted in 2026. |

## Further Reading

- [A2A specification](https://a2a-protocol.org/latest/specification/) — the canonical spec
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — April 2025 launch post
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — reference implementations and SDKs
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP, ACP, A2A, ANP compared
