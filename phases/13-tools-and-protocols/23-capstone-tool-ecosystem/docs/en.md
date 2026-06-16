# Capstone — Building a Complete Tool Ecosystem

> Phase 13 taught every piece individually. This capstone stitches them into a production-shaped system: an MCP server with tools + resources + prompts + tasks + UI, OAuth 2.1 at the edge, an RBAC gateway, a multi-server client, an A2A sub-agent call, OTel traces into a collector, tool-poisoning detection in CI, and an AGENTS.md + SKILL.md bundle. By the end, you can defend every architectural choice.

**Type:** Build
**Languages:** Python (standard library, end-to-end ecosystem scaffolding)
**Prerequisites:** Phase 13 · 01 through 21
**Time:** ~120 minutes

## Learning Objectives

- Compose an MCP server exposing tools, resources, prompts, and a task with a `ui://` app.
- Front the server with an OAuth 2.1 gateway that enforces RBAC and pinned hashes.
- Write a multi-server client that traces end-to-end with OTel GenAI attributes.
- Delegate part of the workload to an A2A sub-agent; verify opacity is preserved.
- Package the entire stack with AGENTS.md + SKILL.md so other agents can drive it.

## The Problem

Deliver the "research and report" system:

- A user asks: "Summarize the three most-cited papers on agent protocols from arXiv 2026."
- The system: searches arXiv via MCP; delegates paper summarization to a specialized writer agent via A2A; aggregates results; renders an interactive report as an MCP Apps `ui://` resource; logs every step to OTel.

Every Phase 13 primitive shows up. This is not a toy — production-grade research-assistant systems shipped by Anthropic (Claude Research product), OpenAI (GPTs with Apps SDK), and third parties in 2026 are exactly this shape.

## The Concept

### Architecture

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long-running)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Trace Hierarchy

```
agent.invoke_agent
 ├── llm.chat (opening)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (internal, opaque)
 ├── mcp.call -> tools/call generate_report (task-enhanced)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

One trace ID. Every span has the correct `gen_ai.*` attributes.

### Security Posture

- OAuth 2.1 + PKCE with a resource indicator pinning the audience to the gateway.
- The gateway holds upstream credentials; users never see them.
- RBAC: `alice` has `research:read`, `research:write` and can call all tools. `bob` has `research:read` and cannot call `generate_report`.
- Pinned description manifest: drops any server whose tool hashes have changed.
- Rule of Two audit: no tool combines untrusted input, sensitive data, and consequential action simultaneously.

### Rendering

The final `generate_report` task returns content blocks plus a `ui://report/current` resource. The client's host (Claude Desktop, etc.) renders the interactive dashboard in a sandboxed iframe. The dashboard contains a sorted paper list, citation counts, and a button that calls `host.callTool('summarize_paper', {arxiv_id})` for any paper the user clicks.

### Packaging

The entire system ships as:

```
research-system/
  AGENTS.md                     # Project conventions
  skills/
    run-research/
      SKILL.md                  # Top-level workflow
  servers/
    research-mcp/               # MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

Users deploy with `docker compose up`. Claude Code, Cursor, Codex, and opencode users can drive the system by triggering the `run-research` skill.

### What Each Phase 13 Lesson Contributes

| Lesson | What the capstone uses |
|--------|------------------------|
| 01–05 | Tool interfaces, provider portability, parallel calls, schemas, linting |
| 06–10 | MCP primitives, server, client, transport, resources + prompts |
| 11–14 | Sampling, roots + elicitation, async tasks, `ui://` apps |
| 15–17 | Tool poisoning, OAuth 2.1, gateways + registries |
| 18 | A2A sub-agent delegation |
| 19 | OTel GenAI tracing |
| 20 | LLM-layer routing gateway |
| 21 | SKILL.md + AGENTS.md packaging |

## Use It

`code/main.py` stitches patterns from earlier lessons into a runnable demo. All standard library, all in-process, so you can read it end to end. It runs the full research-and-report scenario: handshake with the gateway, OAuth 2.1 simulation, tools/list merge, generate_report as a task, A2A call to the writer, ui:// resource return, OTel span emission.

What to look at:

- A single trace ID threading through every hop.
- The gateway policy blocking the second user from writing.
- The task lifecycle going working → completed while returning both text and ui:// content.
- The A2A call's internal state remaining opaque to the orchestrator.
- AGENTS.md and SKILL.md being the only files another agent needs to reproduce this workflow.

## Ship It

This lesson produces `outputs/skill-ecosystem-blueprint.md`. Given a product requirement (research, summarization, automation), this skill produces the full architecture: which MCP primitives, which gateway controls, which A2A calls, which telemetry, and which packaging.

## Exercises

1. Run `code/main.py`. Note the single trace ID and how spans nest. Count how many Phase 13 primitives the demo touches.

2. Extend the demo: add a second backend MCP server (e.g., `bibliography`); confirm the gateway merges its tools into the same namespace.

3. Replace the fake A2A writer agent with a real one running in a subprocess. Use Lesson 19's scaffolding.

4. Between the orchestrator and LLM, add a PII redaction step to the routing gateway. Confirm that email addresses in the user query are sanitized.

5. Write an AGENTS.md for a teammate who will maintain this system. It should be readable in under five minutes and give them everything needed to drive this capstone (in Cursor or Codex).

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Capstone | "Phase 13 integration demo" | End-to-end system using every primitive |
| Research and report | "The scenario" | Search, summarize, render pattern |
| Ecosystem | "All the pieces together" | Server + client + gateway + sub-agent + telemetry + packaging |
| Trace hierarchy | "Single trace ID" | Spans across every hop share the trace; parent-child via span ID |
| Gateway-issued token | "Transitive auth" | The client only sees the gateway's token; the gateway holds upstream credentials |
| Merged namespace | "All tools in one flat list" | Multi-server merge at the gateway, prefixed on conflict |
| Opacity boundary | "A2A calls hide internals" | The sub-agent's reasoning is invisible to the orchestrator |
| Three-layer stack | "AGENTS.md + SKILL.md + MCP" | Project context + workflow + tools |
| Defense-in-depth | "Multiple security layers" | Pinned hashes, OAuth, RBAC, Rule of Two, audit log |
| Spec compliance matrix | "What we deliver vs. what the spec requires" | Checklist mapping deliverables to 2025-11-25 requirements |

## Further Reading

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — Consolidated reference
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — Where the protocol is headed
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 reference
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Authoritative tracing conventions
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — Production agent runtime patterns
