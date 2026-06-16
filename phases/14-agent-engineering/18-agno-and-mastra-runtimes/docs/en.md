# Agno and Mastra: Production Runtimes

> Agno (Python) and Mastra (TypeScript) are the 2026 production runtime duo. Agno targets microsecond agent instantiation and stateless FastAPI backends. Mastra provides agents, tools, workflows, unified model routing, and composite storage on top of the Vercel AI SDK.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Learning Objectives

- Identify Agno's performance targets and when they matter.
- Name Mastra's three primitives — Agents, Tools, Workflows — and the supported server adapters.
- Explain why a stateless, session-scoped FastAPI backend is Agno's recommended production path.
- Choose between Agno and Mastra for a given tech stack (Python-first vs TypeScript-first).

## The Problem

LangGraph, AutoGen, and CrewAI all feel framework-heavy. Teams that want "just an agent loop, fast, in my runtime" reach for Agno (Python) or Mastra (TypeScript). Both trade some framework-owned primitives for raw speed and tighter integration with the surrounding stack.

## The Concept

### Agno

- Python runtime, formerly Phi-data.
- "No graphs, chains, or convoluted patterns — just pure python."
- Performance targets from their docs: ~2 μs agent instantiation, ~3.75 KiB memory per agent, ~23 model providers.
- Production path: stateless, session-scoped FastAPI backend. Each request spins up a fresh agent; session state lives in the database.
- Native multimodal (text, image, audio, video, files) and agentic RAG.

The speed targets matter when you have thousands of short-lived agents per second (chat fan-in, eval pipelines). They matter less when a single agent runs for 10 minutes.

### Mastra

- TypeScript, built on the Vercel AI SDK.
- Three primitives: **Agents**, **Tools** (Zod-typed), **Workflows**.
- Unified model router — 3,300+ models across 94 providers (March 2026).
- Composite storage: memory, workflows, and observability can each go to different backends; ClickHouse recommended for observability at scale.
- Apache 2.0, with an `ee/` directory under a source-available enterprise license.
- Server adapters for Express, Hono, Fastify, Koa; first-class Next.js and Astro integrations.
- Ships Mastra Studio (localhost:4111) for debugging.
- 22k+ GitHub stars, 300k+ weekly npm downloads at 1.0 (January 2026).

### Positioning

Neither wants to be LangGraph. They compete on:

- **Language fit.** Agno for Python-first teams; Mastra for TypeScript-first.
- **Runtime ergonomics.** Agno = near-zero overhead; Mastra = Vercel ecosystem integration.
- **Observability.** Both integrate with Langfuse/Phoenix/Opik (Lesson 24), but Mastra Studio is first-party.

### When to Choose Which

- **Agno** — Python backend, many short-lived agents, strong performance requirements, FastAPI teams.
- **Mastra** — TypeScript backend, Next.js / Vercel deployments, unified multi-provider model routing, Zod-typed tools.
- **LangGraph** (Lesson 13) — when durable state and explicit graph reasoning matter more than raw speed.
- **OpenAI / Claude Agent SDK** — when you want the provider's productionized shape (Lessons 16–17).

### Where This Pattern Breaks Down

- **Performance for performance's sake.** Choosing Agno because "2 μs" sounds impressive when the workload is a single slow agent call per request. Overhead is not the bottleneck.
- **Ecosystem lock-in.** Mastra's Vercel-flavored integrations are a plus on Vercel, a minus elsewhere.
- **Enterprise license confusion.** Mastra's `ee/` directory is source-available, not Apache 2.0. Read the license if you plan to fork.

## Build It

This lesson is primarily comparative — no single code artifact covers both frameworks well. See `code/main.py` for a side-by-side toy: a minimal "run an agent, stream output, persist session" flow implemented twice (once in Agno shape, once in Mastra shape).

Run it:

```
python3 code/main.py
```

Two structurally different but functionally equivalent traces.

## Use It

- **Agno** — Python backends needing speed and FastAPI shape.
- **Mastra** — TypeScript backends with many providers and workflow primitives.
- Both offer first-party observability hooks. Both integrate with Langfuse.

## Ship It

`outputs/skill-runtime-picker.md` picks between Agno, Mastra, LangGraph, or a provider SDK based on tech stack, latency budget, and ops shape.

## Exercises

1. Read Agno's docs. Port the standard-library ReAct loop (Lesson 01) to Agno. What disappears? What remains?
2. Read Mastra's docs. Port the same loop to Mastra. What changes with tool typing (Zod vs nothing)?
3. Benchmark: measure agent instantiation latency on your stack. Does Agno's 2 μs matter for your workload?
4. Design a migration: if you've been running CrewAI in Python, what breaks moving to Agno?
5. Read Mastra's `ee/` license terms. Which restrictions would affect an open-source fork?

## Key Terms

| Term | Common description | What it actually is |
|------|----------------|------------------------|
| Agno | "fast Python agents" | Stateless, session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "multi-provider access" | Single client for 3,300+ models across 94 providers |
| Composite storage | "multiple backends" | Memory/workflows/observability each to different stores |
| Mastra Studio | "local debugger" | localhost:4111 UI for introspecting agents |
| Source-available | "not OSS" | License allows reading source but restricts commercial use |

## Further Reading

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) — performance targets, FastAPI integration
- [Mastra docs](https://mastra.ai/docs) — primitives, server adapters, Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — stateful graph alternative
- [Comet Opik](https://www.comet.com/site/products/opik/) — observability comparison referenced by Mastra integrations
