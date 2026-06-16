# Production Runtimes: Queue, Event, Cron

> Production agents run on six runtime shapes: request-response, streaming, durable execution, queue-based background, event-driven, and scheduled. Pick the shape before picking the framework. Observability is load-bearing in every shape.

**Type:** Learn
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 minutes

## Learning Objectives

- Name six production runtime shapes and match each to a framework/product pattern.
- Explain why durable execution (LangGraph) matters for long-span tasks.
- Describe event-driven runtimes and when Claude Managed Agents is appropriate.
- Explain the "observability is load-bearing" claim in multi-step agents.

## The Problem

Production agents fail in ways Jupyter notebooks don't expose: network timeout at step 37, user hangs up mid-voice-call, cron job hangs on machine restart, background worker runs out of memory. Runtime shape determines which failures are survivable.

## The Concept

### Request-Response

- Synchronous HTTP. User waits for completion.
- Only viable for short tasks (<30s).
- Stack: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- Observability: standard HTTP access logs + OTel spans.

### Streaming

- SSE or WebSocket for progressive output.
- LiveKit extends this to WebRTC for voice/video (Lesson 22).
- Stack: any streaming-capable framework + a frontend handling SSE/WS.
- Observability: per-chunk timing, time-to-first-token, tail latency.

### Durable Execution

- State checkpointed after every step; automatic recovery on failure.
- AutoGen v0.4 actor model isolates failures to a single agent (Lesson 14).
- LangGraph's core differentiator (Lesson 13).
- Essential when step count is unknown and recovery cost is high.

### Queue-Based / Background

- Jobs enqueue, workers pick up, results flow back via webhook or pub/sub.
- Essential for long-span agents (tens to hundreds of steps per task, per Anthropic's computer use announcement).
- Stack: Celery (Python), BullMQ (Node), SQS + Lambda (AWS), custom.
- Observability: queue depth, per-job latency distribution, DLQ size.

### Event-Driven

- Agent subscribes to triggers: new email, PR opened, cron fires.
- Claude Managed Agents covers this out of the box (Lesson 17).
- CrewAI Flows (Lesson 15) structure event-driven deterministic workflows.
- Observability: trigger source, event-to-start latency, agent latency.

### Scheduled

- Cron-shaped agents that run periodically.
- Combined with durable execution so a failed nightly run resumes on next trigger.
- Stack: Kubernetes CronJob + a durable framework; managed (Render cron, Vercel cron).

### 2026 Deployment Patterns

- **CrewAI Flows** for event-driven production.
- **Agno** stateless FastAPI for Python microservices.
- **Mastra** server adapters (Express, Hono, Fastify, Koa) for embedding.
- **Pipecat Cloud / LiveKit Cloud** for managed voice (Lesson 22).
- **Claude Managed Agents** for managed long-running async.

### Observability Is Load-Bearing

Without OpenTelemetry GenAI spans (Lesson 23) plus a Langfuse/Phoenix/Opik backend (Lesson 24), you cannot debug a multi-step agent that fails at step 40. This is not optional for production. It's the difference between "we debug fast" and "we add more logging and replay from scratch."

### Where Production Runtimes Fail

- **Wrong shape choice.** Request-response for a 5-minute task. User hangs up; workers pile up; retries stack.
- **No DLQ.** Queue workers without dead-letter. Failed jobs vanish.
- **Opaque background work.** Background agents run without exporting traces. Failures are invisible until users report.
- **Skipping durable state.** Any run > 30 seconds that can't afford restart needs durable execution.

## Build It

`code/main.py` is a standard-library multi-shape demo:

- Request-response endpoint (plain function).
- Streaming handler (generator).
- Queue-based worker with DLQ.
- Event trigger registry.
- Cron-shaped scheduler.

Run it:

```bash
python3 code/main.py
```

Output: five traces showing how each shape behaves on the same task. Same agent logic, different shell. Durable execution (the sixth shape) is intentionally covered in Lesson 13 with LangGraph checkpointing.

## Use It

- **Request-response** for chat-style UX.
- **Streaming** for progressive responses.
- **Durable** for long-span tasks.
- **Queue** for batch/async/long-running.
- **Event** for agent reactivity.
- **Cron** for chores (memory consolidation, evaluation, cost reports).

## Ship It

`outputs/skill-runtime-shape.md` picks a runtime shape for a task and wires up observability requirements.

## Exercises

1. Port your Lesson 01 ReAct loop into all six shapes in your stack. Which shape fits which product surface?
2. Add a DLQ to the queue-based demo. Simulate 10% job failures; expose DLQ size.
3. Write a cron-triggered eval agent that runs nightly against the day's top 20 traces.
4. Implement streaming with backpressure: if the client is slow, pause the agent. How does this interact with turn budget?
5. Read the Claude Managed Agents docs. When would you migrate a self-hosted long-span agent to managed?

## Key Terms

| Term | Common usage | What it actually is |
|------|----------------|------------------------|
| Request-response | "Synchronous" | User waits; short tasks only |
| Streaming | "SSE / WS" | Progressive output; better UX; latency observable per chunk |
| Durable execution | "Recover from failure" | Checkpointed state; restart from last step |
| Queue-based | "Background jobs" | Producer / worker pool / DLQ |
| Event-driven | "Trigger-based" | Agent reacts to external events |
| DLQ | "Dead-letter queue" | Parking lot for failed jobs |
| Claude Managed Agents | "Managed harness" | Anthropic-hosted long-running async with caching + compaction |

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — durable execution details
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — managed long-running async
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "tens to hundreds of steps per task"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor model fault isolation
