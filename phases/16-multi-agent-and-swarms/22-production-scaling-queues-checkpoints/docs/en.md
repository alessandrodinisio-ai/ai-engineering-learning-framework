# Production Scaling — Queues, Checkpoints, Persistence

> Scaling multi-agent systems to thousands of concurrent runs requires **durable execution**. LangGraph's runtime writes a checkpoint keyed by `thread_id` after each super-step (Postgres by default); a crashing worker releases a lease and another worker resumes. Agents can sleep indefinitely awaiting human input. **MegaAgent** (arXiv:2408.09955) runs a per-agent producer-consumer queue with three states (Idle / Processing / Response) and two-tier coordination (intra-group chat + inter-group manager chat). **Fiber/async** beats "thread-per-job" on LLM streaming: threads sit idle 99% of the time waiting for tokens; fibers cooperatively yield on I/O. Counterpoint: Ashpreet Bedi's "Scaling Agentic Software" argues **FastAPI + Postgres, nothing else** until load proves you need more—simple architectures go further than expected. This lesson builds a durable checkpoint log, a per-agent work queue with state transitions, an async-vs-thread demo, and grounds the pragmatic "start simple" rule.

**Type:** Learn + Build
**Languages:** Python (standard library, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 minutes

## The Problem

A prototype multi-agent system works fine on a laptop with an in-memory event loop running three agents. You move to production:

- Agents sometimes run for hours (long research, human-in-the-loop waits).
- Worker processes crash. Restart loses state.
- Peak load is 10x average; you need horizontal scaling.
- Users pay per agent-run; you need exactly-once billing semantics.

In-memory event loops can't do any of these. You need a durable execution layer underneath. The 2026 standard options are:

1. Workflow engines with checkpointing (Temporal, LangGraph runtime).
2. Message queues with state stores (Postgres + SQS/RabbitMQ).
3. Actor-model frameworks (MegaAgent's per-agent producer-consumer).
4. Hand-rolled FastAPI + Postgres (Bedi's argument).

This lesson builds a miniature version of each.

## The Concept

### Durable execution, the pattern

A durable execution engine persists full program state after each "step" (called a super-step in LangGraph's terminology). On crash:

```
worker crashes mid-step
  -> lease times out
  -> another worker picks up this thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

For this to work you need:

- **Serializable state.** All agent state must be persistable. Function closures with live DB connections won't survive.
- **Deterministic resumption.** Given the same state and the same inputs, the agent produces the same actions (or LLM calls are delegated to an external deterministic oracle).
- **Idempotent side effects.** External calls (tool calls, payments) must be idempotent or use a deduplication key.

LangGraph writes checkpoints after each super-step; Temporal writes after each activity; Restate uses event-sourcing logs. All three implement the same pattern.

### LangGraph's runtime

Each agent has a `thread_id`; state is a typed dict; each super-step writes a row to the checkpoints table. On resumption, the runtime replays from the last checkpoint rather than starting over. Agents can `interrupt()` to wait for human input; the runtime persists and releases the worker. When input arrives, any worker can resume.

This is the reference production design as of April 2026.

### MegaAgent's per-agent queues

arXiv:2408.09955 describes a scale experiment: thousands of concurrent agents in a cluster. Architecture:

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages sent to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat       (agents in the same group)
  inter-group manager chat (high-level routing)
```

Two-tier coordination lets intra-group conversations be dense while inter-group stays sparse—the pattern that keeps cost linear over thousands of agents.

### Async vs thread-per-job

LLM calls are I/O-bound. A thread waiting for the next token is idle 99% of the time. Threads cost ~1MB each in memory; 10k concurrent calls is 10GB of stacks alone.

Fibers (Python `asyncio`, Go goroutines, Rust `tokio`) cooperatively yield on I/O. The same 10k calls fit comfortably in-process. At LLM-agent scale, async isn't an optimization—it's the architecture itself.

Exception: CPU-bound post-processing (embedding, tokenizer tricks) still wants threads or processes. Separate your I/O tier from your CPU tier.

### Bedi's counterpoint

"Scaling Agentic Software" (Ashpreet Bedi, 2026) argues most teams over-engineer before measuring load. The pragmatic default:

- FastAPI + Postgres.
- Each agent run is a row; state is updated in place with optimistic concurrency.
- Background tasks via `pg_notify` or a simple Celery worker.
- Retry strategy in application code.

For loads of ~100 or fewer concurrent agent-runs on controllable tasks, this is often all you need. Upgrade when you measure that it can't keep up.

Rule: adopt a durable execution framework when you hit a specific problem simple architecture can't solve. Premature adoption burns time on ceremony that doesn't pay back.

### Exactly-once semantics

For billed agent runs, you need "effectively exactly-once" (at-least-once delivery + idempotent consumers). Engineering moves:

- **Deduplication key per run.** Embed it in every side-effect call.
- **Outbox pattern.** Side effects are first written to a table, then executed by a separate process. Both steps are idempotent.
- **Compensating transactions.** When a side effect succeeds but its tracking write fails, schedule a compensation.

These are database engineering patterns, not LLM-specific. The LLM tax is simply that LLM calls are slow; the rest is standard distributed systems.

### Rainbow deploys

Anthropic's multi-agent research system uses "rainbow deploys": multiple versions of the agent runtime run concurrently so long-running agents don't have to be killed on every code deploy. Canary a new version on a small slice of traffic; retire old versions after their agents finish.

This is standard for long-running stateful systems; the 2026 adaptation is that agents can live for hours, so deploy cycles must accommodate that.

### Standard production checklist

- Durable state (checkpoints, snapshots, or outbox + replayable logs).
- Idempotent side effects.
- Async I/O tier for LLM calls.
- At-least-once delivery with deduplication.
- Rainbow/canary deploys for stateful workloads.
- Observability: per-agent traces, super-step audit, retry counters.

## Build It

`code/main.py` implements:

- `CheckpointStore` — SQLite-backed checkpoint log keyed by thread-id. Each super-step appends a row.
- `run_with_checkpoint(agent, thread_id)` — simulates a mid-run crash; a second worker resumes from the last checkpoint.
- `AgentQueue` — per-agent Idle / Processing / Response state machine with a small work queue.
- `demo_async_vs_threads()` — runs 500 concurrent simulated "LLM calls" via asyncio and via threads; reports wall-clock time and peak memory (approximate).

Run:

```
python3 code/main.py
```

Expected output: checkpoint recovery succeeds after simulated crash; async version handles 500 concurrent calls in < 1 second; threaded version takes several seconds and uses orders of magnitude more memory per concurrent unit.

## Use It

`outputs/skill-scaling-advisor.md` advises on durable execution choices: FastAPI + Postgres, LangGraph runtime, Temporal, or custom. Calibrated by load, state retention needs, and deploy frequency.

## Ship It

Standard production hardening:

- **Start simple (Bedi's rule).** FastAPI + Postgres until you measure it can't keep up.
- **Instrument everything before optimizing.** Per-run latency histogram, per-step time, retry counts, failure classification.
- **Outbox pattern for side effects.** Especially payments and external API calls.
- **Rainbow deploys.** Never kill in-flight agent runs during deploys.
- **Adopt durable execution engines (Temporal / LangGraph / Restate) when you hit a concrete problem:** multi-hour human-in-the-loop waits, cross-region coordination, complex retry/compensation strategies.
- **Async for the I/O tier.** Threads only for CPU-bound post-processing.

## Exercises

1. Run `code/main.py`. Confirm checkpoint recovery works; measure the concurrency difference between async and threads.
2. Implement an **outbox** table: each tool call first writes to outbox, then a separate goroutine/task executes it. Verify idempotency by running the tool call twice.
3. Simulate a **rainbow deploy**: two concurrent runtime versions; route half of new thread_ids to each; confirm in-flight threads on the old version aren't interrupted.
4. Read LangGraph's runtime docs (link below). Identify which runtime features are hardest to replicate in a hand-rolled FastAPI + Postgres version. Is this a reason to adopt it, or can you defer?
5. Read MegaAgent (arXiv:2408.09955) Section 3. The two-tier coordination (intra-group + inter-group manager chat) is explicit. Sketch how you'd map it to a message queue with two queue families.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Durable execution | "Persisted program state" | Engine writes state after each super-step; crash recovery is deterministic. |
| Super-step | "Transaction boundary" | Unit of work between checkpoints. LangGraph terminology. |
| thread_id | "Agent run identifier" | Key that ties checkpoints and resumption logic together. |
| Idempotency | "Safe to retry" | Repeating a side effect produces the same result as a single attempt. |
| Outbox pattern | "Decoupled side effects" | Write intent to a table; a separate executor processes and marks done. |
| At-least-once delivery | "Might have duplicates" | Message queue semantics; dedup keys make consumers effectively exactly-once. |
| Rainbow deploy | "Version overlap" | Multiple runtime versions run concurrently during long-running workloads. |
| Async fiber | "Cooperative yield" | User-space concurrency; cheaper than threads for I/O-bound workloads. |
| Checkpoint | "State snapshot" | Serialized state at a super-step boundary; key to recovery. |

## Further Reading

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph runtime design
- [MegaAgent](https://arxiv.org/abs/2408.09955) — Per-agent producer-consumer queues; two-tier coordination at thousands of concurrent agents
- [Matrix](https://arxiv.org/abs/2511.21686) — Decentralized framework with message queues as coordination substrate
- [Temporal docs](https://docs.temporal.io/) — Reference workflow engine for durable execution
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — Production lessons including rainbow deploys
