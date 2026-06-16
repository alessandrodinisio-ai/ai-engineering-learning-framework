# Long-Running Background Agents: Durable Execution

> Production long-horizon agents don't run on `while True`. Every LLM call becomes an activity with checkpointing, retry, and replay. Temporal's OpenAI Agents SDK integration went GA in March 2026. Claude Code Routines (Anthropic) run scheduled Claude Code invocations without a persistent local process. Sessions pause while waiting for human input, survive deployments, and resume from the latest checkpoint keyed by `thread_id`. Behind this new ergonomics sits an old pattern—workflow orchestration—with one new input: treating LLM calls as non-deterministic activities that must be deterministically replayed on resume.

**Type:** Learn
**Languages:** Python (standard library, minimal durable execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 minutes

## The Problem

Imagine an agent that runs for four hours. It calls three tools, prompts the user twice, and makes forty LLM calls. Halfway through, its host restarts. What happens?

- In a naive `while True` loop: everything is lost. The run restarts from scratch. Those three tool calls (with real side effects) execute again. The things the user already approved are asked again. Forty LLM calls are re-billed.
- With durable execution: the run resumes from the latest checkpoint. Completed activities are not re-executed; their results are replayed from the persistent log. The user doesn't re-approve what's already approved. Already-made LLM calls are not re-billed.

This is the same pattern workflow engines have delivered for a decade (Temporal, Cadence, Uber's Cherami). What's new is that LLM calls are now an activity type—non-deterministic, expensive, side-effecting—and they fit the pattern cleanly.

The through-line for this lesson: long-horizon reliability decays (METR observes a "35-minute degradation"—success rates drop roughly quadratically with horizon). Durable execution lets you run longer than the reliability profile supports, which is a new way to fail safely if designed right, and a new way to fail unsafely if designed wrong.

## The Concept

### Activities, Workflows, and Replay

- **Workflow**: Deterministic orchestration code. Defines the sequence of activities, branches, waits. Must be deterministic so it can replay from the event log without unexpected divergence.
- **Activity**: A non-deterministic, potentially failing unit of work. LLM calls, tool calls, file writes, HTTP requests. Each activity is logged with its input and (once complete) its output.
- **Event log**: Persistent backing store. Every activity start, completion, failure, retry, and every workflow decision is recorded.
- **Replay**: On resume, the workflow code reruns from the top; every already-completed activity returns its logged result without re-executing. Only not-yet-completed activities actually run.

This is the same shape as React re-rendering against a virtual DOM, or Git rebuilding a working tree from commits. Determinism in the orchestrator is what makes persistence cheap.

### Why LLM Calls Fit the Pattern

An LLM call is:
- Non-deterministic (temperature > 0; even temperature 0 drifts across model versions).
- Expensive (money and latency).
- Can fail (rate limits, timeouts).
- Has side effects (if it calls tools).

This is exactly the activity profile. Wrap each LLM call as an activity and you get retry with exponential backoff, checkpointing across restarts, and a replayable trace for debugging.

### Checkpoints Keyed by `thread_id`

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects, and Claude Code Routines all converge on the same API shape: a `thread_id` (or equivalent) identifies the session; every state transition persists to a backend (PostgreSQL by default, SQLite for dev, Redis for caching); on resume, the latest checkpoint is read.

Backend choice matters:

- **PostgreSQL**: Durable, queryable, survives deployments. LangGraph's default.
- **SQLite**: Local dev only; loses data across hosts.
- **Redis**: Fast but ephemeral unless AOF/snapshot configured.
- **Cloudflare Durable Objects**: Transparently distributed; scoped to a unique key; survives hours to weeks.

### Human Input as First-Class State

Propose-then-commit (Lesson 15) requires a persistent "waiting for human" state. The workflow pauses, an external queue holds the pending request, and one approval resumes from that exact point. Without persistence this is best-effort; with it, an overnight approval arrives in the morning and the workflow continues.

### The 35-Minute Degradation

METR observes that every class of agent measured shows reliability decay after approximately 35 minutes of continuous execution. Double the task duration and the failure rate roughly quadruples. Durable execution doesn't fix this; it lets you run longer than the reliability profile supports. The safe pattern is combining persistence with "checkpoints that require fresh HITL on re-entry" and with "budget kill switches that cap total compute regardless of wall-clock time" (Lesson 13).

### When Durable Execution Is the Wrong Answer

- Runs shorter than a few minutes with no human input. Overhead > benefit.
- Strictly read-only information retrieval.
- Tasks whose correctness requires end-to-end completion within one context window (certain reasoning tasks; certain one-shot generations).

## Use It

`code/main.py` implements a minimal durable execution engine in standard-library Python. It supports:

- An `@activity` decorator that logs input and output to a JSON event log.
- A workflow function that sequences activities.
- A `run_or_replay(workflow, event_log)` function that replays completed activities without re-executing them.

The driver simulates a three-activity workflow, crashes mid-run, and shows (a) a naive retry re-executes everything vs. (b) a replay runs only the missing activity.

## Ship It

`outputs/skill-durable-execution-review.md` reviews a proposed long-running agent deployment for correct durable-execution shape: activities, determinism, checkpoint backend, human-input state, and HITL-on-resume policy.

## Exercises

1. Run `code/main.py`. Observe the difference in activity execution counts between naive retry and replay. Change the crash point and show the replay count changes accordingly.

2. Modify the toy engine to use an explicit `thread_id`. Simulate two concurrent sessions sharing the engine and confirm their event logs don't collide.

3. Take one activity in the toy engine. Introduce a non-determinism (put a wall-clock timestamp in a workflow decision). Demonstrate replay divergence. Explain how real engines handle this (side-effect registration, `Workflow.now()` API).

4. Read LangChain's "Runtime behind production deep agents" post. List every piece of state the runtime persists and identify which failure mode each covers.

5. Design a checkpoint strategy for a 6-hour autonomous coding task. Where do you checkpoint? What does resume after a crash look like? What needs fresh HITL?

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| Workflow | "The agent's script" | Deterministic orchestration code; replayable from event log |
| Activity | "One step" | Non-deterministic unit (LLM call, tool call); logged before and after |
| Event log | "Backing store" | Persistent record of every state transition |
| Replay | "Resume" | Reruns workflow; completed activities return logged results without re-executing |
| Checkpoint | "Save point" | State persisted keyed by thread_id; latest wins on resume |
| thread_id | "Session key" | Identifier that scopes persistent state |
| 35-minute degradation | "Reliability decay" | METR: success rate drops roughly quadratically with horizon |
| Non-determinism | "Replay drift" | Wall-clock, randomness, LLM output; must be registered as side effects |

## Further Reading

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Budgets, turns, and resume semantics.
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent shape.
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — Concrete runtime requirements.
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — Activity shape for LLM calls.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Source for 35-minute degradation.
