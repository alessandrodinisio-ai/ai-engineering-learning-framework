# Parallel / Swarm / Networked Architecture

> Contrast with supervisor: no central decision-maker. Agents read a shared event bus, pick up work asynchronously, write results back. LangGraph explicitly supports "Swarm Architecture" for decentralized, dynamic environments. Matrix (arXiv:2511.21686) represents both control flow and data flow as serialized messages passed through distributed queues, eliminating the orchestrator bottleneck. The tradeoff is explicit: trade determinism and traceability for scalability. Swarms suit tasks with many independent sub-problems; they don't suit tasks requiring a single, coherent plan.

**Type:** Learn + Build
**Languages:** Python (standard library, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Time:** ~75 min

## The Problem

A supervisor scales to a few workers. What about hundreds? The supervisor itself becomes the bottleneck: every "who does what" decision funnels through a single agent. A slow planning step stalls the entire system.

The swarm architecture flips the design. Instead of a central planner dispatching, workers pick up work from a shared queue. "Coordination" is baked into the event bus semantics. No orchestrator; the system scales until the queue can't.

## The Concept

### The Shape

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

No orchestrator. Each worker loops: pick a task, process it, write results (optionally re-enqueue follow-up tasks).

### When Swarms Fit

- **Many independent tasks.** Crawling, transforming, classifying. Tasks don't depend on each other.
- **Variable-duration work.** If some tasks take 100ms and others 10s, a swarm auto-balances — fast workers pick up the next item. A supervisor would have to predict durations.
- **Throughput over determinism.** You care about total completion time, not strict ordering.

### When Swarms Fail

- **Ordered workflows.** If step 3 needs step 2's output, a swarm risks step 3 firing before step 2 completes.
- **Global planning tasks.** A complex research question benefits from a planner. A swarm of researchers produces isolated facts, not a coherent report.
- **Debugging.** Without centralized logs and with asynchronous work, reproducing a bug is expensive.

### Matrix (arXiv:2511.21686)

Matrix is the 2025 paper that pushes swarms to their natural end: control flow and data flow are both serialized messages on distributed queues. No central coordinator. Fault tolerance comes from message durability. Scalability is the message broker's problem, not the system's.

Contribution: a programming model where multi-agent coordination becomes "what message topics does this agent subscribe to?" rather than "which agent does the supervisor pick next?" This makes the system look like a pub/sub event grid.

### LangGraph's Swarm Architecture

LangGraph 2025 docs explicitly describe "Swarm Architecture" as one of the multi-agent patterns: agents are nodes, but edges form a directed graph with cycles, and any node can be activated from the pool. Workers pick up available work based on conditions, rather than being assigned by a supervisor.

### Failure Mode: Starvation and Hot-spotting

If all workers grab the fastest available task, long-running tasks won't be picked up until they're the only ones left. Classic queue starvation.

Mitigations:
- Priority queues with explicit aging (longer wait → higher priority).
- Worker specialization: some workers only take "long" tasks.
- Back-pressure: limit how many fast tasks enter the queue.

### Connection to Content-Based Routing

Swarms pair naturally with content-based routing (Lesson 22). Instead of one generic queue, have one queue per message type. Specialized workers subscribe only to their type. This is the foundation of the message-bus architecture that scales to thousands of agents.

## Build It

`code/main.py` implements a 4-worker-thread swarm pulling from a shared `queue.Queue`. Tasks have variable durations (fast and slow). The demo compares three approaches:

- **Serial baseline:** one worker processes all tasks sequentially.
- **Fixed assignment:** each task pre-assigned to a specific worker (supervisor-style).
- **Swarm:** workers pull from a shared queue.

The swarm auto-balances load; fixed assignment leaves fast workers idle when their assigned tasks are slow.

Run:

```
python3 code/main.py
```

Output shows per-worker task counts (swarm distributes unevenly but optimally) and wall-clock time.

## Use It

`outputs/skill-swarm-fit.md` evaluates whether a task should use a swarm or a supervisor. Inputs: task independence, duration variance, ordering requirements, debuggability needs.

## Ship It

Checklist:

- **Priority queue with aging.** Prevent long-task starvation.
- **Idempotent workers.** If a worker crashes mid-run, a task may be picked up more than once. Workers must be idempotent.
- **Durable queue.** In production use Kafka, Redis Streams, or a DB-backed queue. `queue.Queue` is in-memory only.
- **Per-task observability.** Each task gets a trace ID; each worker logs start/end with it.
- **Back-pressure.** If the queue grows faster than workers drain it, slow down the producer.

## Exercises

1. Run `code/main.py`. How much faster is the swarm than serial on variable-duration load? Than fixed assignment?
2. Add a priority queue variant (use `queue.PriorityQueue`). Assign priority by a task "importance" field. Observe whether low-priority tasks starve under sustained load.
3. Implement a hot-spot detector: log when any worker processes 3x more tasks than the slowest worker. What does this tell you about the task duration distribution?
4. Read the Matrix paper (arXiv:2511.21686) abstract and Section 3. Identify one specific tradeoff Matrix accepts (scalability gain) and one it gives up (traceability, determinism).
5. Modify the swarm demo to use a `queue.Queue` holding `(task_type, payload)` tuples where workers only subscribe to specific types. What routing rules make sense when tasks are heterogeneous?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Swarm architecture | "Decentralized agents" | Workers pull from shared queue; no central orchestrator. |
| Event bus | "Agents subscribe to topics" | A message broker that routes tasks to workers by type or content. |
| Starvation | "Task never runs" | Low-priority tasks never get picked up because higher-priority work keeps arriving. |
| Hot-spotting | "One worker is drowning" | Uneven load where one worker gets most tasks. |
| Back-pressure | "Slow down the producer" | Mechanism to signal upstream to stop producing when the queue is full. |
| Idempotent worker | "Safe to re-run" | Processing a task twice produces the same result. Required because workers can crash mid-run. |
| Durable queue | "Survives crashes" | Queue backed by disk or replicated storage; tasks don't vanish when a worker dies. |
| Matrix framework | "All-message swarm" | Data flow and control flow are both serialized messages on distributed queues. |

## Further Reading

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — explicit swarm support
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — the all-message swarm
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) — a concrete production system that explicitly chose supervisor over swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — event-driven actor rewrite, closer to swarm than v0.2's GroupChat
