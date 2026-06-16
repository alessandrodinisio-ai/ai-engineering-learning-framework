# LangGraph: Stateful Graphs and Durable Execution

> LangGraph is the 2026 reference for low-level stateful orchestration. An agent is a state machine; nodes are functions; edges are transitions; state is immutable and checkpointed after every step. Resume from any failure at exactly that point.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Learning Objectives

- Describe LangGraph's core model: state machine with immutable state, function nodes, conditional edges, checkpointing after every step.
- Name the four capabilities the documentation emphasizes: durable execution, streaming, human-in-the-loop, full memory.
- Explain the three orchestration topologies LangGraph supports: supervisor, peer-to-peer (swarm), hierarchical (nested subgraphs).
- Implement a state graph using the standard library with immutable state, conditional edges, and a checkpoint/resume cycle.

## The Problem

Agents and workflows share a problem: when a 40-step run fails at step 38, you want to resume from step 38, not start over. Second-class state models force operators to hack retries around a library that assumes fresh runs.

LangGraph's design answer: state is a first-class typed object, mutations are explicit, and checkpoints are persisted after every node. Resuming is a single `load_state(session_id)` call.

## The Concept

### The Graph

A graph is defined by:

- **State type.** A typed dictionary (or Pydantic model) that every node reads and mutates.
- **Nodes.** Pure functions `(state) -> state_update`. After return, the update is merged into state.
- **Edges.** Conditional or direct transitions between nodes.
- **Entry and exit.** `START` and `END` sentinel nodes mark boundaries.

Example: an agent with `classify`, `refund`, `bug`, `sales`, `done` nodes — a routing workflow expressed as a graph.

### Durable Execution

After every node returns, the runtime serializes state and writes it to a checkpointer (SQLite, Postgres, Redis, custom). On failure at step N, the runtime can `resume(session_id)` and continue from step N+1 with the exact state.

LangGraph documentation explicitly highlights this as important for production users: Klarna, Uber, J.P. Morgan. The claim is not about the graph shape; it is that the graph shape plus checkpointing makes recovery cheap.

### Streaming

Every node can yield partial output. The graph streams per-node incremental events to the caller, so UIs update as the graph runs.

### Human-in-the-Loop

Inspect and modify state between nodes. Implementation: pause before a critical node, expose state to a human, accept modifications, resume. The checkpointer makes this easy because state is already serialized.

### Memory

Short-term (within a single run — conversation history in state) and long-term (across runs — persisted via the checkpointer plus a separate long-term store). LangGraph integrates with external memory systems (Mem0, custom) via tools.

### Three Topologies

1. **Supervisor.** A central router LLM dispatches to expert sub-agents. `create_supervisor()` in `langgraph-supervisor` (though the 2026 LangChain team recommends doing this directly via tool calls for more context control).
2. **Swarm / peer-to-peer.** Agents hand off directly to each other through a shared tool surface. No central router.
3. **Hierarchical.** Supervisors manage sub-supervisors, implemented as nested subgraphs.

### Where This Pattern Breaks

- **Checkpoints too small.** Checkpointing only conversation turns leaves tool state and memory writes unrecoverable. The full state must be serialized.
- **Non-deterministic nodes.** Resumption assumes node inputs produce the same state update. Random seeds, wall clocks, and external APIs must be captured.
- **Overuse of conditional edges.** A graph where every edge is conditional is a state machine that cannot be reasoned about. Prefer linear chains with occasional branches.

## Build It

`code/main.py` implements a stateful graph using the standard library:

- `State` — a typed dictionary with `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — callables that take state and return an update dictionary.
- `StateGraph` — nodes + edges + conditional edges + run + resume.
- `SQLiteCheckpointer` (in-memory fake) — serializes state after every node; `load(session_id)` resumes.
- A demo graph: classify -> branch(refund / bug / sales) -> human gate -> send.

Run it:

```
python3 code/main.py
```

The trace shows the first run failing at the human gate, persisting, then resuming and producing final output.

## Use It

- **LangGraph** — the reference, production-ready. Use `create_react_agent`, `create_supervisor`, or build your own graph.
- **AutoGen v0.4** (Lesson 14) — actor-model alternative for high-concurrency scenarios.
- **Claude Agent SDK** (Lesson 17) — managed harness with built-in session storage.
- **Custom** — when you need precise control over state shape or checkpointer backend.

## Ship It

`outputs/skill-state-graph.md` generates a LangGraph-shaped state graph on any target runtime, with checkpointing and resume wired in.

## Exercises

1. Add a conditional edge from `classify` to `end` when classification confidence is below a threshold. Resume the run after a human manually sets the `route`.
2. Swap the SQLite-like fake for a real SQLite checkpointer. Measure per-step serialization overhead.
3. Implement parallel edges: two nodes run concurrently, merged with a custom reducer. What does immutable state buy you here?
4. Read the `langgraph-supervisor` reference. Port the toy to `create_supervisor`. Compare trace shapes.
5. Add streaming: each node yields partial state as it runs. Print increments as they arrive.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducer |
| Checkpointer | "Persistence backend" | Serializes state after every node; enables resume |
| Reducer | "State merger" | Function that merges current state with a node's update |
| Conditional edge | "Branch" | Edge selected by a function of state |
| Subgraph | "Nested graph" | A graph used as a node inside another graph |
| Durable execution | "Resume from failure" | Restart from the last successful node with exact state |
| Supervisor | "Router LLM" | Central dispatcher to expert sub-agents |
| Swarm | "P2P agents" | Agents hand off via shared tools; no central router |

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reference documentation
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — supervisor pattern API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model alternative
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — session storage and sub-agents
