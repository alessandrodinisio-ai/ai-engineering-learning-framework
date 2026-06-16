# The Multi-Agent Primitive Model

> Every multi-agent framework in use in 2026 — AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework — is a point in a four-dimensional design space. The primitives are exactly four: agent, handoff, shared state, orchestrator. This lesson builds them from scratch, runs a toy system through all four, then maps every major framework to the same axes so you can read any new release in one paragraph.

**Type:** Learn
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 min

## The Problem

A new multi-agent framework drops every six months. AutoGen in 2023. CrewAI in 2024. LangGraph and OpenAI Swarm in 2024. Google ADK in April 2025. Microsoft Agent Framework RC in February 2026. Every press release claims to be "the right abstraction."

If you try to learn them one by one, you'll burn out. APIs look wildly different, docs disagree on what "agent" even means. One framework calls its shared memory a "blackboard," another calls it a "message pool," a third calls it "StateGraph." You start suspecting the field is just spinning in circles.

It isn't. Beneath the marketing, those four primitives are stable. Learn them once, read every new framework in one paragraph.

## The Concept

### The Four Primitives

1. **Agent** — a system prompt plus a tool list. Stateless; each run starts from its system prompt and the current message history.
2. **Handoff** — a structured transfer of control from one agent to another. Mechanically, either a tool call that returns a new agent, or a conditional graph edge.
3. **Shared state** — any data structure readable (and sometimes writable) by more than one agent. Message pool, blackboard, key-value store, vector memory.
4. **Orchestrator** — the role that decides who speaks next. Options: explicit graph (deterministic), LLM speaker selector (flexible), the last speaker's handoff call (OpenAI Swarm), or a scheduler atop a queue (swarm architecture).

That's the entire design space. Each framework picks defaults for each axis; the rest is surface syntax.

### How Every 2026 Framework Maps to It

| Framework | Agent | Handoff | Shared State | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | Tool returns Agent | Caller manages | LLM's next handoff call |
| AutoGen v0.4 / AG2 | `ConversableAgent` | Speaker selector on GroupChat | Message pool | Selector function (LLM or round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task output chaining | Manager LLM or static sequence |
| LangGraph | Node functions | Graph edges + conditions | `StateGraph` reducer | The graph itself, deterministic |
| Microsoft Agent Framework | Agent + orchestration patterns | Varies by pattern | Thread / context | Varies by pattern |
| Google ADK | Agent + A2A card | A2A tasks | A2A artifacts | Host decides |

The surface differences look enormous. Underneath: the same four knobs.

### Why This Matters

Once you see through to the primitives, framework comparison becomes a short checklist:

- Does the orchestrator trust the LLM to route (Swarm), or nail routing in code (LangGraph)?
- Is shared state full-history (GroupChat) or projected (StateGraph reducer)?
- Can agents modify each other's prompts (CrewAI manager), or only handoff (Swarm)?

These three questions answer 80% of "which framework fits a given problem." You stop shopping for "the best multi-agent framework" and start designing for the axis you actually care about.

### The "Stateless" Insight

Aside from shared state, every primitive is stateless. An agent is a function of (prompt, tools). A handoff is a function call. An orchestrator is a scheduler. **The only stateful thing in the system is shared state.** All the interesting bugs live there: memory poisoning (Lesson 15), message ordering, versioning, write races.

Frameworks that hide shared state (Swarm) push the problem to the caller. Frameworks that centralize it (LangGraph checkpointer, AutoGen pool) make it inspectable but shift coordination cost to the shared-state implementation.

### Anatomy of a Single Primitive

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

No memory. No state. Two agents with the same system prompt and tools are interchangeable. Everything that looks like "per-agent state" actually lives in shared state or in the handoff protocol.

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

Three major implementations:

- **Function return** — a tool returns the next agent. This is OpenAI Swarm's pattern. The agent carries routing in its own tool schema.
- **Graph edge** — LangGraph. Edges are declarative. The LLM produces a value, a condition picks the next node.
- **Speaker selection** — AutoGen GroupChat. A selector function (sometimes itself an LLM call) reads the pool and picks who speaks next.

#### Shared State

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

At minimum, a message list. Often more: structured artifacts (CrewAI Task outputs), typed context (LangGraph reducer), external memory (MCP, vector DB).

Two topologies: **full pool** (every agent sees every message) and **projected** (agents see a role-specific slice). Full pool is simple but scales poorly. Projected pool scales but requires upfront schema design.

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

Four flavors:

- **Static** — the graph is fixed at build time (LangGraph deterministic, CrewAI Sequential).
- **LLM selection** — an LLM reads the pool and picks the next speaker (AutoGen, CrewAI Hierarchical).
- **Handoff-driven** — the current agent decides by calling a handoff tool (Swarm).
- **Queue-driven** — workers pick up work from a shared queue; no explicit "next speaker" (swarm architecture, Matrix).

### What Varies Across Frameworks

Once primitives are fixed, the remaining design decisions are:

- **Memory strategy** — ephemeral vs durable checkpointing (LangGraph checkpointer).
- **Security boundary** — who can approve a handoff (human-in-the-loop).
- **Cost accounting** — per-agent token budgets.
- **Observability** — tracing handoffs, persisting state for replay.

All implementable on top of primitives. None are new primitives.

## Build It

`code/main.py` implements the four primitives in ~150 lines of standard-library Python. No real LLM — each agent is a scripted strategy so the focus stays on coordination structure.

The file exports:

- `Agent` — a dataclass with name, system prompt, tools, strategy function.
- `Handoff` — a function that returns the next agent.
- `SharedState` — a thread-safe message pool.
- `Orchestrator` — three variants: `StaticOrchestrator`, `HandoffOrchestrator`, `LLMSelectorOrchestrator` (simulated).

The demo runs the same three-agent pipeline (research → write → review) through all three orchestrator types and prints the message pool at the end. You can see: the only difference between outputs is *who picks next*; agents and shared state are identical across runs.

Run:

```
python3 code/main.py
```

Expected output: three orchestrator runs, one per mode. Each prints the final message pool. The handoff-driven run will reach fewer agents if the researcher decides early that it's done — that's the micro-version of the LLM routing tradeoff.

## Use It

`outputs/skill-primitive-mapper.md` is a skill that reads any multi-agent codebase or framework documentation and returns a four-primitive mapping. Run it on a new framework release to get a one-paragraph understanding before diving into docs.

## Ship It

Before adopting a new framework, write out its primitive mapping. If you can't, either the docs are incomplete or the framework is inventing a fifth primitive (rare — check if it's some flavor of shared state you haven't seen).

Pin this mapping in your architecture docs. When new members join, hand them the mapping before the API docs. When the framework version changes, diff the mapping, not the changelog.

## Exercises

1. Run `code/main.py` three times with different agent strategies. Observe how the orchestrator's choice changes which agents run.
2. Implement a fourth orchestrator type: a queue-driven one where agents poll shared state for work. What deadlock emerges, and how do you detect it?
3. Take the LangGraph quickstart (https://docs.langchain.com/oss/python/langgraph/workflows-agents) and rewrite it in terms of these four primitives. Which LangGraph abstractions map 1:1 and which are convenience wrappers?
4. Read the OpenAI Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents). Identify which of the four primitives Swarm handles best and which it pushes to the caller.
5. Find a framework in the table that completely hides shared state. Articulate what breaks when agents need to coordinate across handoffs without re-reading history.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Agent | "An LLM with tools" | A `(system_prompt, tools, model)` triple. Stateless. |
| Handoff | "Control transfer" | A structured call specifying the next agent and optional payload. Three implementations: function return, graph edge, speaker selection. |
| Shared state | "Memory" / "context" | The only stateful part of a multi-agent system. Message pool or blackboard. |
| Orchestrator | "Coordinator" | The role that decides who runs next. Static graph, LLM selector, handoff-driven, or queue-driven. |
| Primitive | "Abstraction" | One of the four axes every framework parameterizes. Not a feature of any single framework. |
| Message pool | "Shared chat history" | Full-history shared state. Easy to reason about, scales poorly. |
| Projected state | "Filtered view" | A role-specific view into shared state. Scales well, requires schema design. |
| Speaker selection | "Who speaks next" | An orchestrator pattern where a function (often LLM) picks the next agent from a group. |

## Further Reading

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — the clearest exposition of handoff-driven orchestration
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + speaker selection is the reference for LLM-selector orchestration
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — graph-edge orchestration and reducer-based shared state
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory agents, Sequential / Hierarchical processes
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — the still-updated AutoGen v0.2 lineage after Microsoft moved v0.4 to maintenance
