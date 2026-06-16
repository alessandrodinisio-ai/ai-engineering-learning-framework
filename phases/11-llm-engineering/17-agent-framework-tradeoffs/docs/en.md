# Agent Framework Tradeoffs — LangGraph vs CrewAI vs AutoGen vs Agno

> Every framework sells the same demo (a research agent producing a report) and hides the same bug (the state schema fighting the orchestration layer). Pick the framework whose abstraction matches the shape of your problem; everything else is glue you'll write twice.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 min

## The Problem

You have a task that requires more than one LLM call. Maybe it's a research workflow (plan, search, summarize, cite). Maybe it's a code review pipeline (parse diff, critique, patch, verify). Maybe it's a multi-turn assistant that books flights, writes emails, and files expenses. You pick a framework.

Three days later, you discover the framework's abstractions are leaking. CrewAI gives you roles but fights you when the "researcher" needs to hand a structured plan to the "writer." AutoGen gives you conversations between agents but has no first-class state, so your checkpoints are pickled conversation logs. LangGraph gives you a state graph but forces you to name every transition before you know what the agent will do. Agno gives you a single-agent abstraction that screams when you want to fan out to three concurrent workers.

The solution isn't "pick the best framework." It's matching the framework's core abstraction to the shape of your problem. This lesson draws that map.

## The Concept

![Agent framework matrix: core abstractions vs problem shapes](../assets/framework-matrix.svg)

Four frameworks dominate the 2026 landscape. Their core abstractions are not the same.

| Framework | Core Abstraction | Best For | Worst For |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — typed state, nodes, conditional edges, checkpointer. | Workflows with explicit state and human-in-the-loop interrupts; production agents that need time-travel debugging. | Loose, role-driven brainstorming where the topology is unknown. |
| **CrewAI** | `Crew` — roles (goal, backstory), tasks, process (sequential or hierarchical). | Persona-driven workflows with a short linear/hierarchical plan. | Any stateful requirement beyond the crew's turn history; complex branching. |
| **AutoGen** | `ConversableAgent` pairs — two or more agents take turns speaking until an exit condition. | Multi-agent *dialogue* (teacher-student, proposer-critic, actor-reviewer) where thinking emerges from conversation. | Deterministic workflows with a known DAG; anything requiring persistent state across restarts. |
| **Agno** | `Agent` — a single LLM + tools + memory, composable into teams. | Fast-to-scaffold single agents and lightweight teams; strong multimodal and built-in storage drivers. | Deep, explicitly-branching graphs with custom reducers. |

### What "abstraction" actually means

A framework's core abstraction is the thing you draw on the whiteboard when you explain the architecture to your team.

- **LangGraph** → You draw a graph. Nodes are steps, edges are transitions, and the state object at every point is typed. The mental model is a state machine.
- **CrewAI** → You draw an org chart. Each role has a job description, and a manager routes tasks. The mental model is a small team of specialists.
- **AutoGen** → You draw a Slack DM. Two agents send messages to each other; a third joins when moderation is needed. The mental model is a chat.
- **Agno** → You draw a single box with tools hanging off it. Put several boxes side by side for a team. The mental model is "batteries-included agent."

### The state problem

State is where most framework choices break in production.

- **LangGraph.** Typed state (`TypedDict` or Pydantic model), per-field reducers, first-class checkpointer (SQLite/Postgres/Redis). Recovery, interrupts, and time travel are free. *(See Phase 11 · 16.)*
- **CrewAI.** State flows between tasks as strings via the `context` field, or structured via `output_pydantic`. No per-crew persistent storage out of the box; you must bolt one on if a crew must survive restarts.
- **AutoGen.** State is the chat history plus any user-defined `context`. Conversation logs persist; arbitrary workflow state does not unless you write an adapter.
- **Agno.** Built-in storage drivers (SQLite, Postgres, Mongo, Redis, DynamoDB) attach via `storage=` to an `Agent` — conversation sessions and user memories persist automatically. Not a full graph checkpointer; it's a session store.

### The branching problem

Every non-trivial agent branches. Who decides the branch matters.

- **LangGraph** — you decide, via conditional edges. Routing is a Python function with named branches. Branching is first-class in the compiled graph; the checkpointer records which branch was taken.
- **CrewAI** — the manager decides in hierarchical mode; you decide at build time in sequential mode. Routing is implicit in the task list; there's no first-class "if" beyond the manager's prompt.
- **AutoGen** — agents decide through conversation. Branching emerges from who speaks next. The `GroupChatManager` picks the next speaker; you can hand-write a `speaker_selection_method`, but default is LLM-driven.
- **Agno** — the agent decides by which tool to call next. Teams have coordinator/router/collaborator modes; branching beyond that is the developer's responsibility.

### The observability problem

- **LangGraph** — OpenTelemetry via LangSmith or any OTel exporter. Every node transition is a trace span; checkpoints double as replayable traces. LangSmith is the first-party option; Langfuse/Phoenix have adapters too.
- **CrewAI** — first-class OpenTelemetry support since late 2025; integrations with Langfuse, Phoenix, Opik, AgentOps.
- **AutoGen** — OpenTelemetry via `autogen-core`; AgentOps and Opik have connectors. Tracing granularity is per-agent message, not per-node.
- **Agno** — built-in `monitoring=True` toggle plus OpenTelemetry exporters; tight Langfuse integration for session traces.

### Cost and latency

All four frameworks add per-call overhead (framework logic, validation, serialization). Rough order of increasing overhead: Agno ≈ LangGraph < CrewAI ≈ AutoGen. The difference is mostly driven by how much extra LLM routing the framework does. CrewAI's hierarchical manager spends tokens deciding who goes next; AutoGen's `GroupChatManager` does the same. LangGraph only spends tokens where you write `llm.invoke`. Agno's single-agent path is thin.

When per-run cost matters, prefer explicit routing (LangGraph's edges, AutoGen's `speaker_selection_method`) over LLM-chosen routing.

### Interoperability

- **LangGraph** ↔ **LangChain** tools, retrievers, LLMs. First-class MCP adapter (tools imported as MCP servers).
- **CrewAI** ↔ Tools inherit from `BaseTool`; LangChain tools, LlamaIndex tools, and MCP tools can all be adapted in. Crew-to-crew delegation via `allow_delegation=True`.
- **AutoGen** → `FunctionTool` wraps any Python callable; MCP adapter exists. Tightly coupled to the AG2 ecosystem for agent-to-agent patterns.
- **Agno** → `@tool` decorator or BaseTool subclass; MCP adapter; tools can be shared across agents and teams.

## Ship It

> You can explain in one sentence why a given framework is right for a given agent problem.

Pre-build checklist:

1. **Draw the shape.** Is it a graph (typed state, named transitions)? A role-play (specialists handing off work)? A chat (agents talk until done)? A single agent with tools?
2. **Decide who branches.** Developer-decided branching → LangGraph. Manager-agent-decided → CrewAI hierarchical. Chat-emergent → AutoGen. Tool-call-decided → Agno.
3. **Check the state budget.** Do you need checkpoint recovery? Time travel? Mid-run human interrupts? If yes, LangGraph is the default; Agno's sessions cover conversation-scoped state.
4. **Check the cost budget.** LLM-chosen routing costs extra tokens per turn. If the agent runs thousands of times daily, prefer explicit routing.
5. **Budget for framework overhead.** Every framework is another dependency. If the task is two LLM calls and a tool, write 30 lines of plain Python; no framework is cheaper than no framework.

Refuse to reach for a framework until you can draw the graph, the org chart, the chat, or the agent box. Refuse to pick one that will force you to fight its state model for what you actually need.

## Decision Matrix

| Problem Shape | Preferred Framework | Why |
|---------------|---------------------|-----|
| Typed-state, human-approval, long-running workflow DAG | LangGraph | First-class state, checkpointer, interrupts, time travel. |
| Research/writing pipeline with clear roles | CrewAI (sequential) or LangGraph subgraph | One-role-per-task is low-ceremony in CrewAI; LangGraph scales when branching gets complex. |
| Proposer-critic or teacher-student dialogue | AutoGen | Two-agent chat is its native shape. |
| Single agent with tools, sessions, memory | Agno | Thinnest configuration with built-in storage and memory. |
| Thousands of parallel fan-outs with reducers | LangGraph + `Send` | Only one with a first-class parallel-dispatch primitive. |
| Quick prototype, no framework lock-in | Plain Python + provider SDK | No framework is the fastest framework. |

## Exercises

1. **Easy.** Take the same task — "research Anthropic's headquarters, write a 200-word brief, cite sources" — and implement it in both LangGraph (four nodes: plan, search, write, cite) and CrewAI (three roles: researcher, writer, editor). Report token cost and lines of code per run.
2. **Medium.** Build the same task in AutoGen (researcher ↔ writer chat, editor joins via `GroupChat`) and Agno (a single agent with `search_tools` and `write_tools`, plus a session store). Rank all four implementations on: (a) per-run cost, (b) ability to recover from a crash, (c) ability to inject human approval before the write step.
3. **Hard.** Build a decision-tree script `pick_framework.py` that takes a short problem description (JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`) and returns a recommendation with a one-sentence rationale. Validate it against six use cases of your own design.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| Orchestration | "how agents coordinate" | The layer that decides which node/role/agent runs next. |
| Persistent state | "survives restarts" | State that outlives process death, backed by a checkpoint or session store. |
| LLM-chosen routing | "let the model decide" | A planner LLM picks the next step each turn; flexible but costs tokens per decision. |
| Explicit routing | "developer decides" | A Python function or static edge picks the next step; cheap and auditable. |
| Crew | "a CrewAI team" | Roles + tasks + process (sequential or hierarchical) bound into a single runnable. |
| GroupChat | "AutoGen's multi-agent chat" | A managed conversation among N agents with a speaker selector. |
| Team (Agno) | "Agno's multi-agent" | Route / coordinate / collaborate modes over a set of agents. |
| StateGraph | "LangGraph's graph" | The abstraction with typed state, nodes, conditional edges, and a checkpointer. |

## Further Reading

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph, checkpointers, interrupts, time travel.
- [CrewAI documentation](https://docs.crewai.com/) — Crews, Flows, Agents, Tasks, Processes.
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, teams, tools.
- [Agno documentation](https://docs.agno.com/) — Agent, Team, Workflow, storage, memory.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — Framework-agnostic pattern library (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer).
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — The loop every framework dresses up differently.
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen's design paper.
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — The role-playing foundation CrewAI-style persona stacks build on.
- Phase 11 · 16 (LangGraph) — The framework this lesson benchmarks against.
- Phase 11 · 19 (Reflexion) — A pattern that maps cleanly to LangGraph but awkwardly to CrewAI.
- Phase 11 · 22 (Production Observability) — How to instrument whichever framework you pick.
