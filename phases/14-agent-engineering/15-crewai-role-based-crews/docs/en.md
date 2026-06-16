# CrewAI: Role-Based Crews and Flows

> CrewAI is the 2026 role-based multi-agent framework. Four primitives: Agent, Task, Crew, Process. Two top-level shapes: Crew (autonomous, role-based collaboration) and Flow (event-driven, deterministic). The documentation is blunt: "For any production-ready application, start with Flow."

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## Learning Objectives

- Name CrewAI's four primitives (Agent, Task, Crew, Process) and what each governs.
- Distinguish Sequential, Hierarchical, and the planned Consensus process; pick each by workload.
- Distinguish Crew (autonomous, role-based) from Flow (event-driven, deterministic) and explain the documentation's production recommendation.
- Wire tools using the `@tool` decorator and `BaseTool` subclass; weigh structured output vs. free text.
- Name CrewAI's four memory types and when each earns its cost.
- Implement a three-agent crew (researcher, writer, editor) that produces a briefing using the standard library.
- Identify CrewAI's three failure modes: prompt bloat, manager-LLM tax, brittle handoffs.

## The Problem

Teams adopting multi-agent frameworks hit the same wall. "Autonomous collaboration" sounds great in demos. Then a customer files a bug, and you need deterministic replay. Or finance asks what a single run of an LLM-routed crew costs. Or the on-call engineer needs to know which agent got stuck at 3 AM.

Free-form LLM-routed crews answer none of these well. A pure DAG answers all of them, but loses the exploratory shape a brainstorming agent needs.

CrewAI's split is honest about this tradeoff. Crews for collaborative, role-based, exploratory work. Flows for event-driven, code-owned, auditable production. Same framework, two shapes, pick by scenario.

## The Concept

### The Four Primitives

CrewAI's surface area is small. Memorize this and the rest is configuration.

- **Agent.** `role + goal + backstory + tools + (optional) llm`. The backstory is load-bearing. It shapes tone, judgment, and when the agent stops. Tools are functions the agent can invoke (detailed below).
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`. A reusable unit of work. `expected_output` is the contract. `context` lists upstream tasks whose output is passed in. `output_pydantic` enforces a structured shape.
- **Crew.** The container. Owns the `agents` list, `tasks` list, `process`, and optional `memory` + `verbose` + `manager_llm` settings.
- **Process.** The execution strategy. Sequential, Hierarchical, Consensus (planned). Determines the run shape.

Agents do not see each other directly. Tasks reference agents. Crews order tasks. Processes decide who picks the next task. That is the entire mental model.

> **Validation baseline** CrewAI 0.86 (2026-05). Newer versions may rename or merge process types; check [CrewAI Processes docs](https://docs.crewai.com/concepts/processes) before depending on a specific shape.

### Sequential vs Hierarchical vs Consensus

- **Sequential.** Tasks run in declaration order. Task N's output can serve as `context` for Task N+1. Lowest cost. Most predictable. Use when the order is fixed.
- **Hierarchical.** A manager Agent (separate LLM call) routes among experts. CrewAI derives the manager either from your `manager_llm` config or a default. The manager picks the next task each round and can reject or re-route. Use when you have four or more experts and the order genuinely depends on prior output.
- **Consensus.** Planned; not currently implemented in the public API. The documentation reserves this name for a future voting-based process. Do not depend on it today.

Hierarchical adds a per-round LLM call (the manager) on top of each expert call. Token cost can triple in a five-step run. Only pay for it when you need the routing.

### Crew vs Flow

This is the 2026 documentation's opening framing.

- **Crew.** LLM-driven autonomy. The framework picks the shape at runtime. Good for: research, brainstorming, first drafts, and anywhere "the path itself is part of the answer." Hard to replay. Hard to test. Cheap to prototype.
- **Flow.** An event-driven graph you own. `@start` marks entry. `@listen(topic)` marks a step triggered when another step emits that topic. Each step is plain Python (which can internally invoke a Crew). Good for: production. Observable. Testable. Deterministic.

The 2026 documentation's production recommendation: start with Flow. Fold Crews in as `Crew.kickoff()` calls inside Flow steps when autonomy earns its cost. Flow gives you audit trails, Crew gives you exploration. Combine, don't choose.

### Tool Integration

Three ways to wire tools to an Agent. Pick the simplest that fits.

1. **`@tool` decorator.** A plain function becomes a tool. The signature is the schema; the docstring is the description the LLM sees. Best for one-off small helpers.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.** Class-based tool with explicit argument schema, async support, retries. Use when the tool has state (a client, a cache) or needs structured arguments.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in tool packages.** CrewAI provides first-party adapters: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. One import and you are wired.

Structured output uses Pydantic. Pass `output_pydantic=MyModel` on the Task. CrewAI validates the LLM response against the model, either coercing or retrying. Pair it with a tight `expected_output` string. Free-text output is fine for first drafts; structured output is what downstream Flows can consume.

### Memory Hooks

CrewAI ships four memory types out of the box. They are composable: a Crew can enable all four at once.

> **Validation baseline** CrewAI 0.86 (2026-05). Recent versions route everything into a unified `Memory` system that wraps these four stores. The conceptual model below still holds, but the public class surface may consolidate into a single `Memory` entry point in newer versions; see [CrewAI memory docs](https://docs.crewai.com/concepts/memory) for the current API.

- **Short-term.** Conversation buffer within a single run. Cleared on run end.
- **Long-term.** Persists across runs. Stored in a vector database (Chroma by default, swappable). Retrieved by similarity to the current task.
- **Entity.** Per-entity facts. "Customer X is on the Enterprise plan." Keyed by entity, not similarity. Survives across runs.
- **Contextual.** Retrieved at assembly time. Pulls relevant memory at the moment an Agent needs it, rather than preloading.

Enable with `memory=True` on the Crew or configure by type. Backed by an embedding provider you configure (OpenAI by default, swappable to local). Memory is one of the places where CrewAI earns its keep over thinner frameworks; raw LangGraph requires you to wire each of these yourself.

### When CrewAI Fits

- Three to six agents with named roles and collaborative workflows. Drafting, review, planning, brainstorming.
- Routing scenarios where the LLM's judgment about the next step is itself part of the value (Hierarchical).
- Any setting where the team reads `role + goal + backstory` more comfortably than a graph definition.

### When CrewAI Does Not Fit

- Deterministic DAGs with strict ordering. Use LangGraph (Lesson 13). The graph shape is the right abstraction; CrewAI's role framing is friction.
- Sub-second latency budgets. Hierarchical adds round trips. Even Sequential serializes prompts that include backstory and prior output.
- Single-agent loops. Skip the framework; an agent loop (Lesson 1) plus a tool registry is shorter.

Lesson 17 (Agent Framework Tradeoffs) lays this out in a matrix. Short version: CrewAI sits in the "collaborative, role-based" corner.

### Dependency Shape

Independent of LangChain. Python 3.10 to 3.13. Uses `uv`. Star count: see [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) (snapshot as of 2026-05). AWS Bedrock integration is documented; vendor benchmarks report meaningful speedups over LangGraph on QA workloads, but the methodology (dataset, hardware, evaluation metrics) is not public, so treat framework-vendor numbers as directional only.

### Where This Pattern Breaks

- **Backstory-driven prompt bloat.** A 2000-word backstory per agent, times a five-agent crew, burns the context budget before the first tool call. Keep backstories under 200 words. Reuse phrasing across agents; do not repeat team-style five times.
- **Manager-LLM token tax.** The Hierarchical process adds a manager LLM call before each expert call. On a five-task crew that is six LLM calls instead of five, and the manager call carries the full task list plus prior output. Switch to Sequential unless routing depends on output.
- **Brittle handoffs.** Task N's `expected_output` is "an outline." Task N+1 reads it as `context` and tries to parse three sections. The LLM produces four. The downstream Agent improvises. Fix with `output_pydantic` on Task N so Task N+1 reads a typed object, not free text.
- **Crew as production.** A free-form Crew deployed to production without a Flow wrapper. Output varies widely; replay is impossible; on-call cannot diff a bad run against a good one. Wrap it in Flow.

## Build It

`code/main.py` implements standard-library versions of both shapes plus a three-agent crew.

Shapes:

- `Agent`, `Task` dataclasses matching CrewAI's surface.
- `SequentialCrew.kickoff(inputs)` runs tasks in declaration order, threading output as `context`.
- `HierarchicalCrew.kickoff(topic)` adds a manager Agent that picks the next expert each round and stops at "done."
- `Flow` with `@start` and `@listen(topic)` decorators, a mini event loop, and a trace.
- `tool(name)` decorator mirroring CrewAI's `@tool` shape.
- `Memory` with `short_term`, `long_term`, `entity` stores; mock similarity via numpy.
- Mock LLM responses are hardcoded strings keyed by role + input prefix. No network. Deterministic.

Specific demo: a researcher, writer, editor crew produces a briefing on "agent engineering 2026." Researcher pulls (mock) sources. Writer drafts. Editor tightens. The same crew runs once through a Flow to demonstrate the deterministic shape.

Run it:

```bash
python3 code/main.py
```

The trace covers: sequential crew threading output via `context`, hierarchical crew with manager selection (researcher, writer, editor, then "done"), flow running the same three steps with explicit topics (`researched`, `drafted`, `edited`), tool calls routed via `@tool`, and long-term memory surviving across two kickoffs.

The Crew trace is fluid; the manager could in principle reorder. The Flow trace is fixed. That choice is the point of this lesson.

## Use It

- **CrewAI Flow** for production. Even if the Flow only has one step calling `Crew.kickoff()`. Flow gives you the audit boundary.
- **CrewAI Crew (Sequential)** for collaborative work where the order is clear, especially first drafts and review loops.
- **CrewAI Crew (Hierarchical)** when routing depends on output and you have four or more experts.
- **LangGraph** (Lesson 13) for explicit state machines, durable recovery, strict ordering.
- **AutoGen v0.4** (Lesson 14) for actor-model concurrency and fault isolation.
- **OpenAI Agents SDK** (Lesson 16) for OpenAI-first products with handoff and guardrails.
- **Claude Agent SDK** (Lesson 17) for Claude-first products with sub-agents and session storage.

## Ship It

`outputs/skill-crew-or-flow.md` picks between Crew and Flow for a task and scaffolds the minimal implementation. Hard-rejects "Crew without backstory," "Flow without explicit topics," and "Hierarchical with fewer than three experts."

## Pitfalls

- **Backstory as seasoning.** It shapes output. Test three variants per agent; the variance is real. Pick one, freeze it.
- **Skipping `expected_output`.** Without a per-task contract, downstream tasks pick up whatever the LLM produced. The Crew runs; the audit does not pass.
- **Memory always on.** Long-term memory writes on every run. The vector database grows. Retrieval becomes noisy. Limit writes to tasks whose facts are genuinely durable.
- **Manager prompt drift.** The Hierarchical manager prompt is implicit. If routing gets weird, turn on verbose mode and dump it to read.
- **Tool side effects inside Crews.** A Crew may invoke a tool more times than expected. POSTs, DELETEs, and payments belong in Flow steps, never as Crew tools.

## Exercises

1. Convert the Sequential crew to a Flow. Count the touchpoints where variance decreases. Note where readability drops.
2. Add entity memory to the crew: facts about a customer persist across kickoffs. Verify that retrieval pulls the correct entity.
3. Implement a Hierarchical process where the manager refuses to route to the editor until the writer's output has at least three paragraphs. Trace the retry.
4. Wire a `BaseTool` subclass for a (mock) web search. Compare its trace shape to the `@tool` decorator version.
5. Add `output_pydantic=Brief` to the editor task, where `Brief` has `title`, `summary`, `sections`. Have the writer task emit malformed JSON; verify CrewAI's retry behavior in the trace.
6. Read CrewAI's documentation introduction. Port the toy to the real `crewai` API. What guarantees did the standard-library version skip?
7. Wire AgentOps or Langfuse (Lesson 24) to a real run. What traces are you missing in the standard-library version?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Container of Agents + Tasks + Process |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus (planned) |
| Flow | "Deterministic workflow" | Event-driven, code-owned, testable |
| Backstory | "Persona prompt" | Agent's tone and judgment shaper |
| `@tool` | "Function tool" | Decorator turning a function into an Agent-callable tool |
| `BaseTool` | "Class tool" | Class-based tool with argument schema, retries, async support |
| Entity memory | "Per-entity facts" | Memory scoped to a customer / account / issue |
| Long-term memory | "Cross-run memory" | Vector-backed memory surviving between kickoffs |
| Contextual memory | "Just-in-time retrieval" | Memory pulled at the moment an Agent needs it |
| Manager LLM | "Router agent" | Extra LLM in Hierarchical process picking the next task |
| `expected_output` | "Task contract" | String telling the Agent (and auditors) what shape to return |

## Further Reading

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction): concepts and recommended production path
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows): event-driven shape, `@start`, `@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, built-in packages
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory): short-term, long-term, entity, contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents): when multi-agent helps and when it does not
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): state-machine alternative
