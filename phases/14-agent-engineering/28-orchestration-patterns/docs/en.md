# Orchestration Patterns: Supervisor, Swarm, Hierarchical

> Four orchestration patterns recur across frameworks in 2026: supervisor-worker, swarm/peer-to-peer, hierarchical, and debate. Anthropic's advice: "The key is building the right system for your needs." Start simple; add topology only when "a single agent plus five workflow patterns" isn't enough.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 minutes

## Learning Objectives

- Name four recurring orchestration patterns and when each is appropriate.
- Describe LangChain's 2026 recommendation: tool-call-based supervision vs. supervisor libraries.
- Explain Anthropic's "build the right system" rule and how it constrains topology choices.
- Implement all four patterns using the standard library against a scripted LLM.

## The Problem

Teams reach for "multi-agent" before they need it. Four patterns recur across frameworks; once you can name them, you can pick the right one — or skip topology entirely.

## The Concept

### Supervisor-Worker

- A central routing LLM dispatches to specialist agents.
- Decisions: route back to self, hand off to specialist, terminate.
- Specialists don't talk to each other; all routing goes through the supervisor.

Frameworks: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**LangChain's 2026 recommendation:** Do supervision via direct tool calls rather than `create_supervisor`. Gives you finer context engineering control — you decide exactly what each specialist sees.

### Swarm / Peer-to-Peer

- Agents hand off directly to each other through a shared tool surface.
- No central router.
- Lower latency than supervisor (fewer hops).
- Harder to reason about (no single control point).

Frameworks: LangGraph swarm topology, OpenAI Agents SDK handoff (when all agents can hand off to all others).

### Hierarchical

- Supervisor manages sub-supervisors managing workers.
- Implemented as nested subgraphs in LangGraph; nested crews in CrewAI.
- Scales to large agent populations at the cost of operational complexity.

When you need it: when a single supervisor's context budget can't fit descriptions of all specialists.

### Debate

- Parallel proposers + iterative cross-critique (Lesson 25).
- Not true orchestration — more validation — but surfaces as a topology choice in frameworks.

### CrewAI Crew vs Flow

CrewAI formalizes two deployment modes:

- **Flow** for deterministic event-driven automation (recommended starting point for production).
- **Crew** for autonomous, role-based collaboration.

This is orthogonal to the four patterns above but maps to topology: Flow is typically supervisor or hierarchical; Crew is typically supervisor with an LLM router.

### Anthropic's Advice

"In the LLM domain, success isn't about building the most complex system. It's about building the right system for your needs."

Decision order:

1. Single agent + workflow patterns (Lesson 12) — start here.
2. Supervisor-worker — when you have 2-4 specialists.
3. Swarm — when latency matters more than reasoning clarity.
4. Hierarchical — only when supervisor context budget can't hold.
5. Debate — when accuracy matters more than cost.

### Where This Pattern Goes Wrong

- **Topology-first thinking.** "We need multi-agent" before identifying what problem multi-agent solves.
- **Ping-pong handoffs in swarm.** A -> B -> A -> B. Use a hop counter.
- **Fake hierarchy.** Three layers because "enterprise"; only two teams in practice. Flatten.

## Build It

`code/main.py` implements all four patterns using the standard library against a scripted LLM:

- `Supervisor` — central router.
- `Swarm` — peer-to-peer with direct handoff.
- `Hierarchical` — supervisor of supervisors.
- `Debate` — parallel proposers + critique.

Each pattern handles the same three-intent task (refund / bug / sales). Trace shapes differ.

Run it:

```
python3 code/main.py
```

Output: per-pattern trace + operation count. Supervisor is cleanest; swarm is shortest; hierarchical is deepest; debate is most expensive.

## Use It

- **LangGraph** for supervisor and hierarchical (nested subgraphs).
- **OpenAI Agents SDK** for "handoff as tool" (supervisor-shaped).
- **CrewAI Flow** for production determinism.
- **Custom** for debate or when you want precise control.

## Ship It

`outputs/skill-orchestration-picker.md` picks a topology and implements it.

## Exercises

1. Convert a supervisor-worker into a swarm by removing the router. What broke? What improved?
2. Add a hop counter to the swarm: reject after 3 handoffs. Does it catch A->B->A ping-pong?
3. Build a two-level hierarchical system for a 12-specialist domain. Where does context budget break without nesting?
4. Profile all four patterns on a production-shaped workload. Which wins on which metric (latency, cost, accuracy, debuggability)?
5. Read Anthropic's "Building Effective Agents" post. Map each of your production flows to one of the four. Any that don't map cleanly?

## Key Terms

| Term | Common usage | What it actually is |
|------|----------------|------------------------|
| Supervisor-worker | "Router + specialists" | Central LLM dispatches to specialists; they don't talk to each other |
| Swarm | "Peer-to-peer" | Direct handoff via shared tools; no central router |
| Hierarchical | "Supervisor of supervisors" | Nested subgraphs for large populations |
| Debate | "Proposers + critique" | Parallel proposers, cross-critique (Lesson 25) |
| Tool-call-based supervision | "Library-free supervisor" | Implement supervisor as direct tool calls for context control |
| Crew | "Autonomous team" | CrewAI role-based collaboration mode |
| Flow | "Deterministic workflow" | CrewAI event-driven production mode |

## Further Reading

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — five patterns + agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate pattern
