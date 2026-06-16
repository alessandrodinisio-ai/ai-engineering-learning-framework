# Handoffs and Routines — Stateless Orchestration

> OpenAI's Swarm (October 2024) distilled multi-agent orchestration into two primitives: **routine** (instructions-as-system-prompt + tools) and **handoff** (a tool that returns another Agent). No state machines, no branching DSL — the LLM routes by calling the right handoff tool. OpenAI Agents SDK (March 2025) is its production successor. Swarm itself remains the conceptually cleanest reference — its entire source code is a few hundred lines. The pattern went viral because its API surface is roughly "agent = prompt + tools; handoff = function returning agent." Limitation: stateless, so memory is the caller's problem.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 min

## The Problem

Every multi-agent framework wants you to learn its DSL: LangGraph's nodes and edges, CrewAI's crews and tasks, AutoGen's GroupChat and manager. These DSLs are real abstractions, but they make the whole thing feel heavier than it needs to be.

Swarm pushes the other direction: use the tool-calling capability the model already has. A handoff becomes a tool call. The orchestrator is whichever agent currently holds the conversation. The state machine is implicit in the agents' system prompts.

## The Concept

### Two Primitives

**Routine.** A system prompt defining the agent's role and available tools. Think of it as scoped instructions: "You are a triage agent; if the user asks about refunds, handoff to the refund agent."

**Handoff.** A tool the agent can call that returns a new Agent object. The Swarm runtime detects the Agent return and switches the active agent on the next turn.

That's the entire abstraction.

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

The triage agent's system prompt makes it pick the right handoff based on the user message. The LLM's tool call does the routing.

### Why It Went Viral

- **Tiny API.** Two concepts to learn.
- **Uses what the model already knows.** Tool calling is production-grade across providers.
- **No state machine overhead.** You don't describe the graph; agents' prompts describe who they hand off to.

### The Stateless Deal

Swarm is explicitly stateless between runs. The framework keeps message history during a run, but it persists nothing. Memory, continuity, long-running tasks — all the caller's problem.

In production (OpenAI Agents SDK, March 2025), this is one of the biggest changes: the SDK adds built-in session management, guardrails, and tracing while preserving the handoff primitive.

### When Swarm/Handoffs Fit

- **Triage patterns.** A frontline agent routes the user to a specialist.
- **Skill-based handoffs.** "If the task needs code, call the coder; needs research, call the researcher."
- **Short, bounded conversations.** Customer service, FAQ-to-ticket, simple workflows.

### When Swarm Struggles

- **Long sessions with shared memory.** A handoff resets conversation state to the new agent's prompt plus history. Without caller-managed memory, there's no persistent state across agents.
- **Parallel execution.** Handoffs are one-at-a-time — the active agent switches. Parallelism requires the caller to orchestrate multiple Swarm runs.
- **Audit and replay.** Stateless runs are hard to replay exactly; the LLM's handoff choices are nondeterministic.

### OpenAI Agents SDK (March 2025)

The production successor adds:

- **Session state.** Threads persisted across runs.
- **Guardrails.** Input/output validation hooks.
- **Tracing.** Every tool call and handoff is logged.
- **Handoff filters.** Control what context transfers during a handoff.

The handoff primitive survives; production ergonomics are layered around it.

### Swarm vs GroupChat

Both use LLM-driven routing, but they differ on **who picks next**:

- GroupChat: a selector (function or LLM) picks the next speaker externally.
- Swarm: the current agent picks its successor by calling a handoff tool.

Swarm is "agent decides what's next"; GroupChat is "manager decides what's next." Swarm's decision lives in the active agent's tool call; GroupChat's lives in the `GroupChatManager`.

## Build It

`code/main.py` implements Swarm from scratch: an Agent dataclass, a handoff mechanism (tool returns Agent), and a run loop that detects agent switches.

Demo: a triage agent routes to refund, sales, or support specialist agents. Each specialist has its own tools. The run loop prints each handoff.

Run:

```
python3 code/main.py
```

## Use It

`outputs/skill-handoff-designer.md` designs a handoff topology for a given task: which agents exist, what handoffs each can call, what context transfers.

## Ship It

Checklist:

- **Handoff logging.** Every handoff writes a trace event with from-agent, to-agent, context snapshot.
- **Context transfer rules.** Decide what moves on handoff: full history (expensive), last N messages, or a summary.
- **Guardrails on handoffs.** A handoff to a specialist agent with different tool permissions should require auth — otherwise prompt injection can force unintended handoffs.
- **Loop detection.** Two agents handing back and forth is a common failure; detect with a simple last-K ring check.
- **Fallback agent.** If a handoff target doesn't exist, fall back to a safe default.

## Exercises

1. Run `code/main.py`, triage to the refund agent. Confirm the active agent on the second turn is refund.
2. Add a loop detection rule: if the same two agents hand off 3 times consecutively, force exit. Design the fallback.
3. Read the OpenAI Agents SDK docs on handoff filters. Implement a "summarize on handoff" version: the outgoing agent compresses context into bullet points before the incoming agent takes over.
4. Compare Swarm's handoff vs GroupChatManager's selector. Which pattern makes prompt injection worse, and why?
5. Read the Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents). Identify one explicit design decision Swarm made that OpenAI Agents SDK changed or preserved.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Routine | "Agent's prompt" | System prompt + tool list. Defines the role and available handoffs. |
| Handoff | "Transfer to another agent" | A tool the active agent can call that returns a new Agent. Runtime switches the active agent. |
| Stateless | "No memory between runs" | Swarm persists nothing; memory is the caller's responsibility. |
| Active agent | "Who's talking now" | The agent currently holding the conversation. A handoff changes it. |
| Context transfer | "What moves on handoff" | Strategy for what history the incoming agent sees: all, last N, or summary. |
| Handoff loop | "Agents ping-pong" | Failure mode where two agents keep handing back and forth. |
| OpenAI Agents SDK | "Production Swarm" | March 2025 successor; adds sessions, guardrails, tracing atop the handoff primitive. |
| Handoff filter | "Gate the transfer" | SDK feature that inspects and modifies context at the handoff boundary. |

## Further Reading

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — the reference exposition
- [OpenAI Swarm repo](https://github.com/openai/swarm) — original implementation, preserved as conceptual reference
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — production successor with sessions and tracing
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — how Claude Code subagents use a handoff-like pattern via `Task`
