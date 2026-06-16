# Group Chat and Speaker Selection

> AutoGen GroupChat and AG2 GroupChat let N agents share a conversation; a selector function (LLM, round-robin, or custom) picks who speaks next. This is the prototype for emergent multi-agent conversation — agents don't know their role in some static graph, they simply react to the shared pool. AutoGen v0.2's GroupChat semantics are preserved in the AG2 fork; AutoGen v0.4 rewrote them into an event-driven actor model. Microsoft moved AutoGen to maintenance mode in February 2026 and merged it with Semantic Kernel into Microsoft Agent Framework (RC February 2026). The GroupChat primitive survives in both AG2 and Microsoft Agent Framework — learn once, use everywhere.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 min

## The Problem

Static graphs (LangGraph) are great when the workflow is known. Real conversations aren't static: sometimes the coder asks the reviewer, sometimes the researcher, sometimes the writer. Hard-coding every possible handoff leads to edge explosion. What you want is *agents reacting to a shared pool*, with some function deciding who talks next.

That's exactly what AutoGen GroupChat does.

## The Concept

### The Shape

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

Every agent sees every message. A selector function is called each turn to pick who speaks next.

### Three Selector Flavors

**Round-robin.** Fixed rotation. Deterministic. Scales linearly in N but ignores context — the coder gets a turn even if the topic is legal review.

**LLM selection.** An LLM call reads the recent pool and returns the best next speaker. Context-aware but slow: adds one LLM call per turn. AutoGen's default.

**Custom.** A Python function with whatever logic you write. Typical: LLM selection with fallback rules ("verifier always follows coder").

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

The `GroupChatManager` holds the selector. When an agent finishes a turn, the manager calls the selector, which returns the next agent. The loop continues until a termination condition is met.

### Termination

Three common patterns:

- **Max rounds.** Hard cap on total turns.
- **`TERMINATE` token.** An agent can emit a sentinel message; the manager stops when it sees it.
- **Goal-achievement check.** A lightweight verifier runs each turn and halts the chat when done.

### AutoGen → AG2 Split and Microsoft Agent Framework Merger

In early 2025, Microsoft began the major AutoGen rewrite (v0.4) around an event-driven actor model. The community forked AutoGen v0.2's GroupChat semantics into AG2, preserving the API that early adopters had already integrated.

In February 2026, Microsoft announced AutoGen would move to maintenance mode, with the event-driven actor model merging into **Microsoft Agent Framework** (RC February 2026, now merged with Semantic Kernel). The GroupChat concept survives in both lineages; implementation details differ. AG2 is the preferred upstream for v0.2-compatible code.

### When GroupChat Fits

- **Emergent conversation.** You don't want to pre-wire every possible next-speaker.
- **Mixed-role tasks.** Coder asks researcher, researcher asks archivist, archivist asks coder back. The flow isn't a DAG.
- **Exploratory problem-solving.** Think "brainstorming session" not "pipeline."

### When It Fails

- **Strict determinism.** LLM selectors can be inconsistent. Same prompt, different runs, different next-speaker.
- **Sycophancy cascade.** Agents collapse toward whoever sounds most confident. Prompt adversarial roles explicitly.
- **Context bloat.** Every agent reads every message; after 10 turns context is huge. Use projection (Lesson 15) to trim views.
- **Hot speaker.** One agent dominates the conversation because the selector favors its expertise. Introduce speaker balancing as a selector feature.

### GroupChat vs Supervisor

Same primitives, different defaults:

- Supervisor: one agent plans, others execute. Selector is "ask the planner what to do."
- GroupChat: all agents are peers; selector is a function over the shared pool.

Both use Lesson 04's four primitives. GroupChat defaults to LLM-selector orchestration and full-pool shared state.

## Build It

`code/main.py` implements a GroupChat from scratch with the standard library. Three agents (coder, reviewer, manager), both round-robin and LLM-selector variants, and `TERMINATE`-token-based termination.

The demo prints the conversation log plus the selector's decision trace for both variants.

Run:

```
python3 code/main.py
```

## Use It

`outputs/skill-groupchat-selector.md` configures a GroupChat selector for a given task — round-robin vs LLM-selection vs custom, and which selector inputs to use (recent messages, agent expertise, turn count).

## Ship It

Checklist:

- **Max-rounds cap.** Always have one. Typical: 10-20 for most tasks.
- **Speaker balance metric.** Track turns per agent; alert when imbalance exceeds threshold.
- **Termination token.** `TERMINATE` or a dedicated verifier agent.
- **Projected or trimmed memory.** After ~10 messages, consider giving each agent a trimmed view to prevent context bloat.
- **Selector logging.** For the LLM-selector variant, log both the selector's input and its choice. Otherwise undebugable.

## Exercises

1. Run `code/main.py`. Compare conversation under round-robin vs LLM-selection. Which agent dominates under each?
2. Add a "max turns per agent" rule to the selector. How does it affect the conversation log?
3. Implement goal-achievement termination: stop when the reviewer returns "approved." How often does it trigger before hitting the max-rounds cap?
4. Read AutoGen stable docs on GroupChat (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html). Identify the default selector `GroupChatManager` uses.
5. Read the AG2 repo (https://github.com/ag2ai/ag2) and compare its v0.2 GroupChat to the v0.4 event-driven version. What specific property (throughput, fault tolerance, composability) does v0.4 add?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GroupChat | "Agents in the same chat room" | Shared message pool + selector function. AutoGen / AG2's primitive. |
| Speaker selection | "Who talks next" | The function that picks the next agent. Round-robin, LLM-selection, or custom. |
| GroupChatManager | "Meeting facilitator" | AutoGen component holding the selector and looping through turns. |
| ConversableAgent | "Base agent" | AutoGen's base class; an agent that can send and receive messages. |
| Termination token | "The stop word" | Sentinel string that ends the chat (typically `TERMINATE`). |
| Hot speaker | "One agent dominates" | Failure mode where the selector keeps picking the same agent. |
| Context bloat | "Pool grows unbounded" | Every agent reads every prior message; context grows with turns. |
| Projection | "Filtered view" | A role-specific view into the shared pool to prevent context bloat. |

## Further Reading

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — reference implementation
- [AG2 repo](https://github.com/ag2ai/ag2) — community AutoGen v0.2 continuation
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — the merged successor, RC February 2026
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — event-driven actor model rewrite details
