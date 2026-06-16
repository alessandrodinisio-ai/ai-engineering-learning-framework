# LangGraph — State Machines for Agents

> A hand-rolled ReAct loop is a `while True`. A LangGraph ReAct loop is a graph you can checkpoint, interrupt, branch, and time-travel through. The agent hasn't changed. What changed is the harness around it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 min

## The Problem

You ship a function-calling agent. It runs for three turns, then something goes sideways: the model tries a tool that returns a 500, the user changes their mind mid-task, or the agent decides to refund an order without waiting for sign-off. The `while True:` loop has no hooks. You can't pause it, rewind it, or fork it to see "what if the model had picked the other tool."  The moment you ship it beyond a demo, the agent is a black box that either succeeds or fails.

Once you see it, the next step is obvious. The agent already is a state machine — the system prompt plus message history plus pending tool calls plus next action. Make the state machine explicit: nodes for "the model thinks," "a tool runs," "a human approves," and edges for the conditional transitions between them. Once the graph is explicit, the framework gives you four things for free: checkpointing (save state between steps), interrupts (pause for a human), streaming (stream tokens and intermediate events), and time travel (rewind to a prior state, try a different branch).

LangGraph is the library that delivers this abstraction. It is not an agent framework in the LangChain "here's an AgentExecutor, good luck" sense. It is a graph runtime with first-class state, first-class persistence, and first-class interrupts. The agent loop is something you draw, not something you hand-write.

## The Concept

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

A `StateGraph` has three things.

1. **State.** A typed dictionary (TypedDict or Pydantic model) that flows through the graph. Each node receives the full state, returns a partial update, and LangGraph merges it using a per-field *reducer* — `operator.add` for lists that should accumulate, override by default.
2. **Nodes.** Python functions `state -> partial_state`. Each is a discrete step: "call the model," "run tools," "summarize."
3. **Edges.** Transitions between nodes. Static edges go to one place. Conditional edges take a routing function `state -> next_node_name`, letting the graph branch based on model output.

You compile the graph. Compilation binds the topology, attaches a checkpointer (optional but critical for production), and returns a runnable. You invoke it with an initial state and a `thread_id`. Every step of execution persists a checkpoint keyed by `(thread_id, checkpoint_id)`.

### Four superpowers

**Checkpointing.** Every node transition writes the new state to a store (memory for tests, Postgres/Redis/SQLite for production). Invoking the graph again with the same `thread_id` resumes. The graph picks up where it paused.

**Interrupts.** Tag a node with `interrupt_before=["human_review"]` and execution halts before that node runs. State is persisted. Your API responds to the user with "awaiting approval." A later request to the same `thread_id` with `Command(resume=...)` resumes execution.

**Streaming.** `graph.stream(state, mode="updates")` yields state deltas as they happen. `mode="messages"` streams LLM tokens from model nodes. `mode="values"` yields full snapshots. You pick what to surface in the UI.

**Time travel.** `graph.get_state_history(thread_id)` returns the full checkpoint log. Pass any prior `checkpoint_id` to `graph.invoke` and you fork from that point. Great for debugging ("what if the model had picked tool B?") and for regression-testing production traces by replaying them.

### Reducers are the key

Every state field has a reducer. Most defaults are fine — new values overwrite old ones. But message lists need `operator.add` so new messages append rather than replace. Parallel edges merge their updates through the reducer. If two nodes both update `messages` and you forgot `Annotated[list, add_messages]`, the second silently wins and you lose half the conversation. Reducers are the one subtle thing in this library; get them right and everything else composes.

### The four-node ReAct graph

A production ReAct agent is four nodes and two edges:

1. `agent` — calls the LLM with the current message history. Returns an assistant message (possibly with tool_calls).
2. `tools` — executes whatever tool_calls are in the last assistant message, appending tool results as tool messages.
3. A conditional edge from `agent` that routes to `tools` if the last message has tool_calls, otherwise to `END`.
4. A static edge from `tools` back to `agent`.

That's it. You get the full ReAct loop (Thought → Action → Observation → Thought → …) with checkpointing, interrupts, and streaming in ~40 lines of code.

### StateGraph vs Send (fan-out)

`Send(node_name, state)` lets a node dispatch parallel sub-graphs. Example: the agent decides to query three retrievers at once. Each `Send` spawns a parallel execution of the target node; their outputs merge through state reducers. This is how LangGraph expresses the orchestrator-workers pattern without thread primitives.

### Sub-graphs

A compiled graph can be a node inside another graph. The outer graph sees a single node; the inner graph has its own state and its own checkpoints. This is how teams build supervisor-worker agents: the supervisor graph routes user intent to a domain-specific worker sub-graph.

## Build It

### Step 1: State and nodes

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` is the reducer that makes the message list accumulate rather than overwrite. Forgetting it is the most common LangGraph bug.

### Step 2: Running with a thread

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Each update is a dict `{node_name: state_delta}`. Your frontend can stream these to the UI so the user sees "agent thinking… calling search_web… got results… answering."

### Step 3: Adding a human-in-the-loop interrupt

Tag a node so execution pauses before it runs.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect the proposed tool call.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If rejected: write a rejection message then resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

State, checkpoints, and thread all persist across the interrupt. Nothing lives in memory except during execution.

### Step 4: Time travel for debugging

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # go back three steps
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point onward
```

Passing `None` as input replays from the given checkpoint; passing a value appends it as an update to that checkpoint's state before resuming. This is how you reproduce a bad agent run without re-running the entire conversation.

### Step 5: Swapping the checkpointer for production

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis, and Postgres are available out of the box. `MemorySaver` is for tests. Anything that needs to persist across restarts wants a real store.

## Ship It

> You build agents as graphs, not as `while True` loops.

Before reaching for LangGraph, do a 60-second design:

1. **Name the nodes.** Every discrete decision or side-effecting action is a node. "Agent thinks," "tool runs," "reviewer approves," "response streams." If you can't list them, the task isn't agent-shaped yet.
2. **Declare state.** A minimal TypedDict with reducers on every list field. Don't stuff everything into `messages`; promote task-specific fields (a work-in-progress `plan`, a `budget` counter, a `retrieved_docs` list) to top level.
3. **Draw the edges.** Use static edges unless the next step depends on model output. Each conditional edge needs a routing function with named branches.
4. **Pick the checkpointer early.** `MemorySaver` for tests, Postgres/Redis/SQLite for everything else. Do not go to production without one — no checkpointer means no recovery, no interrupts, no time travel.
5. **Decide interrupts before tool runs, not after.** Approval goes on the edge entering a side-effecting node so you can cancel before causing harm; validation goes on the edge leaving the model so you can reject bad calls cheaply.
6. **Stream by default.** `mode="updates"` for the UI, `mode="messages"` for token-level streaming inside model nodes, `mode="values"` for full snapshots during evaluation.

Refuse to ship any LangGraph agent without a checkpointer. Refuse to ship any agent that interrupts *after* a side effect instead of before. Refuse to ship any `messages` field without `add_messages` as its reducer.

## Exercises

1. **Easy.** Implement the four-node ReAct graph above with a calculator tool and a web search tool. Verify that `list(app.get_state_history(config))` returns at least four checkpoints for a two-turn conversation.
2. **Medium.** Add a `planner` node that runs before `agent` and writes a structured `plan: list[str]` into state. Have `agent` mark plan steps as completed. Fail the test if `plan` is lost after a checkpoint resume (wrong reducer).
3. **Hard.** Build a supervisor graph that routes between three sub-graphs (`researcher`, `writer`, `reviewer`) using `Send`. Each sub-graph has its own state and checkpointer. Add an `interrupt_before=["writer"]` on the outer graph so a human can approve the research brief. Confirm time travel from a prior checkpoint replays only the forked branch.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| StateGraph | "LangGraph's graph" | The builder object you add nodes and edges to before compilation. |
| Reducer | "how a field merges" | A function `(old, new) -> merged` applied when a node returns an update for a field; default is overwrite, `add_messages` is append. |
| Thread | "a conversation ID" | A `thread_id` string that scopes all checkpoints for one session. |
| Checkpoint | "a paused state" | A persisted snapshot of full graph state after a node transition, keyed by `(thread_id, checkpoint_id)`. |
| Interrupt | "pause for a human" | `interrupt_before` / `interrupt_after` halts execution at a node boundary; `Command(resume=...)` continues. |
| Time travel | "fork from a prior step" | `graph.invoke(None, config_with_old_checkpoint_id)` replays from that checkpoint onward. |
| Send | "parallel sub-graph dispatch" | A construct a node can return to spawn N parallel executions of a target node. |
| Sub-graph | "compiled graph as node" | A compiled StateGraph used as a node in another graph; retains its own state scope. |

## Further Reading

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — The authoritative reference for StateGraph, reducers, checkpointers, and interrupts.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — The mental model used in this lesson, straight from the source.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Details on Postgres/SQLite/Redis stores, checkpoint namespaces, and thread IDs.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)`, and edit-state patterns.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — The pattern every LangGraph agent implements; read it for reasoning trace rationale.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — Which graph shape to prefer (chain, router, orchestrator-workers, evaluator-optimizer) and when.
- Phase 11 · 09 (Function Calling) — The tool-call primitive each LangGraph agent node reuses.
- Phase 11 · 14 (Model Context Protocol) — External tool discovery plugged into a LangGraph `ToolNode` via MCP adapters.
- Phase 11 · 17 (Agent Framework Tradeoffs) — When to pick LangGraph over CrewAI, AutoGen, or Agno.
