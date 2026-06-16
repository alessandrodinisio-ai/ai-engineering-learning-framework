# Supervisor / Orchestrator-Worker Pattern

> A lead agent plans and dispatches; specialized workers execute in parallel contexts and report back. This is the pattern behind Anthropic's Research system (Claude Opus 4 as lead, Sonnet 4 as subagents), which measured +90.2% versus single-agent Opus 4 on internal research evals. Anthropic's engineering blog notes that 80% of variance on BrowseComp is explained by token usage alone — multi-agent wins largely because each subagent gets a fresh context window. This lesson builds the supervisor pattern from primitives and covers the engineering lessons from 2026 production deployments.

**Type:** Learn + Build
**Languages:** Python (standard library, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~75 min

## The Problem

Research is the canonical task that defeats single-agent systems. You ask "What changed in multi-agent systems between 2023 and 2026?" The single agent serially reads five papers, fills half its context with their body text, then must reason over all of it. By paper five it's forgotten paper one. It can't parallelize.

The supervisor pattern fixes this: a lead agent plans the search, dispatches each sub-question to a worker, then synthesizes. Each worker gets its own 200k-token window for a narrow question. The lead never sees the raw papers — only worker summaries.

Anthropic's production Research system reports +90.2% over single Opus 4 on internal research evals. The same blog post notes that 80% of variance on BrowseComp is explained by *token usage*. The fresh context per subagent is the primary mechanism.

## The Concept

### The Pattern

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

The lead never reads raw material. Workers never see each other's work before the lead synthesizes. Each arrow is a handoff with a narrow artifact.

### Why It Wins

Three mechanisms:

1. **Fresh context per subagent.** A worker exploring "FIPA-ACL heritage" isn't burdened by the 40k tokens the lead spent on planning. It gets a 200k window for one question.
2. **Specialization via prompt.** The lead's prompt is "decompose and synthesize," not "research." Each worker's prompt is narrow: "Find what changed in X." Focused prompts produce focused outputs.
3. **Parallelism.** Workers run concurrently. Wall-clock time is roughly `max(worker_times) + plan + synthesis`, not `sum(worker_times)`.

### Engineering Lessons (Anthropic 2025)

Anthropic's blog listed several production lessons that still apply in 2026:

- **Match effort to query complexity.** Simple queries: one agent, 3-10 tool calls. Complex queries: 10+ agents. The lead must estimate this, not the caller.
- **Go wide, then narrow.** Decompose into broad sub-questions first; if answers merit deeper exploration, spawn more workers per sub-question.
- **Rainbow deployment.** Agents are long-running and stateful. Traditional blue-green doesn't work. Anthropic uses rainbow deployments: new versions roll out gradually while old versions drain.
- **Token usage is the dominant factor.** Multi-agent uses ~15x the tokens of single-agent. Only run it when task value justifies the cost.

### LangGraph's Pivot

LangGraph initially shipped a `langgraph-supervisor` library with a high-level `create_supervisor` helper. In 2025 LangChain changed the recommendation to implementing the supervisor pattern directly via tool calls, because tool calls give stronger control over *what the supervisor sees* (context engineering). The library still works; the docs now recommend the tool-call form.

### Failure Modes

- **Lead hallucinates the plan.** If the lead generates sub-questions that don't actually decompose the real problem, workers do precise research on the wrong targets.
- **Worker over-exploration.** Without explicit scope bounds, a worker drifts from its assigned sub-question, polluting the synthesis step.
- **Synthesis conflicts.** Two workers return contradictory facts. The lead must either re-query (adds a round) or explicitly note the disagreement. Worst failure is silently picking a side: the user never knows there was a disagreement.

### When Supervisor Is Wrong

- **Serial tasks.** If step 2 genuinely needs step 1's output, parallelism buys nothing. Use a pipeline (CrewAI Sequential, LangGraph linear graph).
- **Simple queries.** A single agent handles them faster and cheaper. Use the lead's "match effort" check before spawning workers.
- **Strict determinism.** The supervisor uses LLM-selector dispatch. When audit/replay matters more than adaptability, a static graph is better.

## Build It

`code/main.py` implements a supervisor managing three parallel workers with `threading`. The lead decomposes a query into sub-questions, workers run concurrently on one sub-question each, and the lead synthesizes. No real LLM — workers are scripted to simulate "fetch and summarize."

Key structure:

- `Lead.plan(query)` decomposes a query into 3 sub-questions.
- `Worker.run(sub_q)` returns a fake summary (in production could be any tool-using agent).
- `Lead.run(query)` launches workers in threads, joins, then synthesizes.

Run:

```
python3 code/main.py
```

Output shows the plan, parallel worker traces with start/end timestamps, and the final synthesis. You can see the wall-clock benefit: three 0.3s workers complete in ~0.35s rather than 0.9s.

## Use It

`outputs/skill-supervisor-designer.md` takes a user query and produces a supervisor pattern design: lead system prompt, worker roles, sub-question decomposition rules, and synthesis template. Use it before building a new research-style agent system.

## Ship It

Checklist before deploying the supervisor pattern:

- **Model pairing.** Lead uses a reasoning-tier model (Opus-class, `o3`-class). Workers use faster, cheaper models (Sonnet, `o4-mini`).
- **Worker timeout.** Any worker exceeding 2x the median runtime gets killed; the lead either re-spawns with narrower scope or continues without it.
- **Per-worker token cap.** A hard ceiling (e.g., 10x the expected synthesis input) prevents a runaway worker from burning the budget.
- **Observability.** Trace the lead's plan, each worker's tool calls, and the synthesis. This is the basis for any post-hoc debugging.
- **Rainbow rollout.** Stateful long-running agents need versioned drain-and-replace, not hot-swap.

## Exercises

1. Run `code/main.py`, then change the lead to spawn 5 workers instead of 3. Observe the wall-clock effect. At what worker count does spawn overhead outweigh parallelism savings in this demo?
2. Implement a worker timeout: kill any worker running beyond 0.5s and have the lead synthesize the remaining results. What observability do you need to know a worker was cut?
3. Add a conflict detection step to the lead's synthesis: if two workers return contradictory answers, the lead notes the disagreement rather than picking one. How do you detect contradiction without calling an LLM?
4. Read Anthropic's Research system engineering blog. List three practices this toy demo would need to adopt for production.
5. Compare LangGraph's `create_supervisor` (old way) vs the new tool-call recommendation. Which gives you better control over what the supervisor sees? Why does Anthropic explicitly pass only sub-answers, not raw worker context, into synthesis?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor | "Lead agent" | An orchestrator agent that plans, dispatches, and synthesizes. Does not do work itself. |
| Worker | "Subagent" | A focused agent spawned by the supervisor with narrow scope and its own context window. |
| Orchestrator-worker | "Supervisor pattern" | Same thing, different name. 2026 literature uses both. |
| Fresh context | "Clean window" | A worker's context starts from its system prompt and assigned question, without the lead's history. |
| Rainbow deployment | "Gradual rollout" | Long-running stateful agents need versioned drain-and-replace, not blue-green. |
| Token dominance | "Context is the variable" | Per Anthropic, 80% of research eval variance comes from total token usage, not model choice. |
| Scale effort | "Match agent count to complexity" | The lead estimates query difficulty and spawns 1 or 10+ workers accordingly. |
| Synthesis conflict | "Workers disagree" | Two workers return contradictory facts; the lead must surface the disagreement, not silently pick one. |

## Further Reading

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production reference for the supervisor pattern
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — tool-call supervisor is now the recommended form
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — the old helper, still in production use in 2026
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — handoff-based supervisor variant
