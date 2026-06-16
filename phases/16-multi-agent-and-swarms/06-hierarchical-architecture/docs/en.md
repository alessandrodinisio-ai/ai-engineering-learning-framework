# Hierarchical Architecture and Its Failure Modes

> Hierarchical is just nested supervisors. A manager agent manages sub-managers, sub-managers manage workers. CrewAI's `Process.hierarchical` is the textbook version: a `manager_llm` dynamically dispatches tasks and validates outputs. Its LangGraph equivalent is `create_supervisor(create_supervisor(...))`. When the task itself mirrors a real org chart, this is the natural pattern. It's also the pattern most prone to collapsing into "management churn" — manager agents that dispatch poorly, misread subordinate outputs, or fail to reach consensus. Serial often beats it.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern)
**Time:** ~60 min

## The Problem

Once the supervisor pattern clicks, the next step is natural: "What if workers are themselves supervisors?" Teams have sub-teams; companies have departments within departments. The hierarchical architecture mirrors exactly this.

The problem: LLM managers aren't like human managers. A human manager has stable priors about what subordinates know. An LLM manager re-reasons the entire organization from scratch every turn, based on what's in its context. A tiny drift in that context and the whole tree dispatches to the wrong places.

## The Concept

### The Shape

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

Each internal node plans, dispatches, and synthesizes. Only leaves do work.

### Where It Shines

- **Clear organizational mapping.** If the real task is departmental ("legal reviews this doc, finance reviews this doc, engineering reviews this doc, then summarize for executives"), the hierarchy is explicit.
- **Local summarization.** Each sub-manager synthesizes its team's outputs before the top-level manager sees them. The top-level manager sees three sub-manager summaries, not fifteen worker outputs.

### Where It Breaks

2026 post-mortems repeatedly surface three failure modes:

1. **Task mis-assignment.** The manager reads the goal, hallucinates a decomposition, and dispatches work to the wrong sub-manager. Because sub-managers dutifully execute on what they receive, the error only surfaces at top-level synthesis — one layer away from where a human could have caught it.
2. **Output misreading.** A sub-manager returns "Could not verify claim X." The top-level manager summarizes as "Claim X was not confirmed." Meaning drifts at every layer.
3. **Consensus loops.** Two sub-managers disagree; the top-level manager asks them to reconcile; they re-dispatch downward; workers re-run; sub-managers return slightly different answers; loop. CrewAI's `Process.hierarchical` uses a step limit to prevent this, but the limit itself is now a hyperparameter.

### The Deciding Question

Serial (linear pipeline) vs hierarchical: does your task actually have mutually independent sub-teams, or is it a linear chain pretending to be a tree? If the latter, use serial. If the former, use hierarchical — but budget for explicit reconciliation rules.

### CrewAI's Implementation

`Process.hierarchical` puts a manager LLM above specialized crews. The manager:

- Receives the top-level task,
- Assigns sub-tasks to crews,
- Evaluates crew outputs,
- Decides to accept, re-dispatch, or iterate.

Docs: https://docs.crewai.com/en/introduction (look for "Hierarchical Process" under Core Concepts).

### LangGraph's Implementation

LangGraph uses nested `create_supervisor` calls. The inner supervisor has its own graph; the outer supervisor treats the inner graph as an opaque node. This is cleaner for debugging (you can step through each graph independently) but harder to express dynamic reshaping of the tree.

Reference: https://reference.langchain.com/python/langgraph-supervisor.

## Build It

`code/main.py` runs a 3-layer hierarchy:

- Top-level manager: decomposes a task into "engineering" and "legal" branches,
- Engineering sub-manager: decomposes into "frontend" and "backend" workers,
- Legal sub-manager: one worker.

The demo contrasts the happy path (everyone agrees) with a **perturbed path**: the top-level manager's decomposition mislabels "legal" as "finance," and you watch the error cascade — the sub-manager dutifully does finance work, the top-level synthesizer reports finance findings, the original legal question goes unanswered.

Run:

```
python3 code/main.py
```

Output shows both paths, clearly contrasting "what was asked" vs "what was delivered."

## Use It

`outputs/skill-hierarchy-fitness.md` evaluates whether a given task should use hierarchical, serial, or flat supervisor. Input: task description, org structure, reconciliation budget. Output: pattern recommendation with specific failure modes to guard against.

## Ship It

If you're going hierarchical:

- **Limit tree depth to 2.** Three layers already hides most errors beyond observability.
- **Explicit reconciliation budget.** Set a max number of rounds before the top-level manager must decide. Usually 2.
- **Provenance on every synthesis.** Each node's summary must cite which leaf outputs produced it.
- **Alert on decomposition drift.** Log the manager's decomposition step-by-step; diff against the user query. If the decomposition no longer covers the query, fire an alert.

## Exercises

1. Run `code/main.py`, compare the happy and perturbed paths. How many layers of manager handoff does it take before the top-level output is entirely unrelated to the user's question?
2. Add a third layer (top → sub → sub-sub → worker). As depth increases, measure what fraction of perturbed-path runs self-correct vs diverge completely.
3. Implement a "canary" worker at each sub-manager that is always asked the verbatim user question. Use the canary's answer to detect decomposition drift. How should the manager react when the canary disagrees with the synthesized answer?
4. Read CrewAI's `Process.hierarchical` docs. Identify one specific guardrail CrewAI applies (step limit, manager_llm constraint) and describe which failure mode it targets.
5. Compare nested LangGraph supervisors vs CrewAI hierarchical. Which makes reconciliation loops cheaper to detect?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Hierarchical | "Org-chart pattern" | Supervisors of supervisors; only leaves do work. |
| Manager LLM | "The boss" | The LLM at internal nodes that decomposes, assigns, and validates. |
| Decomposition drift | "The boss went off-topic" | The top-level manager's split no longer covers the original question. |
| Reconciliation loop | "Endless meetings" | Sub-managers disagree; top re-dispatches; workers re-run; loop until budget exhaustion. |
| Depth-2 ceiling | "Don't go past 2 layers" | Empirical guardrail: 3+ layers and observability collapses. |
| Canary question | "Ground truth at every layer" | A worker always asked the verbatim original query, used to detect drift. |
| Provenance chain | "Who said what" | Tracing from each synthesis back to the leaf outputs that produced it. |

## Further Reading

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — textbook hierarchical with manager LLM
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — nested supervisors via `create_supervisor`
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) — why Anthropic deliberately chose flat supervisor over hierarchical
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy; the coordination-failures section documents decomposition drift
