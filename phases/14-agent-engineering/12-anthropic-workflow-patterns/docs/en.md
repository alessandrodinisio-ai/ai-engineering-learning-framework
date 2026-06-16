# Anthropic's Workflow Patterns: Simple Beats Complex

> Schluntz and Zhang (Anthropic, December 2024) distinguish workflows (predefined paths) from agents (dynamic tool use). Five workflow patterns cover most cases. Start with a direct API call. Add an agent only when steps are unpredictable.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## Learning Objectives

- Name Anthropic's five workflow patterns: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.
- Explain the agent vs. workflow distinction and the engineering cost of each.
- Recognize when to pick a workflow over an agent (and vice versa).
- Implement all five patterns using the standard library against a scripted LLM.

## The Problem

Teams reach for multi-agent frameworks for problems that only need a single function call. The cost is real: frameworks add layers that obscure prompts, hide control flow, and invite premature complexity. Schluntz and Zhang's December 2024 post is the most-cited industry pushback: start simple, add complexity only when it earns its cost.

## The Concept

### Workflows vs Agents

- **Workflow.** LLMs and tools orchestrated through predefined code paths. The engineer owns the graph.
- **Agent.** The LLM dynamically directs its own tools and steps. The model owns the graph.

Both have their place. Workflows are cheaper, faster, and easier to debug. Agents unlock open-ended problems but make failure modes harder to reason about.

### The Augmented LLM

The foundation for all five patterns: an LLM wired with three capabilities — search (retrieval), tools (actions), memory (persistence). Any API call can use these.

### The Five Patterns

1. **Prompt chaining.** Call 1's output is Call 2's input. Use when a task has a clean linear decomposition. Optionally add programmatic gates between steps.

2. **Routing.** A classifier LLM picks which downstream LLM or tool to invoke. Use when categorically different inputs need different handling (tier-1 support vs. refund vs. bug vs. sales).

3. **Parallelization.** Run N LLM calls concurrently, aggregate results. Two flavors: sectioning (different chunks) and voting (same prompt, N runs, majority/synthesis).

4. **Orchestrator-workers.** An orchestrator LLM dynamically decides which workers (also LLMs) to run and synthesizes their output. Similar to an agent loop, but the orchestrator does not loop indefinitely.

5. **Evaluator-optimizer.** One LLM proposes an answer, another LLM evaluates it. Iterate until the evaluator passes. This is the generalization of Self-Refine (Lesson 05).

### Where Workflows Beat Agents

- **Predictable tasks.** If you can enumerate the steps, you should.
- **Cost-constrained tasks.** Workflows have bounded step counts; agents can spin out of control.
- **Compliance-constrained tasks.** Auditors want to read the graph, not infer it from traces.

### Where Agents Beat Workflows

- **Open-ended research.** When the next step depends on what the previous step returned.
- **Variable-length tasks.** Minutes to hours of work, unknown number of steps.
- **Novel domains.** When you don't know the correct workflow yet — explore first, codify later.

### The Companion Discipline of Context Engineering

"Effective context engineering for AI agents" (Anthropic 2025) formalizes the adjacent discipline: the 200k window is a budget, not a container. What to include, when to compact, when to let context grow. Covered in detail in the Phase 14 lesson on context compression (that was earlier Phase 14 Lesson 06 before this curriculum was renumbered).

## Build It

`code/main.py` implements all five workflow patterns against a `ScriptedLLM`:

- `prompt_chain(input, steps)` — sequential.
- `route(input, classifier, handlers)` — classify + dispatch.
- `parallel_vote(prompt, n, aggregator)` — N runs, aggregate.
- `orchestrator_workers(task, workers)` — orchestrator picks workers.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — loop until pass.

Run it:

```
python3 code/main.py
```

Each pattern prints its own trace. Code for each pattern is roughly 10–15 lines; a framework's cost is measured in thousands.

## Use It

- Use a direct API call for most tasks.
- Reach for a framework only when the pattern genuinely needs persistent state (LangGraph), actor-model concurrency (AutoGen v0.4), or role templates (CrewAI).
- Use Claude Agent SDK when you want the Claude Code harness shape without reinventing it.

## Ship It

`outputs/skill-workflow-picker.md` picks the right pattern for a given task description, includes decision rationale, and provides a path to convert into an agent when a workflow is insufficient.

## Exercises

1. Implement routing with a confidence threshold. Below threshold -> escalate to a human. Where does the threshold land for a tier-1 support use case?
2. Add a timeout to `parallel_vote`. What happens when one call hangs? How do you aggregate with a missing vote?
3. Turn `evaluator_optimizer` into a bandit: retain the top-2 outputs across iterations so a late good result doesn't get overwritten by a late bad one.
4. Combine prompt chaining and routing: a router picks one of three chains. Compare token cost vs. a single large-prompt approach.
5. Pick one of your production features. Draw the workflow graph. Count steps. Would an agent genuinely be better here?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Workflow | "Predefined flow" | Engineer-owned graph of LLM and tool calls |
| Agent | "Autonomous AI" | Model-owned graph; dynamic tool direction |
| Augmented LLM | "LLM with tools" | LLM + search + tools + memory; the atomic unit |
| Prompt chaining | "Sequential calls" | Call N's output is Call N+1's input |
| Routing | "Classifier dispatch" | Pick which chain/model handles this input |
| Parallelization | "Fan-out" | N concurrent calls; aggregate by sectioning or voting |
| Orchestrator-workers | "Scheduler agent" | Orchestrator LLM dynamically picks expert LLMs |
| Evaluator-optimizer | "Proposer + judge" | Iterate until evaluator passes; generalization of Self-Refine |

## Further Reading

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — the five workflow patterns
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — the companion discipline
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — when stateful graphs earn their cost
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — productionized orchestrator-workers pattern
