# ReWOO and Plan-and-Execute: Decoupled Planning

> ReAct interleaves thinking and acting in a single stream. ReWOO separates them: one big plan first, then execution. 5× fewer tokens, +4% accuracy on HotpotQA, and you can distill the planner into a 7B model. Plan-and-Execute generalizes it; Plan-and-Act extends it to web navigation.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (The Agent Loop)
**Time:** ~60 minutes

## Learning Objectives

- Explain why ReWOO's Planner / Worker / Solver split saves tokens and is more robust than ReAct's interleaved loop.
- Implement a plan DAG, a dependency-ordered executor, and a solver that combines worker outputs — all in the standard library.
- Use the 2026 "five workflow patterns" framework (Anthropic) to decide whether a task should run plan-then-execute or interleaved ReAct.
- Recognize when long-horizon web or mobile tasks require Plan-and-Act's synthetic plan data.

## The Problem

ReAct's interleaved "think-act-observe" loop is simple and flexible, but every tool call must carry the full preceding context — including every earlier thought. Token usage grows quadratically with depth. Worse: when a tool fails mid-loop, the model must re-derive the entire plan from the error observation.

ReWOO (Xu et al., arXiv:2305.18323, May 2023) noticed this and made a bet: plan the entire thing upfront, fetch evidence in parallel, then combine the answer. One LLM call for planning, N tool calls for evidence (parallelizable), one LLM call for solving. The trade-off is reduced flexibility (the plan is static) in exchange for much better token efficiency and cleaner failure modes.

## The Concept

### Three Roles

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

The Planner produces a DAG. Each node specifies a tool, its arguments, and which earlier nodes it depends on (references like `#E1`, `#E2`). Workers execute nodes in topological order. The Solver stitches everything together.

### Why 5× Fewer Tokens

ReAct's prompt length grows linearly with step count. By step 10, the prompt contains thought 1 + action 1 + observation 1 + thought 2 + action 2 + observation 2, and so on. Each intermediate step also redundantly carries the original prompt again.

ReWOO pays for one planner prompt (large), N small worker prompts (each is just the tool call, no chain), and one solver prompt. The paper measures ~5× fewer tokens on HotpotQA while achieving +4 absolute accuracy.

### Why It Is More Robust

In ReAct, if worker 3 fails, the loop must reason its way out of the error mid-stream. In ReWOO, worker 3 returns an error string; the solver sees it in the context of the original plan and can degrade gracefully. Failure localization is per-node, not per-step.

### Planner Distillation

The paper's second result: because the planner never sees observations, you can fine-tune a 7B model on planner outputs from a 175B teacher. The small model handles planning; the large model is no longer needed at inference time. This is now standard practice — many production agents in 2026 use a small planner with a large executor, or vice versa.

### Plan-and-Execute (LangChain, 2023)

LangChain's August 2023 post generalized ReWOO into a pattern name: Plan-and-Execute. An upfront planner emits a step list, an executor runs each step, and an optional replanner can revise after observing results. This is closer to ReAct than ReWOO (the replanner brings observations back into planning), but retains the token savings.

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act extends the pattern to long-horizon web and mobile agents. The key contribution is synthetic plan data: an annotated trajectory generator produces training data where plans are explicitly visible. This fine-tunes planner models so they remain coherent over 30–50 steps in WebArena-style tasks — where a single ReAct trajectory would have lost coherence long ago.

### When to Choose Which

| Pattern | When to use |
|---------|------|
| ReAct | Short tasks, unknown environment, need reactive exception handling |
| ReWOO | Structured tasks with known tools, token-sensitive, evidence parallelizable |
| Plan-and-Execute | Like ReWOO, but replans after partial execution |
| Plan-and-Act | Long-horizon (>30 steps), web/mobile/computer-use |
| Tree of Thoughts | When search is worth the token multiplier (Lesson 04) |

Anthropic's December 2024 advice: start with the simplest thing. If the task is one tool call plus a summary, don't build ReWOO. If the task is a 40-step research assignment, don't just use ReAct.

## Build It

`code/main.py` implements a toy ReWOO:

- `Planner` — a scripted policy that emits a plan DAG from a prompt.
- `Worker` — dispatches each node's tool call through a registry.
- `Solver` — scripted combination logic that reads evidence and produces the final answer.
- Dependency resolution — references like `#E1` are replaced with earlier worker outputs.

The demo answers "What is the population of the capital of France, rounded to the nearest million?" with a two-step plan: (1) look up the capital, (2) look up the population, then solve.

Run it:

```
python3 code/main.py
```

The trace shows the full plan first, then worker results, then solver combination. Compare token count (we print a rough character count) against a ReAct-style interleaved run — ReWOO wins on this structured task.

## Use It

LangGraph offers Plan-and-Execute as a recipe (ReAct via `create_react_agent`, plan-execute via custom graph). CrewAI's Flows directly encode this pattern: you define tasks upfront and the Flow DAG executes them. Plan-and-Act's synthetic data approach is largely still in research; the runtime pattern (explicit plan DAG) has reached production via LangGraph and CrewAI Flows.

## Ship It

`outputs/skill-rewoo-planner.md` generates a ReWOO plan DAG from a user request given a tool catalog. It validates the plan (no cycles, every reference resolves, every tool exists) before handing off to an executor.

## Exercises

1. Parallelize worker execution for plan nodes that are independent of each other. On a 6-node DAG with 2 parallel groups, what speedup do you get?
2. Add a replanner node that triggers when any worker returns an error. What is the minimal change to turn ReWOO into Plan-and-Execute?
3. Swap the `Planner` for a small model (7B-class) and keep the `Solver` on a frontier model. Compare end-to-end quality — where does the split break down?
4. Read ReWOO paper Section 4 on planner distillation. Conceptually reproduce the 175B → 7B result: what training data do you need, and how do you score plan quality?
5. Port this toy to Plan-and-Act's trajectory shape: the plan is a sequence, not a DAG. Which trade-offs change?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| ReWOO | "Reasoning Without Observation" | Plan first, fetch evidence in parallel, then solve — no observations in the planning prompt |
| Plan-and-Execute | "LangChain's plan-execute pattern" | ReWOO with an optional post-execution replanner node |
| Plan-and-Act | "extended plan-execute" | Explicit planner/executor split with synthetic plan training data for long-horizon tasks |
| Evidence reference | "#E1, #E2, ..." | Plan-node placeholders replaced with earlier worker outputs at dispatch time |
| Planner distillation | "small planner, big executor" | Fine-tuning a small model on planner traces from a large teacher |
| Token efficiency | "fewer round trips" | ~5× fewer tokens vs ReAct on HotpotQA in the paper |
| DAG executor | "topological dispatcher" | Runs plan nodes in dependency order; parallelizes within each level |

## Further Reading

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — the canonical paper
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — extended planner-executor with synthetic plans
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — framework recipe
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — pick the simplest pattern that works
