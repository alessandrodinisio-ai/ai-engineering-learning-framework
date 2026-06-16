# Planning with HTN and Evolutionary Search

> Symbolic planning handles cases where plans can be proven correct. Evolutionary code search handles cases where a fitness function is machine-verifiable. ChatHTN (2025) and AlphaEvolve (2025) demonstrate what each unlocks when paired with LLMs.

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 minutes

## Learning Objectives

- Explain Hierarchical Task Networks (HTN): tasks, methods, operators, preconditions, effects.
- Describe ChatHTN's hybrid loop — symbolic search with LLM fallback decomposition.
- Explain AlphaEvolve's evolutionary loop, and why it only works when there is a programmatic evaluator.
- Implement a toy HTN planner plus a toy evolutionary search using only the standard library.

## The Problem

ReWOO (Lesson 02), Plan-and-Execute, and ReAct cover most agent planning. Two cases they cover poorly:

1. **Provably correct plans.** Scheduling, route planning, compliance workflows — the plan must be sound by construction. An LLM plan that is fluent but occasionally hallucinates a step is unacceptable.
2. **Optimization with machine-verifiable fitness functions.** Matrix multiplication, scheduling heuristics, compiler passes — the goal is not "a correct plan" but "the best plan."

HTN planning and AlphaEvolve solve these two different problems. Both use the LLM as an amplifier, not a replacement.

## The Concept

### Hierarchical Task Networks

An HTN consists of:

- **Tasks** — compound (to be decomposed) and primitive (directly executable).
- **Methods** — ways to decompose a compound task into subtasks, with preconditions.
- **Operators** — primitive actions with preconditions and effects.
- **State** — a set of facts.

Planning: given a goal task and an initial state, find a decomposition such that the resulting primitive operators' preconditions are satisfied in sequence.

HTN predates LLMs and remains the reference for provably correct plans.

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN (arXiv:2505.11814) interleaves symbolic HTN with LLM queries:

1. Attempt to decompose the current compound task using existing methods.
2. If no method applies, ask the LLM: "How would you decompose `task` in state `s`?"
3. Translate the LLM's answer into candidate subtasks.
4. Validate against operator schemas; reject invalid decompositions.
5. Recurse.

The paper's core claim: every produced plan is provably sound because LLM suggestions enter only as candidate decompositions, never as direct edits to the plan. The symbolic layer governs correctness; the LLM expands the method library.

Online method learning (OpenReview `gwYEDY9j2x`, 2025 follow-up) adds a learner that generalizes LLM-produced decompositions via regression — cutting LLM query frequency by up to 75%.

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve (arXiv:2506.13131, DeepMind, June 2025) is a different beast: evolutionary code search orchestrated by a Gemini 2.0 Flash/Pro ensemble.

The loop:

1. Start with a seed program + a programmatic evaluator (returns a fitness score).
2. The LLM ensemble proposes mutations.
3. Run mutations through the evaluator.
4. Keep the best; mutate again.

Published results:

- First improvement over Strassen for 4x4 complex matrix multiplication in 56 years (48 scalar multiplications).
- Recovered 0.7% of Google's compute via a Borg scheduling heuristic.
- 32% speedup for FlashAttention on a frontier workload.

Hard constraint: the fitness function must be machine-verifiable. Running evolutionary search on prose answers will not converge.

### When to Use Which

| Problem class | Use | Why |
|---------------|-----|-----|
| Scheduling with hard constraints | HTN + ChatHTN | Provably sound |
| Compiler optimization | AlphaEvolve | Machine-verifiable fitness |
| Multi-step task execution | ReAct / ReWOO | LLM in the loop, no formal guarantees |
| Code improvement with tests | AlphaEvolve | Tests are the evaluator |
| Policy-constrained automation | HTN | Preconditions encode policy |

### Where This Pattern Breaks

- **HTN without operators.** Without precondition/effect schemas, the soundness claim collapses. ChatHTN's "LLM suggests decomposition" requires schemas to reject invalid actions.
- **AlphaEvolve without a real evaluator.** "Ask the LLM if the code is better" is not a fitness function. The evaluator must be deterministic and fast.
- **Over-engineering.** Most agent tasks need neither. Start with ReAct or ReWOO.

## Build It

`code/main.py` implements two toys:

- A standard-library HTN planner with operators, methods, preconditions, effects, plus an `LLMFallback` that kicks in when no method matches a compound task. The "LLM" is a scripted decomposer so the planner runs offline.
- A standard-library evolutionary search over arithmetic programs: grow expressions whose output minimizes `|f(x) - target|` on a test set. The evaluator is deterministic.

Run it:

```
python3 code/main.py
```

The trace shows the HTN planner decomposing a compound task (with one mid-plan LLM fallback), and the evolutionary loop converging to a target expression.

## Use It

- **HTN planners** — `pyhop`, `SHOP3`, or roll your own for domain-specific policy enforcement.
- **ChatHTN** — research code; the pattern (symbolic + LLM fallback) ports cleanly to any HTN planner.
- **AlphaEvolve** — DeepMind paper; the pattern (ensemble + evaluator) is reproducible. OpenEvolve and similar open-source forks are emerging.
- **Agent frameworks** — none currently offer HTN or AlphaEvolve as first-class citizens. Build it as a sub-agent or background worker.

## Ship It

`outputs/skill-hybrid-planner.md` generates a hybrid planner scaffold (HTN or evolutionary) with the LLM's role explicitly scoped.

## Exercises

1. Extend the HTN planner with backtracking: when an operator's postcondition fails at runtime, roll back and try the next method.
2. Add an LLM method cache to ChatHTN: when the LLM decomposes task `T` under state pattern `P`, store the result. Re-check the method library before invoking the LLM next time.
3. Swap the evolutionary search evaluator for a real test suite. Evolve a sorting function that passes 20 test cases; report the number of generations needed to converge.
4. Read AlphaEvolve's evaluator design notes. Design an evaluator for a domain you care about (SQL query optimization, test suite minimization, deployment YAML).
5. Combine: use HTN to decompose a compound task into subtasks, then run evolutionary search on each subtask's primitive operator. Where does it shine, and where is it over-engineering?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| HTN | "Hierarchical planner" | Task decomposition with operators, preconditions, effects |
| Method | "Decomposition rule" | A way to break a compound task into subtasks |
| Operator | "Primitive action" | A concrete step with preconditions and effects |
| ChatHTN | "LLM + HTN" | Symbolic planner asks LLM when no method matches |
| AlphaEvolve | "Evolutionary code search" | LLM ensemble mutates code; deterministic evaluator selects |
| Fitness function | "Evaluator" | Deterministic, machine-verifiable score on output |
| Online method learning | "Cached LLM decompositions" | Store and generalize LLM plans to cut query cost |

## Further Reading

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — symbolic + LLM hybrid planner
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — evolutionary code search with LLM mutations
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — when to use a planner vs. a simple loop
