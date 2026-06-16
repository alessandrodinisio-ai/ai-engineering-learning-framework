# Self-Refine and CRITIC: Iterative Output Improvement

> Self-Refine (Madaan et al., 2023) has a single LLM play three roles in a loop — generate, feedback, refine. Average gain: +20 absolute across 7 tasks. CRITIC (Gou et al., 2023) hardens the feedback step by routing verification to external tools. In 2026 this pattern appears in every framework as "evaluator-optimizer" (Anthropic) or guardrail loops (OpenAI Agents SDK).

**Type:** Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 01 (The Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 minutes

## Learning Objectives

- Name Self-Refine's three prompts (generate, feedback, refine) and explain why history matters for the refine prompt.
- Explain CRITIC's key insight: LLMs are unreliable at self-verification without external anchoring.
- Implement a Self-Refine loop with history and optional external verification using the standard library.
- Map this pattern to Anthropic's "evaluator-optimizer" workflow and OpenAI Agents SDK's output guardrails.

## The Problem

An agent produces an almost-correct answer. Maybe a line of code has a syntax error. Maybe a summary is too long. Maybe a plan misses an edge case. What you want is: the agent critiques its own output and fixes it.

Self-Refine shows this can be done with a single model, no training data, no RL. But there is a catch: LLMs are poor at self-verification on hard facts. CRITIC provides the fix — route the verification step to external tools (search, code interpreter, calculator, test runner).

Together, these two papers define the 2026 default for iterative improvement: generate, verify (externally when possible), refine, stop when the verifier passes.

## The Concept

### Self-Refine (Madaan et al., NeurIPS 2023)

One LLM, three roles:

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

Key detail: `refine` sees the full history — all prior outputs and critiques — so it does not repeat mistakes. The paper runs an ablation: remove history and quality drops sharply.

Headline: +20 absolute averaged across 7 tasks (math, code, acronyms, dialogue), including GPT-4. No training, no external tools, single model.

### CRITIC (Gou et al., arXiv:2305.11738, v4, Feb 2024)

Self-Refine's weakness: the feedback step is the LLM scoring itself. This is unreliable for factual claims (a hallucination often looks plausible to the model that produced it). CRITIC replaces `feedback(task, output)` with `verify(task, output, tools)`, where `tools` includes:

- Search engine for factual claims.
- Code interpreter for code correctness.
- Calculator for arithmetic.
- Domain-specific verifiers (unit tests, type checkers, linters).

The verifier produces a structured critique anchored in tool results. The refiner then acts on this critique.

Headline: CRITIC beats Self-Refine on factual tasks because the critique is anchored. On tasks without an external verifier (creative writing, formatting), CRITIC degrades to Self-Refine.

### Stop Conditions

Two common shapes:

1. **Verifier passes.** An external test returns success. Prefer when available (unit tests, type checkers, guardrail assertions).
2. **No feedback emitted.** The model says "output is fine." Cheaper but unreliable; pair with a max-iteration cap.

The 2026 default: combine them. "Stop if verifier passes OR model says no issues AND iterations >= 2 OR iterations >= max_iterations."

### Evaluator-Optimizer (Anthropic, 2024)

Anthropic's December 2024 post lists this as one of five workflow patterns. Two roles:

- Evaluator: scores the output and produces a critique.
- Optimizer: revises the output based on the critique.

Loop until the evaluator passes. This is Self-Refine/CRITIC in Anthropic's framing. The key engineering detail Anthropic adds: evaluator and optimizer prompts should be structurally different to prevent the model from rubber-stamping.

### OpenAI Agents SDK Output Guardrails

OpenAI Agents SDK exposes this pattern as "output guardrails." A guardrail is a validator that runs on the agent's final output. If the guardrail trips (throws `OutputGuardrailTripwireTriggered`), the output is rejected and the agent can retry. The guardrail can call tools (CRITIC-style) or be a pure function (Self-Refine-style).

### Pitfalls in 2026

- **Rubber-stamp loops.** The same model with the same prompt style doing both generation and critique converges to "looks good to me." Use structurally different prompts, or a smaller cheaper model for critique.
- **Over-refinement.** Each refinement pass adds latency and tokens. Budget 1–3 passes; beyond that escalate to human review.
- **CRITIC on trivial tasks.** If there is no external verifier, CRITIC degrades to Self-Refine; do not pay the latency for a stub verifier.

## Build It

`code/main.py` implements Self-Refine and CRITIC on a toy task: given a topic, produce a short bullet-point list. The verifier checks formatting (3 bullets, each under 60 characters). CRITIC adds an external "fact checker" that penalizes known hallucinations.

Components:

- `generate` — scripted producer.
- `feedback` — LLM-style self-critique.
- `verify_external` — CRITIC-style anchored verifier.
- `refine` — rewrites output based on history.
- Stop condition — verifier passes or max 4 iterations.

Run it:

```
python3 code/main.py
```

Compare the Self-Refine and CRITIC runs. CRITIC catches a factual error that Self-Refine misses because the external verifier has anchoring that the self-critic lacks.

## Use It

Anthropic's evaluator-optimizer is this pattern in Claude-friendly terminology. OpenAI Agents SDK's output guardrails are the CRITIC shape (guardrails can call tools). LangGraph offers a reflection node that reads like Self-Refine. Google's Gemini 2.5 Computer Use adds a step-by-step safety evaluator, which is a CRITIC variant: each action is verified before being committed.

## Ship It

`outputs/skill-refine-loop.md` configures an evaluator-optimizer loop given the task shape, verifier availability, and iteration budget. Produces prompts for the generator, evaluator/verifier, and optimizer, plus a stop strategy.

## Exercises

1. Run the toy with max_iterations=1. Does CRITIC still help?
2. Replace the external verifier with a noisy one (30% random false positives). What does the loop do? This is the reality for most 2026 guardrail stacks.
3. Implement a "different model for critique" variant: large model generates, small model critiques. Can it beat the same-model setup?
4. Read CRITIC Section 3 (arXiv:2305.11738 v4). Name three categories of verification tools and give one example of each.
5. Map OpenAI Agents SDK's `output_guardrails` to CRITIC's verifier role. Where does the SDK get it wrong, and where does it get it right?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Self-Refine | "LLM that fixes itself" | Generate → feedback → refine loop in a single model, with history |
| CRITIC | "tool-anchored verification" | Replaces feedback with external verifiers (search, code, calc, tests) |
| Evaluator-Optimizer | "Anthropic workflow pattern" | Two roles — evaluator scores, optimizer revises — loop to convergence |
| Output guardrail | "post-hoc check" | OpenAI Agents SDK validator run on agent's output after production |
| Verify step | "critique phase" | The load-bearing decision: anchored or self-eval |
| Refine history | "what the model already tried" | Prior outputs + critiques prepended to the refine prompt; removing it tanks quality |
| Rubber-stamp loop | "self-agreement failure" | Same-prompt critique returns "looks good"; fix with structurally different prompts |
| Stop condition | "convergence test" | Verifier passes OR no feedback AND iteration cap; never use a single condition |

## Further Reading

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — the canonical paper
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) — tool-anchored verification
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — evaluator-optimizer workflow pattern
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — output guardrails as CRITIC-shape validators
