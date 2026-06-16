# STaR, V-STaR, Quiet-STaR — Self-Taught Reasoning

> The smallest self-improvement loop hides in the rationale. The model generates a chain of thought, keeps the ones that land on correct answers, and fine-tunes on those. That's STaR. V-STaR adds a verifier for better inference-time filtering. Quiet-STaR pushes rationales down to every token. All three work. But none are magic—the loop retains any shortcut that happens to reach the right answer.

**Type:** Learn
**Languages:** Python (standard library, bootstrap loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Reasoning & CoT), Phase 15 · 01 (Long-horizon framework)
**Time:** ~60 minutes

## The Problem

The direct approach to teaching models to reason is collecting human-written reasoning traces. This is expensive, slow, and capped by how many high-quality chains of thought humans are willing to write.

STaR (Self-Taught Reasoner, Zelikman et al., 2022) asks: what if the model writes its own rationales and we score them against known answers? The loop:

1. Sample a reasoning trace plus an answer.
2. If the final answer is correct, keep the trace.
3. Fine-tune on kept traces.
4. Repeat.

It works. GSM8K and CommonsenseQA both improve without additional human annotation. But the loop has a built-in bias: any rationale that produces the correct answer gets kept, regardless of whether the reasoning itself is sound. V-STaR (Hosseini et al., 2024) patches this with a learned verifier; Quiet-STaR (Zelikman et al., 2024) generalizes the idea to per-token internal rationales.

## The Concept

### STaR: Bootstrapping on What Works

Start with a base model that has weak reasoning ability. For each training problem, sample a rationale plus an answer. If the answer matches the label, keep the (problem, rationale, answer) triple. Fine-tune the model on the kept set. Repeat.

One detail is critical. If the model can never get a problem right, the loop can't learn on it. STaR adds **rationalization**: for problems the model gets wrong, inject the correct answer as a hint and re-prompt the model to generate a rationale leading to it. Rationalized rationales are added to the training set.

Original paper results (Zelikman et al., 2022): a GPT-J base model improved from 5.8% to 10.7% on GSM8K through repeated STaR rounds with rationalization—about 5 percentage points absolute improvement. On CommonsenseQA, STaR-trained GPT-J 6B reached 72.5%, matching a GPT-3 175B fine-tuned on hand-written rationales (~73%)—roughly 30x larger.

### V-STaR: Training a Verifier with DPO

STaR discards incorrect rationales. Hosseini et al. (2024) noticed those are data too: each (rationale, "is this correct") pair can train a verifier. They use Direct Preference Optimization (DPO) on both correct and incorrect solutions to build a ranker. At inference time, sample N rationales and pick the verifier's top choice.

Reported improvements: +4 to +17 percentage points over prior self-improvement baselines on GSM8K and MATH, with most gains from using the verifier for inference-time filtering rather than additional generator fine-tuning.

### Quiet-STaR: Per-Token Internal Rationales

Zelikman et al. (2024) ask: what if the model learns to generate a small internal rationale at every token position, not just between problem and answer? Quiet-STaR trains the model to emit a hidden "thought" before each predicted token, then blends the thought-informed prediction with the baseline prediction through a learned weight.

Results: Mistral 7B without task-specific fine-tuning improved from 5.9% to 10.9% zero-shot on GSM8K and from 36.3% to 47.2% on CommonsenseQA. The model learns "when to think"—hard tokens get longer internal rationales; easy ones get almost none.

### Why All Three Share the Same Safety Concern

All three methods use the final answer as the gradient signal. A rationale that reaches the correct answer via flawed reasoning—shortcuts, lucky guesses, or non-generalizable patterns—gets positively reinforced. On in-distribution problems, shortcuts work. On out-of-distribution problems, they silently collapse.

V-STaR's verifier mitigates by learning to rank rationales, but the verifier is trained on the same labels. It may learn to prefer well-formatted incorrect reasoning over honest uncertainty. Safer designs combine STaR-style data with both: (a) process-supervised reward models (rewarding intermediate steps, not just answers), and (b) held-out OOD evaluations that break simple shortcuts.

### Comparison

| Method | Training signal | Inference cost | Data waste | Known failure mode |
|---|---|---|---|---|
| STaR | Keep (rationale, answer) if correct | 1x | Discards all wrong rationales | Shortcut rationales |
| STaR + rationalization | Above + retry with correct answer hint | 1x | Less | Rationalized rationales may be unfaithful |
| V-STaR | STaR + DPO verifier trained on both classes | Nx (best-of-N) | Minimal | Verifier may reinforce confident errors |
| Quiet-STaR | Per-token rationale + mixing weight | 1.5-3x | Minimal | Gradient still conditioned on answer |

### Where It Fits in the 2026 Stack

STaR is old. But the pattern appears everywhere in 2025-2026. RL on verifiable math problems (DeepSeek-R1, Kimi-k1.5, o1) is STaR's answer-conditioned gradient signal at scale. Process reward models (Lightman et al., 2023; OpenAI's "Let's verify step by step") are the process-supervised alternative. AlphaEvolve (Lesson 3) is STaR for code with a program evaluator replacing labels. Darwin Godel Machine (Lesson 4) is STaR for agent scaffolding itself.

Understand STaR and all of these click. It's the minimum viable self-improvement loop.

## Use It

`code/main.py` runs a simulated STaR loop on a toy arithmetic task. You can observe:

- How accuracy climbs with bootstrap rounds.
- How shortcuts sneak in: the simulator includes a class of "lazy" rationales that reach the correct answer 40% of the time but generalize poorly. Watch whether STaR keeps them.
- How a verifier (V-STaR style) helps at inference time but can't fully prune shortcuts introduced during training.

## Ship It

`outputs/skill-star-loop-reviewer.md` helps you audit a self-taught reasoning pipeline before you train on it.

## Exercises

1. Run the simulator. Set shortcut frequency to zero, then to 0.4. How much does final accuracy diverge between the two runs, even though both hit >90% on the training distribution?

2. Add a held-out OOD test to the simulator. Draw problems from a different distribution and evaluate the bootstrapped model on both in-distribution and OOD sets. Quantify the gap.

3. Read Quiet-STaR paper (arXiv:2403.09629) Section 3. Explain the "end-of-thought" token and the mixing weight head in three sentences each.

4. Compare STaR's "keep if correct" filter with a process-supervised alternative that independently rewards each reasoning step. Identify differences in annotation cost and potential quality.

5. Design an evaluation that can catch shortcut rationales in a deployed model. It doesn't need to be perfect—it needs to break the simplest shortcuts that a STaR loop would reinforce.

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| STaR | "Self-Taught Reasoner" | Fine-tune on model-generated rationales that land on correct answers; repeat |
| Rationalization | "Hint-assisted retry" | For problems the base model gets wrong, inject the correct answer and prompt for a rationale |
| V-STaR | "STaR with a verifier" | Train a DPO verifier on correct and incorrect rationales for inference-time filtering |
| Quiet-STaR | "Per-token rationale" | Generate hidden thoughts at each token position; blend with baseline prediction |
| Answer-conditioned gradient | "Outcome-based signal" | Training loop rewards the final answer, not reasoning steps |
| Process reward model | "Step-level verifier" | Reward model trained on per-step correctness rather than outcome—contrast with STaR |
| Shortcut rationale | "Right answer, wrong reasoning" | Rationale reaching the label via non-generalizable patterns; STaR keeps these |

## Further Reading

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — The original paper.
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — Adds a DPO verifier for inference-time filtering.
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — Per-token internal rationales.
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Process reward models, an alternative gradient signal.
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — RL on verifiable tasks, scaling STaR to frontier training.
