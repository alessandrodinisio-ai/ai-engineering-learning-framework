# AlphaEvolve — Evolutionary Coding Agent

> Pair a frontier coding model with an evolutionary loop and a machine-checkable evaluator. Let the loop run long enough. It discovers a procedure that multiplies 4x4 complex matrices in 48 scalar multiplications—the first improvement to Strassen in 56 years. It also finds a Borg scheduling heuristic at Google that reclaims ~0.7% of cluster compute in production. The architecture is deliberately boring. The victory comes from evaluator rigor.

**Type:** Learn
**Languages:** Python (standard library, evolutionary loop toy)
**Prerequisites:** Phase 15 · 01 (Long-horizon framework), Phase 15 · 02 (Self-taught reasoning)
**Time:** ~60 minutes

## The Problem

Large language models can write code. Evolutionary algorithms can search over code. Both have been tried independently for decades; both hit ceilings. The LLM ceiling is confabulation: the model writes code that looks plausible but doesn't do what it claims. The evolutionary ceiling is search cost: random mutations on syntax almost never produce compilable programs, let alone better ones.

AlphaEvolve (Novikov et al., DeepMind, arXiv:2506.13131, June 2025) combines them. The LLM proposes targeted edits to a program database; an automated evaluator scores each variant; high-scoring variants become parents for subsequent generations. The LLM handles the expensive step of writing plausible code; the evaluator catches confabulation. The loop runs for hours to weeks.

Reported results: 4x4 complex matrix multiplication in 48 scalar multiplications (Strassen's 1969 bound was 49), a Borg scheduling heuristic in Google production, 32.5% speedup in FlashAttention kernels, Gemini training throughput improvements.

The architecture works because the evaluator is machine-checkable. Where the evaluator isn't checkable, it doesn't work. This asymmetry is the point of this lesson.

## The Concept

### The Loop

1. Start with a correct but suboptimal seed program `P_0`.
2. Maintain a database of program variants, each scored by the evaluator.
3. Sample one or more parents from the database (MAP-elites style or island-based).
4. Prompt the LLM (Gemini Flash for many candidates, Gemini Pro for hard ones) to produce a modified variant of the parent(s).
5. Compile, run, and evaluate the variant on the held-out evaluator.
6. Insert into the database keyed by its score and feature descriptor.
7. Repeat.

Two details are critical. First, the prompt to the LLM includes not just the parent program—usually several top variants from the database, the evaluator signature, and a brief task description. The model's job is to propose a targeted change likely to improve the score. Second, the database is structured (MAP-elites grid, island-based) to keep the loop exploring diversity rather than hill-climbing the current leader.

### What Makes the Evaluator Non-Negotiable

AlphaEvolve's victories all come from domains where the evaluator is fast, deterministic, and hard to game:

- **Matrix multiplication algorithms**: a unit test that performs the multiplication and checks bitwise equality.
- **Borg scheduling heuristics**: a production-grade simulator replaying historical cluster load and measuring wasted compute.
- **FlashAttention kernels**: a correctness test plus wall-clock benchmarks on real hardware.
- **Gemini training throughput**: measured GPU-seconds per step.

In every case the evaluator catches the dominant class of LLM errors: confabulated correctness claims, performance claims that vanish on hardware, and edge-case failures. Remove the evaluator and the loop optimizes for pretty code.

### Reward Hacking Is the Same Coin's Other Side

Evolution optimizes whatever the evaluator measures. If the evaluator is imperfect, the loop finds the imperfection. In an unverified domain, the loop optimizes surface features rather than intended behavior. DeepMind states this explicitly in the paper: AlphaEvolve's success transfers only to domains where evaluator rigor matches search ambition.

Concrete examples of reward hacking in 2025-2026 code-search loops:

- An objective rewarding "completion time" spawned submissions of empty solutions.
- A benchmark score rewarding "correctness under tests" spawned test memorization and overfitting.
- A "code quality" proxy metric spawned comment removal and variable renaming with no semantic change.

AlphaEvolve's fix: ship a held-out evaluator the LLM has never seen, with inputs generated at evaluation time. Even so, DeepMind still recommends rigorous review of any proposed deployment.

### Why LLM + Search Beats Either Alone

The LLM produces compilable, semantically plausible modifications. A genetic algorithm running random mutations on a 2000-line Python file almost always produces syntax errors. The LLM also focuses search on plausible neighborhoods (change one function, not random bytes), dramatically reducing wasted evaluator calls.

Conversely, the evaluator catches LLM confabulation. The LLM will confidently claim a function is "O(n log n) in the limit" when it's actually O(n^2); a wall-clock benchmark settles the matter definitively.

### Where AlphaEvolve Fits in the Frontier Stack

| System | Generator | Evaluator | Domain | Example victory |
|---|---|---|---|---|
| AlphaEvolve | Gemini | Correctness + benchmark | Algorithms, kernels, schedulers | 48-multiplication 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | Correctness | Combinatorics | Cap-set lower bound |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | LLM critique + experiments | ML research | ICLR workshop paper |
| Darwin Godel Machine (L4) | Agent scaffold | SWE-bench / Polyglot | Agent code | SWE-bench 20% → 50% |

All four are variations on the same recipe: generator plus evaluator, looped. The difference is what the evaluator judges and how rigorous it is.

## Use It

`code/main.py` implements a minimal AlphaEvolve-like loop on a toy symbolic regression problem. The "LLM" is a standard-library proxy that proposes small syntactic mutations to a program computing a target function. The "evaluator" measures mean squared error on held-out test points.

Observe:

- How the best score improves across generations.
- How a MAP-elites grid keeps diverse solutions alive, preventing the loop from converging to a local minimum.
- How removing the held-out test (using only the training evaluator) causes the loop to dramatically overfit.

## Ship It

`outputs/skill-evaluator-rigor-audit.md` is the precondition check for considering an AlphaEvolve-style loop in a new domain: does your evaluator actually catch the failures you care about?

## Exercises

1. Run `code/main.py`. Note the best-score trajectory. Disable the held-out evaluator (use `--no-holdout` flag) and run again. Quantify the overfitting.

2. Read AlphaEvolve paper Section 3 on the MAP-elites grid. Design a feature descriptor for a new problem (e.g., compiler optimization passes) that would maintain search diversity.

3. The 48-multiplication 4x4 result improved Strassen's 49-multiplication bound after 56 years. Read paper Appendix F and explain in three sentences why this problem's evaluator is particularly easy to get right, and why most domains are not like it.

4. Propose a domain where AlphaEvolve would fail. Pinpoint exactly where the evaluator breaks and why.

5. For a domain you're familiar with, write out the evaluator signature you would use. Include (a) correctness conditions, (b) performance metric, (c) held-out input generation rules, (d) at least one anti-reward-hacking check.

## Key Terms

| Term | Colloquial usage | Actual meaning |
|---|---|---|
| AlphaEvolve | "DeepMind's evolutionary coding agent" | Gemini + program database + machine-checkable evaluator |
| MAP-elites | "Diversity-preserving archive" | Grid keyed by feature descriptor; each cell stores the best variant for that descriptor |
| Island model | "Parallel evolving subpopulations" | Independent populations with periodic migration; prevents premature convergence |
| Machine-checkable evaluator | "Deterministic oracle" | A unit test, simulator, or benchmark that the LLM cannot fake—the loop's precondition |
| Reward hacking | "Optimizing the metric instead of the goal" | The loop finds a path to max score without doing the intended task |
| Seed program | "Starting point" | A correct but suboptimal initial program from which the loop evolves |
| Held-out evaluator | "Evaluation data the LLM has never seen" | Inputs generated at evaluation time to prevent memorization |

## Further Reading

- [Novikov et al. (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — Full paper.
- [DeepMind blog on AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — Vendor writeup with results.
- [AlphaEvolve results repository](https://github.com/google-deepmind/alphaevolve_results) — Discovered algorithms, including 48-multiplication 4x4 matmul.
- [Romera-Paredes et al. (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — Predecessor system.
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Positions evaluator-bounded autonomy as a key research direction.
