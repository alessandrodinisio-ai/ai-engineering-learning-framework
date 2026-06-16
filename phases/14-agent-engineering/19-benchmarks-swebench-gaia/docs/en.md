# Benchmarks: SWE-bench, GAIA, AgentBench

> Three benchmarks anchor agent evaluation in 2026. SWE-bench tests code patching. GAIA tests generalist tool use. AgentBench tests multi-environment reasoning. Understand their composition, their contamination status, and what they fail to measure.

**Type:** Learn
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 06 (Tool use)
**Time:** ~60 minutes

## Learning Objectives

- Name SWE-bench's test harness (FAIL_TO_PASS) and explain why it gates on unit tests.
- Explain why SWE-bench Verified (OpenAI, 500 tasks) exists and what it removes.
- Describe GAIA's design: easy for humans, hard for AI; three difficulty levels.
- Name AgentBench's eight environments and its main bottleneck for open-source LLMs.
- Summarize SWE-bench+'s contamination findings and their implications.

## The Problem

Leaderboards tell you which model wins on a benchmark. They don't tell you:

- Whether the benchmark is contaminated (solutions in training data, test leakage).
- Whether the benchmark measures what you care about (code vs browsing vs generalist).
- Whether the evaluator is robust (AST matching, state checks, human review).

Understand these three anchor benchmarks and their failure modes before citing any numbers.

## The Concept

### SWE-bench (Jimenez et al., ICLR 2024 oral)

- 2,294 real GitHub issues from 12 popular Python repositories.
- The agent receives: the codebase at the pre-fix commit + the natural-language issue description.
- The agent produces: a patch.
- Evaluator: applies the patch, runs the repository's test suite. The patch must flip FAIL_TO_PASS tests (previously failing, now passing) while not breaking PASS_TO_PASS tests.

SWE-agent (Yang et al., 2024) scored 12.5% at launch by emphasizing the agent-computer interface (file editor commands, search syntax the model can understand).

### SWE-bench Verified

OpenAI, August 2024. A human-curated 500-task subset. Removes ambiguous issues, flaky tests, and tasks with unclear fixes. The primary benchmark for "can your agent ship real patches?"

### Contamination

- Over 94% of SWE-bench issues predate most models' cutoff dates.
- **SWE-bench+** found that 32.67% of successful patches leaked the solution in the issue text (the model saw the fix in the description), and another 31.08% were suspicious due to thin test coverage.
- Verified is cleaner but not contamination-free.

Practical implication: a model scoring 50% on SWE-bench may score 35% on SWE-bench+. If you claim SWE-bench performance, always report both.

### GAIA (Mialon et al., November 2023)

- 466 questions; 300 held out for the private leaderboard at huggingface.co/gaia-benchmark.
- Design philosophy: "conceptually simple for humans (92%) but hard for AI (GPT-4 with plugins: 15%)."
- Tests reasoning, multimodal, web, and tool use.
- Three difficulty levels; Level 3 requires long tool chains across modalities.

GAIA is what you use to measure "generalist capability." Don't conflate it with code-specific benchmarks.

### AgentBench (Liu et al., ICLR 2024)

- 8 environments spanning code (Bash, DB, KG), games (Alfworld, LTP), web (WebShop, Mind2Web), and open-ended generation.
- Multi-turn, ~4k–13k turns per split.
- Key finding: long-horizon reasoning, decision-making, and instruction following are the main bottlenecks for open-source LLMs catching up to commercial models.

### What These Don't Measure

- Real-world operational costs (tokens, wall-clock time).
- Safety behavior under adversarial conditions.
- Performance on your domain (use your own evals, Lesson 30).
- Tail failures (benchmarks average; production ops care about the worst 1%).

### Where Benchmarking Breaks Down

- **Single-number fixation.** SWE-bench 50% tells you less than the P50/P75/P95 cost + step-count distribution.
- **Contaminated claims.** Reporting SWE-bench without mentioning Verified or SWE-bench+ is misleading.
- **Benchmarks as development targets.** Optimizing for a benchmark drifts away from production usefulness.

## Build It

`code/main.py` implements a toy SWE-bench-style harness:

- Synthetic bug-fix tasks (3 of them).
- A scripted "agent" that proposes patches.
- A test runner that checks FAIL_TO_PASS (bug is now fixed) and PASS_TO_PASS (nothing broke).
- A GAIA-style difficulty classifier based on question decomposition depth.

Run it:

```
python3 code/main.py
```

Output shows per-task, per-difficulty solve rates and makes the evaluator rules concrete.

## Use It

- **SWE-bench Verified** for code agents. Always report the Verified score.
- **GAIA** for generalist agents. Use the private leaderboard split.
- **AgentBench** for multi-environment comparison.
- **Custom evals** (Lesson 30) for the real shape of your product.

## Ship It

`outputs/skill-benchmark-harness.md` builds a SWE-bench-style harness for any "codebase–task" pair with FAIL_TO_PASS / PASS_TO_PASS gates.

## Exercises

1. Port the toy harness to run on a real repository (pick one of your own). Write 3 FAIL_TO_PASS tests for known bugs.
2. Add a step-count metric. Across your 3 tasks, how many agent steps does each solve take?
3. Read the SWE-bench+ paper. Implement a solution-leakage check (pattern-match issue text against the diff).
4. Download one GAIA question from the public split. Trace what a GPT-4-class agent would do. What tools does it need?
5. Read AgentBench's per-environment breakdown. Which environment maps to your product surface? What does "SOTA" look like there?

## Key Terms

| Term | Common description | What it actually is |
|------|----------------|------------------------|
| SWE-bench | "code agent benchmark" | 2,294 GitHub issues; patch must flip FAIL_TO_PASS tests |
| SWE-bench Verified | "clean SWE-bench" | 500 human-curated tasks, OpenAI |
| FAIL_TO_PASS | "fix gate" | Tests that previously failed and must pass after the patch |
| PASS_TO_PASS | "no-regression gate" | Tests that previously passed and must still pass |
| GAIA | "generalist benchmark" | 466 questions easy for humans, hard for AI, with multi-tool use |
| AgentBench | "multi-environment benchmark" | 8 environments; long-horizon multi-turn |
| Contamination | "training-set leakage" | Benchmark tasks appearing in model training |
| SWE-bench+ | "contamination audit" | Found 32.67% solution leakage in successful SWE-bench patches |

## Further Reading

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) — the original benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — the curated subset
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) — the generalist benchmark
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) — the multi-environment suite
