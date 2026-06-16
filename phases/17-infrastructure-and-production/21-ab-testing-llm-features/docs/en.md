# A/B Testing LLM Features — GrowthBook, Statsig, and the "Ship by Gut" Problem

> Traditional A/B testing wasn't built for non-deterministic LLMs. The key distinction: evals answer "can the model do this job?" A/B tests answer "do users care?" You need both; the era of shipping by gut feeling is over. What to test in 2026: prompt engineering (wording), model selection (GPT-4 vs GPT-3.5 vs OSS; accuracy vs cost vs latency), generation parameters (temperature, top-p). Real-world cases: a chatbot reward-model variant delivered +70% conversation length and +30% retention; Nextdoor's AI subject-line experiment yielded +1% CTR after reward-function tuning; Khan Academy's Khanmigo iterated on the latency-vs-math-accuracy axis. Platform breakdown: **Statsig** (acquired by OpenAI for $1.1B in September 2025) — sequential testing, CUPED, all-in-one. **GrowthBook** — open-source, warehouse-native, Bayesian + frequentist + sequential engines, CUPED, SRM checks, Benjamini-Hochberg + Bonferroni corrections. You choose based on your warehouse-SQL preference and whether "owned by OpenAI" matters to your organization.

**Type:** Learn
**Languages:** Python (standard library, a toy sequential-test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Delivery)
**Time:** ~60 minutes

## Learning Objectives

- Distinguish evals ("can the model do this job") from A/B tests ("do users care").
- List three testable axes (prompt, model, parameters) and pick metrics for each.
- Explain CUPED, sequential testing, and Benjamini-Hochberg multiple-comparison correction.
- Choose Statsig vs GrowthBook based on warehouse-SQL attitude and enterprise-acquisition stance.

## The Problem

You hand-tuned a system prompt. It felt better. You shipped it. Conversion moved within noise. You blamed the metrics. Or you shipped a new model, conversion didn't move — did the model regress, or was the change too small to detect? You don't know, because you shipped without an A/B test.

Evals answer whether a model can complete a task on a labeled set. They don't answer whether users prefer the output. Only controlled online experiments answer that — and only when the experiment has sufficient power, controls for non-determinism, and corrects for multiple comparisons.

## The Concept

### Eval vs A/B Test

**Eval** — offline, labeled set, judge (rubric or LLM-as-judge or human). Answers: "Is the output correct / helpful / safe on this fixed distribution?"

**A/B test** — online, live users, randomized. Answers: "Did the new variant move that user-level metric that matters?"

You need both. Evals catch regressions before exposure; A/B confirms product impact afterward.

### What to Test

1. **Prompt engineering** — wording, system-prompt structure, examples. Metrics: task success rate, user retention, cost per request.
2. **Model selection** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS. Metrics: accuracy (task) + cost per request + latency P99. Multi-objective.
3. **Generation parameters** — temperature, top-p, max_tokens. Metrics: task-specific (output diversity vs determinism).

### CUPED — Variance Reduction

Controlled-experiments Using Pre-Experiment Data. Regresses away pre-experiment variance before comparing post-experiment data. Typical variance reduction: 30-70%. Effective sample size goes up for free.

Implementation: both Statsig and GrowthBook implement it.

### Sequential Testing

Classic A/B assumes fixed sample size. Sequential testing ("peek and decide") controls false-positive rate under repeated looks. Always-valid sequential procedures (mSPRT, Howard's confidence sequences) let you stop early when there's a clear winner.

### Multiple-Comparison Correction

Run 20 A/B tests at 95% confidence and you'll get one false positive by chance. Bonferroni tightens per-test alpha; Benjamini-Hochberg controls the false discovery rate. GrowthBook implements both.

### SRM — Sample Ratio Mismatch

Assignment hashing randomizes users to variants. If a 50/50 split yields 47/53, something is broken — an SRM check flags it. Both platforms implement this.

### Statsig vs GrowthBook

**Statsig**:
- Acquired by OpenAI for $1.1B (September 2025). Hosted, SaaS.
- Sequential testing, CUPED, holdout groups.
- All-in-one: feature flags + experiments + observability.
- Best for: teams that already want a bundled product and don't mind OpenAI ownership.

**GrowthBook**:
- Open-source (MIT); warehouse-native (reads directly from Snowflake/BigQuery/Redshift).
- Multi-engine: Bayesian, frequentist, sequential.
- CUPED, SRM, Bonferroni, BH correction.
- Self-hosted or managed cloud.
- Best for: warehouse-SQL teams, data teams that own the metric layer, want OSS.

### Non-Determinism Complicates Power Calculations

The same prompt produces different outputs. Traditional power calculations assume IID observations. With LLM non-determinism, effective sample size is lower than nominal. Multiply required sample size by ~1.3-1.5x as a safety buffer.

### Real-World Results

- Chatbot reward-model variant: +70% conversation length, +30% retention.
- Nextdoor subject lines: +1% CTR after reward-function tuning.
- Khan Academy Khanmigo: iterative tradeoff on latency vs math accuracy.

### Anti-Pattern: Shipping by Gut

Every senior engineer can name a feature that shipped because it "felt better" without an A/B test. Most of them caused product metric regressions the team didn't notice for months. A/B is the forcing function.

### Numbers You Should Remember

- Statsig acquired by OpenAI: $1.1B, September 2025.
- GrowthBook: open-source MIT; Bayesian + frequentist + sequential.
- CUPED variance reduction: 30-70%.
- LLM non-determinism → +30-50% sample-size buffer.

## Use It

`code/main.py` simulates a sequential A/B test with fixed and sequential boundaries. Demonstrates how sequential testing lets you stop early.

## Ship It

This lesson produces `outputs/skill-ab-plan.md`. Given a feature change, workload, and baseline, it picks a platform, gates, and sample size.

## Exercises

1. Run `code/main.py`. For a baseline 3% conversion rate with an expected 5% lift, how many samples are needed to reach 80% power?
2. Choose Statsig or GrowthBook for a healthcare-regulated, on-premises customer.
3. Design an A/B test comparing GPT-4 vs GPT-3.5 on "cost per resolved ticket." What's the primary metric, guardrail metric, and secondary metric?
4. Your canary passed, but the A/B shows -1.2% conversion. Do you ship? Write the escalation criteria.
5. Apply CUPED to an experiment with a pre-period whose variance is 60% of post. Calculate the effective sample-size boost.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| Eval | "Offline test" | Labeled-set assessment of model capability |
| A/B test | "Experiment" | Live randomized comparison on users |
| CUPED | "Variance reduction" | Pre-period regression to reduce variance |
| Sequential testing | "Peekable test" | Always-valid procedure allowing early stopping |
| Multiple comparisons | "Family-wise error" | Running many tests inflates false positives |
| Bonferroni | "Tight correction" | Divide alpha by number of tests |
| Benjamini-Hochberg | "BH FDR" | False discovery rate control, less conservative |
| SRM | "Bad split" | Sample ratio mismatch; assignment bug |
| Statsig | "OpenAI-owned" | Commercial all-in-one, acquired 2025 |
| GrowthBook | "The OSS one" | MIT warehouse-native platform |
| mSPRT | "Sequential probability ratio test" | Classic sequential procedure |

## Further Reading

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
