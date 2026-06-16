# Evaluation & Coordination Benchmarks

> Five 2025-2026 benchmarks cover the multi-agent evaluation space. **MultiAgentBench / MARBLE** (ACL 2025, arXiv:2503.01935) evaluates star/chain/tree/graph topologies with milestone KPIs; **graph works best for research**, cognitive planning yields ~3% milestone achievement lift. **COMMA** evaluates multimodal asymmetric-information coordination; state-of-the-art models including GPT-4o struggle to beat random baselines. **MedAgentBoard** (arXiv:2505.12371) covers four medical task categories, frequently finding multi-agent no better than single LLM. **AgentArch** (arXiv:2509.10769) benchmarks enterprise agent architectures combining tool use + memory + orchestration. **SWE-bench Pro** ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) has 1865 problems across 41 repositories spanning commercial apps, B2B services, dev tools; frontier models score ~23% on Pro vs 70%+ on Verified — a reality check against contamination. Reportedly, Claude Opus 4.7 (April 2026) achieves **64.3%** on Pro with explicit agent team coordination (Anthropic has not published a first-party source — treat as preliminary); Verdent (agent scaffold) reaches **76.1% pass@1** on Verified ([Verdent technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)). **AAAI 2026 Bridge Program WMAC** (https://multiagents.org/2026/) is the 2026 community focal point. This lesson builds on MARBLE's metrics, runs a topology-vs-metrics sweep, and nails down the rule: "passing SWE-bench Verified alone is not evidence of generalization."

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 15 (Voting & Debate Topologies), Phase 16 · 23 (Failure Modes)
**Time:** ~75 minutes

## The Problem

When a paper claims "our multi-agent system is better," the question is: better than what, on what, measured how? The 2023-2024 era of multi-agent evaluation was chaos — everyone picked their own metrics, their own baselines, their own task sets. The 2025-2026 benchmarks impose structure.

Without shared benchmarks, you can't meaningfully compare two multi-agent systems. Worse, without hold-out benchmarks, frontier models get contaminated. SWE-bench Verified was partially contaminated by training corpora by mid-2025; frontier scores were inflated; Pro was designed as an uncontaminated reality check.

This lesson enumerates the five standard 2026 benchmarks, states what each measures, and teaches you to read benchmark claims with a skeptical eye.

## The Concept

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935. Evaluates four coordination topologies (star, chain, tree, graph) on research, coding, and planning tasks. Milestone-based KPIs track partial progress, not just final success.

Measured results:

- **Graph** topology works best for research scenarios; supports any-to-any critique.
- **Chain** works best for step-by-step refinement coding.
- **Star** works best for fast-answer factual aggregation.
- **Coordination tax** appears beyond ~4 agents on graph.
- **Cognitive planning** yields ~3% milestone achievement lift across topologies.

When to use: you want to compare coordination topologies apples-to-apples. The MARBLE repository (https://github.com/ulab-uiuc/MARBLE) provides evaluators.

### COMMA — Multimodal Asymmetric Information

Covers tasks where agents have different observational modalities and must coordinate without fully shared information. Reported results are uncomfortable: frontier models including GPT-4o struggle to beat **random baselines** on COMMA's agent-agent collaboration. The signal: multi-agent modality training is undertrained and under-evaluated — LLMs handle single-modality cooperation fine; multimodal coordination collapses.

When to use: your system has multimodal or asymmetric-information coordination. COMMA's null result is a warning: measure first, claim later.

### MedAgentBoard — Domain Stress Test

arXiv:2505.12371. Four medical task categories: diagnosis, treatment planning, report generation, patient communication. Compares multi-agent vs single LLM vs traditional rule-based systems.

Finding: on most categories, multi-agent is no better than single LLM. The multi-agent advantage is narrow — task decomposition helps when subtasks are cleanly separable (diagnosis + treatment); it hurts when coordination overhead exceeds specialization gains (report generation).

When to use: your domain has a clear single-LLM baseline. If MedAgentBoard's lesson generalizes, many proposed multi-agent systems are over-engineered.

### AgentArch — Enterprise Architecture

arXiv:2509.10769. Enterprise scenarios stacking tool use, memory, and orchestration. The benchmark isolates each layer's contribution: how much does adding tools help? Adding memory? Adding multi-agent orchestration?

When to use: you're designing an enterprise agent stack and need to justify each layer. AgentArch helps you avoid buying features you can't measure the value of.

### SWE-bench Pro — Reality Check

arXiv:2509.16941. 1865 problems across 41 repositories spanning commercial apps, B2B services, dev tools. Designed to be **uncontaminated** against later training cutoffs. Frontier models score ~23% on Pro vs 70%+ on Verified. That gap is the contamination signal.

April 2026 scores:
- Claude Opus 4.7 on Pro: **64.3%** (reported with explicit agent team coordination; Anthropic has not published a first-party source — treat as preliminary).
- Verdent (agent scaffold) on Verified: **76.1% pass@1** ([technical report](https://www.verdent.ai/blog/swe-bench-verified-technical-report)).
- Frontier models on Pro without agent scaffolding, raw scores: ~23-35% ([SWE-bench Pro paper](https://arxiv.org/abs/2509.16941)).

Takeaway: "We beat SWE-bench Verified" is no longer evidence of capability. Pro is the current gate. Agent team scaffolding produces measurable lift on Pro (~30-40 point delta), which is one of the strongest empirical arguments for multi-agent coordination in 2026.

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop on Multi-Agent Coordination (https://multiagents.org/2026/). The 2026 community focal point for multi-agent AI research. Accepted papers and workshop proceedings are the standard venue for evaluating new approaches; trust WMAC-accepted claims over arXiv preprints when making production decisions.

### Reading Benchmark Claims with a Skeptical Eye — 2026 Checklist

When someone claims a multi-agent result:

1. **Which benchmark, which split?** SWE-bench Verified vs Pro matters a lot. Numbers reported on the wrong split are worthless.
2. **Contamination check.** Was the benchmark published after the model's training cutoff? If not, be skeptical.
3. **Baseline comparison.** Against single-LLM baselines, against random, against prior multi-agent work. Not "against the same system untuned."
4. **Statistical significance.** N trials, p-values, confidence intervals. Frontier models have high variance; single runs mislead.
5. **Task diversity.** One task or many? Generalization matters for production.
6. **Cost disclosure.** Tokens per task, wall-clock time. A 90% solution at 20× cost is a business decision, not a capability claim.

### What All Benchmarks Measure Poorly

- **Long-horizon coordination.** Days of wall-clock interaction. All current benchmarks run short.
- **Adversarial resilience.** What happens when one agent is malicious or compromised?
- **Drift under deployment.** Benchmarks are static; production distributions shift.
- **Cost-normalized performance.** Most benchmarks report raw accuracy, not accuracy per dollar.

Building your own internal benchmark for the axis you actually care about is often the right move.

## Build It

`code/main.py` is a non-interactive walkthrough:

- Simulates 3 multi-agent systems on a toy task.
- Computes MARBLE-style milestone metrics for each.
- Runs a contamination check by withholding tasks from a "training" set.
- Explicitly compares against a random baseline.
- Prints a benchmark claims scorecard.

Run:

```bash
python3 code/main.py
```

Expected output: system scorecard with raw accuracy, milestone achievement, cost per task, delta over random baseline, and a contamination check note.

## Use It

`outputs/skill-benchmark-reader.md` reads any multi-agent benchmark claim and applies the scrutiny checklist. Output: a rating and caveats.

## Ship It

Production evaluation discipline:

- **Build an internal benchmark** reflecting your actual production distribution. Public benchmarks provide reference but don't substitute.
- **Include a random baseline in every comparison.** If you can't significantly beat random on a coordination task, the task may not be well-posed.
- **Report cost alongside accuracy.** Token cost and wall-clock time. Ops teams need both.
- **Rebuild benchmarks quarterly.** Production distributions shift; stale benchmarks mislead.
- **Avoid overfitting to public benchmarks.** If your team optimizes specifically for SWE-bench Pro numbers, you'll regress in production.

## Exercises

1. Run `code/main.py`. Identify which of the three simulated systems is most cost-efficient per milestone. Is it the same system with the highest raw accuracy?
2. Read MultiAgentBench (arXiv:2503.01935). For your own task domain, determine which of the four topologies MARBLE would recommend. Argue from the paper's results.
3. Read the SWE-bench Pro paper. What specifically makes it contamination-resistant? Could the same technique apply to other benchmarks you care about?
4. Read COMMA's findings on multimodal coordination. Design a simple multimodal coordination task you could add to an internal benchmark. What counts as a useful signal?
5. Apply the benchmark claims checklist to a recent multi-agent paper's headline result. What rating would you give the claim?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MARBLE | "MultiAgentBench" | ACL 2025; star/chain/tree/graph topologies with milestone KPIs. |
| COMMA | "Multimodal benchmark" | Multimodal asymmetric-information coordination; frontier models struggle to beat random. |
| MedAgentBoard | "Domain stress test" | Four medical categories; frequently finds multi-agent no better than single LLM. |
| AgentArch | "Enterprise benchmark" | Tool + memory + orchestration stacked. |
| SWE-bench Pro | "Contamination-resistant" | 1865 problems, 41 repos; ~23% vs 70%+ on Verified (contamination signal). |
| Milestone achievement | "Partial credit" | Benchmarks that reward progress, not just final success. |
| Contamination | "Benchmark leaked into training" | Post-publication benchmarks drifting into training corpora; inflated scores. |
| WMAC | "AAAI 2026 Bridge Program" | Workshop on Multi-Agent Coordination; community focal point. |

## Further Reading

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — Topology benchmarks with milestone KPIs
- [MARBLE repository](https://github.com/ulab-uiuc/MARBLE) — Reference implementation
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — Domain stress test; multi-agent often doesn't win
- [AgentArch](https://arxiv.org/abs/2509.10769) — Enterprise agent architectures
- [SWE-bench leaderboards](https://www.swebench.com/) — Frontier model Verified and Pro scores
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — The 2026 community focal point
