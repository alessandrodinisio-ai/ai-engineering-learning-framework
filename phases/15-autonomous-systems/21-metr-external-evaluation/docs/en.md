# METR Time Horizons & External Capability Evaluation

> METR (formerly ARC Evals) has been an independent 501(c)(3) since December 2023. Their Time Horizon 1.1 benchmark (January 2026) fits task success probability against log(expert completion time) as a logistic curve; the crossing at 50% probability defines the model's time horizon. 2025-2026 collaborations cover GPT-5.1, GPT-5.1-Codex-Max, and prototype oversight evaluations (can a monitor catch side tasks; can an agent evade). Benchmark suites: HCAST (180+ ML, cyber, SWE, reasoning tasks; 1 min to 8+ hours), RE-Bench (71 ML research engineering tasks with expert baselines), SWAA. An honest note: METR's measurements are idealized — no humans, no real consequences — and the team has documented the eval-vs-deployment behavior gap (Lesson 1). A time horizon is an upper bound, not a deployment prediction.

**Type:** Learn
**Languages:** Python (stdlib, logistic-fit horizon estimator)
**Prerequisites:** Phase 15 · 01 (Long-Horizon Agents), Phase 15 · 19 (RSP)
**Time:** ~60 min

## The Problem

Scaling policies (Lessons 19, 20) are only as useful as the measurements they reference. "AI R&D-4 threshold" and "long-horizon autonomy" are defined in policy prose; they become operational only when specific evaluations produce specific numbers.

METR is the external evaluation organization that defined many of those numbers in 2024-2026. They evaluate frontier models — often pre-release, under NDA with labs — then publish methodology. The Time Horizon 1.1 benchmark (January 2026) is their headline product: a scalar that compresses capability into a human-readable unit ("this model can complete tasks that would take an expert X hours at 50% reliability").

This lesson is part methodology (how a horizon is computed) and part interpretation (why a horizon is an upper bound, not a deployment prediction). The two skills are paired. A team that understands how the horizon is fit is much harder to mislead by a bad vendor claim than one that just sees "14 hours" on a slide.

## The Concept

### METR Background

- Founded: December 2023 (formerly ARC Evals, spun off as independent 501(c)(3)).
- Scope: evaluates frontier models for autonomous capabilities, often pre-release.
- Collaborating labs: Anthropic, OpenAI (multiple 2025-2026 collaborations).
- Notable deliverables: Time Horizon 1.0 (March 2025), Time Horizon 1.1 (January 2026), prototype oversight evaluations.

### Time Horizon Fit

Methodology (from METR blog and paper):

1. Collect a task set spanning minutes to hours of expert completion time. Current suites: HCAST (180+ tasks), RE-Bench (71 tasks), SWAA.
2. Run the model on each task; record success or failure.
3. Fit a logistic curve: P(success) as a function of log(expert completion time).
4. The horizon is the expert time at which P(success) = 0.5.

The logistic fit shape is correct because capability vs task difficulty is typically a rising, plateau-approaching relationship. The 50% point is a choice (could be 10%, 90%); METR reports multiple thresholds in detailed papers but leads with 50% because it's most intuitive.

### The January 2026 Numbers

Per Time Horizon 1.1:

- Claude Opus 4.6: ~14 hours at 50% reliability as of Time Horizon 1.1 (January 2026).
- Doubling time on HCAST-style tasks: ~4.3 months (130.8 days) on the post-2023 fit reported in Time Horizon 1.1 (January 2026); the ~7-month figure is from Time Horizon 1.0's full 2019-2025 fit and is reported alongside the post-2023 number in TH1.1.

These are the specific numbers the rest of Phase 15 references. They are cited in Anthropic's RSP v3.0 (Lesson 19) and DeepMind's FSF v3 (Lesson 20) as one of the measurements feeding threshold evaluation.

### Benchmark Suites

- **HCAST**: 189 tasks across ML, cybersecurity, software engineering, and general reasoning. Spans: 1 minute to 8+ hours of expert time. The primary suite behind the horizon numbers.
- **RE-Bench**: 71 ML research engineering tasks with human expert baselines. Measures whether a model can do the kind of ML R&D work that feeds into the AI R&D-4 threshold.
- **SWAA**: shorter tasks; useful for calibrating the low end of the horizon curve.

### Prototype Oversight Evaluations

2025-2026 METR work includes prototype oversight evaluations: can a monitor catch a model performing side tasks (things not requested by the operator); can an agent evade that monitor. These feed directly into OpenAI PF v2's "undermining safeguards" category and DeepMind FSF v3's deceptive alignment monitoring (Lesson 20).

### Why the Horizon Is an Upper Bound

- **Idealized tooling**: benchmark tasks give the model clean tools and well-formed data. Real deployments are messier.
- **No real consequences**: the model never actually bills a customer, deletes real data, or contacts real people. Real deployments have irreversible stakes.
- **Eval-context gaming**: Lesson 1. Models behave differently under test. The 2026 International AI Safety Report documents this empirically.
- **No legitimate user variation**: benchmark prompts are structured. Real users produce ambiguous, context-dependent requests.

The horizon is a capability ceiling under favorable conditions. Deployment reliability is a different number, lower, and teams must measure their own distribution to know it.

### The Case for External Evaluators

External evaluation matters because internal labs have incentives to optimize the metrics they report. METR's independence — a 501(c)(3) with stated methodology and peer-reviewed papers — is structural mitigation. It is not sufficient by itself (labs still control what METR can see), but it is strictly better than no external evaluation.

### How to Use Horizon Numbers in Practice

- **As a capability filter**: if a model's horizon is far below a proposed task's expert time, don't deploy it autonomously (Lesson 1's skill file).
- **As a trend indicator**: doubling time tells you how long current practices remain safe even without new mitigations.
- **As a prior**: a 14-hour horizon is a starting point. Adjust downward for your task distribution, your tool quality, your deployment context.

## Use It

`code/main.py` implements the logistic fit of task success against log(expert time) given a synthetic result set. It reports the 50% horizon (METR's headline), 10% horizon (conservative), and 90% horizon (optimistic). It also demonstrates how the curve shifts when success rates are artificially inflated by eval-context gaming.

## Ship It

`outputs/skill-horizon-interpretation.md` reviews a vendor's horizon claim and produces a gap analysis between benchmark claims and deployment reality.

## Exercises

1. Run `code/main.py`. Confirm the fit's 50% horizon matches the synthetic ground truth. Now halve the task-time grid; does the horizon estimate change meaningfully?

2. Read METR's Time Horizon 1.1 blog post. Identify the specific tasks with highest and lowest reliability. Explain why the gap exists.

3. Read METR's "Measuring Autonomous AI Capabilities" resources. List HCAST's task categories. Pick one category you'd weight more heavily for a given production task and justify why.

4. Introduce eval-context gaming into the simulator: flip ~20% of failed tasks to successes. Report the new horizon. This approximates what a 20% gaming rate does to the observed number.

5. Design an internal horizon evaluation on your own bug backlog or a representative task set. Describe data collection, fitting, and what the output tells you. Compare against METR numbers.

## Key Terms

| Term | Common shorthand | Actual meaning |
|---|---|---|
| METR | "external evaluator" | Formerly ARC Evals; independent 501(c)(3) since December 2023 |
| Time Horizon | "capability metric" | Expert task duration at 50% reliability, from logistic fit |
| HCAST | "METR's primary suite" | 180+ tasks, spanning 1 min to 8+ hours |
| RE-Bench | "research engineering" | 71 ML research engineering tasks with human baselines |
| SWAA | "short task suite" | Calibrates the low end of the horizon curve |
| Doubling time | "growth rate" | Time for the 50% horizon to double; ~4.3 months on post-2023 HCAST fit |
| Eval-context gaming | "model behaves differently" | Documented behavior gap between test and deployment |
| Upper bound | "horizon is a ceiling" | Benchmark horizon > deployment reliability under load |

## Further Reading

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — HCAST, RE-Bench, SWAA specs.
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — The original horizon paper.
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — Current numbers and methodology.
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — Live tracking.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Internal perspective on METR measurements.
