# Shadow Traffic, Canary Rollouts, and Progressive Deployment for LLMs

> LLM rollouts combine the hardest parts of software deployment: no unit tests, diffuse failure modes, delayed signals. The sequence is (1) shadow mode — copy production requests to the candidate model, log, compare, zero impact to users; catches obvious distribution issues but is not a quality guarantee; (2) canary rollout — progressive traffic shift 10% → 25% → 50% → 75% → 100% with gates at each step; track latency percentiles, per-request cost, error/refusal rate, output length distribution, user feedback rate; (3) after stability confirmation, A/B test for radically different alternatives. Non-determinism is irreducible — identical inputs yield up to 15% accuracy variance across runs due to GPU floating-point non-associativity and batch-size variance. Cost is a variable, not a constant — a model that's 20% better may cost 3x per call. Rollback speed is decisive: if rollback requires redeployment, you're too slow. Policy lives in config/flags; models live in a registry with pinned digests; rollback = flip policy + revert threshold + pin back to old model in seconds.

**Type:** Learn
**Languages:** Python (stdlib, a toy canary progression simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Time:** ~60 min

## Learning Objectives

- Distinguish shadow mode (zero-impact comparison), canary (live traffic progression), and A/B (post-stability comparison).
- List five LLM-specific canary metrics (latency, per-request cost, error/refusal, output length distribution, user feedback).
- Explain why LLM non-determinism (up to 15%) changes what "stable" means during rollout.
- Design a rollback path that takes seconds (flip policy) rather than hours (redeploy).

## The Problem

You ship a new model. Offline evals show a 3% accuracy gain. You turn it on in production. Within 24 hours, cost is up 40%, user thumbs-down is up 8%, and three customer tickets report "weird answers." You roll back. Redeployment takes 3 hours. Your weekend is ruined.

Every link in this chain was avoidable. Shadow mode would have caught that 40% cost spike before any user saw it. Canary would have stopped at 10% when thumbs-down moved. Policy-flag rollback would have taken 30 seconds. This discipline is what fills the gap between "offline eval looks good" and "real users are happy."

## The Concept

### Shadow Mode

The candidate model receives the same requests as production; outputs are logged, not returned to users. Zero user impact. Log:

- Output content (diff against production).
- Token counts (cost delta).
- Latency.
- Refusals and errors.

Catches: cost explosions, length regressions, obvious refusal changes, hard errors. Doesn't catch: user-perceptible quality deltas. Shadow is a smoke test, not a quality test.

### Canary Rollout

Progressive traffic shift with gates. Typical progression: 1% → 10% → 25% → 50% → 75% → 100%. Each step gates on 5 metrics:

1. **Latency percentiles** — P50, P95, P99. Breach: canary P99 > 1.5x baseline.
2. **Per-request cost** — blended $. Breach: >20% above baseline.
3. **Error / refusal rate** — 5xx plus explicit refusals. Breach: 2x baseline.
4. **Output length distribution** — mean + P99. Breach: distribution shift.
5. **User feedback rate** — thumbs-down / ticket submissions. Breach: 1.5x baseline.

### Non-determinism Is the New Variance

Identical inputs produce non-identical outputs. Causes:

- GPU floating-point non-associativity (reduction order changes with batch).
- Batch-size variance (same prompt in a batch of 128 vs 16).
- Sampling (temperature > 0).

Measured: up to 15% accuracy variance across runs on the same eval set. "Stable" during rollout means metrics are within expected variance, not identical to baseline. Set gates above the noise floor.

### Cost Is a Variable

A model that's 20% better may cost 3x per call. Per-request cost is one of the five gates. Shipping a "better" model that breaks unit economics is a rollback reason.

### Rollback Is the Weapon

- Policy flag (feature flag system): flip percentage in config; seconds.
- Model pin (registry digest): pinned models don't auto-upgrade.
- Rollback = revert flag + set pinned digest to previous. Seconds, not hours.

If your stack requires redeployment to roll back, fix that before you roll out.

### Tools

**Argo Rollouts** / **Flagger** — Kubernetes progressive delivery controllers. Integrate with Istio/Linkerd weighted routing.

**Istio weighted routing** — traffic splitting at the service mesh layer.

**KServe / Seldon Core** — model serving with built-in canary.

**Feature flags** — LaunchDarkly, Flagsmith, Unleash. Policy-level flips without redeployment.

### Metric Cadence

Canary gates check every 5-15 minutes depending on traffic volume. 1% traffic at 10 requests/minute gives 50-150 data points per window — enough for latency, noisy for user feedback. 10% gives ~10x more. Progression should pause at each step long enough to accumulate sufficient samples.

### The A/B Step Is Optional

If the new model is radically different (different behavior, different cost curve, different tone), A/B test it at 50% after canary passes. If it's just an improved version, canary gates passing is sufficient to go to 100%.

### Numbers You Should Remember

- Canary progression: 1% → 10% → 25% → 50% → 75% → 100%.
- Non-determinism ceiling: up to 15% variance on identical inputs across runs.
- Five canary metrics: latency, cost, error/refusal, output length, user feedback.
- Cost gate: >20% above baseline is a breach.
- Rollback: seconds, not hours.

## Use It

`code/main.py` simulates a canary rollout with injected regressions. Reports at which stage the rollout halts and which gate triggered.

## Ship It

This lesson produces `outputs/skill-rollout-runbook.md`. Given a candidate model, baseline, and risk tolerance, design a shadow → canary → 100% plan.

## Exercises

1. Run `code/main.py`. Inject a 25% cost regression. At which stage does the canary halt?
2. Your new model has 3% offline accuracy gain but +18% per-request cost. Should it ship? Depends on strategy — write out both paths.
3. Design a rollback that completes in under 60 seconds end-to-end. List the infrastructure required.
4. Non-determinism shows ±7% on your eval. Set canary gates that won't false-alarm. What multiplier do you use?
5. Shadow mode catches a 40% cost spike before canary. Write the alert rule that fires in shadow.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Shadow mode | "mirror to new model" | Send to candidate at zero impact for logging |
| Canary | "progressive traffic" | Gradual user-facing rollout with gates |
| Gate | "rollout check" | Metric threshold that blocks progression |
| Non-determinism | "LLM variance" | Irreducible run-to-run differences |
| Policy flag | "flip-flag rollback" | Config-level rollback, seconds not hours |
| Model pin | "registry digest" | Immutable reference to a specific model version |
| Argo Rollouts | "K8s progressive" | Kubernetes-native canary/rollback controller |
| KServe | "inference K8s" | Model serving with canary primitives |
| Istio weighted | "mesh split" | Service mesh traffic splitter |

## Further Reading

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
