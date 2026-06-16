# EAGLE-3 Speculative Decoding in Production

> Speculative decoding pairs a fast draft model with the target model. The draft proposes K tokens; the target verifies them in a single forward pass; accepted tokens are effectively free. In 2026, EAGLE-3 is the production-grade variant — it trains the draft head on the target model's hidden states rather than raw tokens, pushing acceptance rate alpha into the 0.6-0.8 range on general chat. The right question is not "how fast is the draft" but "what is alpha on my traffic." If alpha drops below approximately 0.55, speculative decoding is a net negative at high concurrency — because every rejected draft costs a second target forward. This lesson teaches you to measure alpha before flipping the switch.

**Type:** Learn
**Languages:** Python (standard library, a toy-level acceptance rate simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## Learning Objectives

- Name the three generations of speculative decoding and explain what EAGLE-3 changed versus EAGLE-2 and versus classic draft models.
- Define acceptance rate alpha, compute expected speedup from alpha and K (draft length), and find the break-even alpha at your target concurrency.
- Explain why speculative decoding is opt-in (not default) in 2026 vLLM, and why enabling it without measuring alpha is a production anti-pattern.
- Write a measurement plan: which benchmark, which prompt distribution, which concurrency point, which metric as the gate.

## The Problem

Decode is memory-bound. An H100 running Llama 3.3 70B FP8 reads approximately 140 GB/s of weights to produce one token per decode step. GPU compute is nearly idle during decode — the bottleneck is HBM bandwidth, not matmul throughput.

Speculative decoding exploits this slack. A cheap draft model generates K candidate tokens, then the target model verifies all K in a single forward pass. Each accepted token is effectively free (amortized into one batch-of-K forward pass that the target would have done anyway).

The classic draft-model approach uses a smaller model from the same family (Llama 3.2 1B drafting for Llama 3.3 70B). It works, but acceptance rates are mediocre — the small model's distribution diverges from the target. EAGLE, then EAGLE-2, then EAGLE-3, trains a lightweight draft head directly on the target model's internal states, so the draft distribution tracks the target much more closely. This is why alpha rises from 0.4 with draft models to 0.6-0.8 with EAGLE-3.

The trap: EAGLE-3 is opt-in in 2026 vLLM. `speculative_config` must be set explicitly. No flag, no speedup. Teams that enable it without measuring alpha on their actual traffic often see tail latency degrade rather than improve.

## The Concept

### What Speculative Decoding Actually Buys

Without spec decode, the per-token cost is one target forward. With spec decode, draft length K, and acceptance rate alpha, the expected tokens per target forward is `1 + K * alpha`. Speedup is `(1 + K * alpha) / (1 + epsilon)`, where epsilon is the overhead of drafting plus verification. At K=5, alpha=0.7: `(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`. Real numbers cluster around 2-3x because alpha is rarely that high on production traffic and epsilon grows at large batch sizes.

### Why Alpha Is the Only Metric That Matters

Rejected tokens don't vanish — they force a second target forward for the first rejected token. On a workload where alpha drops to 0.4, you pay draft overhead plus verification plus re-roll. At high concurrency (say 256), the decode batch is already large enough that the memory bandwidth gap between "target alone" and "target with verification" narrows. Below alpha 0.55, spec decode is a net negative on most 2026 hardware.

Alpha varies by workload. On ShareGPT-style general chat with an EAGLE-3 trained on ShareGPT, it hits 0.6-0.8. On domain-specific traffic (code, medical, legal), a generic-trained draft head drops to 0.4-0.6. Training a domain-specific draft head recovers alpha — a lightweight, fast training task compared to target fine-tuning.

### EAGLE Generations at a Glance

- **Classic draft model**: smaller model from the same family. Alpha 0.3-0.5. Simple infrastructure — load two models, draft runs K forwards per target forward.
- **EAGLE-1 (2024)**: single draft head trained on target hidden states (last layer). Alpha ~0.5-0.6. Small parameter overhead on top of target.
- **EAGLE-2 (2025)**: adaptive draft length and tree-based drafting (verifying multiple branches in one target pass). Alpha ~0.6-0.7. More complex draft scheduler.
- **EAGLE-3 (2025-2026)**: draft head trained on multiple target layers (not just the last). Better alignment. Alpha ~0.6-0.8 on general chat.

### The 2026 Production Recipe

1. Ship the target model naively first. Measure baseline TTFT, ITL, throughput at target concurrency.
2. Enable EAGLE-3 draft via vLLM's `speculative_config`. Re-run benchmarks.
3. Record acceptance rate alpha. vLLM V1 reports it as `spec_decode_metrics.accepted_tokens_per_request`. Divide by the request's draft length to get alpha.
4. If alpha < 0.55 on your production traffic distribution, disable spec decode or train a domain-specific EAGLE-3 draft.
5. Re-run at production concurrency. Confirm P99 ITL has not degraded.

### Production Pitfall: P99 Tail

Mean ITL drops with spec decode. P99 may degrade if you don't tune. Rejected drafts trigger a two-pass sequence (draft + verification failure + re-roll). At full batch, these two passes serialize. Watch P99 ITL, not P50.

### Where EAGLE-3 Is Already Deployed

Google deployed speculative decoding into AI Overviews in 2025 (same quality, faster responses). vLLM V1 ships `speculative_config` as a documented interface; N-gram GPU speculative decoding in V1 is the chunked-prefill-compatible variant. SGLang supports EAGLE-3 and recommends it as the draft path for prefix-heavy workloads.

### The One-Line Break-Even Math

Expected speedup: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. Set `S = 1` and solve for alpha: `alpha_breakeven = verify_overhead / K`. For typical verify_overhead ~0.15 and K=5: `alpha_breakeven = 0.03`. But that's bare-decode math. At high concurrency, verification overhead rises and the decode batch already amortizes memory reads across sequences, so the effective alpha_breakeven climbs to ~0.45-0.55 in practice.

### When Not to Use Speculative Decoding

- Batch-1 offline generation where latency doesn't matter. Use naive target.
- Very short outputs (under 50 tokens). Draft overhead and verification cost dominate.
- Specialized domains without a domain-trained draft head. Alpha is too low.
- vLLM v0.18.0 with draft-model spec decode plus `--enable-chunked-prefill`. This combination won't compile. The documented exception is N-gram GPU spec decode in V1.

## Use It

`code/main.py` simulates a decode loop with and without speculative decoding across a range of alpha values and draft lengths K. It prints break-even alpha, measured speedup, and tail behavior. Run it on several (alpha, K) combinations to see exactly where speculative decoding stops paying off.

## Ship It

This lesson produces `outputs/skill-eagle3-rollout.md`. Given a target model, a traffic distribution description, and a concurrency target, it produces a phased EAGLE-3 rollout plan — baseline benchmarks, enablement config, measure alpha, gate on alpha >= 0.55, watch P99 ITL.

## Exercises

1. Run `code/main.py`. At K=5, what alpha do you need for 2x speedup? For 3x? How sensitive is this to verify_overhead?
2. Imagine production traffic that is 70% general chat and 30% code. General chat hits alpha 0.7 with a ShareGPT-trained EAGLE-3; code hits alpha 0.4. What is the blended alpha, and is spec decode a net positive?
3. Read vLLM's `speculative_config` documentation. Name the three modes (draft model, EAGLE, N-gram) and which one is compatible with chunked prefill.
4. You observe that enabling EAGLE-3 drops mean ITL by 25% but raises P99 ITL by 15%. Diagnose and propose a mitigation.
5. Calculate the memory overhead of an EAGLE-3 draft head for Llama 3.3 70B. How does it compare to running Llama 3.2 1B as a classic draft?

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| Speculative decoding | "Draft and verify" | Use a cheap model to propose K tokens, one target forward verifies all K |
| Acceptance rate alpha | "Spec acceptance" | Fraction of draft tokens accepted by the target; the only metric that matters |
| Draft length K | "Spec k" | How many tokens the draft proposes per target forward; typically 4-8 |
| Verification overhead epsilon | "Spec overhead" | Extra cost of verify-and-reroll vs naive target forward; grows with batch size |
| EAGLE-3 | "Latest EAGLE" | 2025-2026 variant; trains draft head on multiple target layers; alpha 0.6-0.8 on general chat |
| `speculative_config` | "vLLM spec config" | Explicit opt-in in vLLM V1; no default means no speedup |
| N-gram spec decode | "N-gram draft" | GPU-side drafting using N-gram lookup from the prompt; compatible with chunked prefill |
| Break-even alpha | "Dead alpha" | The alpha at which spec decode speedup is zero; watch it at production concurrency |
| Rejected-draft double-pass | "Re-roll cost" | Two target forwards when draft is rejected; inflates P99 tail |

## Further Reading

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — Authoritative source on `speculative_config` and chunked prefill compatibility in V1.
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — Exact field set.
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — Original EAGLE draft head formulation.
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — Adaptive drafting and trees.
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — Efficient LLM systems with speculative decoding.
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — Production rollout checklist.
