# Speculative Decoding and EAGLE-3

> Phase 7 · Lesson 16 proved the math: the Leviathan rejection rule exactly preserves the verifier's distribution. This lesson is the training-stack perspective on production speculative decoding in 2026. EAGLE-3 turns the draft model from a cheap approximation into a purpose-built small network trained on the verifier's own hidden states, then adds a training-time test loop to align its training and inference distributions. Result: end-to-end 3x to 6.5x speedups, per-token acceptance rate above 0.9 on chat, no distribution tradeoff. Every production inference stack in 2026 ships it by default.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (Speculative Decoding math), Phase 10 · 12 (Inference Optimization)
**Time:** ~75 minutes

## Learning Objectives

- State the Leviathan theorem in one sentence and prove the speculative loop produces samples identically distributed to the verifier.
- Walk the two-year evolution from vanilla speculative decoding (Leviathan 2023) through EAGLE, EAGLE-2, to EAGLE-3, pointing out the exact limitation removed at each step.
- Compute expected speedup from acceptance rate `α` and draft-to-verifier cost ratio `c`, and choose optimal draft length `N` for each regime.
- Implement the full speculative loop from scratch: drafting, verification, residual rejection sampling, KV cache rollback on rejection, bonus token emission on full acceptance.

## The Problem

Autoregressive decode on a 70B model runs perhaps 35 tokens/second on an H100. The GPU is far from saturated. Memory bandwidth is the ceiling: load 70B weights from HBM per token, do one step of arithmetic, produce one float. Compute units are idle most of the time.

Speculative decoding turns this into a throughput problem you can actually solve. A cheap draft proposes `N` tokens in `N` small forward passes. The verifier runs once on the prefix plus all `N` drafts. If the verifier's distribution at position `i` agrees with the draft's (in a statistical sense we'll make precise), we accept; if not, we reject and sample a correction from the residual distribution. One large-model forward produces up to `N+1` accepted tokens instead of one.

The theorem that matters is Leviathan, Kalman, Matias (ICML 2023): the output distribution is identical to sampling from the verifier directly. Not approximate. Identical. That's the entire reason speculative decoding is acceptable in production — it's a pure latency optimization with no quality tradeoff.

Phase 7 · Lesson 16 gave you the math. This lesson gives you the training stack. A good draft is worth 2x more speedup than a cheap draft. EAGLE, EAGLE-2, and EAGLE-3 (Li et al., 2024–2025) turned "draft = a smaller version of the same model" into a precise engineering discipline. Production inference servers in 2026 default to EAGLE-3.

## The Concept

### The invariant: Leviathan rejection sampling

Let `p(t)` be the draft's distribution over the next token given some prefix, `q(t)` the verifier's. Sample a draft token `d ~ p`. Accept with probability `min(1, q(d) / p(d))`. On rejection, sample from the residual distribution `(q - p)_+ / ||(q - p)_+||_1`. The resulting sample is distributed according to `q`. This holds regardless of how bad `p` is — the worse it is, the more often you reject, but the output stays exact.

Stack `N` such calls back-to-back using a single verifier forward on `prefix + d_1 + ... + d_N`. The verifier returns `q_1, q_2, ..., q_{N+1}` simultaneously. Walk left to right. On the first rejection at position `j`, sample from `residual(q_j, p_j)` and stop. On full acceptance, sample a bonus token from `q_{N+1}`.

### What determines the speedup

Let `α` be the expected acceptance rate per drafted token. Let `c = cost(draft) / cost(verifier)` be the cost ratio. Expected tokens accepted per verifier forward:

```
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

Expected total wall-clock per accepted token: `(N * c + 1) / E[accepted]`. Minimize over `N` to find the sweet spot. For `α = 0.8, c = 0.05`: optimal `N` is about 5-7, speedup 3.2x. For `α = 0.95, c = 0.02`: optimal `N` is about 8-10, speedup approaches 5x.

The single biggest lever is `α`. Going from `α = 0.6` (vanilla draft) to `α = 0.9` (EAGLE-3) at fixed `N = 5` takes you from 2.2 expected accepted tokens per verifier forward to 4.1. Nearly 2x throughput from the same verifier.

### The two-year evolution

**Vanilla speculative (Leviathan, 2023).** Draft model is a smaller LLM from the same family, independently trained. Easy to plug in, `α ≈ 0.6`, speedup capped at ~2x.

**EAGLE-1 (Li et al., 2024).** The draft is a small transformer — usually one or two layers — that takes the verifier's last-layer hidden state as input and directly predicts the next token. Because the draft sees the verifier's feature representation, its distribution is closer to the verifier's. `α` climbs to 0.7-0.8.

**EAGLE-2 (Li et al., 2024).** Adds a dynamic draft tree: instead of proposing a single sequence of `N` tokens, it proposes a small tree of candidates, scoring each in one verifier forward (tree attention), and walking the highest-probability path. Draft length becomes adaptive per step. `α` per token on the accepted path climbs past 0.85.

**EAGLE-3 (Li et al., 2025, NeurIPS).** Two more changes. First, drops the feature-prediction loss entirely — EAGLE-1/2 trained the draft to match the verifier's hidden states, which caps how much the data can help. EAGLE-3 trains directly on token prediction. Second, training-time test (TTT): during draft training, the draft's own previous predictions are fed back as input across multiple steps, mimicking how it operates at inference. This aligns training and test distributions, stopping error accumulation. Measured speedup: up to 6.5x on chat, 38% throughput gain in SGLang batch-64 on H100.

### KV cache rollback

Verification extends the verifier's KV cache by `N` entries in one shot. If rejection occurs at position `j`, cache contents beyond position `j-1` are now wrong. Two common implementations: write to a scratch buffer and commit on acceptance (vLLM, TensorRT-LLM), or maintain a physical KV cache with a logical length that truncates on rejection. Either way, rollback cost is a few bytes per layer per head, negligible vs the forward pass cost.

For EAGLE-2 tree search, the verifier runs attention with a non-causal mask that respects the tree topology. Engineering is tricky, but the compute is one standard flash-attention call with a custom mask.

### 2026 draft architectures

| Strategy | Draft Type | `α` | Speedup | Training Cost |
|----------|-----------|-----|---------|---------------|
| Vanilla | Separate small LLM | 0.55-0.70 | 1.8-2.3x | None (reuse existing small model) |
| Medusa | Extra LM heads on verifier | 0.65-0.75 | 2-3x | ~1B SFT tokens |
| EAGLE-1 | 1-layer transformer on hidden states | 0.70-0.80 | 2.5-3x | ~60B tokens |
| EAGLE-2 | EAGLE-1 + dynamic draft tree | 0.80-0.88 | 3-4x | ~60B tokens |
| EAGLE-3 | Multi-layer feature fusion + TTT | 0.88-0.92 | 3.5-6.5x | ~60-200B tokens |
| Lookahead | No draft (Jacobi iteration) | N/A | 1.3-1.6x | None |

In 2026 production: vLLM and SGLang default to EAGLE-3 when available, EAGLE-2 otherwise. TensorRT-LLM has the fastest Medusa path for Meta and NVIDIA published models. llama.cpp ships vanilla drafting for CPU deployment.

## Build It

See `code/main.py`. This is the complete Leviathan speculative loop with all parts: draft N, verifier parallel pass, per-position rejection, residual sampling, bonus token, KV rollback, and empirical verification that output distribution matches sampling from `q` directly.

### Step 1: Rejection rule

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### Step 2: Residual distribution

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### Step 3: A complete speculative step

The `spec_step` function drafts `N` tokens from `p`, then verifies all of them in a single parallel `q` evaluation. It applies the rejection rule for each drafted token, sampling a correction from the residual on the first rejection. If all are accepted, it emits a bonus token from `q_{N+1}`.

### Step 4: KV rollback bookkeeping

The simulator tracks a logical `kv_length` per worker. Accepting `k` drafts means `kv_length += k`. A rejection at position `j` means the cache was already written past `j`, but the logical length is set to `prefix_length + j + 1` — one past the correction token. Subsequent reads truncate to logical length.

### Step 5: Leviathan check

Run 50,000 speculative steps. Tally the empirical distribution of accepted tokens. Compare to 50,000 direct samples from `q`. Chi-squared statistic should be well below the critical value. The theorem holds in practice.

### Step 6: Speedup vs. α

Sweep draft quality by perturbing `p` away from `q` at varying magnitudes. Measure `α`, then plot expected tokens per verifier call as a function of `α` and `N`. The code prints a table showing how EAGLE-3-level draft quality (`α ≈ 0.9`) unlocks 4-5 tokens per verifier call.

## Use It

Production `vllm serve` with EAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

SGLang at batch 64 on H100 with EAGLE-3: approximately 1.38x throughput over batch-64 vanilla decode, per the EAGLE-3 paper.

When to use speculative decoding:

- Any interactive chat workload where p50 latency matters more than peak throughput.
- Code generation and structured output (JSON, SQL). `α` exceeds 0.9 because the target distribution is highly predictable.
- Long-form generation (thousands of tokens). Amortized speedup pays off continuously.

When not to:

- Very small models (< 3B). The draft isn't much cheaper than the verifier.
- Tiny batch-1 CPU deployments. The draft model's memory overhead may not be worth it.
- Extremely high-temperature creative sampling where acceptance rate `α` collapses.

## Ship It

This lesson produces `outputs/skill-eagle3-tuner.md`. Given an inference workload (model, batch size, target latency, task profile), it recommends a speculative decoding strategy and tuning parameters (draft family, `N`, tree depth, temperature-aware switching).

## Exercises

1. Run `code/main.py`. Confirm the Leviathan distribution check's chi-squared statistic stays below the 95% critical value on 50,000 samples.

2. Sweep `N` from 1 to 10 with `α` fixed at 0.9 and `c` fixed at 0.04. Plot expected tokens per verifier call and actual wall-clock per token. Find the `N` that minimizes wall-clock. Explain the shape of the curve.

3. Modify the code to simulate EAGLE-2 tree search: each step the draft proposes a tree of shape `[2, 2, 2]` (eight candidate paths). The verifier runs once, and the highest-probability accepted path wins. Compute the per-leaf `α` and total tokens per verifier call. Compare against linear-chain speculation at equivalent compute.

4. Implement a batched KV rollback simulator for two concurrent sequences. Sequence A has all drafts accepted; sequence B rejects at position 2. Show that `kv_length` for each sequence is correctly updated and no work is wasted.

5. Read the EAGLE-3 paper Section 4 (training-time test). Explain in two sentences why naive draft training without TTT suffers from exposure bias, and why feeding the draft's own predictions back during training fixes it. Connect it to the scheduled-sampling literature in seq2seq.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Leviathan rule | "min(1, q over p)" | Bernoulli accept/reject at probability `min(1, q(d)/p(d))` that exactly preserves verifier distribution when sampling from residual on rejection |
| Residual distribution | "(q minus p) positive part, normalized" | `(q - p)_+` clamped to zero and renormalized — the correct distribution to sample from on rejection |
| Acceptance rate α | "How often draft is right" | Expected per-token Bernoulli success probability under the rejection rule; dominates all speedup math |
| EAGLE-1 | "Hidden-state drafting" | Small transformer draft conditioned on verifier's last-layer hidden states (Li et al., 2024) |
| EAGLE-2 | "Dynamic draft tree" | EAGLE-1 plus a tree of candidate continuations, scored in one verifier pass via tree attention |
| EAGLE-3 | "Training-time test" | Drops feature-prediction loss, trains on direct token prediction, feeds draft's own output back during training |
| Training-Time Test (TTT) | "Exposure bias fix" | Running the draft autoregressively during training so training and test input distributions match — direct analogy to scheduled sampling |
| KV rollback | "Undo rejected drafts" | Bookkeeping that resets the verifier's KV cache to the accepted prefix length after rejection |
| Bonus token | "The free one" | When all `N` drafts are accepted, an extra sample from `q_{N+1}` at no additional verifier cost |
| Tree attention | "Verify many candidates at once" | Attention with a non-causal mask respecting draft tree topology; computes `q_i` for every node in the tree in one forward pass |

## Further Reading

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — The foundational paper and equivalence theorem
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — Concurrently and independently proposed, with a clean proof
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1, hidden-state conditioned drafting
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — Dynamic tree search
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — The 2026 production default
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — The alternative draft-free approach
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — The canonical production reference wiring all strategies together
