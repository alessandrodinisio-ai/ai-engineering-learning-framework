# Speculative Decoding — Draft, Verify, Repeat

> Autoregressive decoding is serial. Every token waits for the last one. Speculative decoding breaks the chain: a cheap model drafts N tokens, and the expensive model verifies all N in a single forward pass. When the draft is right, you trade one big-model forward for N generations.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache and Flash Attention)
**Time:** ~60 minutes

## The Problem

A 70B LLM on an H100 takes ~30 ms to sample one token. A 3B draft model takes ~3 ms. If we let the 3B draft 5 tokens ahead, then run the 70B *once* to verify all 5, the total for up to 5 accepted tokens is `5×3 + 30 = 45 ms`—versus `5×30 = 150 ms` for straight generation. That's the entire pitch: trade a bit of extra GPU memory (the draft model) for 2–4× lower decoding latency.

The trick must preserve the distribution. Speculative sampling, proposed by Leviathan et al. (2023) and independently by Chen et al., guarantees the output sequence has **exactly the same distribution** as the large model generating on its own. No quality tradeoff. Just faster.

Four draft-verifier pairings dominate 2026 inference:

1. **Vanilla speculative (Leviathan 2023).** Separate draft model (e.g., Llama 3 1B) + verifier (e.g., Llama 3 70B).
2. **Medusa (Cai 2024).** Multiple decoding heads on the verifier predict positions `t+1..t+k` in parallel. No separate draft model.
3. **EAGLE family (Li 2024, 2025).** Lightweight draft reusing verifier hidden states; acceptance rate closer than vanilla; typical 3–4×.
4. **Lookahead decoding (Fu 2024).** Jacobi iteration; no draft model at all. Self-speculative. Niche but dependency-free.

Every production inference stack in 2026 ships speculative decoding by default. vLLM, TensorRT-LLM, SGLang, and llama.cpp all support at least vanilla + EAGLE-2.

## The Concept

### The Core Algorithm

Given a verifier `M_q` and a cheaper draft `M_p`:

1. Let `x_1..x_k` be the already-decoded prefix.
2. **Draft**: use `M_p` to autoregressively propose `d_{k+1}, d_{k+2}, ..., d_{k+N}`, with draft probabilities `p_1..p_N`.
3. **Verify in parallel**: run `M_q` once on `x_1..x_k, d_{k+1}, ..., d_{k+N}`, obtaining verifier probabilities `q_1..q_{N+1}` at positions `k+1..k+N+1`.
4. **Accept/reject each draft token left to right**: for each `i`, accept with probability `min(1, q_i(d_i) / p_i(d_i))`.
5. On first rejection at position `j`: sample `t_j` from the normalized "residual" distribution `(q_j - p_j)_+`. All drafts after `j` are discarded.
6. If all `N` are accepted: sample one extra token `t_{N+1}` from `q_{N+1}` (free bonus token).

The residual distribution trick is the mathematical insight that makes the output distribution exact—as if `M_q` sampled from scratch.

### What Determines Speedup

Let `α` = expected acceptance rate per draft token. Let `c` = cost ratio of draft to verifier. Per step:

- Vanilla generation: one big-model call per token.
- With speculation when `α` is high: one big-model call per `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` tokens.

Typical rule of thumb at `α = 0.75`, `N = 5`: 3× fewer big-model calls. Draft cost is 5× cheaper. Total wall-clock drops ~2.5×.

**α depends on:**

- How well the draft approximates the verifier. Same family / same training data significantly raises α.
- Decoding strategy. Greedy draft vs greedy verifier: α is high. Temperature sampling: harder to match; acceptance rate drops.
- Task type. Code and structured output accept more (predictable); free-form creative writing accepts less.

### Medusa — Drafting Without a Draft Model

Medusa replaces the draft model with extra output heads on the verifier. At position `t`:

```
shared backbone → hidden h_t
    ├── head_0: predicts token at t+1  (standard LM head)
    ├── head_1: predicts token at t+2
    ├── head_2: predicts token at t+3
    ├── head_3: predicts token at t+4
```

Each head outputs its own logits. At inference you sample from each head to get a candidate sequence, then verify with a tree-attention scheme that considers all candidate continuations in one forward pass.

Pros: no second model. Cons: adds trainable parameters; requires a supervised fine-tuning phase (~1B tokens); acceptance rate slightly lower than well-matched vanilla speculation.

### EAGLE — Better Drafts via Hidden State Reuse

EAGLE-1/2/3 (Li et al., 2024–2025) makes the draft model a tiny transformer (typically 1 layer) that consumes the verifier's last-layer hidden states. Because the draft sees the verifier's feature representation, its predictions correlate strongly with the verifier's output distribution. Acceptance rate climbs from ~0.6 (vanilla) to 0.85+.

EAGLE-3 (2025) adds tree search over candidate continuations. vLLM and SGLang ship EAGLE-2/3 as the default speculative path for Llama 3/4 and Qwen 3.

### The KV Cache Dance

Verification feeds `N` draft tokens into the verifier in a single forward pass. This extends the verifier's KV cache by `N` entries. If some drafts are rejected, you must roll back the cache to the accepted prefix length.

Production implementations (vLLM's `--speculative-model`, TensorRT-LLM's LookaheadDecoder) handle this with a tentative KV buffer. Write first, commit on acceptance. Conceptually simple but fiddly.

## Build It

See `code/main.py`. We implement the core speculative sampling algorithm (rejection step + residual distribution), including:

- A "big model" that is a deterministic softmax over hand-written distributions (so we can analytically verify the acceptance math).
- A "draft model" that is a perturbed version of the big model.
- An accept/reject loop that produces the same marginal distribution as direct sampling.

### Step 1: Rejection Step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` is a uniform random number. `q_prob` is the verifier's probability of the drafted token. `p_prob` is the draft model's probability. The Leviathan theorem states: this Bernoulli decision, combined with residual sampling on rejection, exactly preserves the verifier's distribution.

### Step 2: Residual Distribution

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Element-wise subtract `p` from `q`, clamp negatives to zero, renormalize. Sample from here on any rejection.

### Step 3: One Speculative Step

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Accept 5 → one bonus → one verifier pass yields 6 tokens.

### Step 4: Measure Acceptance Rate

Run 10,000 speculative steps at varying draft quality levels. Plot acceptance rate vs KL divergence between draft and verifier distributions. You should see a clean monotonic relationship.

### Step 5: Verify Distribution Equivalence

Empirically: the token histogram produced by the speculative loop should match the histogram produced by direct sampling from the verifier. This is the Leviathan theorem in practice. A chi-squared test confirms within sampling error.

## Use It

Production:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

As of mid-2026, TensorRT-LLM has the fastest Medusa path. `faster-whisper` wraps speculative decoding with a small draft for Whisper-large.

**Picking a draft:**

| Strategy | When to pick | Speedup |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | Fast prototype, no training needed | 1.8–2.3× |
| Medusa heads | You can fine-tune the verifier | 2–3× |
| EAGLE-2 / 3 | Production, chasing max speed | 3–4× |
| Lookahead | No draft, no training, no extra params | 1.3–1.6× |

**When NOT to use speculative decoding:**

- Single-sequence generation of 1–5 tokens. Overhead dominates.
- Highly creative / high-temperature sampling (α drops).
- Memory-constrained deployments (draft model adds memory).

## Ship It

See `outputs/skill-spec-decode-picker.md`. This skill picks a speculative decoding strategy (vanilla / Medusa / EAGLE / lookahead) and tuning parameters (N, draft temperature) for a new inference workload.

## Exercises

1. **Easy.** Run `code/main.py`. Confirm that the speculative token distribution matches the verifier's direct sampling distribution within chi-squared p > 0.05 over 50,000 tokens.
2. **Medium.** Plot speedup (tokens per big-model forward) as a function of `N` for `α = 0.5, 0.7, 0.85`. Find the optimal `N` for each α. (Hint: expected tokens per verification call = `(1 - α^{N+1}) / (1 - α)`.)
3. **Hard.** Implement a mini Medusa: take the Lesson 14 capstone GPT, add 3 extra LM heads predicting positions t+2, t+3, t+4. Train on tinyshakespeare with a joint multi-head loss. Compare acceptance rate against a vanilla draft made by truncating the same model.
4. **Hard.** Implement rollback: start from a KV cache of a 10-token prefix, feed 5 draft tokens, simulate a rejection at position 3. Verify that your cache reads for the next iteration correctly match "prefix + first 2 accepted drafts."

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| Draft model | "The cheap one" | Smaller model proposing candidate tokens; typically 10–50× cheaper than the verifier. |
| Verifier | "The big one" | Target model whose distribution we want to preserve; runs once per speculative step. |
| Acceptance rate (α) | "How often the draft is right" | Per-token probability the verifier accepts the draft. Typical 0.7–0.9. |
| Residual distribution | "Fallback on rejection" | Normalized `(q - p)_+`; sampled on rejection to preserve the verifier's distribution. |
| Bonus token | "The free one" | When all N drafts are accepted, sample one more from the verifier's next-step distribution. |
| Medusa | "Draft-free speculation" | Multiple LM heads on the verifier predict positions t+1..t+k in parallel. |
| EAGLE | "Hidden-state draft" | Tiny transformer draft conditioned on verifier last-layer hidden states. |
| Lookahead decoding | "Jacobi iteration" | Self-speculation via fixed-point iteration; no draft model. |
| Tree attention | "Verify multiple candidates at once" | Branching verification considering several draft continuations simultaneously. |
| KV rollback | "Undo rejected drafts" | Tentative KV buffer; commit on acceptance, discard on rejection. |

## Further Reading

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Core algorithm and equivalence theorem.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Independent concurrent work; clean Bernoulli rejection proof.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa paper; tree-attention verification.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; hidden-state-conditioned draft.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; dynamic tree depth.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — Lookahead, draft-free approach.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — Canonical production reference wiring up all four strategies.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — Reference code for EAGLE-1/2/3.

