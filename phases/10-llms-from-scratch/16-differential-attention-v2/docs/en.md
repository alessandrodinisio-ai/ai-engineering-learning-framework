# Differential Attention (V2)

> Softmax attention sprinkles a tiny probability on every unmatched token. Over 100k tokens, that noise accumulates and drowns the signal. Differential Transformer (Ye et al., ICLR 2025) fixes this by computing attention as the difference of two softmaxes, subtracting out the shared noise floor. DIFF V2 (Microsoft, January 2026) is a production-stack rewrite: decode latency matches baseline Transformer, no custom kernels, FlashAttention compatible. This lesson walks V1 to V2 end-to-end, with a toy implementation of the differential operator you can run in stdlib Python.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 15 (Attention Variants), Phase 10 · 14 (Architecture Walkthroughs)
**Time:** ~60 minutes

## Learning Objectives

- State precisely why softmax attention has a noise floor and why it grows with context length.
- Derive the differential attention formula and explain why the subtraction cancels shared noise components while preserving signal.
- Walk through the V1-to-V2 diff: what got faster, what got simpler, what got more stable, and why each change is necessary for production pretraining.
- Implement differential attention from scratch in pure Python and empirically verify the noise-cancellation property on a synthetic signal-plus-noise query.

## The Problem

Standard softmax attention has a mathematical property that becomes an operational headache at scale. For a query `q`, the attention weights are `softmax(qK^T / sqrt(d))`. Softmax can never produce exact zeros — every unmatched token gets some positive mass. That residual mass is noise, and it scales with context length. At 128k tokens, even if each unmatched token gets just 0.001% probability, the 127,999 of them together contribute about 12% of the total mass. The model must learn to work around a noise floor that grows with context.

Empirically this manifests as attention head interference: hallucinated citations in long-context RAG, lost-in-the-middle failures on 100k-token retrieval tasks, and subtle accuracy degradation on needle-in-a-haystack benchmarks beyond 32k. The Differential Transformer paper (arXiv:2410.05258, ICLR 2025) measured the gap: DIFF Transformer reaches lower perplexity, higher long-context accuracy, and fewer hallucinations than same-size baselines.

DIFF V1 had three problems that kept it out of frontier pretraining pipelines. Its value cache needed to be loaded twice per decode step, it required custom CUDA kernels that broke FlashAttention compatibility, and its per-head RMSNorm broke long-run training stability above 70B. DIFF V2 (Microsoft unilm blog, January 20 2026) fixes all three. This lesson walks both versions, builds the differential operator, and benchmarks noise cancellation on a toy query.

## The Concept

### The noise floor of softmax

For a query `q` and keys `K = [k_1, ..., k_N]`, the attention weights are:

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

No `w_i` will be zero. If `k_i` is completely unrelated to `q`, the score `q . k_i` isn't 0 — it fluctuates around zero with variance `||q||^2 / d`. After softmax normalization, each irrelevant token still contributes `O(1/N)` to the weighted sum. The total contribution of irrelevant tokens is `O((N-1)/N) = O(1)` — not a small quantity.

What the model wants is something like hard top-k: high weight on matching tokens, near-zero elsewhere. Softmax is too smooth to do this directly.

### The differential idea

Split the Q and K projections per head into two halves: Q = (Q_1, Q_2) and K = (K_1, K_2). Compute two attention maps:

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Output:

```
DiffAttn = (A_1 - lambda * A_2) V
```

The subtraction cancels any noise distribution shared between the two maps. If both maps have roughly uniform weight over the 127k irrelevant tokens (which they will at random initialization), those cancel out. The signal — the peaked weight on the few truly relevant tokens — only cancels if it appears at the same magnitude in both maps, which the model learns not to do once training begins.

`lambda` is a per-head learnable scalar parameterized as `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`. It can be negative. `lambda_init` defaults to a small positive number like 0.8.

### Why this matches paired noise cancellation

Imagine two noisy microphones recording the same sound. Both pick up the speaker plus correlated background noise. Subtract one from the other and the shared noise drops out. The voice survives because the two signals differ enough in phase or amplitude to prevent full cancellation. The per-head `lambda` learns exactly that balance.

### V1 vs V2: the diff

V1 keeps parameter count equal to a baseline Transformer. To get two queries per head, it halves the head dimension. That sacrifices head expressiveness and — more painfully — halves the value cache per head. Decode loads the value cache twice per step (once per softmax branch). Result: despite parameter-matching, decode is slower than baseline.

V2 doubles the query head count while keeping KV head count unchanged (borrowing parameters from the up-projection). Head dimension stays the same as baseline. After subtraction, the extra dimensions are projected back down to match the baseline Transformer's O_W projection. Three things happen simultaneously:

1. Decode speed matches baseline (KV cache loaded once).
2. FlashAttention runs unmodified (no custom kernels).
3. Arithmetic intensity during decode improves (more compute per byte loaded from HBM).

V2 also removes the per-head RMSNorm that V1 used to stabilize the subtraction. At 70B-scale pretraining, that RMSNorm broke late-training stability. V2 replaces it with a simpler initialization scheme that maintains training stability without extra modules.

### When to use it

| Workload | Benefit |
|----------|---------|
| Long-context RAG (64k+) | Cleaner attention maps, fewer hallucinated citations |
| Needle-in-a-haystack benchmarks | Significant accuracy gains beyond 32k |
| Multi-document QA | Less cross-document interference |
| Code completion at 8k | Marginal, not worth the architecture change |
| Short chat (< 4k) | Essentially no difference from baseline |

Value grows with context length. At 4k tokens the noise floor is small enough that standard attention is fine. At 128k it's hurting you.

### How it stacks with other 2026 knobs

| Feature | Compatible with DIFF V2? |
|---------|------------------------|
| GQA | Yes (V2 adds Q heads, not KV heads) |
| MLA (DeepSeek) | In principle yes, no published paper combining them |
| MoE | Yes (attention is independent of MLP blocks) |
| RoPE | Yes (unchanged) |
| YaRN / long-context scaling | Yes (exactly where DIFF helps most) |
| FlashAttention | V2 yes (V1 no) |
| Speculative decoding | Yes (attention changes are invisible to the spec-decode loop) |

## Build It

`code/main.py` implements differential attention in pure Python. A toy query with known signal-plus-noise structure lets you directly measure the noise cancellation ratio.

### Step 1: Standard softmax attention

Stdlib matrix operations: lists of lists, hand-written matmul, softmax with max-subtraction for numerical stability.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### Step 2: Split Q, K into two halves

V1 style: halve the head dimension. V2 style: keep head dimension, double head count. The toy implementation uses V1 for pedagogical clarity — the math is identical, only the bookkeeping differs.

### Step 3: Two softmax branches + subtraction

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

Note: output weights can be negative. This is fine — the value cache still handles signed contributions. The downstream V projection absorbs the sign.

### Step 4: Noise cancellation measurement

Construct a synthetic sequence of length 1024. Place signal tokens at a known position, fill the rest with noise. Compute (a) standard softmax attention weight on signal positions and (b) differential attention weight. Measure the signal-to-noise ratio in each. DIFF attention reliably produces higher SNR, by a factor of 3 to 10x depending on how differently the two branches are trained.

### Step 5: V1 vs V2 parameter accounting

Given a configuration (hidden=4096, heads=32, d_head=128), print:

- Baseline Transformer: Q, K, V each of size `hidden * hidden`, MLP at 4 * hidden.
- DIFF V1: Q, K each of size `hidden * hidden`, V of size `hidden * hidden` (unchanged), internal head dimension halved. Add per-head `lambda` parameters (O(heads * d_head)).
- DIFF V2: Q of size `2 * hidden * hidden`, K of size `hidden * hidden`, V of size `hidden * hidden`. Extra dimensions projected back before O_W. Add same `lambda` parameters.

The toy measures V2's additional parameter cost (roughly `hidden * hidden` extra per attention block) and prints it.

## Use It

As of April 2026, DIFF V2 is not yet live in every production inference server, but integration into vLLM and SGLang is ongoing. In the meantime the pattern appears in:

- Microsoft's internal long-context production models.
- Several research reproductions of open model training runs targeting 256k+ context.
- Hybrid architectures combining DIFF attention with sliding-window attention on alternating layers.

When you would adopt it in 2026:

- Training a new model from scratch targeting 64k+ effective context. Add differential attention from the start; retraining later is expensive.
- Fine-tuning a long-context model where lost-in-the-middle failures dominate your evals. A LoRA on the Q projection can approximate the DIFF structure.

When you would not:

- You're serving a pretrained dense model whose long-context performance is stable. Retraining cost rarely pays off on existing weights.
- Your context is always under 16k. The noise floor is negligible.

## Ship It

This lesson produces `outputs/skill-diff-attention-integrator.md`. Given a model architecture, target context length, hallucination profile, and training budget, it produces an integration plan for adding differential attention to a new pretraining run or LoRA fine-tune.

## Exercises

1. Run `code/main.py`. Verify that differential attention reports higher SNR than standard softmax attention on the synthetic query. Vary noise amplitude and show the crossover point where standard attention becomes unusable.

2. Calculate the parameter delta from baseline to DIFF V1 and from baseline to DIFF V2 for a 7B-class model (hidden=4096, heads=32, d_head=128, 32 layers). Show which components gain parameters and which stay the same.

3. Read the DIFF V1 paper (arXiv:2410.05258) Section 3 and the DIFF V2 Hugging Face blog Section 2. Explain in two sentences why V1's per-head RMSNorm was necessary, and why V2 can remove it without training divergence.

4. Implement an ablation: compute differential attention with `lambda = 0` (pure first softmax) and `lambda = 1` (full subtraction). On the synthetic query, measure how SNR varies across this sweep. Find the `lambda` that maximizes SNR.

5. Extend the toy to GQA + DIFF V2. Choose 8 KV heads and 32 Q heads. Show that the KV cache size matches a baseline GQA model with the same (8, 32) configuration.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Differential attention | "Two softmaxes subtracted" | Split Q, K into two halves, compute two softmax maps, subtract the second (scaled by lambda) from the first, then multiply by V |
| Noise floor | "The non-zero tail of softmax" | The O(1/N) weight softmax gives to every irrelevant token, summing to O(1) over long contexts |
| Lambda | "The subtraction scale" | Per-head learnable scalar parameterized as `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`; can be negative |
| DIFF V1 | "The ICLR 2025 version" | The original Differential Transformer; halves head dimension to maintain parameter count, needs custom kernels, slower decode |
| DIFF V2 | "The January 2026 fix" | Doubles Q heads, keeps KV heads; matches baseline decode speed and works with FlashAttention |
| Per-head RMSNorm | "V1's stabilizer" | Extra norm applied after subtraction in V1; V2 removes it to prevent late-training instability |
| Signal-to-noise ratio | "How much attention is wasted" | Ratio of weight on true signal positions to mean weight on irrelevant positions |
| Lost in the middle | "Long-context failure mode" | Empirical phenomenon where retrieval accuracy drops for documents in the middle of a long context — DIFF attention reduces this |
| Arithmetic intensity | "FLOPs per byte loaded" | Ratio V2 improves during decode by doubling queries per KV load; matters for memory-bound decode |

## Further Reading

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — Original paper with noise-cancellation theory and long-context ablations
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — Production-stack rewrite, matching baseline decode, FlashAttention compatible
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — Theoretical analysis of why subtraction can recover pretrained attention structure
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — Parameter-sharing variant
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — The baseline Transformer that DIFF subtracts from
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — The long-context benchmark that DIFF attention targets
