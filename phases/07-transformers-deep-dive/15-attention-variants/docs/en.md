# Attention Variants — Sliding Window, Sparse, Differential

> Full attention is a circle. Every token sees every token, and VRAM pays the bill. Four variants bend the circle's shape and claw back half the cost.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## The Problem

Full attention costs `O(N²)` memory and `O(N²)` compute in sequence length. For a 128K-context Llama 3 70B, that's 16 billion attention entries per layer, times 80 layers. Flash Attention (Lesson 12) hides the `O(N²)` activation memory but doesn't change the arithmetic cost—every token still attends to every other token.

Three families of variants change the topology of the attention matrix itself:

1. **Sliding Window Attention (SWA).** Each token attends to a fixed neighbor window, not the entire prefix. Memory and compute drop to `O(N · W)`, where `W` is the window. Gemma 2/3, Mistral 7B's early layers, Phi-3-Long.
2. **Sparse / block attention.** Only selected pairs `(i, j)` are scored; the rest are forced to zero weight. Longformer, BigBird, OpenAI sparse transformer.
3. **Differential attention.** Compute two attention maps with separate Q/K projections and subtract one from the other. Kills the "attention sink" that bleeds weight into the first few tokens. Microsoft's DIFF Transformer (2024).

They coexist. A 2026 frontier model often mixes them: most layers are SWA-1024, every fifth layer is global full attention, and a few differential heads clean up retrieval. Gemma 3's 5:1 SWA-to-global ratio is the current textbook default.

## The Concept

### Sliding Window Attention (SWA)

Each query at position `i` attends only to positions in `[i - W, i]` (causal SWA) or `[i - W/2, i + W/2]` (bidirectional). Tokens outside the window get `-inf` in the score matrix.

```
Full causal:                Sliding window (W=4):
positions 0-7               positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

For `N = 8192` and `W = 1024`, the score matrix has on expectation 1024 × 8192 non-zero entries—an 8× reduction.

**SWA shrinks the KV cache.** Each layer only needs to keep the last `W` tokens of K and V. For a Gemma-3-like config (1024 window, 128K context), KV cache drops 128×.

**Quality cost.** A pure SWA transformer struggles on long-range retrieval. Fix: interleave SWA layers with full attention layers. Gemma 3 uses 5:1 SWA:global. Mistral 7B uses a causal SWA stack where information "flows forward" through overlapping windows—each layer extends the effective receptive field by `W`, so after `L` layers the model can attend back `L × W` tokens.

### Sparse / Block Attention

Pick an `N × N` sparse pattern ahead of time. Three canonical shapes:

- **Local + strided (OpenAI sparse transformer).** Attend to the last `W` tokens, plus every `stride`-th token going back. Captures both local and long-range at `O(N · sqrt(N))` compute.
- **Longformer / BigBird.** Local window + a small set of global tokens (e.g., `[CLS]`) that attend to everyone and are attended by everyone + random sparse connections. Empirically 2× context at equivalent quality.
- **Native Sparse Attention (DeepSeek, 2025).** Learns which `(Q, K)` blocks matter; skips zero blocks at the kernel level. FlashAttention-compatible.

Sparse attention is a kernel engineering story. The math is simple (mask the score matrix); the win comes from never loading zero entries into SRAM. FlashAttention-3 and the 2026 FlexAttention API make custom sparse patterns first-class citizens in PyTorch.

### Differential Attention (DIFF Transformer, 2024)

Standard attention has an "attention sink" problem: softmax forces each row to sum to 1, so tokens that don't want to attend to anything specific dump weight onto the first token (or first few). This steals capacity from real content.

Differential attention fixes this by computing **two** attention maps and subtracting:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

where `λ` is a learned scalar (typically 0.5–0.8). A1 captures real content weights; A2 captures the sink. Subtraction cancels the sink, redistributing weight to relevant tokens.

Reported results (Microsoft 2024): 5–10% lower perplexity, 1.5–2× longer effective context at same training length, sharper needle-in-a-haystack retrieval.

### Variant Comparison

| Variant | Compute | KV Cache | Quality vs Full | Production Use |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | Baseline | Default layer in every model |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl, fine with global layers | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | Mixed | Similar to SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | ~O(N) | Mixed | Matches full at 2× context | Early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active ratio) | O(N) | Within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models |

## Build It

See `code/main.py`. We implement a causal mask comparator that displays full, SWA, local+strided, and differential attention side by side on a toy sequence.

### Step 1: Full Causal Mask (Baseline)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

The baseline from Lesson 07. Lower triangle; zero weight above the diagonal.

### Step 2: Sliding Window Causal Mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

One parameter—`window`. When `window >= n`, you recover full causal attention. When `window = 1`, each token attends only to itself.

### Step 3: Local + Strided Sparse Mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Dense local window plus every `stride`-th token back to the start of the sequence. Receptive field grows in log strides with layer count.

### Step 4: Differential Attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Two attention passes subtracted with a learned mixing coefficient. In the code we compare attention-sink heatmaps between single attention and differential attention, watching the sink collapse.

### Step 5: KV Cache Size

Print per-layer cache size for each variant at `N = 131072`. SWA and sparse variants drop 10–100×. Differential doubles. Pay your VRAM bill with eyes open.

## Use It

2026 production pattern:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at a 5:1 ratio.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention in PyTorch 2.5+ accepts a mask function:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

This compiles into a custom Triton kernel. For common patterns, speed is within 10% of FlashAttention-3, and the mask function is a Python callable.

**When to pick which:**

- **Pure full attention** — every layer at ~16K context or below, or when retrieval quality is paramount.
- **SWA + global mix** — long context (>32K), memory-bound training and inference. The 2026 default above 32K.
- **Sparse block attention** — custom kernels, custom patterns. Reserve for specialized workloads (retrieval, audio).
- **Differential attention** — any workload where attention-sink pollution hurts (long-context RAG, needle-in-a-haystack).

## Ship It

See `outputs/skill-attention-variant-picker.md`. This skill picks an attention topology for a new model based on target context length, retrieval requirements, and training/inference compute profile.

## Exercises

1. **Easy.** Run `code/main.py`. Verify that SWA with `window=4` zeros out everything beyond the last 4 tokens per row. Verify that `window=n` reproduces full causal attention bit-for-bit.
2. **Medium.** Implement causal SWA with `window=1024` on top of the Lesson 07 capstone project. Train for 1,000 steps on tinyshakespeare. How much does loss degrade vs full attention? How much does peak memory drop?
3. **Hard.** Implement Gemma-3-style 5:1 layer mixing (5 SWA, 1 global) in the capstone model. At equivalent parameters, compare loss, memory, and generation quality against pure SWA and pure global baselines.
4. **Hard.** Implement differential attention with a per-head learned `λ`. Train on a synthetic retrieval task (one needle, 2,000 distractors). At equivalent parameters, measure retrieval accuracy vs single-attention baseline.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| Sliding Window Attention (SWA) | "Local attention" | Each query attends to its last `W` tokens; KV cache shrinks to `O(W)`. |
| Effective receptive field | "How far the model can look back" | In a `L`-layer SWA stack with window `W`, at most `L × W` tokens. |
| Longformer / BigBird | "Local + global + random" | Sparse patterns with a few always-attending global tokens; early long-context approaches. |
| Native Sparse Attention | "DeepSeek's kernel trick" | Learns block-level sparsity; skips zero blocks at the kernel level while preserving quality. |
| Differential attention | "Two maps, subtract one" | DIFF Transformer: subtract `λ` times the second attention map from the first to cancel the attention sink. |
| Attention sink | "Weight bleeds to token 0" | Softmax normalization forces rows to sum to 1; uninformative queries dump weight onto position 0. |
| FlexAttention | "Mask as Python" | PyTorch 2.5+ API that compiles arbitrary mask functions into FlashAttention-shaped kernels. |
| Layer type mixing | "5:1 SWA-to-global" | Interleaving sparse and full attention layers in the stack to preserve quality at lower memory. |

## Further Reading

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — Canonical sliding window + global token paper.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — Local + global + random.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI's local+strided pattern.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:global mix.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — The now-textbook 5:1 mix with window=1024.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer paper.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2's learned sparse attention.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — API reference for the mask-as-callable pattern in "Use It."

