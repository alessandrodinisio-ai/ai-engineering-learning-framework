# Multi-Head Attention

> One attention head learns one relationship. Eight heads learn eight. Heads are free — use more.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 min

## The Problem

A single self-attention head computes one attention matrix. That matrix captures one relationship — usually whichever one reduces training loss most. If subject-verb agreement, coreference, long-range discourse, and syntactic chunking are all tangled in your data, a single head smears them into one softmax distribution and drops half the signal.

The 2017 Vaswani paper's fix: run several attention functions in parallel, each with its own Q, K, V projections, then concatenate the outputs. Each head works in a smaller subspace of dimension `d_model / n_heads`. Total parameter count stays the same; expressiveness goes up.

Multi-head attention is the default shipped by every transformer in 2026. The only remaining debate is *how many* heads, and whether keys and values share projections (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## The Concept

![Multi-head attention split, attend, concatenate](../assets/multi-head-attention.svg)

**Split.** Take `X` of shape `(N, d_model)`. Project to Q, K, V each of shape `(N, d_model)`. Reshape to `(N, n_heads, d_head)` where `d_head = d_model / n_heads`. Transpose to `(n_heads, N, d_head)`.

**Attend in parallel.** Run scaled dot-product attention inside each head. Each head produces `(N, d_head)`. The heads work on different subspaces of the embedding and do not communicate during the attention computation itself.

**Concatenate and project.** Stack heads back to `(N, d_model)`, then multiply by a learnable output matrix `W_o` of shape `(d_model, d_model)`. `W_o` is where heads mix.

**Why it works.** Each head can specialize without competing for representation budget. Probing studies from 2019–2024 reveal clear head specialization: positional heads, previous-token heads, copy heads, named-entity heads, induction heads (the mechanism behind in-context learning).

**2026 variant spectrum:**

| Variant | Q heads | K/V heads | Who uses it |
|---------|---------|-----------|-------------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | compressed to low-rank | DeepSeek-V2, V3 |

GQA is the modern default because it cuts KV-cache memory by `N/G` while nearly preserving full quality. MLA goes further, compressing K/V into a latent space and projecting back at compute time — trading FLOPs for even more memory savings.

## Build It

### Step 1: Split multi-head on top of existing single-head attention

Take lesson 02's `SelfAttention` and wrap it with a split/concat pair. See `code/main.py` for the numpy implementation; the logic:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

One reshape plus one transpose — no loops. This is exactly what PyTorch does under `nn.MultiheadAttention`.

### Step 2: Run scaled dot-product attention per head

Each head gets its own slice of Q, K, V. Attention becomes a batched matmul:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

On real hardware `Qh @ Kh.transpose(...)` is a single `bmm`. The GPU sees one batched matmul of shape `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Adding heads is free.

### Step 3: Grouped-Query Attention variant

Only the key and value projections change. Q gets `n_heads` groups; K and V get `n_kv_heads < n_heads` groups, then repeat to align:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

At inference this saves memory because the KV cache only holds `n_kv_heads` copies, not `n_heads`. Llama 3 70B uses 64 query heads with 8 KV heads — cache shrinks 8×.

### Step 4: Probe what each head learns

Run MHA with 4 heads on a short sentence. For each head, print the `(N, N)` attention matrix. You'll see that even with random initialization, different heads pick out different structure — partly signal, partly rotational symmetry in the subspace.

## Use It

In PyTorch, the one-liner:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA in PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K, V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**How many heads?** Rules of thumb for 2026 production models:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` almost always lands at 64 or 128. It is the unit of how much a single head can "see." Below 32, heads start fighting the scaling factor `sqrt(d_head)`; above 256, you lose the benefit of many small specialists.

## Ship It

See `outputs/skill-mha-configurator.md`. This skill recommends head count, KV head count, and projection strategy for a new transformer given parameter budget, sequence length, and deployment target.

## Exercises

1. **Easy.** Take the MHA in `code/main.py`, fix `d_model=64`, and vary `n_heads` from 1 to 16. Plot loss for a tiny single-layer model on a synthetic copy task. Do more heads help, plateau, or hurt?
2. **Medium.** Implement MQA (one KV head shared across all query heads). Measure parameter reduction compared to full MHA. Compute how much inference KV cache shrinks at N=2048.
3. **Hard.** Implement a mini Multi-head Latent Attention: compress K, V to a rank-`r` latent, store the latent in the KV cache, decompress at attention time. At what `r` does cache memory drop below 1/8 of full MHA while keeping quality within 1 bit of validation ppl?

## Key Terms

| Term | How people talk about it | What it actually means |
|------|--------------------------|------------------------|
| Head | "an independent attention circuit" | A Q/K/V projection at dimension `d_head = d_model / n_heads`, with its own attention matrix. |
| d_head | "head dimension" | Per-head hidden width; almost always 64 or 128 in production. |
| Split / Combine | "the reshape trick" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose before and after attention. |
| W_o | "output projection" | The `(d_model, d_model)` matrix applied after concatenating heads; where heads mix. |
| MQA | "one KV head" | Multi-Query Attention: single shared K/V projection. Minimum KV cache, some quality loss. |
| GQA | "the default since Llama 2" | Grouped-Query Attention, `n_kv_heads < n_heads`; repeat to align with Q. |
| MLA | "DeepSeek's trick" | Multi-head Latent Attention: K, V compressed to low-rank latent, decompressed at attention time. |
| Induction head | "the circuit behind in-context learning" | A pair of heads that detect previously seen content and copy what followed. |

## Further Reading

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — the original multi-head spec.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — the MQA paper.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — how to convert MHA to GQA post-training.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA and why it beats MHA/GQA on cache memory.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — a mechanistic look at what heads actually do.
