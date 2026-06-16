# Native Sparse Attention (DeepSeek NSA)

> At 64k tokens, attention eats 70-80% of decode latency. Every open model lab has a plan to fix it. DeepSeek's NSA (ACL 2025 Best Paper) is the one that stuck: three parallel attention branches—compressed coarse-grained tokens, selectively retained fine-grained tokens, and a sliding window for local context—combined through a learned gate. It's hardware-aligned (kernel-friendly), natively trainable (works in pretraining, not bolted on at inference), and runs faster than FlashAttention on 64k decode while matching or exceeding full attention quality. This lesson builds all three branches end-to-end and shows why this sparsity is end-to-end differentiable.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 minutes

## Learning Objectives

- State NSA's three attention branches and what each captures.
- Explain why NSA is "natively trainable" while prior sparse attention methods are inference-only.
- Compute NSA's attention computation savings over full attention at 64k context as a function of compression block size and selection top-k.
- Implement the three-branch combination in stdlib Python on a short synthetic sequence and verify that gating weights behave correctly.

## The Problem

Full attention at sequence length N costs `O(N^2)` per layer and `O(N)` KV cache. At 64k tokens, the compute and memory bandwidth numbers are catastrophic. The NSA paper's theoretical estimate checks out empirically: attention accounts for 70-80% of total decode latency at 64k. Everything downstream—TTFT, tokens/sec, cost per million tokens—is dominated by attention cost.

Sparse attention is the obvious answer. Prior attempts fall into two buckets. Fixed-pattern sparse (sliding window, strided, block-local) throws away information and fails on long-range recall tasks. Inference-time sparse (KV cache pruning, H2O, StreamingLLM) applies to a model pretrained on dense attention and recovers only a fraction of the potential speedup, because the model was never asked to route information through sparse patterns.

Native Sparse Attention (Yuan et al., DeepSeek + PKU + UW, ACL 2025 Best Paper, arXiv:2502.11089) does both: a sparse pattern learned by the model during pretraining, realized as a kernel-aligned algorithm that delivers real compute savings at inference. Two years from now, NSA or its direct descendant will be the default attention in every frontier long-context model.

## The Concept

### Three parallel branches

For each query, NSA runs attention three times against three different views of the KV cache:

1. **Compression branch.** Tokens are grouped into blocks of size `l` (typically 32 or 64). Each block is compressed via a small learned MLP into a single summary token. The query attends over these compressed tokens, giving a coarse-grained view of the entire sequence.

2. **Selection branch.** Using the compression branch's attention scores, identify the top-k blocks most relevant to the current query. Read fine-grained (uncompressed) tokens from those blocks, and the query attends over all of them. Think of the compression branch's attention as the routing signal for selection.

3. **Sliding window branch.** The query attends to the most recent `W` tokens (typically 512) for local context. This branch captures structurally dense short-range patterns (syntax, local coreference) that the other two may miss.

The outputs of three branches are combined via a learned per-position gate:

```
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win` are gating weights from a small MLP on the query. They need not sum to 1—they can independently weight each branch.

### Why this is "natively trainable"

The selection step (top-k blocks) is discrete. Discrete operations break gradient flow. Prior sparse attention work either skipped backpropagation through the selection (limiting training) or used continuous relaxations that don't give real sparsity at inference.

NSA sidesteps this: the compression branch's attention *is itself* a differentiable coarse-grained attention over the entire sequence. The top-k operation merely reuses the compression branch's highest attention scores to pick which fine-grained blocks to load. Gradients flow through the compression branch's scores (which influence both the compression output *and* the selection logic), and the contribution of the selected blocks to the final output is also differentiable. The non-differentiable `top_k` operation is a no-op on the forward compute graph—it only controls which blocks are loaded from memory.

This is why NSA can be used end-to-end in pretraining. The model learns to jointly route information through the three branches, producing a sparse pattern that actually delivers the promised speedup at inference.

### Hardware-aligned kernels

NSA's kernels are designed for modern GPU memory hierarchies. Kernels load queries by GQA group (outer loop), fetch the corresponding sparse KV blocks for each group (inner loop), and run attention on SRAM. Because each query group sees the same selected blocks (selection is per-query-group, not per-query-head), KV loads are amortized within the group. Arithmetic intensity stays high.

The paper reports Triton kernels running 9x faster than FlashAttention on 64k decode, with speedup growing with sequence length. Both forward and backward kernels are provided.

### Compute budget

Let `N` be sequence length, `l` compression block size, `k` top-k selection count, `w` sliding window, `b` selected block size (typically equals `l`).

- Compression branch: `O(N/l)` keys per query, so `O(N * N / l)` total.
- Selection branch: `O(k * b)` keys per query, so `O(N * k * b)`.
- Sliding branch: `O(w)` keys per query, so `O(N * w)`.

Total: `O(N * (N/l + k*b + w))`.

With `N = 64k, l = 64, k = 16, b = 64, w = 512`: per-query cost is `1000 + 1024 + 512 = 2536 keys`. Full attention is `64000 keys`. 25x compute reduction.

With `N = 128k, l = 64, k = 16, b = 64, w = 512`: per-query cost is `2000 + 1024 + 512 = 3536 keys`. Full attention is `128000 keys`. 36x reduction. Gains grow with sequence length, which is the entire point.

### How it compares

| Method | Differentiable | Real inference speedup | Long-range recall |
|--------|---------------|----------------------|-------------------|
| Sliding window only | Yes | Yes | Fails |
| Strided / block sparse | Yes | Yes | Partial |
| KV pruning (H2O, StreamingLLM) | N/A (inference-time) | Yes | Partial |
| MoBA (Moonshot) | Partial | Yes | Good |
| NSA | Yes (native) | Yes (9x at 64k) | Matches full attention |

MoBA (Moonshot, arXiv:2502.13189) was published concurrently, taking a similar "three is better than one" approach by applying MoE principles to attention blocks. NSA and MoBA are the two architectures to know for 2026 long-context pretraining.

## Build It

`code/main.py` implements the three branches on a short synthetic sequence and demonstrates:

- Compression MLP (using a simple mean-pool baseline for pedagogical clarity; real NSA uses a learned MLP).
- Top-k block selection driven by compression branch scores.
- Sliding window attention over the last `w` tokens.
- Gated combination.
- A compute-count printout comparing against full attention.

### Step 1: Compress tokens into blocks

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### Step 2: Compression branch attention

Run softmax attention of query against compressed keys. The compression branch scores double as the routing signal for top-k selection.

### Step 3: Top-k block selection

Pick the indices of the `k` highest-scoring compressed blocks. Load raw uncompressed tokens from those blocks and run attention over them.

### Step 4: Sliding window attention

Take the last `w` tokens, run standard attention over them.

### Step 5: Gating + combination

A small MLP on the query produces three gating weights. Final output is the weighted sum of the three branch outputs.

### Step 6: Compute count

Print the number of keys attended per query per branch and the total. Compare against `N` (full attention). On a 1024-token synthetic with `l = 32, k = 4, w = 128`, NSA attends to `32 + 128 + 128 = 288` keys per query vs. full attention's 1024—3.5x fewer.

## Use It

NSA is live in DeepSeek's own long-context pretraining pipeline. Integration status in public inference stacks as of April 2026:

- **DeepSeek internal**: Native, public weights use NSA or its successor DSA (Deepseek Sparse Attention).
- **vLLM**: Experimental NSA support under development for DeepSeek-V3.x weights.
- **SGLang**: NSA benchmarks published; production path follows vLLM.
- **llama.cpp / CPU**: Not supported; the kernel decomposition overhead isn't worthwhile at CPU throughput.

When to use NSA:

- Pretraining or continued training runs targeting 64k+ context with serious compute budgets.
- Inference on DeepSeek's own long-context checkpoints. The weights are NSA-native.

When not to use:

- Serving an existing dense-attention pretrained model. You can't retrofit NSA without continued training.
- Context below 16k. Three-branch overhead exceeds savings.
- Batch-1 interactive chat. Latency-sensitive decode benefits, but only at long context.

## Ship It

This lesson produces `outputs/skill-nsa-integrator.md`. Given a long-context pretraining run spec, it produces an NSA integration plan: compression block size, top-k, sliding window, gating MLP width, kernel selection, and the specific long-context evals that would justify this architectural change.

## Exercises

1. Run `code/main.py` on a 1024-token synthetic. Sweep `(l, k, w)` across three presets and print compute counts. Find the preset that achieves the lowest per-query key count while maintaining 95% recall against full attention on a needle-in-a-haystack test.

2. Replace the mean-pool compressor with a small learned MLP (2 layers, hidden 32). Train it on a synthetic task where the signal is the block mean. Measure its perplexity gap against the mean-pool baseline on held-out data.

3. Implement the gating MLP. It takes the query as input and outputs three scalars. Show that gate behavior is reasonable: approximately uniform weighting on random queries, heavy weighting on the selection branch when the query hits a late block.

4. Compute the KV cache memory budget for a 70B model with NSA enabled at 128k context. 8 KV heads, head dim 128, BF16. Compare against full attention and MLA (Phase 10 · 14 shows MLA's numbers). Find the sequence length at which NSA's fine-grained branch KV cache equals full attention.

5. Read NSA paper (arXiv:2502.11089) Section 4, and explain in three sentences why the compression branch's attention scores are reused for top-k selection rather than computing a separate routing score. Connect the answer to gradient flow.

## Key Terms

| Term | How people say it | What it actually is |
|------|----------------|------------------------|
| Compression branch | "coarse view" | Attention over block-averaged keys, giving global context at O(N/l) keys per query |
| Selection branch | "top-k blocks" | Fine-grained attention over the `k` blocks with highest compression-branch scores |
| Sliding window | "local context" | Attention over the last `W` tokens for short-range patterns |
| Native trainability | "pretrain with sparsity" | Sparse pattern learned during pretraining, not bolted on at inference |
| Compression block size l | "grouping size for coarse view" | How many tokens merge into one summary; typically 32-64 |
| Top-k | "blocks to keep" | Number of compressed blocks whose uncompressed tokens will be read; typically 16 |
| Sliding window W | "local attention radius" | Usually 512; shorter hurts local coherence, longer wastes compute |
| Branch gate | "how to mix the three" | Per-position MLP outputs weighting each branch's contribution |
| Hardware alignment | "kernel-friendly sparsity" | The chosen sparse pattern lets actual GPU kernels achieve the theoretical speedup |
| DSA | "NSA's successor" | Deepseek Sparse Attention, the next architecture after NSA in the DeepSeek lineage |

## Further Reading

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — The paper
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — The architecture family NSA targets
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — Concurrent work applying MoE principles to attention blocks
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — Origin of the sliding window
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — Inference-time sparsity baseline that NSA improves upon
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — The full-attention baseline NSA kernels beat at 64k
