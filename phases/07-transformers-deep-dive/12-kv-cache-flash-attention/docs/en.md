# KV Cache, Flash Attention, and Inference Optimization

> Training is parallel and FLOP-bound. Inference is serial and memory-bound. Different bottlenecks, different tricks.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 min

## The Problem

A naive autoregressive decoder does `O(N²)` work to generate `N` tokens: each step recomputes attention over the entire prefix. A 4K-token reply is 16 million attention operations, most of them redundant. A prefix token's hidden state is fixed once computed — you only need to dot the new token's query against cached keys and values from everything before it.

Worse, attention itself moves a lot of data. Standard attention materializes an N×N score matrix, an N×d softmax output, and an N×d final output — too many HBM round-trips. At N≥2K, attention becomes memory-bound before it becomes FLOP-bound. Classic attention kernels use only 1/4 to 1/10 of a modern GPU's efficiency.

Two optimizations, both from Dao et al., pushed frontier inference from "slow" to "fast":

1. **KV cache.** Store each prefix token's K and V vectors. Attention for each new token is a single query against cached keys. Inference drops from `O(N²)` to `O(N)` per generation step.
2. **Flash Attention.** Tile the attention computation so the full N×N matrix never hits HBM. Softmax + matmul happen entirely in SRAM. 2–4× wall-clock speedup on A100; 5–10× on H100 with FP8.

Both are ubiquitous by 2026. Every production inference stack (vLLM, TensorRT-LLM, SGLang, llama.cpp) assumes them. Every frontier model ships with Flash Attention enabled.

## The Concept

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV Cache Arithmetic

Per decoder layer, per token, per head:

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

For a 7B model, 32 layers, 32 heads, d_head=128, fp16:

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

For Llama 3 70B (80 layers, d_head=128, GQA with 8 KV heads):

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

That 10 GB is why Llama 3 70B at 128K context, batch size 1, eats most of a 40 GB A100 on KV cache alone.

**GQA is a KV cache win.** 64-head MHA would be 32 GB. MLA compresses further.

Drag the dimensions and watch cache size change. Pull sequence length or batch size up and see how fast it blows past a single GPU's capacity:

```figure
kv-cache-sizer
```

### Flash Attention — The Tiling Trick

Standard attention:

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

Three HBM round-trips. On H100, HBM bandwidth is 3 TB/s; SRAM is 30 TB/s. Each HBM round-trip is a 10× slowdown compared to keeping everything on-chip.

Flash Attention:

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

One HBM round-trip per tile. Total memory footprint drops from `O(N²)` to `O(N)`. The backward pass recomputes some values from the forward pass instead of storing them — another memory win.

**Numerical trick.** Rolling softmax maintains `(max, sum)` across tiles, making the final normalization exact. Not an approximation — Flash Attention produces bit-identical output to standard attention (modulo fp16 non-associativity).

**Version evolution:**

| Version | Year | Key change | Speedup on reference hardware |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | 2× on A100 |
| Flash 2 | 2023 | Better parallelism, causal-first scheduling | 3× on A100 |
| Flash 3 | 2024 | Hopper async, FP8 | 1.5–2× on H100 (~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5-stage pipeline, software exp2 | Inference-first (forward-only at launch) |

Flash 4 launched forward-pass only. Training still uses Flash 3. Flash 4's GQA and varlen support are pending (mid-2026).

### Speculative Decoding — Another Latency Win

A cheap model proposes N tokens. The large model verifies all N in parallel. If verification accepts k tokens, you trade 1 large-model forward pass for k generation steps. Typical k=3–5 on code and prose.

2026 defaults:
- **EAGLE 2 / Medusa.** Integrated draft heads sharing the verifier's hidden states. 2–3× speedup, no quality loss.
- **Speculative decoding with draft model.** 2–4× speedup on consumer hardware.
- **Lookahead decoding.** Jacobi iteration; no draft model needed. Niche but free.

### Continuous Batching

Classic batched inference: wait for the slowest sequence to finish before starting a new batch. Wastes GPU when short replies finish early.

Continuous batching (first in Orca, now in vLLM, TensorRT-LLM, SGLang): swap new requests into the batch as old ones complete. 5–10× throughput improvement on typical chat workloads.

### PagedAttention — KV Cache as Virtual Memory

vLLM's signature feature. KV cache is allocated in 16-token blocks; a page table maps logical positions to physical blocks. Enables sharing KV across parallel samples (beam search, parallel sampling), hot-swapping prefixes for prompt caching, and defragmenting memory. 4× throughput over naive contiguous allocation.

## Build It

See `code/main.py`. We implement:

1. A naive `O(N²)` incremental decoder.
2. An `O(N)` KV-cache decoder.
3. A tiled softmax simulating Flash Attention's rolling-max algorithm.

### Step 1: KV Cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

Simple: append per-token K, V vectors to per-layer, per-head lists.

### Step 2: Tiled Softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention style softmax(qK^T)V with rolling max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

Bit-identical output to one-shot `softmax(qK) V`, but the working set at any moment is a single `tile × d_head` block, not the full `N × d_head`.

### Step 3: Compare Naive vs Cached Decoding on 100-Token Generation

Count attention operations. Naive: `O(N²)` = 5050. Cached: `O(N)` = 100. Code prints both.

## Use It

```python
# HuggingFace transformers enables KV cache automatically on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # FA3 on Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM production:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

Cross-request prefix caching is a major 2026 win — the same system prompt, few-shot examples, or long-context documents reuse KV across calls. For agent workloads with repeated tool prompts, prefix caching often delivers 5× throughput improvement.

## Ship It

See `outputs/skill-inference-optimizer.md`. This skill picks attention implementation, KV cache strategy, quantization, and speculative decoding for a new inference deployment.

## Exercises

1. **Easy.** Run `code/main.py`. Confirm naive and cached decoders produce identical output; note the operation count difference.
2. **Medium.** Implement prefix caching: given a prompt P and several completions, run one forward pass on P to fill the KV cache, then branch per completion. Measure speedup vs re-encoding P for each completion.
3. **Hard.** Implement a toy PagedAttention: KV cache in fixed 16-token blocks plus a free list. When a sequence finishes, return its blocks to the pool. Simulate 1,000 chat completions of varying length. Compare memory fragmentation vs contiguous allocation.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| KV cache | "the trick that makes decoding fast" | Store K and V from each prefix token; new queries attend to them without recomputation. |
| HBM | "GPU main memory" | High Bandwidth Memory; 80 GB on H100, 192 GB on B200. ~3 TB/s bandwidth. |
| SRAM | "on-chip memory" | Fast per-SM memory, ~256 KB per SM on H100. ~30 TB/s bandwidth. |
| Flash Attention | "tiled attention kernel" | Computes attention without materializing N×N in HBM. |
| Continuous batching | "no-wait batching" | Swap finished sequences out and new ones in without draining the batch. |
| PagedAttention | "vLLM's signature" | KV cache allocated in fixed blocks with a page table; eliminates fragmentation. |
| Prefix caching | "reuse long prompts" | Cache KV of shared prefixes across requests; massive cost reduction for agents. |
| Speculative decoding | "draft + verify" | Cheap draft model proposes tokens; large model verifies k in one pass. |

## Further Reading

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5-stage pipeline and software exp2 trick; read the repo README for caveats on the forward-only release mentioned in this lesson.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — The vLLM paper.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Speculative decoding.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1/2 paper for the integrated draft approach cited in this lesson.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — The Medusa approach mentioned alongside EAGLE.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — Canonical deep-dive on the 16-token block and page table design.
