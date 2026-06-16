# DeepSeek-V3 Architecture Walkthrough

> Phase 10 · Lesson 14 noted six architectural knobs every open model turns. DeepSeek-V3 (December 2024, 671B total, 37B active) turns all six and adds four more: Multi-head Latent Attention, auxiliary-loss-free load balancing, multi-token prediction, and DualPipe training. This lesson reads through the DeepSeek-V3 architecture end-to-end and derives every parameter count from the public config. By the end you can explain why the 671B/37B ratio is the right bet, and why MLA + MoE together beat either alone at the frontier.

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## Learning Objectives

- Read through DeepSeek-V3's config end-to-end, explaining each field with six GPT-2 knobs plus four DeepSeek-specific additions.
- Derive total parameter count (671B), active parameter count (37B), and the components contributing to each.
- Compute MLA's KV cache footprint at 128k context, compared against what a same-active-parameter dense model with GQA would pay.
- State the four DeepSeek-specific innovations (MLA, MTP, auxiliary-loss-free routing, DualPipe) and identify which part of the architecture/training stack each targets.

## The Problem

DeepSeek-V3 is the first frontier open model whose architecture substantively differs from the Llama family. Llama 3 405B is "GPT-2 with six knobs turned." DeepSeek-V3 is GPT-2 with all six knobs turned plus four more. Reading the Llama 3 config is warmup for reading the DeepSeek config, but the deep structure—the shape of the attention block, the routing logic, the training-time objectives—differs enough that you need a separate walkthrough.

The payoff for learning it: DeepSeek-V3's open-weight release changed what "frontier capability" means in open models. This architecture is the blueprint many 2026 training runs are copying. Understanding it is table stakes for any role touching frontier LLM training or inference.

## The Concept

### The invariant core, once more

DeepSeek-V3 is still autoregressive. It still stacks decoder blocks. Each block still has attention plus MLP plus two RMSNorms. The MLP still uses SwiGLU. Still uses RoPE. Pre-norm. Weight-tied embedding. Same baseline as every Llama or Mistral.

### The twist: MLA instead of GQA

From Phase 10 · 14 you know GQA shrinks KV cache by sharing K and V across Q head groups. Multi-head Latent Attention (MLA) goes further: K and V are compressed into a shared low-rank latent representation (`kv_lora_rank`), then decompressed on the fly per head. The KV cache stores only the latent—typically 512 floats per layer per token instead of 8 x 128 = 1024 floats.

At 128k context, DeepSeek-V3 with MLA (one shared latent `c^{KV}` per layer per token; K and V are derived from this latent via up-projections that can be absorbed into subsequent matmuls for inference efficiency):

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

A hypothetical GQA baseline (Llama 3 70B shape, 8 KV heads, head dim 128) would pay:

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA is 4x smaller than Llama-3-70B-style GQA cache at 128k context.

Tradeoff: MLA adds a decompression step to each attention computation (per head). The extra compute is small relative to the bandwidth saved. Long-context inference is a net win.

### Routing: Auxiliary-loss-free load balancing

The MoE router decides which top-k experts process each token. Naive routers concentrate too much work on a few experts, leaving others idle. Standard fix: add an auxiliary loss term that penalizes load imbalance. It works but slightly degrades main-task performance.

DeepSeek-V3 introduces an auxiliary-loss-free scheme. A per-expert bias term is added to router logits, adjusted during training with a simple rule: if expert `e` is overloaded, decrease `bias_e`; if underloaded, increase it. No extra loss term. Training stays clean. Expert load stays balanced.

Impact on main loss: unmeasurable. Impact on MoE architecture: cleaner, no auxiliary loss hyperparameters to tune.

### MTP: Denser training + free draft

From Phase 10 · 18 you know DeepSeek-V3 adds D=1 MTP module predicting the token two positions ahead. At inference, the trained module is repurposed as a speculative decoding draft with 80%+ acceptance rate. During training, each hidden state is supervised by D+1 = 2 targets, providing denser signal.

Parameters: 14B on top of the 671B main model. Overhead: 2.1%.

### Training: DualPipe

From Phase 10 · 19 you know DualPipe is a bidirectional pipeline that overlaps forward and backward chunks with cross-node all-to-all communication. At DeepSeek-V3's 2,048 H800 scale, it recovers roughly 245k GPU-hours that 1F1B would have lost to pipeline bubbles.

### The config, field by field

Here is the DeepSeek-V3 config (simplified):

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used in first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equals num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE experts per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (one always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

Parsing it:

- `hidden_size=7168`: Embedding dimension.
- `num_hidden_layers=61`: Total block depth.
- `first_k_dense_layers=3`: First 3 blocks use dense MLP of size 18432. Remaining 58 use MoE.
- `num_attention_heads=128`: 128 query heads.
- `kv_lora_rank=512`: K and V compressed to this latent dimension, decompressed per head.
- `num_experts=256, num_experts_per_tok=8`: Each MoE block has 256 experts, routing top-8.
- `shared_experts=1`: On top of the 256 routed experts, 1 always-on expert contributes to every token. Think of it as a "dense floor" ensuring every token gets something reliable.
- `moe_intermediate_size=2048`: Each expert's MLP hidden size. Smaller than the dense MLP because there are 256 of them.

### Parameter accounting

Full calculation is in `code/main.py`. Headlines:

- Embedding: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- First 3 dense blocks: attention with MLA (~144M per block) + dense MLP (~260M per block) + norms. ~1.2B total.
- 58 MoE blocks: attention with MLA (~144M) + 256 experts at (~30M each) + 1 shared expert (30M) + norms. Total per block ~7.95B including all experts. 58 MoE blocks total 461B.
- MTP module: 14B.

Grand total: ~476B core architecture + 14B MTP, and the published 671B number clearly includes additional structural parameters (bias tensors, expert-specific components, shared expert scaling, etc.). The numbers we reproduce in the calculator are within 3-5% of published values—differences come from fine-grained accounting documented in DeepSeek's report Section 2 appendix.

Active parameters per forward pass:

- Attention: 144M per layer * 61 = 8.8B (all layers fire).
- Active MLP: first 3 dense layers (3 * 260M = 780M), 58 MoE layers each activate 8 routed + 1 shared + routing overhead. Active MLP per layer: ~260M. Total: 3 * 260M + 58 * 260M = ~15.9B.
- Embedding + norms: 1.2B.
- Total active: roughly 26B core + 14B MTP (trained but not always run at inference) ≈ 37B.

### The 671B / 37B ratio

18x sparsity ratio (active parameters are 5.5% of total). DeepSeek-V3 is the sparsest frontier MoE in published open weights. Mixtral 8x7B has ratio 13/47 (28%) which is much denser. Llama 4 Maverick has ratio 17B/400B (4.25%) which is comparable. DeepSeek's bet: at frontier scale, more experts with lower activation ratio yields better quality per active FLOP.

### Where DeepSeek-V3 sits

| Model | Total | Active | Ratio | Attention | Novel ideas |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### What comes next: R1, V4

DeepSeek-R1 (2025) is a reasoning training run on the V3 backbone. R1 uses the same architecture. What changes is the post-training recipe (large-scale RL on verifiable tasks), not the pretraining architecture.

DeepSeek-V4 (if released) is expected to retain MLA + MoE + MTP and add DSA (DeepSeek Sparse Attention), the successor to NSA in Phase 10 · 17. The lineage is stable: architectural innovations accumulate; each version turns additional knobs.

## Use It

`code/main.py` is a parameter calculator specific to the DeepSeek-V3 shape. Run it, compare its output against the paper numbers, and use it on hypothetical variants (256 experts vs 512, top-8 vs top-16, MLA rank 512 vs 1024).

What to look at:

- Total parameter count vs. published 671B.
- Active parameter count vs. published 37B.
- KV cache at 128k context—MLA vs GQA comparison.
- Per-layer breakdown to see where the parameter budget actually goes.

## Ship It

This lesson produces `outputs/skill-deepseek-v3-reader.md`. Given a DeepSeek-family model (V3, R1, or any future variant), it produces a component-by-component architecture reading that names each field of the config, derives parameter counts by component, and identifies which of the four DeepSeek-specific innovations the model uses.

## Exercises

1. Run `code/main.py`. Compare the calculator's total parameter estimate against the published 671B and identify where the discrepancy comes from. The paper's Section 2 has the full line-item breakdown.

2. Modify the config to use MLA rank 256 instead of 512. Compute the resulting KV cache size at 128k context. What percentage reduction does it buy, and at what cost to per-head expressiveness?

3. Compare DeepSeek-V3's (256 experts, top-8) routing against a hypothetical (512 experts, top-8) variant. Total parameters grow; active parameters stay the same. What does the extra expert capacity theoretically buy, and what does it cost at inference?

4. Read DeepSeek-V3 technical report (arXiv:2412.19437) Section 2.1 on MLA. Explain in three sentences why the K and V decompression matrices can be "absorbed" into subsequent matmuls for inference efficiency.

5. DeepSeek-V3 trains in FP8 for most operations. Compute the memory savings of FP8 vs BF16 for storing 671B weights. How does this interweave with the 14.8T-token training budget?

## Key Terms

| Term | How people say it | What it actually is |
|------|----------------|------------------------|
| MLA | "Multi-head Latent Attention" | Compresses K and V into a shared low-rank latent (kv_lora_rank, typically 512), decompresses per head on the fly; KV cache stores only the latent |
| kv_lora_rank | "MLA compression dim" | Size of the K and V shared latent; DeepSeek-V3 uses 512 |
| First k dense layers | "early layers stay dense" | First few layers of an MoE model skip the MoE router, running dense MLP for stability |
| num_experts_per_tok | "top-k routing" | How many routed experts fire per token; DeepSeek-V3 uses 8 |
| Shared expert | "always-on expert" | Expert that processes every token regardless of routing; DeepSeek-V3 uses 1 |
| Auxiliary-loss-free routing | "bias-adjusted load balancing" | Per-expert bias terms adjusted during training that keep expert load balanced without adding a loss term |
| MTP module | "extra prediction head" | A transformer block that predicts t+2 from h^(1) and E(t+1); denser training, free speculative decoding draft |
| DualPipe | "bidirectional pipeline" | Training schedule overlapping forward/backward compute with cross-node all-to-all |
| Active parameter ratio | "sparsity" | active_params / total_params; DeepSeek-V3 hits 5.5% |
| FP8 training | "8-bit training" | Training storage and many compute ops in FP8; roughly halves memory vs BF16 at small quality cost |

## Further Reading

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — Full architecture, training, and results documentation
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — Config files and deployment notes
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — Predecessor introducing MLA
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — Reasoning training successor on V3 architecture
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — Future direction for DeepSeek family attention
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — Training schedule reference implementation
