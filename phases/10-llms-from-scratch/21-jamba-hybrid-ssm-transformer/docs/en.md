# Jamba — Hybrid SSM-Transformer

> State space models (SSMs) and transformers want different things. Transformers trade quadratic-cost attention for quality. SSMs trade a recurrence for linear-time inference and constant memory, but quality lags. AI21's Jamba (March 2024) and Jamba 1.5 (August 2024) put them in the same model: one Transformer layer for every 7 Mamba layers, MoE every other block, and a 256k context window that fits on a single 80GB GPU. Mamba-3 (ICLR 2026) tightens the SSM side with complex-valued state spaces and MIMO projections. This lesson reads both architectures end-to-end and explains why the hybrid recipe has survived three years of scaling while pure-SSM and pure-Transformer long-context attempts have not.

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## Learning Objectives

- Explain the three primitives in a Jamba block—Transformer layer, Mamba layer, MoE—and the 1:7:alternating recipe.
- State at a high level what an SSM's recurrence looks like and why it enables constant-memory inference.
- Compute a Jamba model's KV cache footprint at 256k context compared to what a pure Transformer model would require.
- Name three Mamba-3 innovations (exponential-trapezoidal discretization, complex-valued state updates, MIMO) and which problem each targets.

## The Problem

Attention is quadratic in sequence length. State space models are linear. This difference compounds: at 256k tokens, a Transformer attention map has 65 billion entries per head; an SSM's recurrence state is fixed-size regardless of sequence length.

Pure SSM models (Mamba, Mamba-2) match Transformer perplexity at small scale but lag on state-tracking tasks and fail on certain classes of in-context retrieval. Intuition: SSMs compress history into a fixed-size state, and information leaks when history is long. Attention remembers everything precisely, but pays quadratic cost.

The obvious fix: use both. Place Transformer layers where precise recall matters. Use SSM layers everywhere else. Tune the ratio. Jamba is the first production-scale model to ship this hybrid recipe at scale (52B total, 12B active, 256k context, single 80GB GPU). Jamba 1.5 expands the family to 398B total / 94B active. Mamba-3 (ICLR 2026) is the current best pure-SSM baseline that hybrid models can rebuild around.

This lesson reads all three papers and produces the mental model for "picking the right ratio."

## The Concept

### SSM in one page

A state space model processes a sequence `x_1, ..., x_N` through a fixed-size state `h`:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

Each step the state evolves through a linear dynamics `A`, eats input `B x_t`, and emits output `C h_t`. `A, B, C` can be learned. Note the key property: computing `y_t` requires only `h_{t-1}` and `x_t`, not any earlier `x`. Memory is constant. Inference is O(1) per token.

The trick for modeling quality is in the structure of `A`. S4 (Gu 2021) used a highly structured matrix that can be efficiently evaluated as a long convolution during training. Mamba (Gu, Dao 2023) replaced fixed `A, B, C` with data-dependent ones (the "selective" part). Mamba-2 (2024) simplified the structure further. Mamba-3 (2026) adds back complexity in specific places.

Key property: for a decoder LLM, an SSM layer is a drop-in replacement for an attention layer, using a fixed-size per-layer state instead of a growing KV cache.

### The Jamba block

A Jamba block interleaves layers according to two numbers:

- `l`: the attention-to-Mamba ratio. Jamba uses `l = 8`, meaning 1 Transformer layer for every 7 Mamba layers (each group of 7 Mamba + 1 Attention = 8 layers).
- `e`: MoE frequency. Jamba uses `e = 2`, meaning MoE is applied every other layer.

Layer sequence within a block:

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (| marks where MoE is applied)
```

Each Jamba block is 8 layers. At 4 blocks deep (32 layers total), you get 28 Mamba and 4 Attention layers. 16 of them use MoE.

### Why the 1:7 ratio

AI21 ran ablations: which attention-to-Mamba ratio gives the best per-parameter perplexity *and* in-context recall on their long-context evals?

- Too much attention (1:1): quality climbs but memory and speed degrade.
- Too little attention (1:15): memory is excellent but in-context retrieval fails.
- Sweet spot: 1:7 or 1:8.

Intuition: Transformer layers handle precise recall and state tracking. Mamba layers handle cheap bulk processing.

### Positional encoding

Mamba layers are inherently position-aware (through the recurrence). Attention layers in early Mamba-based hybrids didn't use RoPE—the SSM layers provide positional information. Jamba 1.5 adds RoPE to attention layers for longer-context generalization, a post-hoc improvement based on empirical long-context evaluation.

### Memory budget

For a Jamba-1 shape (32 layers: 28 Mamba + 4 Attention, hidden 4096, 32 attention heads):

- KV cache (attention layers only): `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB` at 256k in BF16. Only those 4 attention layers contribute.
- SSM state: `28 * hidden * state_size` per token prefix, but this is fixed-size per layer, doesn't grow with sequence length. Typical Mamba state at 16 per feature, hidden 4096: total `28 * 4096 * 16 * 2 = 3.7 MB`.

Compare against a pure Transformer with 32 layers, same hidden, 32-head full MHA: `2 * 32 * 32 * 128 * 256k * 2 = 128 GB` at 256k in BF16. 8x KV cache reduction. Even comparing against the GQA(8) baseline most 2024 models use (`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`), Jamba's 1:7 mix at 16 GB is still 2x smaller.

This is what AI21 means by "256k context on a single 80GB GPU." A full-MHA pure Transformer's KV cache can't fit; even the GQA baseline doesn't leave room for weights and activations; Jamba's does.

### Mamba-3: The 2026 pure-SSM baseline

Mamba-3 (ICLR 2026, arXiv:2603.15569) introduces three innovations on the pure-SSM side:

1. **Exponential-trapezoidal discretization.** Replaces the Euler-method discretization in Mamba-2 with a more expressive recurrence. Applies a convolution-like operation to state-input within the core recurrence, rather than as an outer convolution on `x_t`.

2. **Complex-valued state updates.** Prior Mamba reduced the state matrix from complex (S4) to real diagonal (Mamba) to scaled identity (Mamba-2). Mamba-3 adds complex values back—equivalent to a data-dependent rotary embedding on the state. This recovers state-tracking capabilities lost by prior real-valued simplifications.

3. **Multi-input multi-output (MIMO) projections.** Instead of per-feature scalar projections, uses matrix-valued projections. Improves modeling capacity and inference-time hardware utilization without increasing decode latency.

At 1.5B parameters, Mamba-3 improves average downstream accuracy over Gated DeltaNet by 0.6 points; the MIMO variant adds 1.2 more, totaling 1.8 points of improvement. At the same state size, Mamba-3 matches Mamba-2 with half the state.

Mamba-3 hasn't shipped in a large-scale production hybrid yet—but it's the obvious candidate for the SSM side of the next Jamba-class model.

### When the hybrid wins

The hybrid wins when:

- Context is long enough that pure Transformer KV cache becomes painful (64k+).
- Task mix has both short-range structure (good for SSM) and long-range recall (needs Transformer).
- You want to deploy on a single-GPU memory budget where Transformer KV cache alone can't fit.

The hybrid loses when:

- Context is short (under 16k). SSM overhead is wasted; pure Transformer is fine.
- Task requires everywhere-to-everywhere attention (deep reasoning, multi-document cross-reference). The sparsity of attention layers in the hybrid hurts.
- You're scaling to trillion-parameter frontier models. Pure Transformer + MLA + MoE (DeepSeek-V3 style) is currently winning the capability race.

### Competitive landscape

| Model | Family | Scale | Unique selling point |
|-------|--------|------|-------------|
| Mamba-2 | Pure SSM | 3B | Linear time, constant memory |
| Jamba | Hybrid | 52B/12B | 256k on 80GB |
| Jamba 1.5 Large | Hybrid | 398B/94B | Enterprise-grade long context |
| Mamba-3 | Pure SSM | 1.5B (paper) | State tracking recovered |
| DeepSeek-V3 | Pure Transformer + MoE | 671B/37B | Frontier capability |

2026 landscape: pure Transformer MoE dominates frontier, but hybrids occupy the 256k+ context niche. Mamba-3's state-tracking wins may push the hybrid ratio lower (more SSM, less attention) in the next generation.

## Use It

`code/main.py` is a memory calculator for hybrid architectures. Given an SSM-Transformer ratio and a hidden-size / layer-count configuration, it computes:

- KV cache at target context.
- SSM state memory.
- Total memory for a series of model shapes at context N.

The calculator supports:

- Pure Transformer baseline (KV cache grows with N).
- Jamba-style 1:7 hybrid.
- Pure SSM (no KV cache at all).

Numbers are direct from the Jamba-1 and Jamba-1.5 papers for published shapes, and extrapolated for hypothetical variants.

Integration considerations for real deployment:

- Most production inference servers (vLLM, SGLang) support Jamba and Mamba. Check specific versions.
- At 256k context, Jamba's memory advantage manifests in concurrent request throughput. Same VRAM fits more Jamba sequences than Transformer sequences.
- Mamba-3 as a standalone model is not yet in production—1.5B research preview.

## Ship It

This lesson produces `outputs/skill-hybrid-picker.md`. Given a workload spec (context length profile, task mix, memory budget), it recommends between pure Transformer, Jamba-style hybrid, and pure SSM, with explicit reasoning about memory and quality tradeoffs.

## Exercises

1. Run `code/main.py` to compute the KV cache for a 32-layer pure Transformer (hidden 4096, 32 heads) and a same-shape Jamba-1 hybrid at 256k context. Verify the ~8x memory reduction claimed in AI21's paper.

2. Modify the calculator to model a 1:3 hybrid (4 Mamba : 1 Attention) and a 1:15 hybrid (14 Mamba : 1 Attention). Plot KV cache vs ratio. At what ratio does the KV cache equal the SSM state memory?

3. Read the Jamba paper (arXiv:2403.19887) Section 3. Explain why AI21 used Mamba-1 instead of Mamba-2, despite Mamba-2 being faster. Hint: the hybrid ablation section documents this.

4. Compute the parameter overhead of MoE-every-other-layer in Jamba 1.5 Large (398B total, 94B active). Compare the active ratio against DeepSeek-V3's (37B/671B) and explain why Jamba's architecture pushes the active ratio higher.

5. Read the Mamba-3 paper (arXiv:2603.15569) Section 3. Explain in three sentences why the complex-valued state update is equivalent to a data-dependent rotary embedding. Connect the answer to Phase 7 · Lesson 04's RoPE derivation.

## Key Terms

| Term | How people say it | What it actually is |
|------|----------------|------------------------|
| State space model (SSM) | "recurrence with fixed-size state" | A layer with learned recurrence `h_t = A h_{t-1} + B x_t`; constant memory per token |
| Selective SSM | "Mamba's trick" | Data-dependent A, B, C parameters giving the model gate-like selectivity in linear time |
| Attention-to-Mamba ratio | "how many attention layers" | In Jamba, `l = 8` means 1 attention layer per 7 Mamba layers |
| Jamba block | "the 8-layer group" | One attention + seven Mamba + MoE at alternating positions |
| SSM state | "the hidden buffer" | Fixed-size per-layer state that Mamba layers use in place of KV cache |
| 256k context | "Jamba's headline number" | The sequence length Jamba-1 can fit on a single 80GB GPU; pure Transformer can't at that size |
| Mamba-3 | "2026 pure SSM" | Current best pure-SSM architecture with complex state + MIMO; baseline hybrid models rebuild around |
| MIMO | "multi-input multi-output" | Mamba-3's innovation using matrix-valued projections instead of per-feature scalars |
| Exponential-trapezoidal discretization | "Mamba-3's recurrence" | A more expressive recurrence that subsumes Mamba-2's Euler-method discretization |
| Hybrid architecture | "mixing attention and SSM" | Any model interleaving Transformer and SSM layers; Jamba is the production archetype |

## Further Reading

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — Original Jamba paper, ratio ablations, 256k context claim
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — Scaled family, 398B/94B and 12B/52B public releases
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — The selective SSM paper Jamba builds on
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — Simplified structured state space successor
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — Complex-valued state, MIMO, 2026 pure-SSM frontier
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — The S4 paper, starting point of the SSM lineage for LLMs
