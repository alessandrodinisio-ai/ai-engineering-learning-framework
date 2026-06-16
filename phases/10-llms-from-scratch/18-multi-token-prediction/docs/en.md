# Multi-Token Prediction (MTP)

> From GPT-2 to Llama 3, every autoregressive LLM trains one loss per position: predict the next token. DeepSeek-V3 adds a second loss per position: predict the token after next. Those extra 14B parameters (on a 671B model) get distilled back into the main model through gradient flow, and the trained MTP head is repurposed at inference as a speculative decoding drafter with 80%+ acceptance rate. 1.8x generation throughput for free. This lesson builds the sequential MTP module from the DeepSeek technical report, computes the loss and parameter layout of the shared head, and explains why MTP preserves the causal chain while Gloeckle et al.'s original parallel MTP breaks it.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pretraining a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Learning Objectives

- State the MTP training objective and derive the joint loss across prediction depths.
- Explain the difference between Gloeckle et al.'s parallel MTP heads (2024) and DeepSeek-V3's sequential MTP module, and why the sequential design preserves the causal chain.
- Compute the parameter and memory overhead of adding MTP modules to a pretraining run.
- Implement an MTP module from scratch: shared embedding, per-depth transformer block, projection, and shared output head.

## The Problem

Next-token prediction is the standard LLM training objective. Each hidden state is supervised to predict exactly one thing: the immediately following token. That is a surprisingly weak signal. Most information in a sequence extends beyond a single token—structure, coherence, factuality, arithmetic flow. The model must learn these by accumulating many single-token signals over trillions of tokens.

MTP asks: what if each hidden state were supervised to predict multiple future tokens at once? Gloeckle et al. (Meta, 2024) showed this helps. Their implementation puts several independent output heads on the backbone, each predicting a different offset. Parallel, simple, but those heads see the same hidden state with no hierarchical refinement—and predictions aren't causally linked between them, so they can't be used for speculative decoding.

DeepSeek-V3 (December 2024) redesigns MTP as a sequential module that preserves the causal chain at each prediction depth. The model predicts `t+1` from `h_i^(0)`, then predicts `t+2` from a new hidden state `h_i^(1)` that combines `h_i^(0)` and the `E(t+1)` embedding, and so on. Each depth is its own small transformer block. Shared embedding and shared output head keep parameter overhead modest. At DeepSeek-V3's scale, 14B extra parameters across MTP modules on top of 671B main model weights. That 2% overhead buys denser training signal *and* a ready-made speculative decoding draft at inference.

This lesson builds the single MTP module and D-depth loss from scratch. The math is clean. The implementation is 150 lines.

## The Concept

### The sequential MTP recipe

DeepSeek-V3 adds `D` MTP modules on top of the main model. Each module `k` (`k = 1..D`) predicts the token at depth `k`—that is, `t_{i+k}` given the prefix up to position `i`.

Module `k` consists of:

- A transformer block `T_k` with its own attention and MLP.
- A projection matrix `M_k` that combines the previous depth's hidden state with the embedding of the next-depth ground-truth token.
- Shared embedding `E` (same as main model).
- Shared output head `Out` (same as main model).

During training, for a prefix up to position `i`, the per-depth hidden states are:

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Per-depth predictions are:

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

Per-depth loss is cross-entropy against ground truth `t_{i+k}`:

```
L_k = CE(logits_{i+k}, t_{i+k})
```

Joint loss across depths:

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` is a small weighting factor—DeepSeek-V3 uses 0.3 for the first 10% of training, then 0.1. Total training loss is `L_main + L_MTP`.

### Why sequential, not parallel

Gloeckle's original parallel MTP has D output heads each applied directly on `h_i^(0)`. Each head predicts `t_{i+k}` from the same backbone hidden state. That trains fine, but predictions don't condition on each other. You can't use `head_1`'s output to help `head_2`—the heads fire in parallel.

DeepSeek-V3's sequential design builds `h_i^(k)` from `h_i^(k-1)` plus the actual next-token embedding `E(t_{i+k})`. That preserves the causal chain: to predict `t_{i+k+1}`, the depth-`k+1` module sees what's at `t_{i+k}`. This is structurally identical to an autoregressive decoder consuming its own outputs—enabling the MTP module to be used directly as a speculative decoding drafter.

At inference: feed `h_i^(k-1)` and the drafted `t_{i+k}` into module `k+1` to get a prediction for `t_{i+k+1}`. Repeat. That's exactly an EAGLE-style draft using the trained MTP module as the draft network. DeepSeek-V3 reports 80%+ acceptance rate on the first MTP module and ~1.8x speedup.

### Parameter accounting

For a model with hidden size `h` and vocabulary `V`:

- Main model: billions of parameters plus one output head of size `V * h`.
- Shared output head: reuses the main model's head. No extra parameters.
- Shared embedding: reuses the main model's embedding. No extra parameters.
- Per MTP module:
  - Projection `M_k`: `(2h) * h = 2h^2`.
  - Transformer block `T_k`: attention (`4h^2` for MHA) plus MLP (typically `8h^2` at SwiGLU ratio 8/3). ~`12h^2` per block.

Total extra per module: `~14h^2`. For DeepSeek-V3's `h = 7168`, D = 1 module: on paper `~14 * 7168^2 = ~720M` parameters. DeepSeek-V3 reports 14B—the difference is mostly because the MTP module's expert layers are also MoE.

### The speculative decoding payoff

During pretraining, MTP modules slow training by ~10% (more forward compute, extra loss). The payoff is twofold:

1. Denser training signal. Each hidden state sees D+1 supervision targets. Measured effect on MMLU, GSM8K, MATH, HumanEval: consistent few-percent improvements in DeepSeek-V3's ablations.

2. Free speculative decoding draft at inference. The MTP module is already trained to predict the next several tokens. Repurposed as a draft network, it delivers 80%+ acceptance rate. At that level, N=3 or N=5 spec decoding gives 1.8x throughput. The 10% training-time cost pays for itself the first time you run inference.

### Relationship to EAGLE

EAGLE trains a small draft model *after* pretraining, separately. MTP bakes the draft into pretraining. Both approaches converge to similar acceptance rates but via different pipelines:

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pretraining | During pretraining |
| Backward compatible with existing weights | Yes | No (requires retraining) |
| Draft parameters | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | 0.80+ at depth 1 |
| Benefit beyond speedup | Speculative decoding only | Denser training signal + speedup |

## Build It

`code/main.py` builds a single MTP module end-to-end: shared embedding, projection, transformer block, shared output head. It then computes per-depth cross-entropy loss on a short synthetic sequence and prints parameter counts by component. A 32-token toy vocabulary keeps numbers readable.

### Step 1: Shared embedding table

A single `vocab_size x hidden` table used by both the main model *and* each MTP module at every depth. Not a second copy—literally the same tensor.

### Step 2: Per-depth combination

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

Real DeepSeek-V3 concatenates the two RMSNorm'd vectors into `[2h]` and projects with an `h x 2h` matrix. The toy uses vector addition for stdlib brevity.

### Step 3: Transformer block at depth k

Self-attention plus MLP. In the toy, a single-layer linear attention block and a SwiGLU MLP keep structure visible without numpy.

### Step 4: Shared output head

Reuses the main model's output projection. Logits over vocabulary.

### Step 5: Per-depth loss

Cross-entropy of softmax(logits) against the ground-truth token at offset `k`. Aggregated across depths with the `lambda / D` scaling factor.

### Step 6: Parameter accounting

Print total parameters, shared (embedding, head) counts, and per-module extra counts. Show the ratio of MTP extra to main model size.

## Use It

MTP is integrated in DeepSeek-V3 (December 2024) and the DeepSeek-R1 family. At inference:

- DeepSeek's own serving stack consumes MTP modules as speculative decoders out of the box.
- As of April 2026, vLLM and SGLang have integration paths for DeepSeek-V3 MTP.
- AMD's ROCm SGLang tutorial shows a concrete MTP speculative decoding configuration with measured 1.8x speedup on V3 checkpoints.

When to use MTP in a new pretraining run:

- You control the full pretraining pipeline and want to bank denser training signal.
- You know you'll serve the model at scale and want free speculative decoding.
- Your hidden size is at least 4096. At 1B scale the overhead hurts more than the gains help.

When not to use:

- Fine-tuning an existing pretrained dense model. The MTP module wasn't trained.
- A research model you want as a clean baseline for comparison. MTP changes the architecture.

## Ship It

This lesson produces `outputs/skill-mtp-planner.md`. Given a pretraining run spec (model size, data, compute), it returns an MTP integration plan: number of depths D, `lambda` schedule, memory overhead, and speculative decoding wiring at inference.

## Exercises

1. Run `code/main.py`. Show that per-depth loss decreases monotonically with synthetic signal enhancement. Alter the synthetic to use a fixed pattern and verify both depth-1 and depth-2 losses converge.

2. Compute the parameter overhead of a D=1 MTP module for a dense 70B model (hidden 8192, 80 layers). Compare against DeepSeek-V3's reported 14B overhead. Explain why DeepSeek's number is higher: the MTP transformer block inherits the same MoE structure, inflating per-module parameter count.

3. Implement D=2 in the toy: add a second MTP module that takes h^(1) and predicts `t_{i+2}`. Verify the joint loss and parameter accounting match the DeepSeek paper's equations 19-21.

4. Switch the toy to parallel MTP (Gloeckle-style): add D output heads on top of the main hidden state, each predicting a different offset. Measure how per-depth loss on the same synthetic signal compares against the sequential version. The sequential version should produce lower depth-k loss for k > 1, because it conditions on intermediate predictions.

5. Use the trained MTP module as an EAGLE-style draft: at inference, call module k to propose `t_{i+k}`. Measure the acceptance rate of these draft tokens against the main model's predictions on a held-out sequence. If you hit 50%+ on the toy, you've reproduced the empirical property of MTP-as-draft.

## Key Terms

| Term | How people say it | What it actually is |
|------|----------------|------------------------|
| MTP module | "extra loss block" | A small transformer block plus projection that predicts a token `k` positions ahead of the main model |
| Prediction depth | "which offset" | Integer `k` such that module `k` predicts `t_{i+k}` from a prefix up to position `i` |
| Parallel MTP | "Gloeckle-style" | D independent heads on the same backbone hidden state, no causal chain |
| Sequential MTP | "DeepSeek-V3 style" | Each module conditions on the previous depth's hidden state plus the next token's embedding; preserves the causal chain |
| Shared output head | "reuse the main head" | MTP modules call the main model's LM head, not a separate output projection |
| Shared embedding | "reuse the main table" | Same vocabulary embedding table used everywhere; no duplicate parameters |
| Projection matrix M_k | "combine hidden + next token" | An `h x 2h` linear layer that folds the previous hidden state and target token embedding into the next depth's input |
| Joint loss L_MTP | "averaged extra loss" | Arithmetic mean of per-depth cross-entropy losses, scaled by `lambda` |
| Acceptance rate at depth 1 | "how often the MTP draft is right" | Fraction of times the D=1 MTP module's top-1 prediction equals the main model's top-1 prediction; 80%+ on DeepSeek-V3 |
| Lambda weighting | "importance of the extra loss" | Per-depth scaling factor; 0.3 at training start, 0.1 afterward on DeepSeek-V3 |

## Further Reading

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — Full sequential MTP description (Section 2.2), with joint loss equations and 1.8x inference speedup
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — The parallel MTP baseline that DeepSeek's design improves upon
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — Total 685B (671B main + 14B MTP), deployment notes
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — The speculative decoding framework that MTP plugs into
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — EAGLE's 2025 draft architecture, the competitor to MTP
