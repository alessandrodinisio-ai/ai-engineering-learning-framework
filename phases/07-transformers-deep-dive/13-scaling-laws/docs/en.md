# Scaling Laws

> The 2020 Kaplan paper said: bigger models, lower loss. The 2022 Hoffmann paper said: you're undertrained. Compute goes into two buckets — parameters and tokens — and the split isn't obvious.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 min

## The Problem

When you have C FLOPs of training compute and want the best model, you face two knobs:

1. **How many parameters (N)?** Bigger model, higher capacity.
2. **How many training tokens (D)?** More data, better use of capacity.

FLOPs grow roughly as `6 × N × D`. You can push N high and keep D low, or push D high and keep N low. Which is better?

Before 2022, the answer was "push N hard." GPT-3 (2020) was 175B parameters trained on ~300B tokens. A ratio of ~1.7 tokens per parameter. Kaplan scaling laws supported this.

Hoffmann et al. (2022) trained a small family of models called Chinchilla and found something different: the optimal ratio is closer to **20 tokens per parameter**. GPT-3 was undertrained by 10×. Chinchilla (70B params, 1.4T tokens) beat GPT-3 (175B, 300B tokens) on every benchmark at 2.5× lower inference cost.

2026 is Chinchilla's world — with an important twist. Llama 3 8B was trained on 15 trillion tokens, a ratio of 1,875 tokens per parameter. 94× beyond Chinchilla-optimal. For models that will be served at massive scale, inference cost matters more than training cost, so overtraining (beyond Chinchilla) for a smaller deployable size is the 2026 default.

## The Concept

![Chinchilla curve: loss vs compute at different N/D ratios](../assets/scaling-laws.svg)

### The Hoffmann Law

From the Chinchilla paper, loss follows:

```
L(N, D) = A / N^α + B / D^β + E
```

- `N` = parameters (non-embedding).
- `D` = training tokens.
- `α ≈ 0.34`, `β ≈ 0.28` (roughly symmetric).
- `E ≈ 1.69`, the irreducible loss floor.
- `A ≈ 406`, `B ≈ 411`.

As you scale, the two terms trade off. Differentiate L w.r.t. `N` under fixed compute (C = 6ND) and solve:

```
N_opt ≈ 0.6 × (C/6)^0.5
D_opt ≈ 0.6 × (C/6)^0.5
D_opt / N_opt ≈ 20
```

Compute-optimal: 20 tokens per parameter.

### Why Overtrain Anyway

Chinchilla-optimal minimizes training loss per training FLOP. But training cost is paid once; inference cost is paid forever.

For a chatbot serving a trillion tokens per month, inference dominates total cost. Llama's approach: train smaller, train longer. 8B with 15T tokens is a deep inference optimization:

- Fits on consumer GPUs.
- Latency is a fraction of a 70B Chinchilla-optimal model.
- Quality is close enough for most tasks.

DeepMind's 2024 paper ("Overtraining is the new optimal") formalizes this. For inference-dominated workloads, the right ratio is closer to 100–500 tokens per parameter, depending on serving volume.

### Emergence vs Smoothness

There's a claim that certain capabilities (arithmetic, multi-step reasoning, chain-of-thought following) "emerge" at a certain scale, appearing suddenly.

Schaeffer et al. (2023) argued this is a measurement artifact: emergence metrics use discontinuous scoring (exact match, threshold accuracy), which masks smooth improvement in the underlying logits. Continuous metrics (cross-entropy) show smooth curves.

The 2026 consensus: predictions via continuous loss are reliable. Jumps on benchmarks are often scorer artifacts. Budget by continuous metrics.

### The 2026 Landscape

Scaling laws still hold, but:

| Factor | What changed |
|--------|-------------|
| Data quality | Curated "good" tokens (Phi-style) shift the curve by >2× effective compute |
| MoE | Total params and active FLOPs decouple; scaling laws apply per active FLOP |
| Post-training | Some capabilities (instruction following, code) shift more with SFT+RLHF than pretraining |
| Multimodal | Image + text tokens scale together; separate curves per modality |
| Synthetic data | Models generate training data; effective compute can compound |

The Muon optimizer (Kimi Moonlight, 2024) shows ~2× effective compute gain over AdamW on equal data. Some 2026 training runs default to Muon. It changes the absolute constants in the scaling law, not its shape.

## Build It

See `code/main.py`. We implement the Chinchilla loss equation and solve for compute-optimal `(N, D)` at several compute budgets.

### Step 1: Chinchilla Loss

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

Plot `L` as contours over `(N, D)` at fixed `C = 6ND`. Find the minimum.

### Step 2: Compute-Optimal Frontier

For compute budgets from `1e17` to `1e25` FLOPs, find the `(N, D)` that minimizes loss under the constraint `6ND = C`. Verify the ratio `D/N ≈ 20`.

### Step 3: The Cost of Overtraining

Compute the extra loss from training a model 10× smaller (1/10 of optimal N, 10× optimal D). Report the inference FLOP savings gained in exchange (proportional to N).

### Step 4: Compare Against Real Models

Plug in the known `(N, D)` pairs for GPT-3, Chinchilla, Llama 3 8B, and DeepSeek-V3 (active params) and compare predicted loss vs reported loss.

## Use It

You're unlikely to train a frontier model yourself. But scaling laws tell you:

1. **Whether your fine-tuning data is enough.** If your task-specific data is below 20 tokens per base-model parameter, expect saturation at some loss floor.
2. **Whether to pick a larger base model.** If you spend your entire budget on inference, bias toward smaller, longer-trained models.
3. **Where diminishing returns kick in.** Beyond 1000× Chinchilla-optimal, log-loss changes become noise.

**2026 research trajectories:**

- **Data-constrained regime.** The number of high-quality tokens on the web is finite (~5–10 trillion English tokens after filtering). Frontier pretraining is approaching this ceiling. Synthetic data, multilingual, multimodal, and RLHF-scaled fine-tuning are the next levers.
- **Compute multiplier tricks.** Muon optimizer, MoE, better data curation — each shifts absolute constants, not asymptotics.
- **Scaling laws for RL.** Open question. Early evidence suggests power laws on RL samples, but exponents differ significantly from pretraining.

## Ship It

See `outputs/skill-training-budget-estimator.md`. This skill picks `(N, D, hours, GPU)` for a new training run given a compute budget, deployment constraints, and target loss.

## Exercises

1. **Easy.** Run `code/main.py`. Print Chinchilla-optimal `(N, D)` for compute budgets `1e20`, `1e22`, `1e24`. Compare against the real-model table.
2. **Medium.** Implement the Hoffmann loss as a function of compute curve. Plot loss vs `log10(C)` for the compute-optimal frontier. Find the point where the law predicts we need `>10^28` FLOPs to drop cross-entropy by another 0.1.
3. **Hard.** Fit your own scaling law on 5 very small models (100K to 10M params) trained on the same dataset. Estimate `α` and `E`. How do your exponents compare to published values?

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| Parameters (N) | "model size" | Non-embedding weight count; determines capacity. |
| Tokens (D) | "training data" | Number of training tokens seen; determines how well params are utilized. |
| Compute (C) | "FLOPs spent" | Roughly `6 × N × D` for a standard transformer. |
| Chinchilla-optimal | "D/N ≈ 20" | The ratio that minimizes loss per pretraining FLOP. |
| Overtraining | "beyond Chinchilla" | Spending more training FLOPs to save inference FLOPs; D/N >> 20. |
| Irreducible loss | "the floor" | The `E` term in the scaling law; entropy of the data itself. |
| Emergent capabilities | "sudden jumps at scale" | Often a scorer artifact; continuous loss is smooth. |
| Effective compute | "training efficiency multiplier" | Better data / optimizer / architecture makes each FLOP go further. |

## Further Reading

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — The first scaling-law paper; undertrained.
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — Emergence as measurement artifact.
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — Why Llama's overtraining is correct for its workload.
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — The 2× compute multiplier.
