# Visual Autoregressive Modeling (VAR): Next-Scale Prediction

> Diffusion models iterate in time (denoising steps). VAR iterates in scale—it predicts a 1×1 token first, then 2×2, then 4×4, up to the final resolution, each scale conditioned on the previous ones. The 2024 paper showed VAR follows GPT-style scaling laws on image generation and beats DiT at matched compute budgets. This lesson builds the core mechanism.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## The Problem

Autoregressive generation dominates language modeling because it scales predictably: more compute, more parameters, lower perplexity, better outputs. Before 2024, image generation had two major AR attempts: PixelRNN/PixelCNN (per-pixel) and DALL-E 1 / Parti / MuseGAN (per-token on VQ-VAE codes).

Both suffered from a generation-order problem. Pixels and tokens live on a 2D grid, but AR models must visit them in 1D raster order. An early corner pixel has no idea what the image will ultimately become. Generation quality scaled worse than GPT-on-text, never reaching diffusion quality at matched compute.

VAR fixes the generation-order problem by changing *what* is generated. Instead of predicting image tokens one-by-one in space, VAR predicts an entire image at increasing resolutions. Step 1: predict a 1×1 token (the "summary" of the entire image). Step 2: predict a 2×2 token grid (coarser features). Step 3: predict a 4×4 grid. Step K: predict the final (H/8)×(W/8) grid.

Each scale attends to all previous scales (causal in "scale order") and is fully parallel within itself. The order problem vanishes: the entire image at scale k is produced in a single transformer forward pass.

## The Concept

### Multi-Scale VQ Tokenizer

VAR requires a **multi-scale discrete tokenizer**. For an image x, it produces a sequence of token grids at progressively higher resolutions:

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

Each z_k uses the same codebook (typically size 4096–16384). Tokenization across scales is not independent—it is trained so that summing each scale's residual reconstructs f:

```
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

This is a **residual VQ** variant. Scale k captures what scales 1..k-1 missed. The decoder takes the sum of all scale embeddings and produces the image.

The multi-scale VQ tokenizer is trained once (like VQGAN), then frozen. All generation work is done by the autoregressive model on top.

### Next-Scale Prediction

The generative model is a transformer that sees all previous-scale tokens and predicts the next scale's tokens.

Input sequence structure:
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Positional embeddings encode both the scale index and the spatial position within a scale. Attention is causal in scale order: a token at scale k, position (i, j) can attend to all tokens at scales 1..k, and within scale k itself to tokens earlier in whatever intra-scale ordering is used (VAR uses fixed positional attention with no intra-scale causality—all positions within a scale are predicted in parallel).

Training loss: at each scale k, predict z_k's tokens given all previous-scale tokens. Cross-entropy loss on discrete VQ codes. Structurally identical to GPT, but the "sequence" is now scale-structured.

### Generation

At inference:
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

For K = 10 scales, generation is 10 transformer forwards. Each produces its entire scale in parallel—no per-token autoregression within a scale. For a 256×256 image this is ~10 forwards vs DiT's 28–50.

### Why Next-Scale Beats Next-Token

Three structural wins:
1. **Coarse-to-fine matches natural image statistics.** Human visual perception and image datasets both exhibit scale-dependent regularity: low-frequency structure is stable and predictable; high-frequency detail is conditioned on low-frequency content. Next-scale prediction exploits this.
2. **Intra-scale parallel generation.** Unlike GPT-style token AR, VAR produces all tokens at a scale in one step. Effective generation length is logarithmic rather than linear.
3. **No generation-order bias.** Tokens at scale k see the full scale k-1; there's no "left-of" or "above" bias forcing early tokens to commit before later context is available.

### Scaling Laws

Tian et al. showed VAR's FID on ImageNet follows a power-law scaling curve—just like GPT for perplexity. Doubling parameters or compute reliably halves error. This is the first image generation model to exhibit this kind of clean scaling behavior like language models. The consequence is that VAR scale predictions become inferable from compute, rather than empirical per-architecture guesses.

### Relationship to Diffusion

VAR and diffusion share the same data-compression story: both factor generation into a sequence of easier subproblems.

- Diffusion: progressively add noise, learn to undo one step.
- VAR: progressively add resolution, learn to predict the next scale.

They are different axes through the same problem. Both yield tractable conditional distributions. Empirically VAR is faster at inference (fewer forwards, full intra-scale parallelism) and matches or beats DiT on class-conditional ImageNet. Text-conditioned VAR (VARclip, HART) is active research.

## Build It

In `code/main.py` you will:
1. Build a mini **multi-scale VQ tokenizer** on synthetic "image" data (2D Gaussian rings).
2. Train a **VAR-style transformer** to next-scale-predict those tokens.
3. Sample by calling the transformer 4 times (4 scales) and decoding.
4. Verify the scale-ordered training enables intra-scale parallel generation.

This is a toy implementation. The point is to see the scale-structured attention mask and intra-scale parallel generation actually working.

## Ship It

This lesson produces `outputs/skill-var-tokenizer-designer.md`—a skill for designing multi-scale tokenizers: number of scales, scale ratios, codebook size, residual sharing, decoder architecture.

## Exercises

1. **Scale count ablation.** Train VAR with 4, 6, 8, 10 scales. Measure reconstruction quality vs autoregressive forward count. More scales = finer residuals = better quality but more forwards.

2. **Codebook size.** Train the tokenizer with codebook sizes 512, 4096, 16384. Larger codebook gives better reconstruction but harder prediction. Find the inflection point.

3. **Intra-scale parallelism check.** For a trained VAR, explicitly measure attention patterns. Within scale k, does the model attend to cross-scale positions but not intra-scale? Verify the mask implementation.

4. **VAR vs DiT scaling.** For the same ImageNet class-conditional task, train VAR and DiT at matched parameter budgets (e.g., 33M, 130M, 458M). Plot FID vs compute. VAR should lead DiT at each size—reproduce the paper's result at small scale.

5. **Text conditioning.** Extend VAR to accept a text embedding (CLIP pooled) as additional conditioning via adaLN. This is the HART recipe. How much does FID improve on text-aligned samples?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| VAR | "visual autoregressive" | Generate images via next-scale prediction over a pyramid of VQ token grids |
| Next-scale prediction | "predict coarse first, then fine" | Model predicts tokens at increasing resolution scales, conditioned on all prior scales |
| Multi-scale VQ tokenizer | "residual VQ" | VQ-VAE producing K token grids at increasing resolutions, decoder sums all scales |
| Scale k | "pyramid level k" | One of K resolution levels, from k=1 at 1×1 to k=K at (H/p)×(W/p) |
| Intra-scale parallelism | "one forward per scale" | All tokens at scale k are predicted in a single transformer forward, not autoregressively |
| Inter-scale causality | "scale-order attention" | Tokens at scale k attend to all of scales 1..k but not scales k+1..K |
| Residual VQ | "additive tokenization" | Each scale's tokens encode the residual left by lower scales; decoder sums all scale embeddings |
| VAR scaling law | "image GPT scaling" | FID follows a predictable power law in compute, like language model perplexity |
| HART | "hybrid VAR + text" | Text-conditioned VAR variant combining MaskGIT-style iterative decoding with VAR's scale structure |
| Scale positional embedding | "(scale, row, col) triple" | Positional encoding carries both the scale index and within-scale spatial coordinates |

## Further Reading

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — the VAR paper, authoritative reference
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT, the diffusion comparison baseline
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN, the tokenizer family that VAR's multi-scale tokenizer extends
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE, foundation for discrete image tokenization
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — text-conditioned VAR
