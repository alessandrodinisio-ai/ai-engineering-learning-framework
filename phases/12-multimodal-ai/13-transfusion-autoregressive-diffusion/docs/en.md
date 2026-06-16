# Transfusion: Autoregressive Text + Diffusion Images in One Transformer

> Chameleon and Emu3 bet everything on discrete tokens. They work, but the quantization bottleneck is visible — image quality plateaus below continuous-space diffusion models. Transfusion (Meta, Zhou et al., August 2024) bets the other way: keep images continuous, throw away VQ-VAE entirely, train one transformer with two losses. Text tokens use next-token prediction. Image patches use flow-matching / diffusion loss. Both objectives optimize the same set of weights. The architecture underlying Stable Diffusion 3 (MMDiT) is a close relative. This lesson reads through the Transfusion argument, builds a toy-scale dual-loss trainer, and traces the attention mask that lets one transformer do two jobs simultaneously.

**Type:** Build
**Languages:** Python (standard library, dual-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## Learning Objectives

- Wire up a transformer running two losses on one backbone (NTP on text tokens, diffusion MSE on image patches).
- Explain why bidirectional attention across image patches plus causal attention across text tokens is the correct mask choice.
- Compare Transfusion-style (continuous images, diffusion loss) vs Chameleon-style (discrete images, NTP) on compute, quality, and code complexity.
- State MMDiT's contributions: per-block modality-specific weights, joint attention on the residual stream.

## The Problem

The discrete vs continuous image token debate is older than LLMs. Continuous representations (raw pixels, VAE latents) preserve detail. Discrete tokens (VQ indices) fit the transformer's native vocabulary but lose detail at the quantization step.

Chameleon / Emu3 went discrete: one loss, one architecture, but image fidelity is capped by tokenizer quality.

Diffusion models went continuous: image quality is superior, but it's a model separate from the LLM, requiring complex noise schedule engineering and unable to integrate cleanly with text generation.

Transfusion asks: can we have both? Keep images continuous, still train one model, with two losses stitched into the same gradient step.

## The Concept

### Dual-Loss Architecture

A single decoder-only transformer processes a sequence containing:

- Text tokens (discrete, from BPE vocabulary).
- Image patches (continuous, 16x16 pixel blocks linearly projected to hidden dimension — same as ViT encoder input).
- `<image>` and `</image>` tags marking where continuous patches live.

One forward pass runs. The loss picks one of two heads for each token:

- For text tokens: standard cross-entropy on the vocabulary logit head.
- For image patches: diffusion loss on continuous patches — predicting the noise added to each patch.

Gradients flow through the shared transformer body. Both losses improve the shared weights simultaneously.

### Attention Mask: Causal Text + Bidirectional Image

Text tokens must be causal — you can't let a text token attend to future text, or teacher forcing breaks. Image patches represent a snapshot; they should attend bidirectionally to each other within the same image block.

The mask:

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

Implemented as a block-triangular mask at both training and inference time.

### Diffusion Loss Inside the Transformer

The diffusion loss is standard: add noise to an image patch, have the model predict the noise (or equivalently predict the clean patch). Transfusion's version uses flow matching — predicting the velocity field from noisy to clean.

During training:
1. For each image patch x0, sample a random timestep t.
2. Sample noise ε, compute xt = (1-t) * x0 + t * ε (flow matching linear interpolation).
3. The transformer predicts v_theta(xt, t); loss = MSE(v_theta(xt, t), ε - x0).
4. Backpropagate together with text NTP loss from the same sequence.

At inference, generation is:
- Text tokens: standard autoregressive sampling.
- Image patches: diffusion sampling loop conditioned on preceding text tokens (typically 10-30 steps).

### MMDiT: The Stable Diffusion 3 Variant

Stable Diffusion 3 (Esser et al., March 2024) shipped MMDiT (Multimodal Diffusion Transformer) around the same time as Transfusion. The two architectures are siblings.

MMDiT's key differences:

- Per-block modality-specific weights. Each transformer block has separate Q, K, V and MLP weights for text tokens and image patches. Attention is joint (cross-modal); everything else is modality-specific.
- Rectified flow training. A specific flow-matching variant with known sampling and simpler math than DDPM.
- Scale. MMDiT is the backbone of SD3 (2B and 8B parameter variants). The Transfusion paper scales to 7B.

Both converge on the same core idea: one transformer running NTP on text and diffusion on continuous image representations.

### Why This Beats Chameleon-Style

The quality gap between continuous diffusion and discrete NTP for image generation is measurable. The Transfusion paper reports:

- At 7B parameters, 3-5 FID points better than same-scale Chameleon-style models.
- No tokenizer training needed — image encoder is simpler (linear projection to hidden dim, same as ViT input layer).
- Inference can parallelize image patch denoising, unlike autoregressive image tokens.

Downside: Transfusion is a dual-loss model with trickier training dynamics. Loss weights need tuning. Schedule mismatches between NTP and diffusion can let one head dominate.

### What Comes After

Janus-Pro (Lesson 12.15) refines Transfusion's idea by decoupling the understanding and generation vision encoders — one uses SigLIP, one uses VQ — while sharing the transformer body. Show-o (Lesson 12.14) replaces diffusion with discrete diffusion (mask prediction). The unified generation family forked rapidly after Transfusion.

The production VLMs shipping image output in 2026 — Gemini 3 Pro, GPT-5, Claude Opus 4.7's image generation paths — almost certainly use a descendant of this family. Details are proprietary.

## Use It

`code/main.py` builds a toy Transfusion on a micro MNIST-style problem:

- Text captions are short integer sequences describing a digit (0-9).
- Images are 4x4 byte grids.
- A pair of shared-weight linear projections acts as a transformer stand-in; NTP loss on text, MSE loss on noised patches.
- The training loop alternates the two losses with an explicit attention mask.
- Generation produces a text caption and a 4x4 image in one forward pass.

The transformer is a toy. The dual-loss pipeline, the attention mask construction, and the inference loop are the real deliverables.

## Ship It

This lesson produces `outputs/skill-two-loss-trainer-designer.md`. Given a new multimodal training task (text + image, text + audio, text + video), it designs a dual-loss schedule (loss weights, mask shapes, shared vs modality-specific blocks) and flags implementation risks.

## Exercises

1. A Transfusion-style model trains on 70% text tokens and 30% image patches. The image diffusion loss is ~10x the magnitude of the text NTP loss. What loss weights would balance them?

2. Implement the block-triangular mask for the sequence `[T, T, <image>, P, P, P, P, </image>, T]`. Label each entry as 0 or 1.

3. MMDiT has modality-specific QKV weights. How much parameter overhead does this add relative to Transfusion's fully shared transformer? At 7B parameters, is it worth it?

4. Generation: given a text prompt, the model runs NTP for 50 tokens, hits `<image>`, then runs diffusion for 256 patches over 20 denoising steps. How many total forward passes?

5. Read SD3 paper Section 3. Describe rectified flow and why it converges in fewer inference steps than DDPM.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Dual-loss training | "NTP + diffusion" | A single transformer optimizing cross-entropy on text tokens and MSE on continuous image patches in the same gradient step |
| Flow matching | "rectified flow" | A diffusion variant predicting the velocity field from noise to clean data; simpler math than DDPM |
| MMDiT | "multimodal DiT" | Stable Diffusion 3's architecture: joint attention, modality-specific MLPs and norms |
| Block-triangular mask | "causal text + bidirectional image" | Attention mask that is causal across text but bidirectional within image regions |
| Continuous image representation | "no VQ" | Image patches as real-valued vectors rather than integer codebook indices |
| Velocity prediction | "v-parameterization" | Network output is the velocity field between noise and data, not the noise itself |

## Further Reading

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
