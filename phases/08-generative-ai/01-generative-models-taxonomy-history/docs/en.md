# Generative Models — Taxonomy & History

> Every image model, text model, video model, and 3D model fits into one of five buckets. Pick the wrong bucket and you'll wrestle the math for weeks; pick the right one and the last twelve years of progress in this field stack up cleanly in your head.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 minutes

## The Problem

A generative model does one thing: given training samples drawn from some unknown distribution `p_data(x)`, produce new samples that look like they came from the same distribution. Faces, sentences, MIDI files, protein structures—squint and it's all the same problem.

The hard part is that `p_data` lives in a space with millions of dimensions (a 512×512 RGB image has ~786k dimensions), samples cluster on a thin manifold inside that space, and you have at most 10 million of them. Computing this density directly is hopeless. Every generative model is a compromise—trading one hard problem for a slightly less hard one.

Five families survived the last twelve years. Knowing which compromise each family makes tells you why it wins on some tasks and collapses on others.

## The Concept

![Five families of generative models — classified by what they model](../assets/taxonomy.svg)

**1. Explicit density, tractable.** Write `log p(x)` as a sum you can actually compute. Autoregressive models (PixelCNN, WaveNet, GPT) decompose `p(x) = ∏ p(x_i | x_<i)`. Normalizing flows (RealNVP, Glow) construct `p(x)` as an invertible transform of a simple base distribution. Pros: exact likelihood, clean training loss. Cons: autoregressive inference is sequential (slow for long sequences); flows require invertible architectures (architecturally constrained).

**2. Explicit density, approximate.** Find a lower bound (ELBO) on `log p(x)` and optimize that bound. VAE (Kingma 2013) uses an encoder-decoder with a variational posterior. Diffusion models (DDPM, Ho 2020) train a denoiser that implicitly optimizes a weighted ELBO. By 2026, diffusion is the dominant backbone for images, video, and 3D.

**3. Implicit density.** Skip density entirely; learn a generator `G(z)` that produces samples and a discriminator `D(x)` that distinguishes real from fake. GAN (Goodfellow 2014). Inference is fast (single forward pass), but training is notoriously unstable. Even in 2026, StyleGAN 1/2/3 remains state-of-the-art for photorealism in fixed domains (faces, bedrooms).

**4. Score-based / continuous-time.** Learn the gradient of log-density `∇_x log p(x)` (the score) directly. Song & Ermon (2019) showed score matching generalizes diffusion into an SDE. Flow matching (Lipman 2023) is the 2024–2026 favorite: simulation-free training, straighter paths, 4–10× faster sampling than DDPM. Stable Diffusion 3, Flux, and AudioCraft 2 all use flow matching.

**5. Token-based autoregressive on discrete codes.** Compress high-dimensional data into a short sequence of discrete tokens via VQ-VAE or residual quantizer, then model that token sequence with a Transformer. Parti, MuseNet, AudioLM, VALL-E, Sora's patch tokenizer—all this lineage. This is bucket 1 plus a learned tokenizer.

## A Brief History

| Year | Model | Why it matters |
|------|-------|----------------|
| 2013 | VAE (Kingma) | First deep generative model with a usable training loss. |
| 2014 | GAN (Goodfellow) | Implicit density, no likelihood—yet razor-sharp samples. |
| 2015 | DRAW, PixelCNN | Sequential image generation. |
| 2017 | Glow, RealNVP | Invertible flows; exact likelihood with depth. |
| 2017 | Progressive GAN | First megapixel faces. |
| 2019 | StyleGAN / StyleGAN2 | Photorealistic faces, still hard to beat in that domain. |
| 2020 | DDPM (Ho) | Diffusion becomes practical. |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image goes mainstream. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning = commoditization. |
| 2022 | ControlNet, LoRA | Fine-grained control over pretrained diffusion. |
| 2023 | SDXL, Midjourney v5, Flow Matching | Scale + better training dynamics. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion; flow matching wins. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Production-grade video. |
| 2026 | Consistency + Rectified Flow | One-step sampling from diffusion backbones. |

## Five-Question Triage

Whenever a new generative model paper appears, answer these five questions before reading the methods section.

1. **What is modeled?** Pixels, latents, discrete tokens, 3D Gaussians, meshes, waveforms?
2. **Is density explicit or implicit?** Do they write out `log p(x)`?
3. **Sampling: one-shot or iterative?** Iterative means slower inference; one-shot usually means adversarial or distilled.
4. **Conditioning: unconditional, class, text, image, pose?** This determines the loss and architectural scaffolding.
5. **Evaluation: FID, CLIP score, IS, human preference, task accuracy?** Each has known failure modes (see Lesson 14).

You'll revisit these five questions for every lesson in this phase. By the end, they'll be reflexive.

## Build It

This lesson's code is a lightweight visualization: fitting a 1D Gaussian mixture from samples using three toy methods (kernel density, discrete histogram, and a "GAN-like" generator that snaps to the nearest sample), letting you see the explicit-vs-implicit density distinction on a problem that fits in one screen.

Run `code/main.py`. It draws 2000 samples from a bimodal Gaussian mixture, then prints:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Notice: the first two let you ask "how likely is this point?" The third cannot. That is the *explicit vs implicit* distinction, and it matters in every lesson ahead.

## Use It

In 2026, which family for which task?

| Task | Best family | Why |
|------|-------------|-----|
| Photorealistic faces, narrow domain | StyleGAN 2/3 | Still sharpest, fastest inference. |
| General text-to-image | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| Fast text-to-image | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM. |
| Text-to-video | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| Speech + music | Token-based autoregressive (AudioLM, VALL-E, MusicGen) or flow matching (AudioCraft 2) | Discrete tokens scale cheaply. |
| 3D scenes | Gaussian splatting fitting, diffusion prior | 3D-GS for reconstruction, diffusion for novel views. |
| Density estimation (no sampling) | Flows | Only family that gives exact `log p(x)`. |
| Simulation / physics | Flow matching, score SDE | Straight paths, smooth vector fields. |

## Ship It

Save as `outputs/skill-model-chooser.md`.

This skill takes a task description and outputs: (1) which family to use, (2) a ranked list of three open-source and three hosted options, (3) likely failure modes to watch for, (4) a compute/time budget.

## Exercises

1. **Easy.** For these five products, identify the family and backbone: ChatGPT Images, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Evidence should come from public technical reports.
2. **Medium.** A paper you'll read tomorrow claims 100× faster sampling than diffusion. Write three questions to verify whether this speedup holds after adding conditioning and high resolution.
3. **Hard.** Pick a domain you care about (e.g., protein structures, CAD, molecules, trajectories). Answer the five-question triage for the current SOTA model in that domain, and sketch what a better model would change.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|----------------------|
| Generative model | "It can create new things" | Learns a sampler for `p_data(x)`, optionally exposing `log p(x)`. |
| Explicit density | "You can compute it" | Model provides a closed-form or tractable `log p(x)`. |
| Implicit density | "The GAN approach" | Only a sampler—no way to compute `p(x)` for a given point. |
| ELBO | "Evidence lower bound" | A tractable lower bound on `log p(x)`; VAEs and diffusion optimize it. |
| Score | "Gradient of log-density" | `∇_x log p(x)`; diffusion and SDE models learn this field. |
| Manifold hypothesis | "Data lives on a surface" | High-dimensional data concentrates on a low-dimensional manifold; this is why dimensionality reduction works. |
| Autoregressive | "Predict the next piece" | Factorizes the joint distribution into a product of conditionals. |
| Latent | "The compressed code" | A low-dimensional representation from which a decoder can reconstruct the input. |

## Production Notes: Five Families, Five Inference Profiles

Each family maps to a different inference-server cost curve. The production inference literature decomposes LLM inference into prefill + decode; the same decomposition applies here:

- **Autoregressive (buckets 1 and 5).** Sequential decode dominates latency; KV-cache, continuous batching, speculative decoding all apply directly.
- **VAE / diffusion / flow matching (buckets 2 and 4).** No decode phase in the LLM sense. Cost = `num_steps × step_cost`, where `step_cost` is one transformer or U-Net forward at full latent resolution. Production knobs are step count (DDIM / DPM-Solver / distillation), batch size, and precision (bf16 / fp8 / int4).
- **GAN (bucket 3).** Single forward pass. No scheduler, no KV-cache. TTFT ≈ total latency. This is why StyleGAN still wins on interactive experience for narrow domains.

When you see "faster than diffusion" in a paper abstract, translate it to "fewer steps × same step cost" or "same steps × cheaper step cost." Everything else is marketing.

## Further Reading

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — The GAN paper.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — The VAE paper.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — The DDPM paper.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — Diffusion as an SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — The flow matching paper.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.
