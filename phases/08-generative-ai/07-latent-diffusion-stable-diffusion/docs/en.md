# Latent Diffusion & Stable Diffusion

> Pixel-space diffusion on 512×512 images is a compute war crime. Rombach et al. (2022) noticed you don't need all 786K dimensions to generate an image—you need just enough to capture semantic structure, plus a separate decoder for the rest. Run diffusion in a VAE's latent space. That single idea is Stable Diffusion.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 minutes

## The Problem

Pixel-space diffusion at 512² means the U-Net runs on tensors of shape `[B, 3, 512, 512]`. For a 500M-parameter U-Net, each sampling step costs ~100 GFLOPS. Fifty steps is 5 TFLOPS per image. Training on a billion images makes the compute bill absurd.

Most of those FLOPs go toward pushing perceptually irrelevant detail through the network—high-frequency textures that a lossy VAE could compress away. Rombach's idea: train a VAE once (*stage 1*), freeze it, then run diffusion entirely in the 4-channel 64×64 latent space (*stage 2*). Same U-Net. 1/16th the pixels. ~64× fewer FLOPs at comparable quality.

This is the Stable Diffusion recipe. SD 1.x / 2.x uses an 860M U-Net on `64×64×4` latents, SDXL uses a 2.6B U-Net on `128×128×4`, and SD3 replaces the U-Net with a Diffusion Transformer (DiT) with flow matching. Flux.1-dev (Black Forest Labs, 2024) ships a 12B-parameter DiT-MMDiT. They all run on the same two-stage foundation.

## The Concept

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**Two stages, trained separately.**

1. **Stage 1 — VAE.** Encoder `E(x) → z`, decoder `D(z) → x`. Goal is compression: downsample each spatial axis by 8× + adjust channel count so total latent size is ~1/16th of pixel count. Loss = reconstruction (L1 + LPIPS perceptual) + KL (small weight—keeps `z` from being forced too Gaussian since we don't need exact sampling from `z`). Often trained with an adversarial loss for sharp decoded images.

2. **Stage 2 — Diffusion on `z`.** Treat `z = E(x_real)` as data. Train a U-Net (or DiT) to denoise `z_t`. At inference: sample `z_0` via diffusion, then `x = D(z_0)`.

**Text conditioning.** Two additional components. A frozen text encoder (SD 1.x uses CLIP-L, SD 2/XL uses CLIP-L+OpenCLIP-G, SD3 and Flux use T5-XXL). A cross-attention injection: each U-Net block takes `[Q = image features, K = V = text tokens]` and mixes them in. These tokens are the only pathway for text to influence the image.

**The loss is identical to Lesson 06.** Same DDPM / flow matching MSE on noise. You just changed the data domain.

## Architecture Variants

| Model | Year | Backbone | Latent Shape | Text Encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | Same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 steps |

Trends: replacing U-Net with DiT (transformer on latent patches), scaling text encoders (T5 beats CLIP on prompt adherence), increasing latent channels (4 → 16 gives more detail headroom).

## Build It

`code/main.py` stacks a toy 1-D "VAE" (identity encoder + decoder for demonstration; a real VAE would be convolutional) on top of Lesson 06's DDPM, adding class conditioning with classifier-free guidance. It demonstrates that the same diffusion loss works whether you run on raw 1-D values or encoded ones—that's the key insight.

### Step 1: Encoder/Decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

A real VAE has trained weights. For pedagogical purposes, this linear mapping suffices to show diffusion operating on `z` regardless of the original data space.

### Step 2: Diffusion in `z` space

Same DDPM as Lesson 06. The network sees data as `z = E(x)`. After sampling `z_0`, decode with `D(z_0)`.

### Step 3: Classifier-free guidance

Drop class label 10% of the time during training (replace with a null token). At inference compute both `ε_cond` and `ε_uncond`, then:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = no guidance (full diversity), `w = 3` = default, `w = 7+` = saturated / over-sharpened.

### Step 4: Text conditioning (conceptual, not code)

Replace the class label with the output of a frozen text encoder. Feed text embeddings to the U-Net via cross-attention:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

This is the only substantive difference between a class-conditional diffusion model and Stable Diffusion.

## Pitfalls

- **VAE scaling mismatch.** SD 1.x's VAE multiplies by a scaling constant after encoding (`scaling_factor ≈ 0.18215`). Forgetting it trains the U-Net on latents with wildly wrong variance. Every checkpoint carries one.
- **Text encoder silent failures.** SD3 needs T5-XXL with token count >=128; falling back to CLIP-only is lossy. Always check `use_t5=True` or prompt fidelity tanks.
- **Mixing latent spaces.** SDXL, SD3, and Flux all use different VAEs. A LoRA trained on SDXL latents won't work on SD3. Hugging Face diffusers 0.30+ will refuse to load mismatched checkpoints.
- **CFG too high.** `w > 10` produces saturated, oily images at the cost of diversity, overfitting the prompt. Sweet spot is `w = 3-7`.
- **Negative prompt leakage.** Empty negative prompt becomes the null token; a filled negative prompt becomes `ε_uncond`. These are not the same thing; some pipelines silently default to the null token.

## Real-World Usage

2026 production stacks:

| Goal | Recommended Backbone |
|--------|----------------------|
| Narrow domain, paired data, train from scratch | SDXL fine-tune (LoRA / full) — fastest to ship |
| Open-domain text-to-image, open weights | Flux.1-dev (12B, Apache / non-commercial) or SD3.5-Large |
| Fastest inference, open weights | Flux.1-schnell (1-4 steps, Apache) or SDXL-Lightning |
| Best prompt adherence, hosted | GPT-Image / DALL-E 3 (still holds), Midjourney v7, Imagen 4 |
| Editing workflows | Flux.1-Kontext (Dec 2024) — natively accepts image + text |
| Research, baselines | SD 1.5 — ancient but well-studied |

## Ship It

Save as `outputs/skill-sd-prompter.md`. The skill accepts a text prompt + target style and outputs: model + checkpoint, CFG strength, sampler, negative prompt, resolution, optional ControlNet/IP-Adapter combo, and a step-by-step QA checklist.

## Exercises

1. **Easy.** Run `code/main.py` with guidance `w ∈ {0, 1, 3, 7, 15}`. Log the mean sample per class. At what `w` does the class mean overshoot the true data mean?
2. **Medium.** Replace the toy linear encoder with a pair of tanh-MLP encoder/decoder with reconstruction loss. Retrain diffusion on the new latents. Does sample quality change?
3. **Hard.** Build a real Stable Diffusion inference with diffusers: load `sdxl-base`, run 30-step Euler, CFG=7, time it. Then switch to `sdxl-turbo`, 4 steps, CFG=0. Same subject, different quality—describe what changed and why.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Stage 1 | "The VAE" | Trained encoder/decoder pair; compresses 512² to 64². |
| Stage 2 | "The U-Net" | Diffusion model operating in latent space. |
| CFG | "Guidance strength" | `(1+w)·ε_cond - w·ε_uncond`; dials conditioning intensity. |
| Null token | "Empty prompt embedding" | Unconditional embedding used as `ε_uncond`. |
| Cross-attention | "How text gets in" | Each U-Net block attends to text tokens as K and V. |
| DiT | "Diffusion Transformer" | Replaces U-Net with a transformer on latent patches; scales better. |
| MMDiT | "Multimodal DiT" | SD3's architecture: text stream and image stream do joint attention. |
| VAE scaling factor | "The magic number" | Divides latents by ~5.4 so diffusion works in unit-variance space. |

## Production Notes: Running Flux-12B on an 8GB Consumer GPU

The reference Flux integration is the classic "I have a consumer GPU, can I ship this?" recipe. The tricks are the same three knobs the production inference literature lists, applied to a diffusion DiT:

1. **Offloading in stages.** Flux has three networks that never need to coexist in VRAM: T5-XXL text encoder (~10 GB at fp32), CLIP-L (small), the 12B MMDiT, and the VAE. Encode prompt, *delete* encoder, load DiT, denoise, *delete* DiT, load VAE, decode. A consumer 8GB GPU fits one stage at a time.
2. **4-bit quantization via bitsandbytes.** Use `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` for both T5 encoder and DiT. 8× memory reduction, imperceptible quality loss for text-to-image per Aritra's benchmarks (linked in the notebook).
3. **CPU offload.** `pipe.enable_model_cpu_offload()` automatically swaps modules between CPU and GPU as forward passes progress. Adds 10-20% latency, but makes the pipeline runnable at all.

The memory math: `10 GB T5 / 8 = 1.25 GB` quantized, `12B params × 0.5 bytes = ~6 GB` quantized DiT, plus activations. In stas00's words this is the extreme end of TP=1 inference—no model parallelism, maximum quantization. In production you'd run TP=2 or TP=4 on H100s; for a single dev laptop, this is the recipe.

## Further Reading

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 family.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — Reference implementations for all checkpoints above.
