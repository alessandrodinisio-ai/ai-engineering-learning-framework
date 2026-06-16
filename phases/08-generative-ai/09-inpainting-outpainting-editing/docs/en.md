# Inpainting, Outpainting & Editing

> Text-to-image creates new things. Inpainting fixes old things. In production, 70% of billable image work is editing—swapping backgrounds, removing logos, extending canvases, redrawing a hand. Inpainting is where diffusion makes money.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 minutes

## The Problem

A client sends a perfect product photo with a distracting sign in the background. You want to erase the sign and keep everything else pixel-identical. You can't run text-to-image from scratch—the result would have different colors, different lighting, different product angle. You want to regenerate *only* the masked region, and you want the regenerated part to respect surrounding context.

That's inpainting. Variants include:

- **Inpainting.** Regenerate inside the mask; pixels outside the mask are preserved.
- **Outpainting.** Regenerate outside the mask (or beyond the canvas); pixels inside are preserved.
- **Image editing.** Regenerate the full image while staying semantically or structurally faithful to the original (SDEdit, InstructPix2Pix).

Every diffusion pipeline in 2026 ships an inpainting mode. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. They all work on the same principle.

## The Concept

![Inpainting: mask-aware denoising + context-preserving re-injection](../assets/inpainting.svg)

### The naive approach (and why it's wrong)

Run standard text-to-image with a mask. At each sampling step, replace the unmasked region of the noisy latent with the forward-diffused clean image. It works… poorly. Boundary artifacts bleed through because the model knows nothing about what's inside the masked region.

### A proper inpainting model

Train a modified U-Net that takes 9 channels instead of 4:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

The extra channels are a copy of the VAE-encoded source image plus a single-channel mask. During training you randomly mask out regions, train the model to denoise only the masked area while the unmasked area serves as a clean conditioning signal. At inference the model can "see" what surrounds the mask and produces coherent completions.

SD-Inpaint, SDXL-Inpaint, and Flux-Fill all use this 9-channel (or similar) input. Diffusers' `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — Free editing

Add noise to the source image up to some intermediate `t`, then run the reverse chain from `t` to 0 with a new prompt. No retraining needed. The choice of starting `t` trades off fidelity vs creative freedom:

- `t/T = 0.3` → nearly identical to source, minor style changes
- `t/T = 0.6` → moderate edits, coarse structure preserved
- `t/T = 0.9` → nearly generated from noise, very little source preserved

### InstructPix2Pix (Brooks et al., 2023)

Fine-tune a diffusion model on `(input image, instruction, output image)` triplets. At inference, condition on both the input image and a text instruction ("make it sunset," "add a dragon"). Two CFG strengths: image strength and text strength.

### RePaint (Lugmayr et al., 2022)

Keep a standard unconditional diffusion model. At each reverse step, resample—occasionally jump back to a noisier state and re-denoise. Avoids boundary artifacts. Use when you don't have a trained inpainting model.

## Build It

`code/main.py` implements a toy 1-D inpainting scheme on 5-dimensional data. We train a DDPM on a 5-D mixture dataset where each sample is 5 floats from one of two clusters. At inference we "mask" 2 of the 5 dimensions, inject the forward-noised version of the unmasked 3 dimensions at each step, and regenerate only the masked dimensions.

### Step 1: 5-D DDPM data

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Step 2: Train denoiser on all 5 dimensions

Standard DDPM. The network outputs 5-D noise prediction for a 5-D noisy input.

### Step 3: Mask-aware reverse at inference

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

This is the naive approach, which works on toy 1-D data. Real image inpainting uses the 9-channel input because texture coherence matters more.

### Step 4: Outpainting

Outpainting is inpainting with the mask inverted: mask the new (previously nonexistent) canvas, fill the rest with the original. The training objective is identical.

## Pitfalls

- **Seams.** The naive approach leaves visible boundaries because gradient information doesn't flow across the mask. Fix: dilate the mask by 8-16 pixels, or use a proper inpainting model.
- **Mask leakage.** If the conditioning image's unmasked area is low quality or noisy, it pollutes in-mask generation. Lightly denoise or blur first.
- **CFG interacts with mask size.** High CFG on small masks = saturated blobs. Lower CFG for small edits.
- **SDEdit fidelity cliff.** Going from `t/T = 0.5` to `t/T = 0.6` can lose subject identity. Sweep and save checkpoints.
- **Prompt mismatch.** The prompt should describe the *entire* image, not just the new content. "A cat sitting on a chair," not "a cat."

## Real-World Usage

| Task | Pipeline |
|------|----------|
| Remove object, small mask | SD-Inpaint or Flux-Fill, standard prompt |
| Replace sky | SD-Inpaint + "blue sky at sunset" |
| Extend canvas | SDXL outpaint mode (8px feathering) or Flux-Fill with outpaint mask |
| Redraw hand / face | SD-Inpaint, prompt re-describes subject + ControlNet-Openpose |
| Restyle a region | SDEdit at `t/T=0.5` on masked area |
| "Make it sunset" | InstructPix2Pix or Flux-Kontext |
| Replace background | SAM mask → SD-Inpaint |
| Highest fidelity | Flux-Fill or GPT-Image (hosted) for hardest cases |

SAM (Meta's Segment Anything, 2023) + diffusion inpaint is the 2026 cutout pipeline. SAM 2 (2024) works on video.

## Ship It

Save as `outputs/skill-editing-pipeline.md`. The skill accepts a source image + edit description + optional mask (or SAM prompt) and outputs: mask generation approach, base model, CFG strengths (image + text), SDEdit-t or inpainting mode, and a QA checklist.

## Exercises

1. **Easy.** Change the fraction of masked dimensions from 0.2 to 0.8 in `code/main.py`. At what fraction does inpaint quality (residual on masked dims) equal unconditional generation?
2. **Medium.** Implement RePaint: every 10th reverse step, jump back 5 steps (add noise) and re-denoise. Measure whether it reduces boundary residual at mask edges.
3. **Hard.** Use Hugging Face diffusers to compare SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill on 20 face-redraw tasks. Score pose adherence and identity preservation separately.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Inpainting | "Fill the hole" | Regenerate inside the mask; unmasked pixels preserved. |
| Outpainting | "Extend the canvas" | Regenerate outside the canvas; inside preserved. |
| 9-channel U-Net | "Proper inpainting model" | U-Net taking `noisy | encoded-source | mask` as input. |
| SDEdit | "img2img with noise level" | Add noise to time `t`, denoise with new prompt. |
| InstructPix2Pix | "Text-only editing" | Diffusion fine-tuned on (image, instruction, output) triplets. |
| RePaint | "No retraining needed" | Periodic re-noising during reverse process to reduce seams. |
| SAM | "Segment Anything" | Generates masks from clicks or boxes; pairs with inpaint. |
| Flux-Kontext | "Edit with context" | Flux variant accepting reference image + instruction for editing. |

## Production Notes: Editing Pipelines Are Latency-Sensitive

Users editing images expect sub-5-second round trips. SDXL-Inpaint at 30 steps on 1024² on an L4 is 3-4 seconds, plus SAM mask generation (~200 ms) and VAE encode/decode (~500 ms total). In the production framework, this is TTFT-bound rather than throughput-bound—batch-1, low concurrency, squeeze every stage:

- **SAM-H is the slow one.** SAM-H at 1024² is ~200 ms; SAM-ViT-B is ~40 ms with slightly lower quality. SAM 2 (video) adds temporal overhead; don't use it for single-image edits.
- **Skip encoding when you can.** `pipe.image_processor.preprocess(img)` encodes to latents. If you have latents from a prior generation (typical in iterative editing UIs), pass them directly via `latents=...` and save one VAE encode.
- **Mask dilation matters for throughput too.** Small masks mean most of the U-Net forward is wasted (unmasked pixels get clamped anyway). `diffusers`' `StableDiffusionInpaintPipeline` runs the full U-Net regardless; only proper 9-channel inpaint variants exploit the mask to save compute.
- **Flux-Kontext is the 2025 answer.** Single forward pass for `(source image, instruction)`—no separate mask, no SDEdit noise sweep. ~1.5s per edit on H100. Architectural lesson: merge these stages away.

## Further Reading

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — Training-free inpainting.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — Text-instruction editing.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, the mask source.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — Video-capable SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — Attention-level editing.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 tooling.
