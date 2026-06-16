# ControlNet, LoRA & Conditioning

> Text alone is a clumsy control signal. ControlNet lets you clone a pretrained diffusion model and steer it with depth maps, pose skeletons, scribbles, or edge maps. LoRA lets you fine-tune a 2B-parameter model by training only 10 million parameters. Together, they turn Stable Diffusion from a toy into the image pipeline every creative company ships in 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundations)
**Time:** ~75 minutes

## The Problem

A prompt like "a woman in a red dress walking a dog on a busy street" gives the model zero information about *where* the dog is, *what pose* the woman has, or *what perspective* the street takes. Text pins down roughly 10% of the information needed to specify an image. The rest is visual, and cannot be efficiently described in words.

Training a new conditional model from scratch for each signal (pose, depth, canny, segmentation) is prohibitively expensive. You want to keep the 2.6B-parameter SDXL backbone frozen, attach a small side-network that reads the condition, and let it nudge the backbone's intermediate features. That's ControlNet.

You also want to teach the model new concepts (your face, your product, your style) without retraining the whole thing. You want an increment 100× smaller. That's LoRA—low-rank adapters plugged into existing attention weights.

ControlNet + LoRA + text = the 2026 practitioner's toolkit. Most production image pipelines stack 2-5 LoRAs, 1-3 ControlNets, and an IP-Adapter on top of a single SDXL / SD3 / Flux base model.

## The Concept

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

Take a pretrained SD. *Clone* the encoder half of the U-Net. Freeze the original. Train the clone to accept an additional conditioning input (edges, depth, pose). Connect the clone back to the original decoder half with *zero convolutions* (1×1 convolutions initialized to zero—a no-op at the start, then learns an increment).

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv initialization means ControlNet starts as an identity—it causes no harm even before training. Train with standard diffusion loss on 1M (prompt, condition, image) triplets.

Each modality's ControlNet is released as a small side-model (~360M for SDXL, ~70M for SD 1.5). You can compose them at inference:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

For any linear layer `W ∈ R^{d×d}` in the model, freeze `W` and add a low-rank delta:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

where `r << d`. Rank 4-16 is standard for attention, 64-128 for heavy fine-tuning. New parameter count: `2 · d · r` instead of `d²`. For SDXL attention `d=640`, `r=16`: 20K params per adapter instead of 410K—20× fewer. Across the full model: a LoRA is typically 20-200MB vs the 5GB base.

At inference you can scale the LoRA: `W' = W + α · B @ A`. `α = 0.5-1.5` is normal. Multiple LoRAs add linearly (with the usual caveat that they interact non-linearly).

### IP-Adapter (Ye et al., 2023)

A tiny adapter that conditions on an *image* (alongside text). It uses a CLIP image encoder to produce image tokens and injects them into cross-attention alongside text tokens. ~20MB per base model. Lets you do "generate an image in the style of this reference" without LoRA.

## Composability Matrix

| Tool | What it controls | Size | When to use |
|------|------------------|------|-------------|
| ControlNet | Spatial structure (pose, depth, edges) | 70-360MB | Precise layout, composition |
| LoRA | Style, subject, concept | 20-200MB | Personalization, style |
| IP-Adapter | Style or subject from a reference image | 20MB | No text can describe what it looks like |
| Textual Inversion | A single concept as a new token | 10KB | Legacy, mostly superseded by LoRA |
| DreamBooth | Full fine-tune on a subject | 2-5GB | Strong identity, high compute |
| T2I-Adapter | Lighter ControlNet alternative | 70MB | Edge devices, tight inference budget |

ControlNet ≈ spatial. LoRA ≈ semantic. Use both.

## Build It

`code/main.py` simulates both mechanisms in 1-D:

1. **LoRA.** A pretrained linear layer `W`. Freeze it. Train a low-rank `B @ A` such that `W + BA` matches a target linear layer. Demonstrates that `r = 1` suffices to perfectly learn a rank-1 correction.

2. **ControlNet-lite.** A "frozen base" predictor and a "side network" that reads an extra signal. The side network's output is gated by a learnable scalar initialized to zero (our version of zero convolution). Train and watch the gate ramp up.

### Step 1: LoRA math

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Step 2: Zero-initialized side network

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

At step 0 the output matches the base model exactly. Early training slowly updates `gate`—no catastrophic drift.

## Pitfalls

- **Over-scaling LoRA.** `α = 2` or `α = 3` is a common "make it stronger" hack that produces over-stylized / broken outputs. Keep `α ≤ 1.5`.
- **ControlNet weight conflicts.** Using a Pose ControlNet at weight 1.0 and a Depth ControlNet at weight 1.0 typically overshoots. Sum of weights ≈ 1.0 is a safe default.
- **LoRA on the wrong base.** An SDXL LoRA silently becomes a no-op on SD 1.5 because attention dimensions don't match. Diffusers warns at 0.30+.
- **Textual Inversion drift.** A token trained on one checkpoint drifts heavily on another. LoRA is more portable.
- **LoRA weight merging vs storage.** You can bake a LoRA into the base weights for faster inference (no runtime addition), but you lose the ability to scale `α` at runtime. Keep both versions.

## Real-World Usage

| Goal | 2026 Pipeline |
|------|---------------|
| Reproduce a brand's art style | Rank-32 LoRA trained on ~30 curated images |
| Put my face in a generated image | DreamBooth or LoRA + IP-Adapter-FaceID |
| Specific pose + prompt | ControlNet-Openpose + SDXL + text |
| Depth-aware composition | ControlNet-Depth + SD3 |
| Reference image + prompt | IP-Adapter + text |
| Precise layout | ControlNet-Scribble or ControlNet-Canny |
| Replace background | ControlNet-Seg + Inpainting (Lesson 09) |
| Fast one-step style | LCM-LoRA on SDXL-Turbo |

## Ship It

Save as `outputs/skill-sd-toolkit-composer.md`. The skill accepts a task (input assets: prompt, optional reference image, optional pose, optional depth, optional scribble) and outputs the tool stack, weights, and a reproducible seed workflow.

## Exercises

1. **Easy.** Change LoRA rank `r` from 1 to 4 in `code/main.py`. At what rank can the LoRA exactly match a rank-2 target delta?
2. **Medium.** Train two independent LoRAs on two target transforms. Load both together and demonstrate their additive interaction. When does this interaction break linearity?
3. **Hard.** Use diffusers to stack: SDXL-base + Canny-ControlNet (weight 0.8) + a style LoRA (α 0.8) + IP-Adapter (weight 0.6). Measure FID vs prompt adherence tradeoff as stack weights vary.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| ControlNet | "Spatial control" | Cloned encoder + zero-conv skip connections; reads a condition image. |
| Zero convolution | "Starts as identity" | 1×1 conv initialized to zero; ControlNet starts as a no-op. |
| LoRA | "Low-rank adapter" | `W + B @ A`, `r << d`; 100× fewer params than full fine-tuning. |
| Rank r | "The knob" | LoRA's compression level; typical 4-16, heavy personalization 64+. |
| α | "LoRA strength" | Runtime scaling of the LoRA delta. |
| IP-Adapter | "Reference image" | Small image-conditioning adapter via CLIP image tokens. |
| DreamBooth | "Full subject fine-tune" | Trains the whole model on ~30 images of a subject. |
| Textual Inversion | "New token" | Learns only a new word embedding; legacy, mostly superseded. |

## Production Notes: LoRA Hot-Swapping, ControlNet Channels, Multi-Tenant Serving

A real text-to-image SaaS serves hundreds of LoRAs and a dozen ControlNets on the same base checkpoint. The serving problem looks a lot like LLM multi-tenancy (what the production literature describes under continuous batching and LoRAX / S-LoRA for the LLM case):

- **Hot-swap LoRAs, don't merge.** Baking `W' = W + α·B·A` into the base saves ~3-5% per-step inference, but freezes `α` and the base. Keep LoRAs as rank-r deltas hot-resident in VRAM; diffusers exposes `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` for per-request activation. Swap cost is those `2 · d · r · num_layers` weights—MB-scale, sub-second.
- **ControlNet as a second attention channel.** The cloned encoder runs in parallel with the base. Two ControlNets at weight 1.0 each = two extra forward passes per step, not one merged pass. Batch-size headroom drops quadratically. Budget ~1.5× single-step cost per active ControlNet.
- **LoRAs quantize too.** If you quantized the base (see Lesson 07, Flux on 8GB), LoRA deltas can also be cleanly quantized to 8-bit or 4-bit. QLoRA-style loading lets you stack 5-10 LoRAs on a 4-bit Flux base without blowing memory.

Flux-specific: Niels' Flux-on-8GB notebook quantizes the base to 4-bit; stacking a style LoRA on that quantized base (`pipe.load_lora_weights("user/style-lora")`), using `weight_name="pytorch_lora_weights.safetensors"`, still works. This is the recipe most SaaS creative companies ship in 2026.

## Further Reading

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (originally for LLMs, portable to diffusion).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — Lighter ControlNet alternative.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — Reference pipelines.
