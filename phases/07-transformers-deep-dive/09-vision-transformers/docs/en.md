# Vision Transformers (ViT)

> An image is a grid of patches. A sentence is a sequence of tokens. The same transformer handles both.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNN), Phase 4 · 14 (Vision Transformers Intro)
**Time:** ~45 min

## The Problem

Before 2020, computer vision meant convolutions. Every SOTA on ImageNet, COCO, and detection benchmarks used a CNN backbone. Transformers were for language.

Dosovitskiy et al. (2020) — "An Image is Worth 16x16 Words" — proved you can ditch convolutions entirely. Cut an image into fixed-size patches, linearly project each patch into an embedding, feed the sequence to a standard transformer encoder. At sufficient scale (ImageNet-21k pre-training or larger), ViT matches or beats ResNet-based models.

ViT started a broader pattern in 2026: one architecture, many modalities. Whisper tokenizes audio. ViT tokenizes images. Robotics tokenizes actions. Video tokenizes pixels. The transformer doesn't care — feed it a sequence, it learns.

By 2026, ViT and its descendants (DeiT, Swin, DINOv2, ViT-22B, SAM 3) dominate most of vision. CNNs still win on edge devices and latency-sensitive tasks. Everything else has a ViT somewhere in the stack.

## The Concept

![Image → patch → token → transformer](../assets/vit.svg)

### Step 1 — Patchify

Cut an `H × W × C` image into a flat patch sequence of shape `N × (P·P·C)`. Typical setup: `224 × 224` image, `16 × 16` patches → 196 patches, each 768 values.

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Patch size is the lever. Smaller patches = more tokens, better resolution, quadratic attention cost. Larger patches = coarser, cheaper.

### Step 2 — Linear embedding

A learned matrix projects each flat patch to `d_model`. Equivalent to a convolution with kernel size `P` and stride `P`. In PyTorch this is `nn.Conv2d(C, d_model, kernel_size=P, stride=P)` — 2-line implementation.

### Step 3 — Prepend `[CLS]` token, add positional embeddings

- Prepend a learnable `[CLS]` token. Its final hidden state is the image representation for classification.
- Add learnable positional embeddings (original ViT) or sinusoidal 2D (later variants).
- 2024+ extends RoPE to 2D for position, sometimes without explicit embeddings.

### Step 4 — Standard transformer encoder

Stack L blocks of `LayerNorm → Self-Attention → + → LayerNorm → MLP → +`. Identical to BERT. No vision-specific layers. That's the paper's pedagogical elegance.

### Step 5 — Head

For classification: take `[CLS]` hidden state → linear → softmax. DINOv2 or SAM drops `[CLS]` and uses patch embeddings directly.

### Notable variants

| Model | Year | What changed |
|-------|------|--------|
| ViT | 2020 | Original. Fixed patch size, full global attention. |
| DeiT | 2021 | Distillation; trains with only ImageNet-1k. |
| Swin | 2021 | Hierarchical with shifted windows. Fixes to sub-quadratic cost. |
| DINOv2 | 2023 | Self-supervised (no labels). Best general-purpose visual features. |
| ViT-22B | 2023 | 22B parameters; scaling laws apply. |
| SigLIP | 2023 | ViT + language pairing, sigmoid contrastive loss. |
| SAM 3 | 2025 | Segment anything; ViT-Large + promptable mask decoder. |

### Why it took a while

ViT needs *lots* of data to match CNNs because it lacks CNN inductive biases (translation invariance, locality). Without >100M labeled images or strong self-supervised pre-training, CNNs still win at equal compute. DeiT fixed this in 2021 with distillation tricks; DINOv2 fixed it permanently in 2023 with self-supervision.

## Build It

See `code/main.py`. Pure-stdlib patchify + linear embedding + sanity check. No training — any realistic-scale ViT requires PyTorch and hours of GPU time.

### Step 1: Fake image

A 24 × 24 RGB image represented as a list of rows of `(R, G, B)` tuples. We use 6×6 patches → 16 patches, each a 108-dim embedding vector.

### Step 2: Patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Raster order: row-major across the grid. Every ViT uses this order.

### Step 3: Linear embedding

Multiply each flat patch by a random `(patch_flat_size, d_model)` matrix. After prepending `[CLS]`, verify output shape is `(N_patches + 1, d_model)`.

### Step 4: Count parameters for a realistic ViT

Print parameter count for ViT-Base: 12 layers, 12 heads, d=768, patch=16. Compare against ResNet-50 (~25M). ViT-Base lands at ~86M. ViT-Large ~307M. ViT-Huge ~632M.

## Use It

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embeddings are the default choice for image features in 2026.** Freeze the backbone, train a small head. Classification, retrieval, detection, captioning all work. Meta's DINOv2 checkpoints outperform CLIP on every non-text vision task.

**How to pick patch size.** Small models use 16×16 (ViT-B/16). Dense prediction (segmentation) uses 8×8 or 14×14 (SAM, DINOv2). Very large models use 14×14.

## Ship It

See `outputs/skill-vit-configurator.md`. This skill picks a ViT variant and patch size for a new vision task based on dataset size, resolution, and compute budget.

## Exercises

1. **Easy.** Run `code/main.py`. Verify patch count equals `(H/P) * (W/P)` and flat patch dimension equals `P*P*C`.
2. **Medium.** Implement 2D sinusoidal positional embeddings — compute a separate set of sinusoidal codes for each patch's `row` and `col`, then concatenate. Feed them into a small PyTorch ViT and compare accuracy against learnable positional embeddings on CIFAR-10.
3. **Hard.** Build a 3-layer ViT (PyTorch) with 4×4 patches, train on 1,000 MNIST images. Measure test accuracy. Now add DINOv2 pre-training on the same 1,000 images (simplified: just train the encoder to predict patch embeddings from masked patches). Does accuracy improve?

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| Patch | "vision transformer's token" | A flat vector of pixel values from a `P × P × C` region of the image. |
| Patchify | "cut + flatten" | Split the image into non-overlapping patches, flatten each into a vector. |
| `[CLS]` token | "image summary" | A learnable token prepended to the sequence; its final embedding is the image representation. |
| Inductive bias | "what the model assumes" | ViT has fewer priors than CNNs; needs more data to close the gap. |
| DINOv2 | "self-supervised ViT" | Trained without labels using image augmentations + momentum teacher. Best general-purpose image features in 2026. |
| SigLIP | "CLIP's successor" | ViT + text encoder trained with sigmoid contrastive loss; beats CLIP at equal compute. |
| Swin | "windowed ViT" | Hierarchical ViT with local attention + shifted windows; sub-quadratic. |
| Register token | "2023 trick" | Extra learnable tokens that absorb attention sinks; improves DINOv2 features. |

## Further Reading

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — The ViT paper.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — The register-token fix for DINOv2.
