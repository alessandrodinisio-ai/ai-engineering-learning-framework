# Vision Transformer and the Patch-Token Primitive

> Before anything multimodal, an image must first become a sequence of tokens a transformer can consume. The 2020 ViT paper gave the answer: 16x16-pixel patches, one linear projection, one positional embedding. Five years later, every 2026 frontier model (native 2576px Claude Opus 4.7, Gemini 3.1 Pro, Qwen3.5-Omni) still starts here—encoders evolved from ViT to DINOv2 to SigLIP 2, register tokens appeared, position schemes became 2D-RoPE, but the primitive never changed. This lesson reads through the patch-token pipeline end to end and builds it with pure standard-library Python, so the rest of Phase 12 has a concrete mental model of "visual tokens."

**Type:** Learn
**Languages:** Python (standard library, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## Learning Objectives

- Convert an HxWx3 image into a sequence of patch tokens with correct positional encoding.
- Given a ViT's (patch size, resolution, hidden dim, depth), compute its sequence length, parameter count, and FLOPs.
- Name three upgrades that took ViT from 2020 research to 2026 production: self-supervised pretraining (DINO / MAE), register tokens, native-resolution packing.
- Choose between CLS pooling, mean pooling, and register tokens for a downstream task.

## The Problem

Transformers process sequences of vectors. Text is already a sequence (bytes or tokens). An image is a 2D pixel grid with three color channels—not a sequence. If you flatten every pixel, a 224x224 RGB image becomes 150,528 tokens, and self-attention at that length is unworkable (quadratic in sequence length).

The pre-2020 approach was to prepend a CNN feature extractor: a ResNet produces a 7x7 feature map of 2048-dim vectors, feeding 49 tokens to the transformer. This works but inherits CNN inductive biases (translation equivariance, local receptive fields) and forfeits the transformer's appetite for scale.

Dosovitskiy et al. (2020) asked a blunt question: what if we skip the CNN entirely? Slice the image into fixed-size patches (e.g. 16x16 pixels), linearly project each patch into a vector, add positional embeddings, and feed the sequence to a vanilla transformer. At the time this was heresy—convolution-free vision. With enough data (first JFT-300M, later LAION), it beat ResNet on ImageNet and kept getting better.

By 2026, the ViT primitive is unquestioned bedrock. Every open-weight VLM's vision tower is one of its descendants (DINOv2, SigLIP 2, CLIP, EVA, InternViT). The question is no longer "should I use patches?" but "which patch size, which resolution scheme, which pretraining objective, which positional encoding."

## The Concept

### Patches as Tokens

Given an image `x` of shape `(H, W, 3)` and patch size `P`, you slice the image into a `(H/P) x (W/P)` non-overlapping grid. Each patch is a `P x P x 3` pixel block. Flatten each block into a `3P^2`-dimensional vector. A shared linear projection `W_E` of shape `(3P^2, D)` maps each patch to the model's hidden dimension `D`.

The canonical ViT-B/16 configuration:
- Resolution 224, patch size 16 → 14x14 grid → 196 patch tokens.
- Each patch is `16 x 16 x 3 = 768` pixel values, projected to `D = 768`.
- Add a learnable `[CLS]` token → sequence length 197.

The patch projection is mathematically identical to a 2D convolution with kernel size `P`, stride `P`, and output channels `D`. Production code implements it exactly this way—`nn.Conv2d(3, D, kernel_size=P, stride=P)`. "Linear projection" is the conceptual description; "conv kernel" is the efficient implementation.

### Positional Embeddings

Patches have no inherent order—the transformer sees them as a bag. Early ViT added learnable 1D positional embeddings (one 768-dim vector per position, 197 total). This works but locks the model to the training resolution: if you change grid size at inference, you must interpolate the position table.

Modern vision backbones use 2D-RoPE (Qwen2-VL's M-RoPE, SigLIP 2's default) or factorized 2D positions. 2D-RoPE rotates query and key vectors based on the patch's (row, col) index, so the model infers relative 2D positions from rotation angles. No position table. At inference the model handles any grid size.

### CLS Token, Pooled Output, and Register Tokens

Which output represents the image? Three choices coexist:

1. `[CLS]` token. Prepend a learnable vector to the patch sequence. After all transformer blocks, the CLS token's hidden state is the image representation. Inherited from BERT. Original ViT and CLIP use this.
2. Mean pool. Average the output hidden states of patch tokens. SigLIP, DINOv2, and most modern VLMs use this.
3. Register tokens. Darcet et al. (2023) found that ViTs trained without explicit sink tokens develop high-norm "artifact" patches that hijack self-attention. Adding 4–16 learnable register tokens absorbs this load, improving dense prediction quality (segmentation, depth). DINOv2 and SigLIP 2 ship with registers.

This choice matters downstream. CLS works for classification. For VLMs that feed patch tokens to an LLM, you skip pooling entirely—every patch becomes an LLM input token. Registers are discarded before handoff (they are scaffolding, not content).

### Pretraining: Supervised, Contrastive, Masked, Self-Distilled

The 2020 ViT was pretrained with supervised classification on JFT-300M. Quickly superseded:

- CLIP (2021): contrastive learning on 400M image-text pairs. See lesson 12.02.
- MAE (2021, He et al.): mask 75% of patches, reconstruct pixels. Self-supervised, runs on images alone.
- DINO (2021) / DINOv2 (2023): teacher-student self-distillation, no labels, no captions. The 2023 DINOv2 ViT-g/14 is the strongest pure-vision backbone and the default for "dense features" use cases.
- SigLIP / SigLIP 2 (2023, 2025): sigmoid-loss CLIP plus native-aspect-ratio NaFlex support. It is the dominant vision tower in 2026 open VLMs (Qwen, Idefics2, LLaVA-OneVision).

Your pretraining choice determines what the backbone excels at: CLIP/SigLIP for semantic matching with text, DINOv2 for dense visual features, MAE as a starting point for downstream fine-tuning.

### Scaling Laws

ViT scaling (Zhai et al. 2022) established: ViT quality follows predictable power laws over model size, data size, and compute. At fixed compute:
- Larger model + more data → better quality.
- Patch size is a lever between sequence length and fidelity. Patch 14 (typical for DINOv2/SigLIP SO400m) produces more tokens per image than patch 16; better for OCR and dense tasks, slower.
- Resolution is another major lever. Going from 224 to 384 to 512 almost always helps, at quadratic FLOPs cost.

ViT-g/14 (1B params, patch 14, resolution 224 → 256 tokens) and SigLIP SO400m/14 (400M params, patch 14) are the two workhorse encoders for 2026 open VLMs.

### ViT Parameter Count

Full computation is in `code/main.py`. For ViT-B/16 at resolution 224:

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Sanity-check every ViT this way before loading a checkpoint. The backbone's size sets the memory floor for any downstream VLM.

### 2026 Production Configuration

Most 2026 open VLMs ship with SigLIP 2 SO400m/14 at native resolution (NaFlex) as the encoder. It has:
- 400M parameters.
- Patch size 14, default resolution 384 → 729 patch tokens per image.
- Mean pool for image-level tasks; all 729 patches flow into the LLM for VQA.
- 4 register tokens, discarded before LLM handoff.
- 2D-RoPE with image-level scaling, supporting native aspect ratios.

Every decision in this configuration traces back to a paper you can read.

## Use It

`code/main.py` is a patch tokenizer plus geometry calculator. It takes (image H, W, patch P, hidden dim D, depth L) and reports:

- Grid shape and sequence length after patching.
- The token sequence for a synthetic 8x8 toy image (walking the flatten + project path).
- Parameter count decomposed by patch embed, positional embed, transformer blocks, and head.
- FLOPs per forward pass at target resolution.
- A comparison table across ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, and SigLIP SO400m/14 @ 384.

Run it. Cross-check param counts against published figures. Vary patch size and resolution to feel the token-count cost.

## Ship It

This lesson produces `outputs/skill-patch-geometry-reader.md`. Given a ViT configuration (patch size, resolution, hidden dim, depth), it outputs token count, parameter count, and memory estimate with reasoning. Use this skill whenever picking a vision backbone for a VLM—it prevents the surprise of "token explosion filling the LLM context."

## Exercises

1. Compute the patch-token sequence length for Qwen2.5-VL at native 1280x720 input, patch size 14. How does it compare to a CLS-only representation?

2. How many tokens does a single 1080p (1920x1080) frame produce at patch 14? For a 5-minute video at 30 FPS, how many total visual tokens? Which strategy saves the most: pooling, frame sampling, or token merging?

3. Implement mean pooling over patch tokens in pure Python. Verify that mean-pooling 196 tokens from DINOv2 output matches the pooled embedding returned by the model's `forward` call.

4. Read Section 3 of "Vision Transformers Need Registers" (arXiv:2309.16588). In two sentences, describe what artifact the registers absorb and why this matters for downstream dense prediction.

5. Modify `code/main.py` to support patch-n'-pack: given a set of images at different resolutions, produce a single packed sequence and block-diagonal attention mask. Validate against lesson 12.06 when you reach it.

## Key Terms

| Term | Common Usage | Actual Meaning |
|------|--------------|----------------|
| Patch | "16x16 pixel block" | A fixed-size, non-overlapping region of the input image; becomes one token |
| Patch embedding | "linear projection" | A shared learnable matrix (or stride-P Conv2d) mapping flattened patch pixels to a D-dim vector |
| CLS token | "class token" | A learnable vector prepended to the sequence whose final hidden state represents the whole image; optional in 2026 |
| Register token | "sink token" | Extra learnable tokens that absorb high-norm attention artifacts ViTs develop during pretraining |
| Position embedding | "positional info" | Per-position vectors or rotations that give the sequence spatial awareness; 2D-RoPE is the modern default |
| Grid | "patch grid" | The (H/P) x (W/P) 2D array of patches given a resolution and patch size |
| NaFlex | "native flexible resolution" | SigLIP 2's ability to serve multiple aspect ratios and resolutions from one checkpoint without retraining |
| Backbone | "vision tower" | The pretrained image encoder whose patch-token output feeds the LLM in a VLM |
| Pooling | "image-level summary" | Strategy for collapsing patch tokens into one vector: CLS, mean, attention pooling, or register-based |
| Patch 14 vs 16 | "finer grid vs coarser grid" | Patch 14 produces more tokens per image with better OCR fidelity but is slower; patch 16 is the classic default |

## Further Reading

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — the original ViT.
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE, self-supervised pretraining.
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — large-scale self-distillation, no labels.
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register tokens and artifact analysis.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — the 2026 default vision tower.
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — empirical scaling laws.
