# Any-Resolution Vision: Patch-n'-Pack and NaFlex

> Real images aren't 224x224 squares. A receipt is 9:16, a chart is 16:9, a medical scan might be 4096x4096, a phone screenshot is 9:19.5. The pre-2024 VLM answer — resize everything to a fixed square — destroys the signals that make OCR, document understanding, and high-resolution scene parsing work. NaViT (Google, 2023) proved you can pack variable-resolution patches into a single transformer batch using block-diagonal masks. Qwen2-VL's M-RoPE (2024) dropped the absolute position table entirely. LLaVA-NeXT's AnyRes slices high-res images into base image + sub-images. SigLIP 2's NaFlex variant (2025) is now the default encoder for open VLMs that want to serve all aspect ratios from a single checkpoint. This lesson implements patch-n'-pack end-to-end.

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 min

## Learning Objectives

- Pack patches from a batch of variable-resolution images into a single sequence and build the block-diagonal attention mask.
- Choose between AnyRes tiling (LLaVA-NeXT), NaFlex (SigLIP 2), and M-RoPE (Qwen2-VL) for a given task.
- Compute token budgets for OCR, charts, and photography images without rescaling.
- Name three failure modes of square rescaling: squished text, cropped content, tokens wasted on padding.

## The Problem

Transformers expect a sequence. A batch is a stack of equal-length sequences. If your images are all 224x224, you get 196 patch tokens every time, no padding, done. Train at 224, infer at 224, never think about resolution again.

But the world doesn't comply. Documents are portrait (8.5x11 inches, ~2:3). Chart screenshots are landscape (16:9). Receipts are tall and narrow (1:3). Medical images ship at 2048x2048 or larger. Mobile device screenshots are 1170x2532 (0.46:1).

The three options before 2024 and why each fails:

1. Resize to fixed square (224x224 or 336x336). Squashing distorts text and faces. Downsampling destroys chart labels and OCR content. This was the standard pre-LLaVA-1.5.
2. Crop to fixed aspect ratio. You throw away most of the image, and choosing the crop location is itself a vision problem.
3. Pad to longest edge. Fixes distortion, but portrait images have 50%+ tokens wasted on padding. All those padding tokens still pay quadratic attention cost.

The 2024-2025 answer: let the transformer eat patches at the image's native resolution, and figure out how to pack a heterogeneous batch into a single sequence without wasting compute.

## The Concept

### NaViT and Patch-n'-Pack

NaViT (Dehghani et al., 2023) is the paper that proved this works at scale. The idea is mechanical:

1. For each image in the batch, compute its native patch grid at the chosen patch size (e.g. 14).
2. Flatten each image's patches into its own variable-length sequence.
3. Concatenate all images' patches into one long sequence for the batch.
4. Build a block-diagonal attention mask so image A's patches only attend to each other.
5. Carry positional information per patch (2D RoPE or fractional position embeddings).

A batch of three images — 336x336 (576 tokens), 224x224 (256 tokens), 448x336 (768 tokens) — becomes a 1600-token sequence with a 1600x1600 block-diagonal mask. No padding. No wasted compute. The transformer handles arbitrary aspect ratios.

NaViT also introduces fractional patch dropout at training time — randomly dropping 50% of patches uniformly across the whole batch — both regularizing and speeding up training. SigLIP 2 inherited this.

### AnyRes (LLaVA-NeXT)

LLaVA-NeXT's AnyRes is the pragmatic alternative. Given a high-res image and a fixed encoder (CLIP or SigLIP at 336), tile the image:

1. Pick a grid layout from a predefined set — (1x1), (1x2), (2x1), (1x3), (3x1), (2x2), etc. — that best matches the image's aspect ratio.
2. Slice the full image into the grid; each tile becomes a 336x336 crop.
3. Also produce a thumbnail: the full image resized to 336x336 as global context tokens.
4. Run each tile through the frozen 336 encoder. Concatenate tile tokens + thumbnail tokens.

A 672x672 image uses a 2x2 grid plus thumbnail: 4 * 576 + 576 = 2880 visual tokens. Expensive but effective — the LLM sees both local detail and global context.

AnyRes is the go-to route when your encoder is frozen and supports only a single resolution. It explodes token count for large images (a 1344x1344 image on a 4x4 grid is 9216 + 576 ≈ 9800 tokens, filling most of an 8k LLM context).

### M-RoPE (Qwen2-VL)

Qwen2-VL introduces Multi-modal Rotary Position Embedding. Instead of NaViT's fractional positions or AnyRes's tile-plus-thumbnail, each patch carries a 3D position (time, height, width). Query/key rotations handle arbitrary H, W, and temporal lengths.

M-RoPE gets native dynamic resolution without retraining. At inference you feed any HxW image, the patch embedder produces H/14 x W/14 tokens, each token gets its (t=0, r=row, c=col) position, RoPE rotates attention with the correct frequencies, done. Qwen2.5-VL and Qwen3-VL continue this. InternVL3's V2PE is the same idea with variable encoding per modality.

Unlike AnyRes, M-RoPE is O(H x W / P^2) tokens at native resolution — no multiplicative overhead from tiling. Unlike NaViT, it still expects one image per forward pass. Cross-resolution batching still requires patch-n'-pack on top.

### NaFlex (SigLIP 2)

NaFlex is the native flexible mode of SigLIP 2 checkpoints. A single model serves multiple sequence lengths (256, 729, 1024 tokens) at inference. Internally it uses NaViT-style patch-n'-pack at training time with fractional absolute positions per patch. The selling point: one checkpoint, pick your token budget per task at inference.

Semantic tasks (classification, retrieval) use 256 tokens. OCR or chart understanding uses 1024 tokens. No retraining.

### Packing Masks

The block-diagonal mask is where most implementations trip up. For a packed sequence of length `N_total` covering images `i=0..B-1` with respective lengths `n_i`, the mask `M` of shape `(N_total, N_total)`: entries are 1 when both indices fall within the same image's block, 0 otherwise. You can build it from a cumulative length list:

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

In PyTorch this is one line with `torch.block_diag` or an explicit gather. FlashAttention's varlen path (`cu_seqlens`) skips the mask entirely, using the cumulative-length tensor to do attention within each sequence — ~10x faster than the dense mask for typical batches.

### Token Budget

Pick your strategy per task:

- OCR / documents: 1024-4096 tokens. SigLIP 2 NaFlex at 1024, or AnyRes 3x3 + thumbnail.
- Charts and UI: 729-1024 tokens at 384-448 native. Qwen2.5-VL dynamic resolution with a max pixel cap.
- Natural photos: 256-576 tokens suffice. The downstream LLM sees enough. Spend tokens where content density is high.
- Video: 64-128 tokens per frame after spatial pooling, 2-8 FPS. Lesson 12.17 covers this.

2026 production rule: pick a max pixel cap per task, encode at native aspect ratio within that cap, pack the batch, skip padding. Qwen2.5-VL exposes `min_pixels` and `max_pixels`, exactly this knob.

## Use It

`code/main.py` implements patch-n'-pack for a heterogeneous image batch (integer pixel coordinates). It:

- Takes a set of (H, W) image sizes.
- Computes the patch sequence length per image at patch size 14.
- Packs them into a single sequence of total length `sum(n_i)`.
- Builds a block-diagonal attention mask (dense, for clarity).
- Compares packing cost vs square resize vs AnyRes tiling.
- Prints a token budget table for a mixed batch (receipt, chart, screenshot, photo).

Run it. The numbers that fall out are why every 2026 open VLM uses patch-n'-pack.

## Ship It

This lesson produces `outputs/skill-resolution-budget-planner.md`. Given a mixed-aspect-ratio workload (OCR, charts, photos, video frames) and a total token budget, it picks the right strategy (NaFlex, AnyRes, M-RoPE, or fixed square) and produces a per-request configuration. Use this skill when specifying a VLM for a product — it prevents the silent 10x token blowup that crashes latency budgets.

## Exercises

1. A receipt is 600x1500 (1:2.5). How many native-resolution tokens at patch size 14? How many after square resize to 336? Which loses more OCR accuracy in practice?

2. Build the block-diagonal mask for a 4-image batch with lengths 256, 576, 729, 1024. Verify the attention matrix is 2585x2585 and has exactly `256^2 + 576^2 + 729^2 + 1024^2` nonzero entries.

3. For a 1792x896, patch-14 image, compare: (a) square resize to 336 then encode, (b) AnyRes 2x1 + thumbnail, (c) native M-RoPE. Which uses the fewest tokens? Which preserves the most detail?

4. Implement fractional patch dropout: given a packed sequence, uniformly randomly drop 50% of tokens and update the block-diagonal mask accordingly. Measure the change in mask sparsity.

5. Read the Qwen2-VL paper (arXiv:2409.12191) Section 3.2. Describe in two sentences what `min_pixels` and `max_pixels` control, and why both bounds matter.

## Key Terms

| Term | Common phrasing | What it actually means |
|------|-----------------|------------------------|
| Patch-n'-pack | "NaViT-style packing" | Concatenating variable-length patch sequences from different images into one batch dimension |
| Block-diagonal mask | "Packing mask" | Attention mask restricting each image's patches to attend only to themselves, not neighbors in the pack |
| AnyRes | "LLaVA-NeXT tiling" | Slicing a high-res image into a grid of fixed-size tiles plus a global thumbnail; encoding each tile with a fixed encoder |
| NaFlex | "SigLIP 2 native flex" | A single SigLIP 2 checkpoint that serves 256/729/1024-token budgets at inference without retraining |
| M-RoPE | "Multimodal RoPE" | 3D rotary position encoding (time, row, col) handling arbitrary H, W, T without a position table |
| cu_seqlens | "FlashAttention packing" | Cumulative-length tensor used by FlashAttention's varlen path to replace the dense block-diagonal mask |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL's per-request knob capping token count for very small or very large inputs |
| Visual token budget | "Tokens per image" | The rough number of patch tokens produced per image; determines LLM prompt budget and attention cost |

## Further Reading

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
