# Vision Encoder Patch Splitting

> A vision model that reads pixels needs a dedicated pixel tokenizer. Patch embedding is that tokenizer. It slices the image into a square grid, flattens each square, passes it through a linear projection, and adds a 2D positional signal so the transformer knows where each square originally sat in the image.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19, Lessons 30-37 (Track B foundations)
**Time:** ~90 minutes

## Learning Objectives

- Tokenize an image into a fixed-length sequence of patch embeddings.
- Implement a `Conv2d`-based patch projection that is mathematically equivalent to unfold followed by a linear layer.
- Build a deterministic 2D sinusoidal positional embedding that encodes spatial position into token order.
- Verify patch count, embedding shape, and `Conv2d`-vs-unfold equivalence on a synthetic fixture.

## The Problem

A transformer consumes a sequence of vectors. An image is a three-channel grid. Treating every pixel as a token causes the sequence length to explode: a 224x224 RGB image yields 150,528 tokens, and the attention cost of a 12-layer transformer simply cannot handle that. Reading the entire image as one massive flat vector destroys locality, and attention layers cannot recover it. The encoder front-end's job is to compress the pixel grid into a few hundred tokens, each summarizing a square region.

Patch embedding solves this with a single linear projection. A 224x224 image split into 16x16 patches produces a 14x14 grid of 196 patches. Each patch is flattened from `(3, 16, 16) = 768` pixel values into a vector, then mapped to the model's hidden dimension by a linear layer. The transformer sees 196 tokens of dimension `hidden` (typically 768), plus a CLS token. That is the sequence the rest of the network can digest.

## The Concept

```mermaid
flowchart LR
  Image[224x224x3 image] --> Cut[Split into 16x16 patches]
  Cut --> Grid[14x14 patch grid]
  Grid --> Flatten[Flatten each patch]
  Flatten --> Proj[Linear projection]
  Proj --> Tokens[196 tokens of dimension hidden]
  Tokens --> Pos[Add 2D sinusoidal position]
  Pos --> Out[Final token sequence]
```

### Why patches instead of pixels

Attention cost grows quadratically with sequence length. A 196-token sequence requires `196 * 196 = 38,416` attention scores per head per layer; a 150,528-token sequence requires `150,528 * 150,528 = 22.6 billion`. Patches reduce attention computation by roughly 590,000x, and a single 16x16 region carries enough signal for high-level visual tasks. The trade-off is losing fine-grained spatial detail within a patch, which is why downstream multimodal systems often run an additional high-resolution branch when precise localization is needed.

### Why a single linear projection suffices

Each patch is treated as an independent vector. The projection learns a set of bases: edge detectors, color filters, simple textures. A single linear layer is small (`768 * 768 = 589,824` parameters for ViT-Base) and trains quickly. Deeper convolutional stems exist (the "hybrid" ViT), but a flat linear projection is the standard, and most modern open-source encoders use exactly this shape.

### The `Conv2d` trick

A `Conv2d(in_channels=3, out_channels=hidden, kernel_size=patch_size, stride=patch_size)` with no padding produces numerically identical results to unfold-then-linear, because each output position is a dot product between patch pixels and a filter. This convolution is the patch projection; most production codebases implement it this way because it is faster on GPU and avoids one reshape.

### Positional embedding

Tokens emerge from the projection with no ordering information. A 2D sinusoidal embedding gives each token a fixed signal encoding its `(row, col)` position. Half the embedding dimensions encode row position with sin/cos at multiple frequencies; the other half encodes column position. This encoding is deterministic, so you can change resolution without retraining, and it interpolates cleanly to grids the model has never seen during training.

| Component | Shape | Parameters |
|-----------|-------|------------|
| Patch projection (`Conv2d`) | `(hidden, 3, patch, patch)` | `3 * P * P * hidden + hidden` |
| Positional embedding (fixed) | `(num_patches, hidden)` | 0 (computed, not learned) |
| CLS token (learned) | `(1, hidden)` | `hidden` |

For ViT-Base/16 at 224 resolution: 590,592 parameters in the projection, 768 for the CLS token, zero for sinusoidal positions. The next lesson (Lesson 59) stacks a 12-layer transformer on top of this front-end.

### Equivalence as a sanity check

The patch step can be written two ways: a `Conv2d` projection, and an explicit unfold followed by a linear layer. Given the same weights, they must produce the same output. If they differ, the unfold math is wrong, and the rest of the encoder is built on sand. This lesson's tests specifically verify that equivalence.

## Build It

`code/main.py` implements:

- `PatchEmbed`, an `nn.Module` wrapping `Conv2d` for patch projection.
- `sinusoidal_2d(grid_h, grid_w, dim)`, a stateless function that builds the 2D positional table.
- `VisionFrontEnd`, which combines patch embedding, CLS prepend, and positional addition in one forward pass.
- `synthesize_image(seed)` helper that builds a deterministic 224x224x3 fixture using `numpy.random`.
- A demo that passes a fixture image through the front-end and prints output shape, CLS token norm, and one row of the positional embedding.

Run it:

```bash
python3 code/main.py
```

Output: the 224x224 fixture is tokenized into a sequence of shape `(1, 197, 768)`. The first token is CLS; the next 196 are patch tokens. The positional embedding norm is uniform within a row — the signature of a sinusoidal signal.

## Use It

The same patch front-end appears in every modern vision-language model: CLIP ViT-L/14, SigLIP, DINOv2, the Qwen-VL family, InternVL systems — all start with a `Conv2d` patch projection plus a positional signal. Differences across families lie downstream (CLS pooling vs. no-CLS pooling, register tokens, patch size 14 vs. 16, dynamic resolution via positional interpolation). This lesson's front-end is the bedrock on which all the above stand.

## Ship It

`code/test_main.py` covers:

- Patch count equals `(image_size / patch_size) ** 2`
- Output shape equals `(batch, num_patches + 1, hidden)`
- `Conv2d` projection matches hand-written unfold-then-linear on a small fixture
- Sinusoidal positional table is deterministic across multiple calls
- CLS token broadcasts over the batch dimension without leaking

Run them:

```bash
python3 -m unittest code/test_main.py
```

## Exercises

1. Replace sinusoidal positions with a learnable `nn.Parameter` and compare first-epoch loss on a small synthetic classification task. Learnable positions win at fixed resolution; sinusoidal wins when resolution changes after training.

2. Replace the `Conv2d` with explicit `nn.Unfold` plus `nn.Linear` and assert outputs match within floating-point tolerance. Same math, two spellings.

3. Add support for non-square patch sizes (e.g., 32x16 for wide-format inputs) and verify the positional table handles non-square grids.

4. Profile the patch step at batch sizes 1, 8, and 64. Patch projection is rarely the bottleneck; downstream attention layers dominate.

5. Use the front-end as a frozen feature extractor and train on a 4-class synthetic shape dataset (circle, square, triangle, star). The CLS token output should be linearly separable.

## Key Terms

| Term | Meaning |
|------|---------------|
| Patch | A square sub-region of the image, typically 14x14 or 16x16 |
| Patch embedding | Linear projection of a flattened patch into hidden dimension |
| Sequence length | Number of tokens after patch tokenization, usually plus CLS |
| Sinusoidal position | Fixed sin/cos signals encoding 2D grid coordinates |
| CLS token | Learnable vector prepended to the sequence, serving as a pooling head |

## Further Reading

- An Image is Worth 16x16 Words (ViT, 2021) — the original patch-embed framework.
- Attention Is All You Need (2017) — the sinusoidal position formula adapted here to 2D.
- DINOv2 paper — register tokens, which you can add as Exercise 6.
