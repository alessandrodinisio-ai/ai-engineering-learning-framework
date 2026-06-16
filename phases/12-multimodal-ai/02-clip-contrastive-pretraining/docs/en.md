# CLIP and Contrastive Vision-Language Pretraining

> OpenAI's CLIP (2021) proved an idea powerful enough to sustain the next five years: align an image encoder and a text encoder into the same vector space using nothing but noisy web image-text pairs and a contrastive loss. Zero supervised labels. 400 million pairs. The resulting embedding space enables zero-shot classification, image-text retrieval, and serves as the vision tower plugged into every 2026 VLM. SigLIP 2 (2025) replaces softmax with sigmoid to push scale past CLIP at lower cost. This lesson walks through the math from InfoNCE to the sigmoid pairwise loss and builds the training step in pure standard-library Python.

**Type:** Build
**Languages:** Python (standard library, InfoNCE + sigmoid loss implementation)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 minutes

## Learning Objectives

- Derive the InfoNCE loss from mutual information and implement a numerically stable vectorized version.
- Explain why the sigmoid pairwise loss (SigLIP) scales to batch 32768+ without the all-gather overhead softmax requires.
- Run a zero-shot ImageNet classification by constructing text templates (`a photo of a {class}`) and argmaxing cosine similarity.
- Name the four levers CLIP / SigLIP pretraining gives you: batch size, temperature, prompt template, data quality.

## The Problem

Vision before CLIP was supervised. Collect a labeled dataset (ImageNet: 1.2M images, 1000 classes), train a CNN, release. Labels are expensive, biased toward what annotators can agree on, and don't transfer to new tasks without fine-tuning.

Billions of loosely-annotated image-text pairs sit free on the web. A photo of a golden retriever captioned "my dog Max at the park" already carries a supervision signal—the text describes the image. The question is: can you turn it into useful training?

CLIP's answer: treat image-text pairs as a matching task. Given a batch of N images and N captions, learn to match each image to its own caption among N-1 distractors. The supervision signal is "these two things are a pair; those N-1 are not." No class labels. No human annotation. Just a contrastive loss.

The resulting embedding space does far more than CLIP trained for. ImageNet zero-shot works because "a photo of a cat" embeds near cat images that were never explicitly labeled as cats. This bet spawned every 2026 VLM.

## The Concept

### Dual Encoder

CLIP has two towers:

- Image encoder `f`: ViT or ResNet, outputting one D-dim vector per image.
- Text encoder `g`: small transformer, outputting one D-dim vector per caption.

Both towers L2-normalize outputs to unit length. Since both sides are unit-norm, similarity is `cos(f(x), g(y)) = f(x)^T g(y)`.

For a batch of N (image, caption) pairs, build the `(N, N)` similarity matrix `S`:

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

where `tau` is a learnable temperature (CLIP initializes to 0.07; learned in log space).

### InfoNCE Loss

CLIP applies a symmetric cross-entropy over rows and columns:

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

This is InfoNCE. The softmax inside CE forces each image to be more similar to its caption than to all other captions in the batch. The "negatives" are all other items in the batch. Larger batch = more negatives = stronger signal. CLIP trains with batch 32k; scale matters.

### Temperature

`tau` controls the softmax sharpness. Low tau → sharp distribution with hard-negative mining effects. High tau → soft, all examples contribute. CLIP learns log(1/tau), clipped to prevent collapse. SigLIP 2 fixes the initial tau and uses a learnable bias instead.

### Why Sigmoid Scales Better (SigLIP)

Softmax requires the entire similarity matrix to be in sync. In distributed training, you must all-gather every embedding to every replica, then softmax. This is communication-quadratic in world size.

SigLIP replaces softmax with element-wise sigmoid: for each pair `(i, j)`, the loss is an "are these two a match?" binary classification with positive labels on the diagonal and negatives everywhere else. Loss:

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` when `i == j`, else 0. Each pair's loss is independent. No all-gather needed. Each GPU computes its local chunk and sums. SigLIP 2 scales cheaply to batch 32k-512k where CLIP needs proportionally growing communication.

### Zero-Shot Classification

Given N class names, construct a text template for each class:

```
"a photo of a {class}"
```

Embed each template with the text encoder. Embed your image with the image encoder. Argmax of cosine similarity = predicted class. No training on target classes.

Prompt templates matter. The CLIP paper uses 80 templates per class (plain, art, photo, painting, etc.) and averages embeddings, gaining +3 on ImageNet. Modern usage typically picks one or two templates.

### Linear Probe and Fine-Tuning

Zero-shot is a baseline. A linear probe (training a linear layer on frozen CLIP features for your target classes) beats zero-shot on in-domain tasks. Full fine-tuning beats linear probe in-domain but may harm zero-shot transfer. Three paradigms, three trade-offs.

### SigLIP 2: NaFlex and Dense Features

SigLIP 2 (2025) adds:
- NaFlex: single model serves variable aspect ratios and resolutions.
- Better dense features for segmentation and depth estimation, targeting use as a frozen backbone in VLMs.
- Multilingual: trained on 100+ languages versus CLIP's English-only.
- Billion-parameter scale where CLIP capped at 400M.

In 2026 open VLMs, SigLIP 2 SO400m/14 is the default vision tower. CLIP remains the default for pure image-text retrieval—provided its specific LAION-2B training distribution matches your query patterns.

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

ALIGN (Google, 2021): same idea as CLIP at 1.8B-pair scale, 90% noisy. Proved noisy data scales. OpenCLIP (LAION): open reproduction of CLIP on LAION-400M / 2B, multiple scales, the go-to open checkpoint. EVA-CLIP: initialized from masked image modeling; a strong backbone for VLMs. BASIC: Google's CLIP+ALIGN hybrid. All the same family, different data and tuning.

### Zero-Shot Ceiling

CLIP-family models cap at ~76% ImageNet zero-shot (CLIP-G, OpenCLIP-G). Beyond that requires either much more data (SigLIP 2 reaches 80%+) or architectural changes (supervised heads, more parameters). This benchmark is saturating; the real value is the embedding space downstream VLMs consume.

## Use It

`code/main.py` implements:

1. A toy dual encoder (hash-based image features, char-level text features) so you can see the InfoNCE shape without numpy.
2. Pure-Python InfoNCE loss (numerically stable via log-sum-exp).
3. Sigmoid pairwise loss for comparison.
4. A zero-shot classification routine: compute cosine similarity against a set of text prompts, argmax for prediction.

Run it and observe the loss curves. Absolute numbers are toy-scale; the curve shapes match what a real CLIP trainer produces.

## Ship It

This lesson produces `outputs/skill-clip-zero-shot.md`. Given a set of images (by path) and a list of target classes, it constructs CLIP templates as text prompts, embeds both sides with a specified checkpoint (e.g. `openai/clip-vit-large-patch14`), and returns top-1 / top-5 predictions with similarity scores. The skill refuses to make any assertion about classes not in the prompt list.

## Exercises

1. Hand-compute InfoNCE for a 4-pair batch. Construct the 4x4 similarity matrix, run softmax, pick the diagonal, compute cross-entropy. Validate against your Python implementation.

2. SigLIP uses a bias parameter `b` in addition to temperature: `S'[i,j] = S[i,j]/tau + b`. When the batch has severe class imbalance (far more negatives per row than positives), what role does `b` play? Read SigLIP Section 3 (arXiv:2303.15343).

3. Build a zero-shot cat vs dog classifier. Try two prompt templates: `a photo of a {class}` and `a picture of a {class}`. Measure accuracy on 100 test images. Does template ensembling beat single-template?

4. Compute the communication cost of softmax InfoNCE vs sigmoid pairwise loss on 512 GPUs with batch 32k. Which is O(N) and which is O(N^2)? Cite SigLIP Section 4.

5. Read the OpenCLIP scaling laws paper (arXiv:2212.07143, Cherti et al.). Reproduce their data-scaling conclusion from the figures: at fixed model size, what is the log-linear relationship between ImageNet zero-shot accuracy and training data scale?

## Key Terms

| Term | Common Usage | Actual Meaning |
|------|--------------|----------------|
| InfoNCE | "contrastive loss" | Cross-entropy over a batch's similarity matrix; each item's positive is its paired item, negatives are everything else |
| Sigmoid loss | "SigLIP loss" | Per-pair binary cross-entropy; no softmax, no all-gather, scales cheaply in distributed training |
| Temperature | "tau" | Scalar that scales logits before softmax/sigmoid; controls distribution sharpness |
| Zero-shot | "no fine-tuning classification" | Constructing class embeddings via text prompts and classifying by cosine similarity; no training on target classes |
| Prompt template | "a photo of a ..." | Text scaffolding around a class name; swings zero-shot accuracy by 1-5 points |
| Dual encoder | "two-tower" | One image encoder + one text encoder, outputs in a shared D-dim space |
| Hard negative | "tricky distractor" | A negative similar enough to the positive that the model must work hard to separate them |
| Linear probe | "frozen + one layer" | Training only a linear classifier on frozen features; measures feature quality |
| NaFlex | "native flexible resolution" | SigLIP 2's ability to ingest images at any aspect ratio and resolution without resizing |
| Temperature scaling | "log-parameterized tau" | CLIP parameterizes `log(1/tau)` for well-behaved gradients; clipped to prevent collapse to near-zero tau |

## Further Reading

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — the CLIP paper.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — multilingual + NaFlex.
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — scaling with noisy web data.
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP scaling laws.
