# Show-o and Discrete Diffusion Unified Models

> Transfusion mixes continuous and discrete representations. Show-o (Xie et al., August 2024) takes another route: text tokens use causal next-token prediction, image tokens use masked discrete diffusion in the spirit of MaskGIT. Both sit in one transformer with a hybrid attention mask. The result is VQA, text-to-image, inpainting, and mixed-modality generation unified on one backbone, one tokenizer per modality, and one loss formulation (next-token extended to mask prediction). This lesson walks through the Show-o design — why masked discrete diffusion is a parallel, low-step image generator — and compares it against Transfusion and Emu3.

**Type:** Learn
**Languages:** Python (standard library, masked discrete diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## Learning Objectives

- Explain masked discrete diffusion: the schedule of uniformly masking tokens then having the transformer recover them.
- Compare parallel image decoding (Show-o, MaskGIT) vs autoregressive image decoding (Chameleon, Emu3) on speed and quality.
- State the three tasks Show-o handles in one checkpoint: T2I, VQA, inpainting.
- Pick a mask schedule (cosine, linear, truncated) and reason about its effect on sample quality.

## The Problem

Transfusion's dual-loss training works, but the dynamics are trickier — the continuous diffusion loss and discrete NTP loss live on different numerical scales. Balancing loss weights is a hyperparameter search. The architecture is effective but complex.

Show-o's answer: keep both modalities discrete (like Chameleon), but generate images in parallel via masked discrete diffusion rather than sequentially. The training objective becomes a single masked-token prediction that naturally generalizes next-token prediction.

## The Concept

### Masked Discrete Diffusion (MaskGIT)

The original MaskGIT trick (Chang et al., 2022) is elegant. Start from a fully masked image (every token is the special `<MASK>` id). At each step, predict all masked tokens in parallel, then keep the top-K most confident predictions and re-mask the rest. After ~8-16 iterations, all tokens are filled. The schedule of how many tokens to unmask per step is tuned — cosine schedule works well.

Training is simple: uniformly sample a mask ratio from [0, 1], apply it to the image's VQ tokens, train the transformer to recover the masked ones. Identical to what BERT does for text, scaled up to image generation.

### Show-o: One Transformer, Hybrid Mask

Show-o puts MaskGIT inside a causal language model transformer. The attention mask is:

- Text tokens: causal (standard LLM).
- Image tokens: fully bidirectional within the image block (so masked tokens can see every other image token when predicting).
- Text-to-image: text attends to previous images, images attend to previous text.

Training alternates between:
1. Standard NTP on text sequences.
2. T2I samples: text → image with masked image tokens, mask-token prediction loss.
3. VQA samples: image → text with masked text tokens (which is just NTP).

The unified loss is cross-entropy on `<MASK>` tokens, which covers both text NTP (only the last token is "masked") and image masked diffusion (random subsets are masked).

### Parallel Sampling

Show-o generates an image in ~16 steps rather than ~1000 (token-by-token autoregressive) or ~20 (diffusion). Each step, predict all masked tokens in parallel; commit the top-K confident ones; repeat.

Comparison:
- Chameleon / Emu3 (autoregressive across tokens): N_tokens forward passes, typically 1024-4096 per image.
- Transfusion (continuous diffusion): ~20 steps, one full transformer forward each.
- Show-o (masked discrete diffusion): ~16 steps, one full transformer forward each.

At comparable model scale, Show-o is faster than Chameleon, roughly matches Transfusion in step count but with lower per-step cost (discrete vocab logits vs continuous MSE loss).

### Tasks in One Checkpoint

Show-o supports four tasks at inference, selected by prompt format:

- Text generation: standard autoregressive text output.
- VQA: image in, text out.
- T2I: text in, image out via masked discrete diffusion.
- Inpainting: image with partially masked tokens, fill in.

Inpainting comes free from the mask-prediction training. Mask a region of the VQ-token grid, feed the rest plus a text prompt, predict the masked tokens.

### Mask Schedule

The schedule of how many tokens to unmask per step shapes quality. Show-o recommends cosine:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

At step 0, all tokens are masked (ratio 1.0). At step T, no mask. Cosine concentrates quality on predicting the most informative mid-range ratios. Linear schedule also works but plateaus faster.

### Show-o2

Show-o2 (2025 sequel, arXiv 2506.15564) scales Show-o: larger LLM base, better tokenizer, improved mask schedule. Same architectural pattern.

### Where Show-o Sits

In the 2026 taxonomy:

- Discrete tokens + NTP: Chameleon, Emu3. Simple but slow inference.
- Discrete tokens + masked diffusion: Show-o, MaskGIT, LlamaGen, Muse. Parallel sampling, still bounded by tokenizer lossiness.
- Continuous + diffusion: Transfusion, MMDiT, DiT. Highest quality, more complex training.
- Continuous + flow matching in VLMs: JanusFlow, InternVL-U. Latest.

Pick by task: Show-o when you want T2I + inpainting + VQA in a single open model at reasonable speed; Transfusion when quality is paramount and you can afford the dual-loss pipeline.

## Use It

`code/main.py` simulates Show-o sampling:

- A toy grid of 16 VQ tokens.
- A mock "transformer" that predicts logits based on prompt and current unmasked tokens.
- Parallel mask sampling over 8 steps with cosine schedule.
- Prints intermediate states (mask pattern evolution) and final tokens.

Run it and watch the mask ablate step by step.

## Ship It

This lesson produces `outputs/skill-unified-gen-model-picker.md`. Given a product that needs both understanding (VQA, captioning) and generation (T2I, inpainting) with open-weight constraints, it picks among Show-o family, Transfusion/MMDiT family, and Emu3/Chameleon family with concrete trade-offs.

## Exercises

1. Masked discrete diffusion samples in ~16 steps. Why not 1 step? What breaks if you unmask everything at step 0?

2. Inpainting is free in masked diffusion. Propose a product use case (real or hypothetical) where Show-o's inpainting beats a dedicated model.

3. Cosine schedule vs linear schedule: trace the number of unmasked tokens per step for T=8. Which is more balanced?

4. A 512x512 Show-o image is 1024 tokens. At vocabulary K=16384, the model outputs 1024 * log2(16384) = 14,336 bits (~1.75 KiB) of data. Stable Diffusion outputs 512*512*24 bits = 6,291,456 bits (~768 KiB) of raw pixels. What's the compression ratio, and what quality does it buy?

5. Read LlamaGen (arXiv:2406.06525). How does LlamaGen's class-conditional autoregressive image model differ from Show-o's masked route?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Masked discrete diffusion | "MaskGIT-style" | Training to predict masked tokens; inference iteratively unmasks the most confident predictions |
| Cosine schedule | "unmask schedule" | Decay of mask ratio over inference steps; concentrates confidence growth in mid-range |
| Parallel decoding | "all tokens at once" | Each step predicts the full masked token sequence in one forward pass, then commits top-K |
| Hybrid attention | "causal + bidirectional" | Mask that is causal across text tokens and bidirectional within image blocks |
| Inpainting | "fill-in generation" | Conditioning on an image with partially masked tokens, predicting the missing ones; comes free from the training objective |
| Commit rate | "top-K per step" | How many tokens are declared "done" each iteration; controls inference vs quality trade-off |

## Further Reading

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
