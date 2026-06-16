# Emu3: Next-Token Prediction for Image and Video Generation

> BAAI's Emu3 (Wang et al., September 2024) is the 2024 result that should have ended the diffusion-vs-autoregressive debate. A single Llama-style decoder-only transformer, trained only with a next-token prediction objective, across a unified vocabulary (text + VQ image tokens + 3D VQ video tokens), beats SDXL on image generation and beats LLaVA-1.6 on perception. No CLIP loss. No diffusion schedule. Classifier-free guidance at inference for quality, but the core training objective is next-token prediction with teacher forcing. Published in Nature. This lesson walks through the Emu3 argument—why a better tokenizer plus scale is all you need—and contrasts it with the diffusion route.

**Type:** Learn
**Languages:** Python (standard library, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## Learning Objectives

- Explain why Emu3's single-loss next-token objective works despite the long-held assumption that image quality requires diffusion.
- Describe the 3D video tokenizer: what a spatiotemporal VQ codebook looks like and why patches span time.
- Compare Emu3 with Stable Diffusion XL on (training compute, inference cost, quality ceiling).
- Name the three roles a single Emu3 model plays: Emu3-Gen (image generation), Emu3-Chat (perception), Emu3-Stage2 (video generation).

## The Problem

The conventional wisdom up to 2024 was: image generation requires diffusion. The argument: discrete image tokens lose too much information to reconstruct fine detail, and autoregressive sampling accumulates error across thousands of tokens. Stable Diffusion, DALL-E 3, Imagen, Midjourney all use some form of diffusion. Chameleon (Lesson 12.11) partially refuted this at small scale but didn't match SDXL on quality.

Emu3 attacks this argument head-on. Its claim: a better visual tokenizer + sufficient scale + next-token loss = image generation that beats diffusion, in the same model that also does perception.

The bet was controversial at publication time. Two years later, open-source unified generation families (Emu3, Show-o, Janus-Pro, Transfusion) are the default research path; production frontier models appear to use some variant.

## The Concept

### The Emu3 Tokenizer

The key ingredient is the visual tokenizer. Emu3 trains a custom IBQ-class tokenizer (Inverse Bottleneck Quantizer, SBER-MoVQGAN family) that does 8× resolution reduction per token. A 512×512 image becomes 64×64 = 4096 tokens at codebook size 32768.

This is more tokens than Chameleon's 1024 per 512×512 image at K=8192, but each token is cheaper (smaller codebook lookup, simpler encode/decode). The key metric: reconstruction PSNR of 30.5 dB, on par with Stable Diffusion's continuous latent space at 32 dB.

For video: a 3D VQ tokenizer encodes a spatiotemporal patch (4×4×4 pixels) into a single integer. A 4-second, 8-FPS clip has 32 frames; at 256×256 with 4× spatial and 4× temporal reduction, token count is (256/4) × (256/4) × (32/4) = 64 × 64 × 8 = 32,768 tokens.

Tokenizer quality is the ceiling. Part of Emu3's contribution is "we trained a very good tokenizer."

### Single-Loss Training

Emu3 trains with one objective: next-token prediction over a shared vocabulary spanning text tokens, 2D image tokens, and 3D video tokens. During training, weights are multiplied by modality-specific coefficients to balance contributions, but the loss function is identical.

Trained on a mix:
- Image generation: `<text caption> <image> image_tokens </image>`
- Image perception: `<image> image_tokens </image> <question> text_tokens`
- Video generation: `<text caption> <video> video_tokens </video>`
- Video perception: similar.
- Pure text: standard NTP.

The model learns from the data distribution when to emit image tokens and when to emit text. Generation emerges from the model predicting image tokens after the `<image>` tag.

### Classifier-Free Guidance and Temperature

Autoregressive image generation benefits enormously from classifier-free guidance (CFG) at inference. Emu3 uses it: generate twice, once with the full caption and once with an empty caption, blend logits with a guidance weight (typical 3.0–7.0). This is the same CFG trick diffusion uses, borrowed into the autoregressive setting.

Temperature matters: too high produces artifacts; too low produces mode collapse. Emu3's recommended temperature is 1.0 for perception and 0.8 for image generation.

### Three Roles, One Model

Emu3 ships as three functionally distinct APIs but with one set of weights underneath:

- Emu3-Gen. Image generation. Input text, output image tokens.
- Emu3-Chat. VQA and captioning. Input image (tokens), output text.
- Emu3-Stage2. Video generation and video VQA. Input text or video, output text or video.

No task-specific heads. Just different prompt templates. Same checkpoint.

### Benchmarks

From the Emu3 paper (September 2024):

- Image generation: beats SDXL on MJHQ-30K FID (5.4 vs 5.6), GenEval overall (0.54 vs 0.55—statistical tie), Deep-Eval composite tie.
- Image perception: beats LLaVA-1.6 on VQAv2 (75.1 vs 72.4), roughly ties on MMMU.
- Video generation: 4-second clip quality competitive with Sora-era publicly benchmarked models on FVD.

Numbers don't always win—Emu3 gives a point here, takes one there—but the claim "next-token prediction is all you need" holds across modalities.

### Compute Cost

Emu3 trains a 7B-parameter model on ~300B multimodal tokens. GPU hours are roughly comparable to Llama-2-7B pretraining (2k–4k GPU-years on A100-class silicon). Diffusion models like Stable Diffusion 3 train in similar budgets but need separate text encoders and more complex pipelines.

At inference, Emu3 is slower per image than SDXL: 4096 image tokens at 30 tok/s ≈ ~2 minutes per 512×512 image vs 2–5 seconds for SDXL. Speculative decoding and KV-cache optimization narrow but don't close the gap. Autoregressive image generation is compute-heavy; this is a persistent trade-off.

### Why It Matters

Emu3's deeper contribution is conceptual. If next-token prediction scales to match diffusion on image generation, then the unified-model path (one loss, one backbone, any modality) is viable. Future models don't need separate text encoders, separate diffusion schedulers, separate VAEs. One transformer, one tokenizer per modality, plus scale.

Show-o, Janus-Pro, and InternVL-U all build on or challenge this argument. Through 2025, Chinese labs (BAAI, DeepSeek) published more aggressively in this direction than US labs.

## Use It

`code/main.py` builds two toy pieces:

- A 2D vs 3D VQ tokenizer counter: given (resolution, patch, clip length, FPS), computes image vs video token counts.
- An autoregressive image-token sampler with classifier-free guidance at temperature.

The CFG implementation matches Emu3's recipe—blending conditional and unconditional logits with a guidance weight.

## Ship It

This lesson produces `outputs/skill-token-gen-cost-analyzer.md`. Given a generation product spec (image or video, target resolution, quality tier, latency budget), it computes token counts, inference cost, and picks between the Emu3 family vs diffusion.

## Exercises

1. Emu3 produces 4096 tokens per 512×512 image at 8× reduction. Compute the equivalent for 1024×1024 and 2048×2048. How does inference latency scale?

2. Read Emu3 Section 3.3 on the video tokenizer. Describe the 3D VQ patch shape and why it is 4×4×4 rather than 8×8×1.

3. Classifier-free guidance weight 5.0 vs 3.0: what's the visual difference? Trace the math in `code/main.py`.

4. Compute training FLOPs for Emu3-7B on 300B tokens and compare with Stable Diffusion 3. Which training is more expensive?

5. Emu3 beats SDXL on FID but underperforms dedicated VLMs on VQAv2. Explain why the unified-loss route shows different strengths relative to specialists across benchmarks.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Next-token prediction | "NTP" | Standard autoregressive loss: given tokens[0..i], predict token[i+1]; works for every modality once tokenized |
| IBQ tokenizer | "inverse bottleneck quantizer" | A class of VQ-VAE with larger codebook (32768+) and better reconstruction than Chameleon |
| 3D VQ | "spatiotemporal quantizer" | A codebook indexed by (time, row, col); one token covers a 4×4×4 pixel cube |
| Classifier-free guidance | "CFG" | Blending conditional and unconditional logits with weight gamma; improves image quality at inference |
| Unified vocabulary | "shared tokens" | Text + image + video all draw from the same integer space; the model predicts which modality comes next |
| MJHQ-30K | "image gen benchmark" | Midjourney-quality benchmark with 30k prompts; where Emu3 reports FID |

## Further Reading

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
