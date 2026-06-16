# Chameleon and Early-Fusion Token-Only Multimodal Models

> Every VLM we've seen so far keeps images and text separate. Visual tokens come from a vision encoder, flow through a projector, then meet text inside the LLM. The visual vocabulary and text vocabulary never overlap. Chameleon (Meta, May 2024) asks: what if they did? Train a VQ-VAE that turns images into a sequence of discrete tokens from a shared vocabulary. Now every multimodal document is one flat sequence—text tokens and image tokens interleaved, one autoregressive loss. The side effect: the model can generate mixed-modality output in a single inference call—alternating text tokens and image tokens. This lesson walks through the early-fusion argument and builds a toy version end to end.

**Type:** Build
**Languages:** Python (standard library, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## Learning Objectives

- Explain why a shared vocabulary + single loss changes what the model can do.
- Describe how VQ-VAE tokenizes an image into a discrete sequence compatible with the transformer's next-token objective.
- Name Chameleon's training stability tricks: QK-Norm, dropout placement, LayerNorm ordering.
- Compare Chameleon with BLIP-2's Q-Former route and describe when each is the right choice.

## The Problem

Adapter-based VLMs (LLaVA, BLIP-2, Qwen-VL) treat text and images as two different things. Text tokens go through `embed(text_token)`; images go through `visual_encoder(image) → projector → ... pseudo_tokens`. The model has two input paths that merge midway.

Three consequences:

1. The LLM can only consume images, not produce them. Output is text-only.
2. Mixed-modality documents (paragraphs and images alternating, like an article) are awkward—you either parse multimodal input outside the model or chain generations.
3. Distribution mismatch. Visual tokens and text tokens live in different regions of hidden space, creating subtle alignment issues.

Chameleon rejects the premise: images are just a sequence of discrete tokens from a shared vocabulary. Train the model on interleaved documents, one loss, one autoregressive decoder, and you unlock mixed-modality generation for free.

## The Concept

### VQ-VAE as Image Tokenizer

The tokenizer is a vector-quantized variational autoencoder. Architecture:

- Encoder: CNN + ViT, maps an image to a spatial feature map, e.g., 32×32 features of dimension 256.
- Codebook: a learned vocabulary of K vectors (Chameleon uses 8192), also dimension 256.
- Quantization: for each spatial feature, find the nearest codebook entry by L2 distance. Replace the continuous feature with the integer index.
- Decoder: CNN, reconstructs pixels from quantized features.

Training: VAE reconstruction loss + commitment loss + codebook loss. The codebook indices form the discrete alphabet for images.

For Chameleon: one image becomes 32×32 = 1024 tokens drawn from a vocabulary of 8192. Concatenated with text tokens (from the LLM's BPE vocabulary, e.g., 32000). Final vocabulary: 40192. The transformer sees one sequence, one loss.

### Shared Vocabulary

Chameleon's vocabulary merges text tokens, image tokens, and modality delimiters. Each token has a single ID. The input embedding layer maps each ID to a D-dimensional hidden vector. The output projection maps hidden to vocabulary logits. Softmax picks the next token regardless of modality.

Delimiters matter: `<image>` and `</image>` tags bracket image token sequences. At generation time, if the model emits `<image>`, downstream software knows the next 1024 tokens are VQ indices to be sent to the decoder to render pixels.

### Mixed-Modality Generation

Inference is next-token prediction over the shared vocabulary. Example prompt: "Draw a cat and describe it." Chameleon emits:

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

The model autonomously picks the order—it may go image-first then text, text-first then image, or interleaved. Same decoder, same loss.

Compare to adapter VLMs that can only generate text. Chameleon reopens the question of model output modality.

### Training Stability — QK-Norm, Dropout, LayerNorm Ordering

Early-fusion training is unstable at scale. Chameleon's paper documents three tricks:

- QK-Norm. Apply LayerNorm to query and key projections before the dot product inside attention. Prevents logit magnitude explosion at depth. Adopted by multiple post-2024 large models.
- Dropout placement. Dropout after every residual addition, not just after attention and MLP. More regularization is needed when gradients from image tokens can dominate.
- LayerNorm ordering. Pre-LN on residual branches (standard), plus an extra LN on the final block skip connection. Stabilizes last-layer gradient flow.

Without these tricks, 34B-parameter Chameleon training diverged at multiple checkpoints. With them, it converges. The training recipe is as much a contribution as the architecture.

### Tokenizer Reconstruction Ceiling

VQ-VAE is lossy. At 8192 codebook entries and 1024 tokens per 512×512 image, reconstruction PSNR caps at ~26–28 dB. This is sufficient for recognizable image generation but noticeably worse than continuous-space diffusion (Stable Diffusion 3 reaches 32+ dB).

The tokenizer is the bottleneck. Better tokenizers (MAGVIT-v2, IBQ, SBER-MoVQGAN) raise the ceiling. Emu3 (Lesson 12.12) achieves SDXL-quality generation with a better tokenizer alone.

### Chameleon vs BLIP-2 / LLaVA

Chameleon (early fusion, shared vocabulary):
- One loss, one decoder.
- Generates mixed-modality output.
- Tokenizer is the quality ceiling.
- Expensive: inference path runs a VQ-VAE decoder for every generated image.

BLIP-2 / LLaVA (late fusion, separate towers):
- Vision in, text-only out.
- Reuses pretrained LLM.
- No tokenizer bottleneck for understanding.
- Cheap: single forward pass.

Choose by task. If you need image generation, pick the Chameleon family. If you only need understanding, adapter VLMs are simpler and reuse more pretraining compute.

### Fuyu and AnyGPT

Fuyu (Adept, 2023) is a related route: it skips the separate vision encoder entirely, feeding raw image patches through the LLM's input projection as if they were tokens, without a tokenizer. Simpler than Chameleon, but loses the shared-vocabulary output generation capability.

AnyGPT (Zhan et al., 2024) extends Chameleon to four modalities: text, image, speech, music. Each uses the same VQ-VAE trick with a shared transformer. Any-to-any generation. Lesson 12.16 covers more.

## Use It

`code/main.py` builds a toy end-to-end early-fusion model:

- A tiny VQ-VAE-style quantizer mapping 8×8 patches to codebook indices (K=16).
- A shared vocabulary: (text ids 0..31) + (image ids 32..47) + (delimiters 48, 49).
- A toy autoregressive decoder (bigram table) trained on synthetic caption + image-token sequences.
- A sampling loop that emits alternating text + image tokens given a prompt.

The code keeps the transformer intentionally minimal (bigram) so you can trace signal flow end to end.

## Ship It

This lesson produces `outputs/skill-tokenizer-vs-adapter-picker.md`. Given a product spec (understanding-only vs understanding + generation, required image quality, cost budget), it picks between the Chameleon family (early fusion) and LLaVA family (late fusion) with quantitative rules of thumb.

## Exercises

1. Chameleon uses K=8192 codebook entries and 1024 tokens per 512×512 image. Estimate the compression ratio relative to a 24-bit RGB image. Is it lossy? How lossy?

2. A 4K image (3840×2160) produces how many image tokens at the same VQ-VAE density? Can a Chameleon-style model generate a 4K image in a single inference call? What breaks first—context, tokenizer quality, or KV cache?

3. Implement QK-Norm in pure Python. Given a 64-dimensional query and key, show the dot product before and after LayerNorm. Why does magnitude control matter at depth?

4. Read Chameleon Section 2.3 on training stability. Describe the exact failure mode the paper observed at 34B without QK-Norm. What characterizes the "norm explosion"?

5. Extend the toy decoder to emit a mixed-modality response given a text-only prompt. Under a training data distribution of 60% text-first / 40% image-first, test how often the model chooses image-first vs text-first.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Early fusion | "unified tokens" | Images are converted to discrete tokens sharing the transformer vocabulary from step one |
| VQ-VAE | "image tokenizer" | CNN + ViT + codebook that maps an image to integer indices a transformer can predict |
| Shared vocabulary | "one dictionary" | A single token ID space covering text + image + modality delimiters |
| QK-Norm | "attention stabilizer" | LayerNorm applied to queries and keys before their dot product, preventing norm blowup |
| Mixed-modality generation | "text + image output" | Inference that autonomously produces interleaved text and image tokens in one forward pass |
| Codebook size | "K entries" | The number of discrete vectors the VQ-VAE can quantize to; trades off compression vs fidelity |
| Tokenizer ceiling | "reconstruction limit" | The best PSNR achievable by decoding VQ tokens; caps the model's image quality |

## Further Reading

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
