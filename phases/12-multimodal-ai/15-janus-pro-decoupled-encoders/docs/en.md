# Janus-Pro: Decoupled Encoders for Unified Multimodal Models

> Unified multimodal models have an unavoidable tension. Understanding wants semantic features — SigLIP or DINOv2 outputs vectors rich in concept-level information. Generation wants reconstruction-friendly encoding — VQ tokens that can be reassembled into sharp pixels. These two goals are incompatible in a single encoder. Janus (DeepSeek, October 2024) and Janus-Pro (DeepSeek, January 2025) argue the fix is to stop forcing it: decouple the two encoders. Share the transformer body across tasks, but route understanding through SigLIP and generation through a VQ tokenizer. At 7B scale, Janus-Pro beats DALL-E 3 on GenEval while matching LLaVA on MMMU. This lesson reads through why two encoders succeed where one fails.

**Type:** Build
**Languages:** Python (standard library, dual-encoder routing + shared-body signals)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## Learning Objectives

- Explain why a single shared encoder compromises either understanding or generation quality.
- Describe Janus-Pro's routing: SigLIP features for understanding on the input side, VQ tokens for generation on both input and output sides.
- Trace the data-recipe scaling that makes Janus-Pro succeed where Janus failed.
- Compare decoupled (Janus-Pro), coupled continuous (Transfusion), and coupled discrete (Show-o) architectures.

## The Problem

Unified models share one transformer body between understanding and generation. Prior attempts (Chameleon, Show-o, Transfusion) used one visual tokenizer to serve both directions. That tokenizer is a compromise:

- Optimized for reconstruction (generation): VQ-VAE captures fine-grained pixel detail but produces tokens with weaker semantic coherence.
- Optimized for semantics (understanding): SigLIP embeddings cluster "cat" images near the "cat" token but don't allow good reconstruction.

Show-o and Transfusion both pay a visible quality tax in one direction for this. Janus-Pro asks: why use one tokenizer when the two tasks have different needs?

## The Concept

### Decoupled Visual Encoding

Janus-Pro's architecture separates two encoders:

- Understanding path. Input image → SigLIP-SO400m → 2-layer MLP → transformer body.
- Generation path. Input image (if conditioning on an existing image) → VQ tokenizer → token IDs → transformer body.
- Output generation. Image tokens predicted by transformer → VQ decoder → pixels.

The transformer body is shared. Everything upstream and downstream of the body is task-specific.

Input is disambiguated by prompt format: an `<understand>` tag routes through SigLIP; `<generate>` routes through VQ. Or routing is implicitly decided by task.

### Why This Works

The understanding loss gets SigLIP features, which CLIP-style pretraining has already tuned for semantic similarity. The model's perceptual baseline improves over Show-o / Transfusion because the input features are better matched to the task.

The generation loss gets VQ tokens, which a tokenizer has already tuned for reconstruction. Image quality improves over Show-o because the VQ encoding can cleanly reassemble into pixels.

The shared transformer body sees both input distributions (SigLIP and VQ) and learns to work with both. The claim is: enough data + enough parameters, and the body can absorb this switching.

### Data Scaling — Janus vs Janus-Pro

Janus (original, arXiv 2410.13848) introduced the decoupling but at small scale (1.3B parameters, limited data). Janus-Pro (arXiv 2501.17811) scaled it:

- 7B parameters (vs 1.3B).
- Stage 1 (alignment) 90M image-text pairs, up from 72M.
- Stage 2 (unified) 72M, up from 26M.
- Stage 3 adds 200K image generation instruction samples.

Result: Janus-Pro-7B matches LLaVA on MMMU (60.3 vs ~58) and beats DALL-E 3 on GenEval (0.80 vs 0.67). One open model, competitive on both sides of the unified spectrum.

### JanusFlow — The Rectified Flow Variant

JanusFlow (arXiv 2411.07975) replaces the VQ generation path with a rectified-flow generation path (continuous). The split becomes "SigLIP for understanding + rectified-flow for generation." Quality ceiling further raised. Architecture is still decoupled encoder–shared body.

### The Shared Body's Job

The transformer body processes a unified sequence but with two input distributions. Its job is:

- For understanding: consume SigLIP features + text tokens → autoregressively output text.
- For generation: consume text tokens + (optional image VQ tokens) → autoregressively output image VQ tokens.

The body has no per-block modality-specific weights. It's the same text-style transformer you'd expect in Qwen or Llama, plus two input adapters.

Interestingly, this means Janus-Pro's body can initialize from a pretrained LLM. Janus-Pro does initialize from DeepSeek-MoE-7B. This choice matters: the LLM contributes reasoning capabilities that a purely from-scratch unified model struggles to achieve.

### Comparison with InternVL-U

InternVL-U (Lesson 12.10) is the 2026 successor. It combines:

- Native multimodal pretraining (InternVL3 backbone).
- Decoupled encoder routing (SigLIP in, VQ + diffusion head out).
- Unified understanding + generation + editing.

InternVL-U subsumes Janus-Pro's architectural choices into a larger framework. The decoupled encoder idea is now the default for scaled unified models.

### Limitations

Decoupled encoders add architectural complexity. Two tokenizers to train, two input paths to maintain, two failure modes. For products that don't need generation, Janus-Pro is over-engineered — pick a LLaVA-family understanding model.

For products that don't need understanding, Janus-Pro is overqualified — pick a Stable Diffusion 3 / Flux model.

For products that need both, Janus-Pro is the reference open architecture today.

## Use It

`code/main.py` simulates Janus-Pro routing:

- Two mock encoders: SigLIP-style (outputs 256-dim semantic vectors) and VQ-style (outputs integer codes).
- A prompt router that picks the encoder based on task tag.
- A shared body (stand-in) that processes the token sequence regardless of which encoder produced it.
- A weighted sampling schedule switching from stage 1 (alignment) to stage 3 (instruction tuning).

Prints the routing path for 3 examples: image QA, T2I, image editing.

## Ship It

This lesson produces `outputs/skill-decoupled-encoder-picker.md`. Given a product wanting unified generation + understanding at near-frontier quality, it picks among Janus-Pro, JanusFlow, and InternVL-U with a concrete data-scale recommendation.

## Exercises

1. Janus-Pro-7B beats DALL-E 3 on GenEval. Explain why a 7B open model can match frontier proprietary models on generation but not on understanding.

2. Implement a routing function: given prompt text, classify as `understand` or `generate`. How do you handle ambiguous prompts like "describe then draw"?

3. JanusFlow replaces the VQ path with rectified flow. What does the transformer body now output, and how does the loss change?

4. Propose a fourth task the Janus-Pro architecture could handle by adding one more decoupled encoder. Examples: image segmentation (DINO-style), depth (MiDaS-style).

5. Read Janus-Pro Section 4.2 on data scaling. Which data stage contributed most to T2I quality gains relative to Janus?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Decoupled encoding | "two vision encoders" | Separate tokenizer or encoder per direction: semantic for understanding, reconstruction for generation |
| Shared body | "one transformer" | Single transformer processes output from either encoder; no modality-specific weights |
| SigLIP for understanding | "semantic features" | CLIP-family vision tower providing rich conceptual features but poor reconstruction |
| VQ for generation | "reconstruction encoding" | Vector-quantized tokens that can cleanly decode back to pixels |
| JanusFlow | "rectified-flow variant" | Janus-Pro with continuous flow-matching generation head replacing VQ |
| Routing tag | "task tag" | Prompt marker selecting the input encoder (`<understand>` / `<generate>`) |

## Further Reading

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
