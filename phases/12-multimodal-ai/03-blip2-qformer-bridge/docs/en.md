# From CLIP to BLIP-2 — Q-Former as a Modality Bridge

> CLIP aligned images and text, but it can't generate captions, answer questions, or hold conversations. BLIP-2 (Salesforce, 2023) solved this with a small trainable bridge: 32 learnable query vectors attend to a frozen ViT's features via cross-attention, then plug directly into a frozen LLM's input stream. A 188M-parameter bridge connecting an 11B LLM to a ViT-g/14. Through 2026, every adapter-based VLM — MiniGPT-4, InstructBLIP, LLaVA's cousins — descends from it. This lesson walks through the Q-Former architecture, explains its two-stage training, and builds a toy version that feeds visual tokens into a frozen text decoder.

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 min

## Learning Objectives

- Explain why placing a trainable bottleneck between a frozen vision encoder and a frozen LLM beats end-to-end fine-tuning on cost and stability.
- Implement a cross-attention block where a fixed set of learnable queries attend to external image features.
- Walk through BLIP-2's two-stage pretraining: representation stage (ITC + ITM + ITG) then generation stage (LM loss with a frozen decoder).
- Compare Q-Former with LLaVA's simpler MLP projector and argue when each wins.

## The Problem

You have a frozen ViT producing 256 patch tokens of dimension 1408 per image. You have a frozen 7B LLM expecting token embeddings of dimension 4096. The most obvious bridge — a linear layer from 1408 to 4096 — works, but feeding all 256 patch tokens into the LLM's context costs 256 tokens per image. A batch of 32 images eats 8192 tokens on the vision modality alone.

BLIP-2's question: can you compress the 256-token image representation into far fewer tokens (say 32) while retaining enough information for the LLM to describe, answer, and reason about the image? And can you train this bridge without touching the frozen backbones, keeping training cost to just the bridge's own parameters?

The answer: a Q-Former. 32 learnable "query" vectors cross-attend to the ViT's patch tokens, producing a 32-token visual summary for LLM consumption. 188M parameters total. Trained with contrastive, matching, and generation objectives before the LLM is ever touched.

## The Concept

### Learnable Queries

The Q-Former's core trick: instead of having the LLM's text tokens attend to image patches, introduce a new set of 32 learnable query vectors `Q` and have *them* attend to the image patches. These queries are model parameters — they're learned during training and shared across all images.

After cross-attention, each query holds a compressed summary of the image — "describe the main object," "describe the background," "count objects," etc. The queries don't literally correspond to semantic labels; they learn whatever encoding minimizes the downstream loss.

### Architecture

The Q-Former is a small transformer (12 layers, ~100M parameters) with two paths:

1. Query path: 32 query vectors go through self-attention (among themselves), then cross-attention to the frozen ViT's patch tokens, then FFN.
2. Text path: a BERT-like text encoder sharing self-attention and FFN weights with the query path. Cross-attention is disabled for the text path.

Both paths run during training. Queries and text interact through shared self-attention, meaning queries can condition on text for tasks that need it (ITM, ITG). At inference time for VLM handoff, only the query stream flows, producing 32 visual tokens.

### Two-Stage Training

BLIP-2 pretrains in two stages:

Stage 1: Representation learning (no LLM). Three losses:
- ITC (Image-Text Contrastive): CLIP-style contrastive loss between pooled query tokens and text CLS token.
- ITM (Image-Text Matching): binary classifier — does this image-text pair match? Hard negative mining.
- ITG (Image-Grounded Text Generation): causal LM head on text, conditioned on queries. Forces queries to encode content that can generate text.

Only the Q-Former trains. ViT is frozen. No LLM involved.

Stage 2: Generative learning. Attach a frozen LLM (OPT-2.7B or Flan-T5-XL etc.). A small linear layer projects the 32 query outputs to the LLM's embedding dimension. Prepend them to the text prompt. Train the linear projection and Q-Former with LM loss on the concatenated prompt + image + caption sequence only.

After Stage 2, Q-Former + projection is the complete visual adapter. At inference: image → ViT → Q-Former → linear projection → prepend to text → frozen LLM produces output.

### Parameter Economics

BLIP-2 with ViT-g/14 (1.1B, frozen) + OPT-6.7B (6.7B, frozen) + Q-Former (188M, trainable) = 8B total, 188M trainable. The Q-Former alone is ~2.4% of the full stack's parameters. Training cost reflects this: a few days on a few A100s, vs. weeks for end-to-end.

Quality: BLIP-2 matches or beats Flamingo-80B on zero-shot VQA while being 50x smaller. The bridge works.

### InstructBLIP and Instruction-Aware Q-Former

InstructBLIP (2023) adds an extra input to the Q-Former: the instruction text itself. During cross-attention, queries now see both image patches and instructions. Queries can specialize per instruction ("count cars," "describe mood") rather than learning one fixed summary. Benchmark gains on held-out tasks.

### MiniGPT-4 and Projection-Only Training

MiniGPT-4 keeps the Q-Former but only trains the output linear projection, freezing everything else. Cheap, at the cost of quality — those queries are BLIP-2's, not yours. Good for quick iteration, not optimal architecture.

### Why LLaVA Went Simpler

LLaVA (2023, Lesson 12.05) replaced the Q-Former with a naive 2-layer MLP, projecting every ViT patch token to LLM space — 576 tokens per image on a 24x24 grid, all fed to the LLM. Worse compression, but lets the LLM attend to raw patches. Controversial at the time; it dominated by late 2023 because visual instruction data (LLaVA-Instruct-150k) proved the MLP can be trained to retain enough signal. The tradeoff: LLaVA fills context faster, but it scales naturally to multi-image and video.

By 2026, the field split: Q-Former survives where token budget matters (long video, multi-image); MLP projectors dominate where per-token raw quality comes first.

### Gated Cross-Attention: Ancestor Flamingo

Flamingo (Lesson 12.04) predates BLIP-2 and uses the same cross-attention idea but places it at every layer of the frozen LLM, not as a single bridge. BLIP-2 proved you can compress to just the input layer and still work. Gemini and Idefics combine both: interleaved input tokens plus optional gated cross-attention for in-context few-shot.

### Descendants in 2026

- Q-Former: BLIP-2, InstructBLIP, MiniGPT-4, and most video-language models for token budget reasons.
- Perceiver resampler: Flamingo's variant (Lesson 12.04); Idefics family, Eagle, OmniMAE.
- MLP projector: LLaVA, LLaVA-NeXT, LLaVA-OneVision, Cambrian-1.
- Attention pooling: VILA, PaliGemma.

All four are valid. The deciding question: are you token-budget-constrained, or per-token-quality-constrained.

## Use It

`code/main.py` builds a Q-Former-style cross-attention with stdlib:

1. Simulates 256 image patch tokens (dim 128).
2. Instantiates 32 learnable queries (dim 128).
3. Runs scaled dot-product cross-attention (Q from queries, K/V from patches).
4. Projects through a linear layer to LLM dimension (512).
5. Outputs 32 visual tokens ready to feed an LLM.

All math in pure Python (nested loops over vectors). A toy, but the shapes are correct. Attention weight matrix is printed so you can see which patches each query draws from.

## Ship It

This lesson produces `outputs/skill-modality-bridge-picker.md`. Given a target VLM configuration (vision encoder token count, LLM context budget, deployment constraints, quality target), it recommends between Q-Former, MLP, and Perceiver resampler, with a short rationale and parameter count estimate for each bridge.

## Exercises

1. Implement this cross-attention block in PyTorch. Verify: with 32 queries and 256 key/values, the attention weight matrix is 32 x 256 and rows sum to 1 after softmax.

2. In BLIP-2 Stage 1, the Q-Former runs three losses simultaneously: ITC, ITM, ITG. Write the forward signature for each in pseudocode. Which one requires the text encoder path to be active?

3. Compare parameter counts: Q-Former (12 layers, 768 hidden) vs a 2-layer MLP projector (1408 → 4096, two layers). At what LLM scale does the Q-Former's 188M cost pay for itself in training efficiency?

4. Read BLIP-2 paper (arXiv:2301.12597) Section 3.2 on how the Q-Former is initialized. Explain why initializing from BERT-base (rather than random) accelerates convergence.

5. For a 10-minute video sampled at 1 FPS (60 frames), compute the per-frame token cost under (Q-Former → 32 tokens/frame) vs (MLP projector → 576 tokens/frame). Which fits in a 128k-token LLM context window?

## Key Terms

| Term | Common phrasing | What it actually means |
|------|----------------|------------------------|
| Q-Former | "Querying transformer" | A small transformer with 32 learnable query vectors that cross-attend to frozen ViT features |
| Learnable queries | "Soft prompts for vision" | A fixed set of parameters serving as the query side of cross-attention; learned by the model, shared across all inputs |
| Cross-attention | "Q from here, K/V from there" | Attention where query, key, value come from different sources; this is how queries extract from ViT patches |
| ITC | "Image-text contrastive" | CLIP-style loss applied between Q-Former pooled query and text CLS |
| ITM | "Image-text matching" | Binary classifier on hard-negative-mined pairs; forces queries to discriminate fine-grained mismatches |
| ITG | "Image-grounded text generation" | Causal LM loss generating text conditioned on queries; forces queries to encode text-decodable content |
| Two-stage pretraining | "Representation then generation" | Stage 1 trains Q-Former alone (ITC/ITM/ITG); Stage 2 attaches frozen LLM and trains projection + Q-Former |
| Frozen backbone | "No fine-tuning" | Vision encoder and LLM weights are fixed; only the bridge trains |
| Projection head | "Linear to LLM dim" | The final linear layer mapping Q-Former outputs to LLM embedding dimension |
| Perceiver resampler | "Flamingo's version" | Similar learnable-query cross-attention, used per-layer in Flamingo rather than as a single bridge |

## Further Reading

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — the core paper.
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — predecessor with the ITC/ITM/ITG trio.
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "align before fuse" — conceptual ancestor of Stage 1 training.
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — instruction-aware Q-Former.
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — projection-only training approach.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — general architecture for learnable-query cross-attention.
