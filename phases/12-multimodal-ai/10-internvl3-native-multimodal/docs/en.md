# InternVL3: Native Multimodal Pretraining

> Every open VLM before InternVL3 followed the same three-step recipe: take a text LLM pretrained on trillions of text tokens, bolt on a vision encoder, and fine-tune the seam. This works, but it incurs alignment debt—the text LLM spent all its pretraining budget on pure text and does not natively understand visual tokens. When you add vision after the fact, the LLM must relearn how to relate visual input to its text reasoning without forgetting text. InternVL3 (Zhu et al., April 2025) refuses the post-hoc route: one pretraining run, text and multimodal interleaved from the first step. The result matches Gemini 2.5 Pro on MMMU-Pro at open scale with 78B parameters. This lesson walks through the rationale for native pretraining and what changes once you commit to it.

**Type:** Learn
**Languages:** Python (standard library, training corpus mixer)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time:** ~120 minutes

## Learning Objectives

- Explain why post-hoc VLM training accumulates alignment debt, citing three measurable symptoms (catastrophic forgetting, answer drift, vision-text inconsistency).
- Describe InternVL3's native pretraining corpus mix and why the text : interleaved : caption ratio matters.
- Compare V2PE (Variable Visual Position Encoding) with Qwen2-VL's M-RoPE.
- Name the two deployment optimizations: Visual Resolution Routing (ViR) and Decoupled Vision-Language Deployment (DvD).

## The Problem

Post-hoc VLM training is the default approach. LLaVA, BLIP-2, Qwen-VL, Idefics—all take an already-pretrained LLM (Llama, Vicuna, Qwen, Mistral) and add vision. Training stages typically look like:

1. Frozen LLM + frozen vision encoder + trainable projector, trained on caption pairs to align embeddings.
2. Unfreeze LLM, train on instruction data (LLaVA-Instruct, ShareGPT4V).
3. Optional task-specific fine-tuning.

Three symptoms of alignment debt surface:

- Catastrophic forgetting. Post-hoc VLMs forget text-only skills. GSM8K scores drop 5–10 points. Hellaswag scores decline. Text-only agent capabilities degrade.
- Answer drift. The same visual question with a minor rephrasing yields different answers. The bond between the visual encoder and LLM is weaker than the LLM's own tokens.
- Vision-text inconsistency. The VLM correctly describes an image, then answers a question that contradicts its own description. Visual tokens do not participate in the LLM's internal consistency checks the way text does.

These symptoms are well-documented. MM1.5 Section 4 quantifies them. LLaVA-OneVision ablations hint at them. Native pretraining is the answer.

## The Concept

### Native Multimodal Pretraining

InternVL3 trains from scratch on a natively multimodal corpus from step one. The mix is:

- 40% pure text data (FineWeb, Proof-Pile-2, etc.)
- 35% interleaved image-text data (OBELICS, MMC4-style)
- 20% paired image-text caption data
- 5% video-text data

Visual tokens, text tokens, and cross-modal interactions all participate in the same loss from the first gradient step. No alignment pretraining, no projector freeze stage, no catastrophic forgetting to recover from.

Base model training is single-stage. Instruction fine-tuning follows, but the base model already understands visual tokens as first-class citizens.

### V2PE (Variable Visual Position Encoding)

Qwen2-VL uses M-RoPE with fixed axis allocation. InternVL3 introduces V2PE: position encoding that varies by modality type (text, image, video) with learnable scaling. In practice:

- Text tokens get 1D positions (text index).
- Image patches get 2D positions (row, col).
- Video frames get 3D positions (temporal, row, col).

All three share the same RoPE frequency base, but the hidden-dimension allocation per band is a learnable parameter rather than a fixed split. This gives pretraining freedom to trade off temporal vs spatial frequency resolution.

V2PE ablations claim 1–2 points over M-RoPE on video benchmarks at equal compute. Not revolutionary, but cleaner.

### Visual Resolution Routing (ViR)

A deployment optimization. Not all images need full-resolution encoding. A photo with a single low-detail object wastes tokens when encoded at native 1280px. ViR is a small classifier that predicts the minimum resolution needed to answer the question, before encoding.

The router has three bins: low-res (256 tokens), medium (576), high (2048+). For 60% of queries in production traffic, low or medium suffices. Net effect: 2–3× throughput at equal quality.

### Decoupled Vision-Language Deployment (DvD)

When you serve a large VLM, the vision encoder runs once per image, but the LLM runs autoregressively for each output token. The two components bottleneck differently (vision = GPU memory bandwidth for conv + attention; LLM = KV cache). DvD splits them onto different GPUs with streaming in between.

For a model with an 8B + 400M encoder, DvD roughly doubles per-node throughput compared to co-located deployment.

### Single-Stage vs Multi-Stage Quality

InternVL3's main benchmark claim: matching Gemini 2.5 Pro on MMMU-Pro at 78B parameters. Matching GPT-4o at 38B. Leading the open 8B leaderboard at 8B. All on a single-stage pretraining + instruction fine-tuning recipe.

The alignment debt hypothesis is measurable: InternVL3-8B loses fewer text benchmark points (MMLU, GSM8K) per unit of visual benchmark gain than Qwen2.5-VL-7B. The model is more of a generalist because training is one block, not two.

### InternVL3.5 and InternVL-U

InternVL3.5 (August 2025) scales the recipe. Same native pretraining route, more data, more parameters. MMMU gains are incremental.

InternVL-U (2026) adds unified generation—image output via an MMDiT head on the same backbone. "U" stands for "understanding + generation," chasing Transfusion-style unified models (Lesson 12.13). The same native pretraining backbone supports both the understanding head and the generation head.

### Trade-offs of Native Pretraining

Native pretraining is not free:

- Compute. Training a new VLM from scratch costs as much as training a text LLM—millions of GPU hours. Post-hoc adaptation reuses existing LLM weights, saving most of that cost.
- Data. Interleaved image-text corpora at scale are scarce. OBELICS is 141M documents; MMC4 is 571M. Text alone has 15T tokens. Multimodal pretraining data scarcity is a hard constraint.
- Base LLM reuse. Native pretraining forfeits the option to swap in a newer LLM later. Post-hoc routes just retrain the adapter to go from Llama-3.1 to Llama-4.

InternVL3's bet is that alignment debt costs more than reuse loss. Benchmarks support the claim. Production cost blocks future labs from cheap replication. Post-hoc VLMs will continue to exist because they're still cheaper for most projects.

## Use It

`code/main.py` is a training corpus mixer plus ViR routing simulator. It:

- Takes a target corpus ratio (%text, %interleaved, %caption, %video) and computes expected steps per modality.
- Simulates ViR routing on a batch of queries (distribution: 50% low-detail, 30% medium, 20% high-detail), reporting average token count.
- Given encoder vs LLM FLOPs, reports DvD throughput estimates.
- Prints a side-by-side comparison of post-hoc vs native pretraining on parameters, compute, data, and expected alignment debt symptoms.

## Ship It

This lesson produces `outputs/skill-native-vs-posthoc-auditor.md`. Given a proposed VLM training plan, it audits whether to go native or post-hoc, flags alignment debt risks, and recommends a corpus mix. Use it when speccing a new open VLM project and needing to pick a training strategy.

## Exercises

1. Estimate the compute gap between InternVL3-8B (native pretraining) and LLaVA-OneVision-7B (post-hoc). What is the approximate ratio in GPU hours? What explains the gap?

2. InternVL3 reports 40% text / 35% interleaved / 20% caption / 5% video. If your target task is video-heavy, propose a new ratio and argue why the base model still needs substantial text and caption data.

3. Read MM1.5 Section 4 on forgetting. Name the exact benchmark where post-hoc training shows the largest degradation. How much does the degradation cost?

4. ViR routes 60% of traffic to low-resolution encoding. What class of queries does it misroute (sending to low-res when high-res is needed)? Propose three routing failure modes.

5. DvD splits vision and LLM onto different GPUs. Under what traffic pattern does DvD hurt throughput rather than help?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Native multimodal pretraining | "train together from scratch" | Text + image + video tokens participate in the loss from step 1, rather than being bolted on post-hoc |
| Alignment debt | "post-hoc penalty" | Measurable degradation in text skills and answer consistency from bolting vision onto a frozen LLM |
| V2PE | "Variable Visual Position Encoding" | Per-modality learnable position encoding allocation; InternVL3's successor to M-RoPE |
| ViR | "resolution routing" | A small classifier that picks the minimum resolution needed per query before encoding, saving inference tokens |
| DvD | "decoupled deployment" | Vision encoder on one GPU, LLM on another, streaming between them; roughly doubles throughput for large VLMs |
| InternVL-U | "unified understanding + generation" | 2026 follow-up adding an image generation head to the native pretraining backbone |
| Interleaved corpus | "OBELICS / MMC4" | Documents with text and images in natural reading order; raw material for native pretraining |

## Further Reading

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
