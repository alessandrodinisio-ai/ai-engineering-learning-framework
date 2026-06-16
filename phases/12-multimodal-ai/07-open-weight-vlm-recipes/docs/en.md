# Open-Weight VLM Recipes: What Actually Matters

> The 2024-2026 open-weight VLM literature is a forest of ablation tables. Apple's MM1 tested 13 combinations of image encoders, connectors, and data mixes. Allen AI's Molmo proved that detailed human captions beat GPT-4V distillation. Cambrian-1 ran 20+ encoder comparisons. Idefics2 formalized the five-axis design space. Prismatic VLMs compared 27 training recipes on a controlled benchmark. Across all this noise, a small set of conclusions hold across papers: image encoder matters more than connector architecture, data mix matters more than either, and detailed human captions beat distilled synthetic data. This lesson reads those tables for you.

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 min

## Learning Objectives

- Name the five-axis VLM design space: image encoder, connector, LLM, data mix, resolution strategy.
- Read an MM1 / Idefics2 / Cambrian-1 ablation table and predict which knob will move a given benchmark.
- Pick a recipe (encoder, connector, data, resolution) for a new VLM given a compute budget and task mix.
- Explain why detailed human captions beat GPT-4V distillation at the same token count.

## The Problem

Hundreds of open-weight VLMs exist. The gap between "good" and "state-of-the-art" is mostly not architecture. It's data, resolution strategy, and encoder choice. Knowing which knob to turn first when your model underperforms can save you a 5M GPU-hour mistake.

The 2023 wave (LLaVA-1.5, InstructBLIP, MiniGPT-4) ran caption-pair pretraining + LLaVA-Instruct-150k. Good baselines. Capped around MMMU 35%.

The 2024 wave (MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs) ran exhaustive ablations. The results are both surprising and practical.

## The Concept

### The Five-Axis Design Space

Idefics2 (Laurençon et al., 2024) named the axes:

1. Image encoder. CLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B. Encoders differ in patch size, resolution, and pretraining objective.
2. Connector. MLP (2-4 layers), Q-Former (32 queries + cross-attention), Perceiver Resampler (64 queries), C-Abstractor (conv + bilinear pooling).
3. Language model. Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. LLM scale is the dominant parameter cost.
4. Training data. Caption pairs (CC3M, LAION), interleaved data (OBELICS, MMC4), instruction data (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. Resolution strategy. Fixed 224/336/448, AnyRes, native dynamic. Progressive increase or constant during training.

Every production VLM makes a choice on each axis. Most variance in MMMU scores is explained by axes 1, 4, and 5 — not which connector you picked.

### Axis 1: Encoder > Connector

MM1 Section 3.2 shows: swapping from CLIP ViT-L/14 to SigLIP SO400m/14 gains 3+ points on MMMU. Swapping connector from MLP to Perceiver Resampler gains less than 1 point. Idefics2 reproduces: SigLIP > CLIP, and Q-Former ≈ MLP ≈ Perceiver at same token count.

Cambrian-1's "Cambrian Vision Encoder Battle" (Tong et al., 2024) ran 20+ encoders on a vision-centric benchmark (CV-Bench). Top of the leaderboard: DINOv2 and SigLIP blends; CLIP in the middle; ImageBind and ViT-MAE at the bottom. The gap from CLIP ViT-L to DINOv2 ViT-g/14 on CV-Bench is ~5-7 points.

The 2026 default encoder for open VLMs is SigLIP 2 SO400m/14 (for semantic + dense features), sometimes concatenated with DINOv2 ViT-g/14 features (Cambrian's "Spatial Vision Aggregator" does this).

### Axis 2: Connector Design Is a Tie

MM1, Idefics2, Prismatic, and MM-Interleaved all reach the same conclusion: at fixed visual token count, connector architecture barely matters. A 2-layer MLP on mean-pooled patches performs within 1 point of a 32-query Q-Former at the same token budget.

What matters is token count. More visual tokens = more LLM compute = better performance, up to a point of diminishing returns. 64 tokens per image is too few for OCR. 576-1024 is the sweet spot for most open VLMs. 2048+ helps only for documents and charts.

Q-Former vs MLP is a cost question, not a quality one: Q-Former caps tokens at 32-64 regardless of image resolution; MLP emits all patch tokens. For high-resolution inputs, Q-Former saves LLM context; for low-resolution, the difference is noise.

### Axis 3: LLM Scale Sets the Ceiling

Doubling the LLM from 7B to 13B reliably adds 2-4 MMMU points in every VLM paper. At 70B you saturate most benchmarks. The VLM's multimodal reasoning ceiling is the LLM's text reasoning ceiling — the vision encoder can only feed it, not reason for it.

This is why Qwen2.5-VL-72B and Claude Opus 4.7 crush MMMU-Pro and ScreenSpot-Pro: the language brain is massive. A 7B VLM cannot make up for a 70B with clever connector design.

### Axis 4: Data — Detailed Human Captions Beat Distillation

Molmo + PixMo (Deitke et al., 2024) is the 2024 result everyone should read. Allen AI had human annotators describe images using a 1-3 minute dense speech-to-text process, producing 712k images with dense captions. No GPT-4V distillation anywhere in the training data.

Molmo-72B beats Llama-3.2-90B-Vision on all 11 benchmarks. The gap isn't architecture — it's caption quality. Detailed human captions contain 5-10x more information per image than short web captions, and remain factually reliable where GPT-4V distillation would hallucinate.

ShareGPT4V (Chen et al., 2023) and Cauldron (Idefics2) played the same game with mixed human + GPT-4V captions. The trend is clear: for 2026 frontier, caption density > caption count > distillation convenience.

### Axis 5: Resolution and Its Strategy

Idefics2's ablation: 384 -> 448 gains 1-2 points. 448 -> 980 with image splitting (AnyRes) gains another 3-5 on OCR benchmarks. Fixed-resolution training plateaus at moderate accuracy; progressive resolution increase (start at 224, end at 448 or native) trains faster and converges higher.

Cambrian-1 ran resolution vs token count tradeoffs: at fixed compute, you can have more tokens at lower resolution or fewer tokens at higher resolution. High-res wins on OCR; lower-res more tokens wins on general scene understanding.

2026 production recipe: Stage 1 at fixed 384, Stage 2 with dynamic resolution up to 1280 for OCR-heavy tasks.

### Prismatic's Controlled Comparison

Prismatic VLMs (Karamcheti et al., 2024) is the paper that controlled all axes. Same 13B LLM, same instruction data, same evaluation — varying one axis at a time. Results:

- Visual tokens per image explains ~60% of variance.
- Encoder choice explains ~20%.
- Connector architecture explains ~5%.
- Everything else (data mix, scheduler, LR) explains the remaining ~15%.

This is a rough decomposition, but it's the cleanest answer in the literature to "what should I ablate first."

### A 2026 Picker

Given the evidence, the default open VLM recipe for a new project in 2026:

- Encoder: SigLIP 2 SO400m/14 with NaFlex for native resolution, concatenated with DINOv2 ViT-g/14 for dense features if you need segmentation/grounding.
- Connector: 2-layer MLP on patch tokens. Skip Q-Former unless you're token-constrained.
- LLM: Qwen2.5 / Llama-3.1 / Gemma 2, 7B for cost, 70B for quality, pick by target latency.
- Data: PixMo + ShareGPT4V + Cauldron, supplemented with task-specific instruction data.
- Resolution: Dynamic (min 256px, max 1280px long edge).
- Schedule: Stage 1 alignment (projector only), Stage 2 full fine-tune, Stage 3 task-specific fine-tune.

Every one of these defaults traces back to a measured ablation in the papers cited at the end of this lesson.

## Use It

`code/main.py` is an ablation table parser and recipe picker. It encodes the MM1 and Idefics2 ablation tables (condensed) and lets you query:

- "Given budget X and task Y, which recipe wins?"
- "If I swap SigLIP for CLIP on a 7B Llama, what's the expected MMMU delta?"
- "For 80% confidence in an answer, which axis should I ablate first?"

Output is a ranked recipe list with expected benchmark deltas and an "ablate this first" recommendation.

## Ship It

This lesson produces `outputs/skill-vlm-recipe-picker.md`. Given a target task mix, a compute budget, and a latency target, it produces a complete recipe (encoder, connector, LLM, data mix, resolution strategy) with each choice citing the ablation that supports it. It prevents engineers from reinventing the Idefics2 ablation table every time they start a new VLM project.

## Exercises

1. Read MM1 Section 3.2. At a fixed 2B LLM and a budget of 50M images, which encoder wins? Does the answer flip at 13B LLM? Why or why not?

2. Cambrian-1 finds that concatenating DINOv2 + SigLIP outperforms either alone on vision-centric benchmarks, but adds no signal on MMMU. Predict which benchmarks will improve and which will stay flat.

3. Your target is a mobile UI agent on a 2B LLM. Pick encoder, connector, resolution, and data mix. Justify each choice with a specific ablation table.

4. Molmo ships 4B and 72B models. The 4B matches closed-source 7B VLMs; the 72B beats Llama-3.2-90B-Vision on 11/11 benchmarks. What does this say about the LLM scale plateau hypothesis?

5. Design an ablation table that isolates data mix quality from encoder quality on a 7B VLM. What is the minimum number of training runs needed? Propose the four axis settings.

## Key Terms

| Term | Common phrasing | What it actually means |
|------|-----------------|------------------------|
| Ablation | "Turn one knob" | Running multiple training runs that differ in exactly one design-space axis, everything else held constant |
| Connector | "Bridge" / "projector" | The trainable module mapping vision encoder output into LLM token space (MLP, Q-Former, Perceiver) |
| Detailed human captions | "Dense captions" | Multi-sentence human-written descriptions (typically 80-300 tokens) richer than web alt text |
| Distillation | "GPT-4V captions" | Training data generated by a stronger proprietary VLM; convenient but inherits hallucinations |
| AnyRes / dynamic resolution | "High-res path" | Strategy for feeding images larger than the encoder's native resolution via tiling or M-RoPE |
| Resolution curriculum | "Progressive" | Training schedule that starts at low resolution and increases, speeding up alignment learning |
| Vision-centric benchmarks | "CV-Bench / BLINK" | Evaluations that stress fine-grained visual perception over language-heavy reasoning |
| PixMo | "Molmo's data" | Allen AI's 712k dense-captioned image dataset; human speech transcribed into dense captions |

## Further Reading

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
