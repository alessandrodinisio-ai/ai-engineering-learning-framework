# LLaVA-OneVision: Single-Image, Multi-Image, and Video in One Model

> Before LLaVA-OneVision (Li et al., August 2024), the open VLM world had several separate threads: LLaVA-1.5 for single images, Mantis and VILA for multi-image, Video-LLaVA and Video-LLaMA for video. Each won its own benchmarks and lost on the others. LLaVA-OneVision argues: one curriculum can train one model that dominates all three scenarios, and the emergent task transfer effects (single-image skills transferring to video, multi-image reasoning transferring to single-image) beat the sum of specialists. The recipe is deceptively simple: a visual token budget that stays constant across scenarios, plus an explicit curriculum from single-image to OneVision (multi-image) to video. This lesson walks through the budget, the curriculum, and those emergent behaviors.

**Type:** Build
**Languages:** Python (standard library, token budget solver + curriculum planner)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Time:** ~180 minutes

## Learning Objectives

- Design a visual token budget that stays constant across single-image, multi-image, and video inputs.
- Lay out a training curriculum that transfers skills from single-image to video without catastrophic forgetting.
- Explain why, when the curriculum is done right, a single model beats specialists at the same parameter count.
- Name the three emergent capabilities reported by LLaVA-OneVision: multi-camera reasoning, set-of-mark prompting, and iPhone screenshot agent.

## The Problem

Images, multi-image, and video each stress the model in different ways.

Single-image wants high-resolution tokens (AnyRes, ~2880 visual tokens) to capture OCR and fine detail. Per-sample budget: one image, 2880 tokens.

Multi-image wants several medium-resolution images (~576 tokens each) so cross-image reasoning fits in context. Per-sample budget: 4–8 images at 576 each, totaling 2300–4600 tokens.

Video wants many low-resolution frames (~196 tokens per frame after pooling) to capture temporal dynamics. Per-sample budget: 8–32 frames at 196 each, totaling 1600–6200 tokens.

If you train separate models, you pick one budget. If you train one model, you need the budget to scale sensibly across scenarios without blowing up context.

Before OneVision, the default answer was "train one scenario, ignore the others." Video-LLaVA bolted video onto an image model via extra training stages. LLaVA-NeXT added multi-image support with tiling. Nobody handled all three cleanly.

## The Concept

### OneVision Token Budget

LLaVA-OneVision picks a unified visual token budget of ~3000–4000 tokens per sample, allocated differently per scenario:

- Single-image: AnyRes-9 (3×3 tiles + thumbnail), each tile at 384 yields 729 patches, with aggressive 2×2 bilinear pooling → 182 per tile. Total: 9 × 182 + 182 = 1820 tokens. Or AnyRes-4 at 729 per tile = 2916 + 729.
- Multi-image: each image at medium resolution (384, no tiling), 729 tokens unpooled. Budget of 6 images → 4374 tokens.
- Video: 32 frames at 384 resolution with aggressive 3×3 bilinear pooling → 81 tokens per frame. Total: 32 × 81 = 2592 tokens.

This allocation keeps total tokens roughly constant. The LLM never sees a batch that blows up its context. The encoder produces different geometries per scenario, but the LLM consumes the same budget.

### Three-Stage Curriculum

LLaVA-OneVision trains in three stages:

1. Single-Image SFT (SI stage). All data is single-image plus text. Trained on high-resolution AnyRes inputs. This teaches perception, OCR, and fine-grained understanding. Uses LLaVA-NeXT data plus OneVision-specific single-image data.
2. OneVision SFT (OV stage). Mixes single-image + multi-image + video (uniformly sampled frames). Trained on the unified token budget. This teaches the model to handle heterogeneous batch shapes. No weight reset—continues from the SI stage.
3. Task Transfer (TT stage). Continues training on a target task mix, usually emphasizing multi-image or video depending on the product. Optional deployment fine-tuning.

Key: curriculum order matters. Training video-first or multi-image-first produces worse image performance than single-image-first, even with identical data. The paper ablates this explicitly.

### Why the Curriculum Works

Single-image training lays the perception foundation. Patch tokens carry fine-grained visual features; the LLM learns to integrate them with text. Multi-image and video introduce structural challenges (which image is which, what happened first) that are hard to learn without a strong perception foundation.

If you train all scenarios together from scratch, the model underfits perception (limited single-image data per batch) and overfits structure (lots of multi-image/video data). The result: a model that follows cross-image reasoning patterns but is visually shallow.

Curriculum ordering gives you perception strength from the SI stage and compositional/temporal reasoning from the OV stage without losing either.

### Emergent Cross-Scenario Skills

The LLaVA-OneVision paper reports three emergent capabilities:

1. Multi-camera reasoning. Trained separately on multi-image + video; asked at inference to reason about a multi-camera driving scene. Despite never seeing this exact format during training, the model correctly integrates the viewpoints.
2. Set-of-mark prompting. Users annotate objects in an image with numbered markers; the model reasons "what is mark 3 doing relative to mark 7." Never trained on markers or annotations; learned from the combination of spatial grounding + multi-image referencing.
3. iPhone screenshot agent. Users provide an iPhone screen capture and request planning the next tap. Trained on UI screenshots, user workflow videos, and multi-image before/after pairs. Generalizes to agent use cases.

These are not trained tasks; they emerge from the combinatorial structure of the curriculum.

### Visual Token Pooling

The token budget requires pooling. OneVision uses bilinear interpolation on the 2D patch grid: 24×24 = 576 patches become 12×12 = 144 (2×) or 8×8 = 64 (3×). Pooling is done in patch-grid space, not token space, to preserve locality.

The pooling factor per scenario is itself a hyperparameter. Less pooling = more tokens = richer representation. More pooling = fewer tokens = more frames/images fit.

### LLaVA-OneVision-1.5

The 2025 follow-up (LLaVA-OneVision-1.5, arXiv 2509.23661) is "fully open" in training data, model weights, and code. It closes part of the gap with proprietary models on some benchmarks and democratizes the recipe. Same curriculum, more data, better base LLM. No architecture changes.

### Comparison with Qwen2.5-VL

Qwen2.5-VL (Lesson 12.09) makes different choices. It uses M-RoPE and dynamic FPS rather than fixed pooling. Its budget scales with input—a 1-minute video uses more tokens than a 5-second one. LLaVA-OneVision fixes the budget and scales pooling. Both work; they trade off differently between configurability and predictability.

## Use It

`code/main.py` is a curriculum and budget planner for OneVision-style VLMs. Given a per-sample token budget and a target scenario mix (e.g., 40% single-image, 30% multi-image, 30% video), it:

- Allocates resolution, pooling factor, and frame count for each scenario.
- Checks that each scenario fits within the shared budget.
- Reports expected token counts, LLM FLOPs, and which scenarios are token-starved.
- Prints a stage-by-stage training plan.

Use it to plan a OneVision fine-tune or sanity-check the per-request cost of a VLM deployment.

## Ship It

This lesson produces `outputs/skill-onevision-budget-planner.md`. Given a target task distribution and a per-sample budget, it outputs AnyRes factors, per-frame pooling, video frame counts, and curriculum stage weights. Use it whenever you train or fine-tune a unified-scenario VLM.

## Exercises

1. Your product supports 80% single-image, 10% multi-image (2–4 images), 10% video (8–16 frames). Design the token budget. Where would you put the extra budget saved from not emphasizing multi-image?

2. Read LLaVA-OneVision Section 4.3 (emergent capabilities). Propose a fourth emergent skill the curriculum likely unlocks but the paper doesn't report.

3. Swap the curriculum order—train multi-image first, then single-image, then video. Predict which benchmarks degrade and why.

4. The paper reports video benchmarks trained at only 8 frames per sample. Does this generalize to 30-second videos at inference? What breaks first—token budget or temporal reasoning?

5. Bilinear-pool a 24×24 patch grid to 12×12—a 4× per-dimension reduction. Implement this pooling in standard-library Python and verify that the mean over each 2×2 block matches the bilinear output.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| OneVision scenario | "single-image, multi-image, or video" | One of three input shapes handled by a unified VLM; budget stays constant across them |
| Token budget | "how many tokens per sample" | Total visual tokens the LLM sees per training/inference sample, typically 3000–4000 |
| Curriculum | "training order" | Stage ordering chosen for emergent transfer (single-image → multi-image → video) |
| Bilinear pooling | "token shrinking" | Bilinear interpolation on the patch grid (2D) to reduce token count while preserving locality |
| Emergent skill | "not trained, still works" | Capabilities appearing at inference due to curriculum composition, with no corresponding training data |
| AnyRes-k | "k-tile setup" | k fixed-resolution sub-tiles plus one thumbnail, typical k ∈ {4, 9} |
| Task transfer | "cross-scenario generalization" | Skills learned on single-image applied to video (and vice versa) via shared backbone |

## Further Reading

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
