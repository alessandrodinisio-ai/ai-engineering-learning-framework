# Video-Language Models: Temporal Tokens and Grounding

> Video is not a stack of photos. A 5-second clip has causal ordering, action verbs, and event timing that image models cannot represent. Video-LLaMA (Zhang et al., June 2023) shipped the first open video LLM with audiovisual grounding. VideoChat and Video-LLaVA scaled the pattern. By 2025, Qwen2.5-VL's TMRoPE closed the gap with frontier proprietary models. Each system handles temporal tokens differently — per-segment Q-former, per-frame concat-pool, per-token TMRoPE. This lesson reads through these patterns, builds a uniform vs dynamic frame sampler, and evaluates on temporal grounding tasks.

**Type:** Build
**Languages:** Python (standard library, frame sampler + temporal grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## Learning Objectives

- Explain why temporal positional encoding, independent of the visual encoder, changes video VLM performance.
- Compare uniform, dynamic FPS, and event-driven frame sampling on tokens-per-second vs grounding accuracy.
- Describe per-segment Q-former (Video-LLaMA) vs per-frame pooling (Video-LLaVA) vs per-token M-RoPE (Qwen2.5-VL) designs.
- State four video benchmarks: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## The Problem

A 1-minute, 30 FPS video is 1800 frames. At 196 visual tokens per frame (ViT-B at 224), that's 352K tokens — larger than any 2024-era LLM context.

Three reduction strategies exist:

1. Frame subsampling (1-8 FPS by content).
2. Aggressive per-frame patch pooling (3x3 or 4x4 bilinear pooling).
3. Q-former compression, which eats a 16-frame segment and outputs 64 tokens.

Each trades differently. Subsampling loses temporal detail. Pooling loses spatial detail. Q-former loses some of both but saves tokens.

Temporal positional encoding is another axis: how does the model know frame 5 comes before frame 6? Options include simple 1D temporal RoPE (Video-LLaMA), learned temporal embeddings (Video-LLaVA), and TMRoPE (Qwen2.5-VL, full 3D).

## The Concept

### Video-LLaMA: Per-Segment Q-former + Audio Branch

Video-LLaMA (2023) is the first open video LLM. Architecture:

- 16-frame segments at 2 FPS (i.e., 8 seconds).
- Per-frame ViT features -> Video Q-former that cross-attends to all 16 frames -> 32 learned queries -> LLM.
- Parallel audio branch: waveform -> ImageBind audio encoder -> Audio Q-former -> 32 queries -> LLM.

Strengths: audiovisual joint reasoning. Weaknesses: fixed segment length, no arbitrary temporal grounding.

### VideoChat and Video-LLaVA

VideoChat keeps Video-LLaMA's idea but drops audio and simplifies. Video-LLaVA (Lin et al., 2023) trains a single vision encoder on both image and video frames ("pre-projection alignment"), giving a unified representation. Both are frozen CLIP encoder + MLP + LLM.

Neither handles long video. Both are 8-16 frame systems.

### Qwen2.5-VL and TMRoPE

Qwen2.5-VL introduces TMRoPE — Temporal-Modal Rotary Position Embedding. Each patch token carries a (t, h, w) position where t is the real-world timestamp (not frame index).

Key differences from simple temporal embeddings:

- Absolute time, not index. The model sees "at 4.2 seconds" rather than "at frame 15."
- Per-token rotation, not per-segment. Each visual token is rotated by its timestamp independently.
- Compatible with dynamic FPS. If you sample at 2 FPS here and 4 FPS there, TMRoPE natively handles the uneven spacing.

TMRoPE enables queries like "at what second does the cat jump?" The model can output "at 4.2 seconds." Video-LLaMA can only say "early in the segment."

### Frame Sampling Strategies

Uniform: sample N frames uniformly over the duration. Simple, misses motion peaks.

Dynamic FPS: adaptively sample based on motion intensity. Optical flow or frame differencing picks high-motion segments for denser sampling. Qwen2.5-VL trains on this.

Event-driven: run a lightweight detector, sample more where actions occur. VideoAgent uses this.

Keyframe + context: sample at shot boundaries + neighboring frames. Used for film content.

### Per-Frame Pooling

At 1 FPS and 576 tokens per frame, a 5-minute clip is 172,800 tokens. Doable with Qwen2.5-VL-72B's 128k context but expensive.

3x3 bilinear pooling reduces to 64 tokens per frame -> 5 minutes at 19,200 tokens. Sweet spot for most tasks.

For agent workflows where spatial detail matters less, pool more aggressively (6x6 -> 16 tokens per frame).

### Four Video Benchmarks

- VideoMME: comprehensive video understanding, short + medium + long.
- TempCompass: fine-grained temporal reasoning, "before"/"after" style questions.
- EgoSchema: long-horizon egocentric video.
- Video-MMMU: multimodal multi-discipline video questions.

A full video VLM evaluation hits all four. They stress different axes — TempCompass is all about ordering, EgoSchema about 3+ minute reasoning, VideoMME spans durations.

### Grounding Output Format

The output format for temporal grounding:

- Free text: "The cat jumps at approximately 4 seconds." Easy to parse but imprecise.
- Structured JSON: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL trains on this.
- Token-based: special `<time>4.1</time>` tokens interleaved with the answer. Qwen2.5-VL's internal format.

Token-based is most accurate for downstream use. Qwen2.5-VL's JSON output format is directly parseable.

### 2026 Best Practices

The 2026 video VLM:

- Encoder: SigLIP 2 with M-RoPE or TMRoPE (Qwen2.5-VL).
- Frame sampling: dynamic FPS (1-4 by motion) with max frame cap.
- Per-frame pooling: 3x3 bilinear.
- Output: structured JSON with time + event fields.
- Benchmarks: VideoMME + TempCompass for general; EgoSchema for long-horizon.

## Use It

`code/main.py` contains:

- Uniform and dynamic FPS frame samplers.
- A toy temporal grounding evaluator: given a ground-truth event at time T and a model output, scores accuracy with tolerance.
- A comparison table across Video-LLaMA (16 frames, Q-former), Video-LLaVA (8 frames, MLP), and Qwen2.5-VL (dynamic FPS + TMRoPE).

## Ship It

This lesson produces `outputs/skill-video-vlm-frame-planner.md`. Given a video task (surveillance, action recognition, temporal grounding, summarization), it picks the frame sampler, pooling multiplier, output format, and expected accuracy tier.

## Exercises

1. For a 3-minute cooking demonstration, pick between uniform vs dynamic FPS. Justify with a token count.

2. What specifically does TMRoPE add that a simple temporal embedding table cannot?

3. Write a temporal grounding JSON schema that a VLM can learn to output. Include error cases.

4. Read Video-LLaVA Section 3 on "pre-projection alignment." Why is this better than training separate image and video encoders?

5. Given the VideoMME leaderboard, how large is the gap between top open and top proprietary models as of 2026? How much of that gap is attributable to temporal encoding vs base LLM scale?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Temporal grounding | "time-located answers" | VLM outputs a specific timestamp range for when an event occurs |
| TMRoPE | "temporal-modal RoPE" | 3D rotary positions with absolute timestamps, used by Qwen2.5-VL |
| Dynamic FPS | "motion-aware sampling" | Sample more frames in high-motion segments, fewer in static ones |
| Frame pooling | "per-frame spatial compression" | Reducing patches per frame via bilinear interpolation before entering LLM |
| Video Q-former | "segment compressor" | Cross-attention bottleneck mapping N frames to K learned queries |
| VideoMME | "video benchmark" | Comprehensive short/medium/long video benchmark, 2500+ samples |

## Further Reading

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
