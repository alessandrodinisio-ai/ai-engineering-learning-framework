# The Qwen-VL Family and Dynamic FPS Video

> The Qwen-VL family—Qwen-VL (2023), Qwen2-VL (2024), Qwen2.5-VL (2025), Qwen3-VL (2025)—is the most influential open vision-language model lineage as of 2026. Each generation made a decisive architectural bet that the open ecosystem copied within twelve months: native dynamic resolution via M-RoPE, dynamic FPS sampling with absolute time alignment, window attention in the ViT, and structured agent output formats. By Qwen3-VL, the recipe has stabilized: a 2D-RoPE-ViT encoder with native aspect-ratio input, an MLP projector into a large Qwen3 language backbone, and training stages that emphasize OCR, grounding, and agent behavior as first-class objectives. This lesson walks through the family chronologically so you understand why each knob is where it is.

**Type:** Learn
**Languages:** Python (standard library, M-RoPE encoder + dynamic FPS sampler)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 minutes

## Learning Objectives

- Compute M-RoPE's three-axis rotation (temporal, height, width) and explain why all three are needed.
- Pick a dynamic FPS sampling strategy for a video and reason about the trade-off between tokens-per-second and event detection accuracy.
- Name the four generational upgrades of Qwen-VL in order and what each unlocked.
- Wire up a Qwen2.5-VL-style JSON agent output format and parse structured tool calls from VLM responses.

## The Problem

Qwen-VL shipped in August 2023 as a direct response to LLaVA-1.5 and BLIP-2. The Qwen team targeted three gaps: resolution, video, and structured output.

Resolution: LLaVA-1.5 ran at 336×336. Fine for photos, useless for a Chinese invoice or a dense spreadsheet screenshot. Qwen-VL's first innovation was 448×448 and bounding-box output with grounding, letting the model point at things.

Video: Video-LLaMA stacked per-frame encoders then fed an LLM. Workable for clips, not for multi-minute videos where the timeline is the signal. The Qwen team wanted a single encoder that understands time.

Structured output: LLaVA emits free-form text. Agents need JSON. Qwen-VL trained on explicit JSON output formats, with bounding-box coordinates trained as text.

Each Qwen-VL generation extends along one of these three axes.

## The Concept

### Qwen-VL (August 2023)

First generation: OpenCLIP ViT-bigG/14 as encoder (2.5B parameters), a Llama-compatible Q-Former (256 queries, one-step), Qwen-7B backbone. Contributions:

- 448×448 resolution (SOTA for open VLMs at the time).
- Grounding: trained on image-text pairs with explicit coordinate token output. "The cat is at <box>(112, 204), (280, 344)</box>."
- Bilingual Chinese-English training from day one.

Benchmarks at the time: competitive with GPT-4V in English, dominant in Chinese. The grounding supervision was the real headline.

### Qwen2-VL (September 2024) — M-RoPE and Native Resolution

Qwen2-VL replaced the fixed-resolution + Q-Former pipeline with a natively dynamic-resolution ViT encoder. Key changes:

- Native dynamic resolution. The ViT accepts any H×W divisible by 28 (patch 14 with 2× spatial merging). A 1120×672 image (40×24 merged patches) produces 960 visual tokens. No resizing, no tiling, no thumbnail.
- M-RoPE (Multimodal RoPE). Each token carries a 3D position (t, h, w) rather than 1D. Images use t=0; videos use t = frame index. RoPE rotates query/key vectors at one frequency per axis. No positional embedding table.
- MLP projector. Drops the Q-Former; uses a 2-layer MLP on merged patch tokens.
- Video with dynamic FPS. Video defaults to 1–2 FPS sampling, but the model accepts any frame count.

Results: Qwen2-VL-7B matched GPT-4o on several multimodal benchmarks and beat it on DocVQA (94.5 vs 88.4). The architecture change was the decisive move.

### Qwen2.5-VL (February 2025) — Dynamic FPS + Absolute Time

Qwen2.5-VL's big pivot is video. Dynamic FPS is not just "sample more frames when needed." The paper formalizes it:

- Absolute time tokens. Instead of positional indices (frame 0, 1, 2…), real timestamps. "At 0:04, the cat jumped." The model sees `<time>0.04</time>` tokens interleaved with frame tokens.
- Dynamic FPS. Slow scenes at 1 FPS, action at 4+ FPS. Chosen by user or trainer; M-RoPE adapts.
- Window attention in the ViT. For throughput, spatial attention is windowed (local within blocks); global attention every few layers.
- Explicit JSON output format. Trained on tool-call data: "{\"tool\": \"click\", \"coords\": [380, 220]}". Agent-ready out of the box.
- MRoPE-v2 scaling. Positions scale with max input size so a 10-minute video doesn't exhaust the frequency range.

Benchmarks: Qwen2.5-VL-72B beats GPT-4o on most video benchmarks, matches Gemini 2.0 on documents, and sets the open-model SOTA for GUI grounding (ScreenSpot: 84% accuracy vs GPT-4o's 38%).

### Qwen3-VL (November 2025)

Qwen3-VL is an incremental upgrade, consolidating rather than reinventing: larger LLM backbone (Qwen3-72B), expanded training data, improved OCR, stronger reasoning via Qwen3 "thinking mode." The ViT and M-RoPE remain unchanged. The paper focuses on data and training improvements, not architecture.

The takeaway for this lineage: by 2025 the Qwen-VL architecture has stabilized. Subsequent generations scale compute and data, not primitives.

### The Math of M-RoPE

Classic RoPE uses paired coordinates, rotating a query `q` of dimension `d` by position `m`:

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE splits the hidden dimension into three frequency bands. For example, `d = 96`. Allocate 32 dims to temporal, 32 to height, 32 to width. Each band is rotated by its own axis position. A patch at (t=5, h=10, w=20) has rotations `R_t(5)`, `R_h(10)`, `R_w(20)` applied to its three bands respectively.

Text tokens use `t = text_index, h = 0, w = 0` (or some normalized choice), staying compatible. Video frames use `t = frame_time, h = row, w = col`. Single images use `t = 0`.

The benefit: one positional encoding handles text, images, and video with no branching code or separate position tables.

### Dynamic FPS Sampling Logic

Given a video of duration `T` seconds and a target token budget `B`:

1. Compute max affordable FPS: `fps_max = B / (T * tokens_per_frame)`.
2. Pick a target FPS from `{1, 2, 4, 8}` satisfying `fps <= fps_max`.
3. If motion is intense (optical flow heuristic or explicit user request), pick higher FPS. If motion is calm, pick lower.
4. Sample uniformly at the chosen FPS; insert `<time>t</time>` tokens between frames.

Qwen2.5-VL trains this logic implicitly; at inference the user controls it via the `fps` parameter. A 60-second action sequence at 4 FPS, 81 tokens per frame = 19440 tokens, manageable in a 32k context.

### Structured Agent Output

Qwen2.5-VL's agent training explicitly targets structured tool calls:

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Parsing is deterministic: JSON.parse on model output. Compare to free-form "click at (1024, 512)", which requires regex and ambiguity handling. This shift is why Qwen2.5-VL's ScreenSpot score jumps from 55% (Qwen2-VL) to 84%.

## Use It

`code/main.py` implements:

- M-RoPE position computation for a packed sequence of mixed text, image patches, and video frames.
- A dynamic FPS sampler: given (duration, budget, motion level), picks FPS and produces frame timestamps.
- A toy Qwen2.5-VL JSON output parser handling tool-call responses with coordinate fields.

Run it, then swap fixed FPS for dynamic FPS on a 5-minute video to feel the difference.

## Ship It

This lesson produces `outputs/skill-qwen-vl-pipeline-designer.md`. Given a video task (surveillance, agent, action recognition, accessibility), it outputs a Qwen2.5-VL configuration (frame budget, FPS strategy, window attention toggle, agent output mode) and a latency estimate. Use it whenever you deploy a Qwen-VL family model for a video product.

## Exercises

1. For a patch at (t=3, h=5, w=7), hidden dimension 48 (16 per band, base theta 10000), compute the M-RoPE rotation. Give the rotation angles for the first three pairs of each band.

2. A 10-minute surveillance video at 1 FPS produces how many frames? At 384 resolution with 3× pooling, how many total tokens? Does Qwen2.5-VL's default 32k context fit it?

3. Pick FPS for a 30-second tennis rally, a 30-second cooking demo, and a 30-second UI-agent screen recording. Justify each using dynamic FPS logic.

4. Qwen2.5-VL dropped the Q-Former entirely. Why does a simple MLP work in 2025 but not in 2023? (Hint: data scale and encoder quality.)

5. Parse three Qwen2.5-VL JSON tool-call outputs into Python dicts. What fails on malformed JSON, and what recovery strategy does the Qwen cookbook recommend?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| M-RoPE | "multimodal RoPE" | 3D rotary positional embedding with temporal, height, and width frequency bands in the hidden dimension |
| Dynamic FPS | "smart sampling" | Frame sampling rate chosen per video based on motion, duration, and token budget |
| Absolute time tokens | "timestamp tokens" | `<time>t</time>` interleaved in the sequence so the model sees real seconds rather than frame indices |
| Window attention | "local attention" | Spatial self-attention restricted to small windows for speed; periodic global attention added |
| Structured agent output | "JSON mode" | Training data supervision that teaches the VLM to emit parseable JSON with coordinates and tool names |
| min_pixels / max_pixels | "resolution bounds" | Qwen2.5-VL's per-request controls that cap total pixel count and thus token count |
| Grounding | "point at it" | Outputting bounding-box coordinates as text tokens; used since Qwen-VL v1 |

## Further Reading

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
