# Video Generation

> An image is a 2-D tensor. A video is a 3-D tensor. The theory is the same; the compute is 10-100× harder. OpenAI's Sora (Feb 2024) proved it works. By 2026, Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0, and WAN 2.2 all produce 1080p video from text—and the open-weights stack (CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2) trails by 12 months.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## The Problem

A 24fps, 10-second, 1080p video is 240 frames of 1920×1080×3 pixels. ~1.5 GB raw per clip. Pixel-space diffusion is out. You need:

1. **Spatiotemporal compression.** A VAE that encodes video, not frames, into a sequence of spatiotemporal patches.
2. **Temporal coherence.** Frames must share content, lighting, and object identity over seconds. The network must model motion.
3. **Compute budget.** At the same model size, video training is 10-100× more expensive than image training.
4. **Conditioning.** Text, image (first frame), audio, or another video. Most production models accept all four.

The architecture that solves this is a **Diffusion Transformer (DiT)** applied to spatiotemporal patches, trained on massive (prompt, caption, video) datasets. The loss is the same diffusion loss from Lesson 06.

## The Concept

![Video diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchification

Encode the video with a 3D VAE (learned spatiotemporal compression). Latent shape is `[T_latent, H_latent, W_latent, C_latent]`. Slice into patches of size `[t_p, h_p, w_p]`. For Sora-style models, `t_p = 1` (per-frame patches) or `t_p = 2` (every two frames). A 10-second 1080p video compresses to ~20K-100K patches.

### Spatiotemporal DiT

A transformer processes this flattened patch sequence. Each patch has a 3D positional embedding (time + y + x). Attention is typically factorized:

- **Spatial attention** within patches of each frame.
- **Temporal attention** across frames at the same spatial position.
- **Full 3D attention** is 16-100× more expensive; used only at low resolution or in research.

### Text conditioning

Cross-attention with a large text encoder (Sora uses T5-XXL, CogVideoX-5B also uses T5-XXL). Long prompts are critical—Sora's training set has GPT-generated dense re-captions averaging 200 tokens per clip.

### Training

Standard diffusion loss (ε or v prediction) on spatiotemporal latents. Data: web video + ~100M curated clips + synthetic text captions. Compute: even a small research run is 10K+ GPU hours; Sora-scale is 100K+.

## 2026 Production Landscape

| Model | Date | Max Duration | Max Resolution | Open Weights? | Highlight |
|-------|------|--------------|---------|---------------|--------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | No | First to show world-simulator properties at scale |
| Sora Turbo | 2024-12 | 20s | 1080p | No | 5× faster inference production Sora |
| Veo 2 (Google) | 2024-12 | 8s | 4K | No | Highest quality + physics as of 2025 |
| Veo 3 | 2025 Q3 | 15s | 4K | No | Native audio and stronger camera control |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | No | Best human motion Q1 2025 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | No | Pro video tooling layer on top |
| Pika 2.0 | 2024-10 | 5s | 1080p | No | Best character consistency |
| CogVideoX (THUDM) | 2024 | 10s | 720p | Yes (2B, 5B) | First open-source 5B-scale video |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | Yes (13B) | Open-source SOTA late 2024 |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | Yes (10B) | Most permissive license |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | Yes | Strongest open-source model mid-2025 |

The open-weights gap is closing faster than in images: by mid-2026, HunyuanVideo + WAN 2.2 LoRAs power most open-source workflows.

## Build It

`code/main.py` simulates the core idea of a spatiotemporal DiT: patchify a small synthetic video, add per-patch positional embeddings, and denoise the entire sequence with a transformer-style cross-patch attention. No numpy; pure Python. We show that even in 1-D, temporal coherence emerges when adjacent frames' patches share a denoiser and positional embeddings.

### Step 1: Patchify a synthetic 1-D "video"

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### Step 2: Per-frame positional embedding

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### Step 3: Denoiser sees the full sequence

Instead of denoising each frame independently, our mini-network concatenates all frame values + their positional embeddings and jointly predicts noise for all frames.

### Step 4: Temporal coherence test

After training, sample a video. Measure frame-to-frame deltas. If the model learned temporal structure, these deltas will be smaller than when sampling frames independently.

## Pitfalls

- **Per-frame independent sampling = flicker.** If you run image diffusion per frame independently, the output flickers because each frame's noise is independent. Video diffusion fixes this by coupling frames through attention or shared noise.
- **Naive 3D attention = OOM.** Full 3D attention on 10-second 1080p latents is hundreds of billions of ops. Factorize into spatial + temporal.
- **Data captions matter more than scale.** Sora's main upgrade over prior work was training on ~10× more detailed captions (GPT-4 re-labeled clips). OpenAI's tech report is explicit about this.
- **First-frame conditioning.** Most production models also accept an image as the first frame. This is "image-to-video" mode; training includes this variant.
- **Physics drift.** Long clips (>10s) accumulate subtle inconsistencies. Sliding-window generation + keyframe anchoring helps.

## Real-World Usage

| Use Case | 2026 Choice |
|----------|-----------|
| Highest quality text-to-video, hosted | Veo 3 or Sora |
| Cinematic with camera control | Runway Gen-3 with motion brush |
| Character consistency across clips | Pika 2.0 or Kling 2.1 |
| Open weights, fast fine-tuning | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V, Kling 2.1 I2V, or Runway |
| Audio-to-video lip sync | Veo 3 (native audio) or a dedicated lip-sync model |
| Video editing | Runway Act-Two, Kling Motion Brush, Flux-Kontext (stills) |

Cost per second of video dropped 20× from 2024 to 2026 at comparable quality.

## Ship It

Save as `outputs/skill-video-brief.md`. The skill accepts a video brief (duration, aspect ratio, style, shot list, subject consistency, audio) and outputs: model + hosting, prompt scaffold (shot language, subject description, motion descriptors), seed + reproducibility workflow, and a frame-level QA checklist.

## Exercises

1. **Easy.** Compare frame-to-frame deltas in `code/main.py` between (a) independent per-frame sampling and (b) joint sequence sampling. Report mean and variance of deltas.
2. **Medium.** Add first-frame conditioning: pin frame 0 to a given value, sample the rest. Measure how the pinned value propagates.
3. **Hard.** Use HuggingFace diffusers to run CogVideoX-2B locally on a GPU. Time 20 inference steps for a 6-second clip at 720p. Profile spatiotemporal attention to locate the bottleneck.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Video VAE | "3D VAE" | Encoder compressing `(T, H, W, C)` into spatiotemporal latents. |
| Patch | "Those tokens" | Fixed-size 3D blocks of the latent; input to the DiT. |
| Factorized attention | "Spatial + temporal" | Attend spatially then temporally; skips full 3D attention. |
| Image-to-video (I2V) | "Animate this photo" | Model takes image + text, outputs video starting from it. |
| Keyframe conditioning | "Anchor frames" | Pin specific frames to control the video's trajectory. |
| Motion brush | "Direction hints" | UI input where users paint motion vectors on an image. |
| Re-captioning | "Dense captions" | Using an LLM to re-label training clips with detailed prompts. |
| Flicker | "Temporal artifact" | Frame-to-frame inconsistency; fixed by coupled denoising. |

## Production Notes: Video Latents Are a Memory-Bandwidth Problem

A 24 fps, 10-second, 1080p clip is 240 frames × 1920 × 1080 × 3 ≈ 1.5 GB raw pixels. After a 4× video VAE compression (`2× spatial × 2× temporal`), latents per request are ~100 MB. Passing it through a spatiotemporal DiT at batch 1 for 30 steps, you move ~3 GB in HBM per step—the bottleneck is memory bandwidth, not FLOPs.

Three production knobs, all directly from the inference chapter of the production literature:

- **TP on DiT.** Text-to-video models are ≥10B parameters. TP=4 across 4× H100s is standard; 405B-scale models use PP=2 × TP=2. Per-step latency drops roughly linearly with TP until you hit the all-reduce wall.
- **Frame batching = continuous batching.** During generation, a video is conceptually a batch of frames tied by attention. Continuous batching (in-flight scheduling) applies: if the model architecture allows sliding-window generation, you can start rendering frame `t+1` while returning frame `t-1`.
- **Clip-level prefill caching.** For image-to-video, first-frame conditioning is analogous to LLM prompt prefill: compute once, reuse across passes of the temporal decoder. This is effectively a video KV-cache.

## Further Reading

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora technical report.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — Mid-2025 open-source SOTA.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — Foundational video diffusion paper.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Precursor to Stable Video Diffusion.
