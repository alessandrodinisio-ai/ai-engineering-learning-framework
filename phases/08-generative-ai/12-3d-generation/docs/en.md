# 3D Generation

> 3D is the modality with the strongest 2D-to-3D leverage. The 2023 breakthrough was 3D Gaussian Splatting. The 2024–2026 generative push stacks multi-view diffusion + 3D reconstruction on top, producing objects and scenes from a single prompt or photograph.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 minutes

## The Problem

3D content is painful:

- **Representation.** Meshes, point clouds, voxel grids, signed distance fields (SDF), neural radiance fields (NeRF), 3D Gaussians. Each has tradeoffs.
- **Data scarcity.** ImageNet has 14M images. The largest clean 3D dataset (Objaverse-XL, 2023) has ~10M objects, most low quality.
- **Memory.** A 512³ voxel grid is 128M voxels; a useful scene NeRF takes 1M samples per ray. Generation is harder than reconstruction.
- **Supervision.** For a 2D image you have pixels. For 3D you typically have only a handful of 2D views and must lift to 3D.

The 2026 stack separates the two problems. First, use a diffusion model to generate *2D multi-view images*. Then fit a *3D representation* (usually Gaussian splatting) to those images.

## The Concept

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### Representation: 3D Gaussian Splatting (Kerbl et al., 2023)

Represent a scene as a cloud of ~1M 3D Gaussians. Each has 59 parameters: position (3), covariance (6, or quaternion 4 + scale 3), opacity (1), spherical-harmonic color (48 at degree 3, or 3 at degree 0).

Rendering = projection + alpha compositing. Fast (~100 fps at 1080p on a 4090). Differentiable. Fit by gradient descent on ground-truth photos. A scene fits in 5–30 minutes on a consumer GPU.

Two 2023–2024 innovations on top:
- **Generative Gaussian splatting.** Models like LGM, LRM, InstantMesh directly predict a Gaussian cloud from one or few images.
- **4D Gaussian splatting.** Gaussians with per-frame offsets for dynamic scenes.

### Multi-View Diffusion

Fine-tune a pretrained image diffusion model to generate multiple consistent views of the same object from a text prompt or single image. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). Typically output 4–16 views around the object, then lift to 3D via Gaussian splatting or NeRF.

### Text-to-3D Pipelines

| Model | Input | Output | Time |
|-------|-------|--------|------|
| DreamFusion (2022) | Text | NeRF via SDS | ~1 hour per asset |
| Magic3D | Text | Mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | Text | Implicit 3D | ~1 min |
| SJC / ProlificDreamer | Text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | Image | Triplane | ~5 sec |
| InstantMesh (2024) | Image | Mesh | ~10 sec |
| SV3D (Stability, 2024) | Image | Novel views | ~2 min |
| CAT3D (Google, 2024) | 1–64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | Image | Mesh | ~1 sec |
| Meshy 4 (2025) | Text + image | PBR mesh | ~30 sec |
| Rodin Gen-1.5 (2025) | Text + image | PBR mesh | ~60 sec |
| Tencent Hunyuan3D 2.0 (2025) | Image | Mesh | ~30 sec |

2025–2026 direction: direct text-to-mesh models with PBR materials suitable for game engines. For general objects, the multi-view diffusion intermediate step remains the best-performing recipe.

### NeRF (as Background)

Neural Radiance Fields (Mildenhall et al., 2020). A mini MLP takes `(x, y, z, viewing direction)` and outputs `(color, density)`. Integrate along rays to render. Outperforms mesh-based novel view synthesis on quality but renders 100–1000× slower. Replaced by Gaussian splatting for most real-time use cases, but still dominant in research.

## Build It

`code/main.py` implements a toy 2D "Gaussian splatting" fit: represent a synthetic target image (a smooth gradient) as a sum of 2D Gaussian splats. Optimize positions, colors, and covariances via gradient descent to match the target. You see two core operations: forward rendering (splat + alpha composite) and gradient-descent fitting.

### Step 1: 2D Gaussian Splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Step 2: Render by Accumulating Splats

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

Real 3D Gaussian splatting sorts Gaussians by depth and alpha-composites in order. Our 2D toy just accumulates.

### Step 3: Gradient Descent Fit

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Pitfalls

- **View inconsistency.** If you generate 4 views independently and they disagree on object structure, the 3D fit will be blurry. Solution: multi-view diffusion with shared attention.
- **Back-face hallucination.** Single-image → 3D must hallucinate the unseen side. Quality is hit-or-miss.
- **Gaussian splatting explosion.** Unconstrained training grows to 10M splats and overfits. Densification + pruning heuristics (from the original 3D-GS paper) are required.
- **Topology issues.** Meshes from implicit fields (SDF) often have holes or self-intersections. Run a remesher (e.g., Blender's voxel remesh) before shipping.
- **Training data licensing.** Objaverse licenses are mixed; commercial use varies by model.

## Use It

| Task | 2026 Pick |
|------|-----------|
| Scene reconstruction from photos | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| Text-to-3D objects for games | Meshy 4 or Rodin Gen-1.5 (PBR output) |
| Image-to-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Novel view synthesis from few images | CAT3D, SV3D |
| Dynamic scene reconstruction | 4D Gaussian splatting |
| Avatars / clothed humans | Gaussian Avatar, HUGS |
| Research / SOTA | Whatever came out last week |

To ship production 3D in a game or e-commerce pipeline: Meshy 4 or Rodin Gen-1.5's PBR meshes drop straight into Unity / Unreal.

## Ship It

Save as `outputs/skill-3d-pipeline.md`. The skill accepts a 3D requirement (input: text / single image / few images; output: mesh / splats / NeRF; use: rendering / game / VR), outputs: pipeline (multi-view diffusion + fitting, or direct mesh model), base model, iteration budget, topology post-processing, material channels needed.

## Exercises

1. **Easy.** Run `code/main.py` with 4, 16, 64 Gaussians. Report final MSE relative to target.
2. **Medium.** Extend to color Gaussians (RGB). Confirm reconstruction matches target color patterns.
3. **Hard.** Using gsplat or Nerfstudio, reconstruct a real object from a 50-photo capture. Report fitting time and final SSIM on held-out views.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | Represent a scene as a cloud of 3D Gaussians; differentiable alpha-compositing render. |
| NeRF | "neural radiance field" | MLP that outputs color + density at a 3D point; renders via ray integration. |
| Triplane | "three 2D planes" | Factorize 3D into three axis-aligned 2D feature grids; cheaper than voxels. |
| SDS | "score distillation sampling" | Use a 2D diffusion model's score as a pseudo-gradient to train a 3D model. |
| Multi-view diffusion | "many views at once" | Diffusion model that outputs a batch of consistent camera views. |
| PBR | "physically-based rendering" | Materials with albedo, roughness, metallic, normal channels. |
| Densification | "grow splats" | 3DGS training heuristic: split/clone splats in high-gradient regions. |

## Production Notes: 3D Has No Shared Foundation Yet

Unlike images (latent diffusion + DiT) and video (spatiotemporal DiT), 3D has no single dominant runtime in 2026. The production decision tree forks on representation:

- **NeRF / triplane.** Inference is ray marching + one MLP forward per sample. A 512² render needs millions of MLP forwards. Batch ray samples aggressively; SDPA/xformers apply.
- **Multi-view diffusion + LRM reconstruction.** Two-stage pipeline. Stage 1 (multi-view DiT) is just a diffusion server like lesson 07. Stage 2 (LRM transformer) is a single-pass forward on those views. The overall latency profile is "diffusion + one-shot"—pick serving primitives for each stage accordingly.
- **SDS / DreamFusion.** Per-asset optimization, not inference. Build batch jobs, not request handlers.

For most 2026 products, the correct answer is "run a multi-view diffusion model per request, reconstruct into 3DGS asynchronously, serve the 3DGS for real-time viewing." This splits the workload cleanly between a GPU inference server (fast) and an offline optimizer (slow).

## Further Reading

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.
