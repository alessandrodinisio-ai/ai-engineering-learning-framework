# Flow Matching and Rectified Flows

> Diffusion models sample in 20–50 steps because they travel a curved path from noise to data. Flow matching (Lipman et al., 2023) and rectified flows (Liu et al., 2022) straighten that path. Straighter path, fewer steps, faster inference. Stable Diffusion 3, Flux.1, and AudioCraft 2 all switched to flow matching in 2024.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 minutes

## The Problem

DDPM's reverse process is a 1000-step random walk from `N(0, I)` back to the data distribution. DDIM compresses it into 20–50 deterministic steps. You want fewer steps—ideally one. The blocker is that the ODE solving the reverse process is stiff; the path is curved.

If you could train the model so that the path from noise to data is a *straight line*, then a single Euler step from `t=1` to `t=0` suffices. Flow matching constructs exactly this: define a straight-line interpolation from `x_1 ∼ N(0, I)` to `x_0 ∼ data`, train a vector field `v_θ(x, t)` to match its time derivative, and integrate at inference.

Rectified flows (Liu 2022) go further: use a reflow procedure to iteratively straighten paths, producing an increasingly linear ODE. After two reflow iterations, a 2-step sampler matches 50-step DDPM quality.

## The Concept

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### Straight-Line Flows

Define:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

where `x_0 ~ data`, `x_1 ~ N(0, I)`. The time derivative along this line is constant:

```
dx_t / dt = x_1 - x_0
```

Define a neural vector field `v_θ(x_t, t)` and train it to match this derivative:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

This is the **conditional flow matching** loss (Lipman 2023). Training is simulation-free: you never unroll the ODE. Just sample `(x_0, x_1, t)` and regress.

### Sampling

At inference, integrate the learned vector field *backward* in time:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

Start from `x_1 ~ N(0, I)`, Euler-step down to `t=0`.

### Rectified Flows (Liu 2022)

Straight-line flows work, but the learned paths are *not actually straight*—they bend because many `x_0` can map to the same `x_1`. The rectified flow reflow step:

1. Train a flow model v_1 with random pairings.
2. Sample N pairs `(x_1, x_0)` by integrating v_1 from `x_1` to its landing point `x_0`.
3. Train v_2 on these paired samples. Because the pairs are now "ODE-matched," the straight-line interpolation between them genuinely becomes flatter.
4. Repeat.

In practice 2 reflow iterations get you near-linear, enabling 2–4 step inference. SDXL-Turbo, SD3-Turbo, LCM are all distilled from flow-matching models.

### Why It Won for Images in 2024

Three reasons:

1. **Simulation-free training**—no ODE unrolling during training, trivial to implement.
2. **Better loss geometry**—straight paths have consistent signal-to-noise ratio, while DDPM's ε-loss has poor SNR at schedule extremes.
3. **Faster inference**—4–8 steps at SDXL-Turbo quality; 1 step with consistency distillation.

## Flow Matching vs DDPM — The Exact Connection

Flow matching with Gaussian conditional paths is diffusion *with a specific noise schedule*. Pick the schedule `x_t = α(t) x_0 + σ(t) x_1`, and flow matching recovers the Stratonovich restatement of diffusion with `v = α'·x_0 - σ'·x_1`. For Gaussian paths, the two are algebraically equivalent.

What flow matching adds: *clarity* of the objective (a plain velocity), a cleaner loss landscape, and permission to experiment with non-Gaussian interpolants.

## Build It

`code/main.py` implements 1D flow matching on a bimodal Gaussian mixture. The vector field `v_θ(x, t)` is a mini MLP trained with the straight-line target. Inference integrates with 1, 2, 4, and 20 Euler steps to compare sample quality.

### Step 1: Training Loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Step 2: Multi-Step Inference

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Step 3: Compare Step Counts

Expected: a 4-step sampler already matches the quality of 20 steps—this matters for latency.

## Pitfalls

- **Time parameterization.** Flow matching uses `t ∈ [0, 1]` with `t=0` at data and `t=1` at noise. DDPM uses `t ∈ [0, T]` with `t=0` at data and `t=T` at noise. Same direction, different scale. Papers get this wrong constantly.
- **Schedule choice.** The rectified-flow straight line is "the" flow matching schedule, but you can use cosine or logit-normal t-sampling (SD3 does this) to better cover scales.
- **Reflow cost.** Generating paired datasets for reflow costs one full inference pass per sample. Only reflow when you genuinely need 1–2 step inference.
- **Classifier-free guidance still applies.** Just swap ε for v in the linear combination: `v_cfg = (1+w) v_cond - w v_uncond`.

## Use It

| Use case | 2026 Stack |
|----------|-----------|
| Text-to-image, best quality | Flow matching: SD3, Flux.1-dev |
| Text-to-image, 1–4 steps | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Real-time inference | Consistency distillation from flow-matching base (LCM, PCM) |
| Audio generation | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Video generation | Flow matching + diffusion hybrid (Sora, Veo, Stable Video) |
| Scientific / physics (particle trajectories, molecules) | Flow matching + equivariant vector fields |

In 2025–2026 whenever a paper says "faster than diffusion," it is almost always flow matching + distillation.

## Ship It

Save as `outputs/skill-fm-tuner.md`. The skill accepts a diffusion-style model spec and converts it to a flow matching training config: schedule choice, time-sampling distribution (uniform / logit-normal), optimizer, reflow plan, target step count, evaluation workflow.

## Exercises

1. **Easy.** Run `code/main.py`, compare 1-step vs 20-step MSE relative to the true data distribution.
2. **Medium.** Replace uniform `t` sampling with logit-normal (concentrate sampling at mid-range t). Does model quality improve?
3. **Hard.** Implement one reflow iteration: integrate the first model to generate paired (x_0, x_1), train a second model on those pairs, compare 1-step sample quality.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Flow matching | "straight-line diffusion" | Train `v_θ(x, t)` to match `x_1 - x_0` along interpolants. |
| Rectified flow | "Reflow" | Iterative process to straighten learned flows. |
| Velocity field | "v_θ" | The model's output—the direction to move `x_t`. |
| Straight-line interpolant | "the path" | `x_t = (1-t)·x_0 + t·x_1`; target derivative is trivially simple. |
| Euler sampler | "first-order ODE solver" | Simplest integrator; works well when paths are straight. |
| Logit-normal t | "SD3 sampling" | Concentrate `t` sampling at mid-range values where gradients are strongest. |
| Consistency distillation | "one-step sampler" | Train a student to map any `x_t` directly to `x_0`. |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; same trick, new variable. |

## Production Notes: Flux.1-schnell Is Flow Matching at Its Fastest

The production payoff of flow matching is Flux.1-schnell—a flow-matching DiT distilled to 1–4 step inference while retaining Flux-dev quality. Niels's "running Flux on 8GB" notebook is the reference deployment recipe: T5 + CLIP encoding, quantized MMDiT denoising (schnell uses 4 steps vs dev's 50), VAE decode. Cost breakdown:

| Variant | Steps | Latency at 1024² on L4 | Total FLOPs (relative) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (original) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× faster) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

Production rule: **flow-matching base + distillation = the default for fast text-to-image in 2026.** Every major provider ships this combo: SD3-Turbo (SD3 + flow + distillation), Flux-schnell (Flux-dev + rectified-flow straightening), CogView-4-Flash. Pure-diffusion base models only survive in legacy checkpoints.

## Further Reading

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flows.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, rectified flows at scale.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — general framework covering FM + diffusion.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — one-step distillation from diffusion/flow.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo variants.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — flow matching in production.
