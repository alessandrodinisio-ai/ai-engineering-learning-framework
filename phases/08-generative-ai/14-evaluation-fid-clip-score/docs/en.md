# Evaluation — FID, CLIP Score, Human Preference

> Every generative model leaderboard cites FID, CLIP score, and win rates from human-preference arenas. Each number has a failure mode an obsessive researcher can exploit. If you don't understand these failure modes, you cannot distinguish a real improvement from a gamed run.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time:** ~45 minutes

## The Problem

A generative model is judged on *sample quality* and *condition adherence*. Neither has a closed-form metric. Your model must render 10k images; something must assign numbers to them; you must trust those numbers across model families, resolutions, and architectures. Three metrics survived the 2014–2026 gauntlet:

- **FID (Fréchet Inception Distance).** Distance between the real and generated distributions in an Inception network's feature space. Lower is better.
- **CLIP score.** Cosine similarity between the CLIP image embedding of a generated image and the CLIP text embedding of its prompt. Higher is better. Measures prompt adherence.
- **Human preference.** Pit two models head-to-head on the same prompt, have humans (or a GPT-4-class model) pick the winner, aggregate into an Elo score.

You'll also see: IS (inception score, mostly retired), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Each corrects a flaw of the previous one.

## The Concept

![FID, CLIP, and preference: three dimensions, different failure modes](../assets/evaluation.svg)

### FID — Sample Quality

Heusel et al. (2017). Steps:

1. Extract Inception-v3 features (2048-dim) for N real images and N generated images.
2. Fit a Gaussian to each pool: compute means `μ_r, μ_g` and covariances `Σ_r, Σ_g`.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

Interpretation: Fréchet distance between two multivariate Gaussians in feature space. Lower = more similar distributions.

Failure modes:
- **Biased at small N.** FID is a mean-square over feature distributions—small N underestimates covariance, giving falsely low FID. Always use N ≥ 10,000.
- **Inception-dependent.** Inception-v3 was trained on ImageNet. Domains far from ImageNet (faces, art, text images) produce meaningless FID. Use a domain-specific feature extractor.
- **Gaming.** Overfitting Inception priors can yield low FID without visual quality improvement. Use CMMD (below) to counter.

### CLIP Score — Prompt Adherence

Radford et al. (2021). For a generated image + prompt:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Average over 30k generated images → a scalar comparable across models.

Failure modes:
- **CLIP's own blind spots.** CLIP has weak compositional reasoning ("a red cube on a blue ball" often fails). A model can score high on CLIP score while not actually following complex prompts.
- **Short-prompt bias.** Short prompts have more CLIP image matches in the wild. Long prompts mechanically score lower.
- **Prompt gaming.** Stuffing "high quality, 4k, masterpiece" into prompts inflates CLIP score without improving text-image binding.

CMMD (Jayasumana et al., 2024) fixes some of these: uses CLIP features instead of Inception, maximum mean discrepancy instead of Fréchet. Better at detecting subtle quality differences.

### Human Preference — The Ground Truth

Pick a pool of prompts. Generate with model A and model B. Show paired results to humans (or a strong LLM judge). Aggregate wins into an Elo or Bradley-Terry score. Benchmarks:

- **PartiPrompts (Google):** 1,600 diverse prompts across 12 categories.
- **HPSv2:** 107k human annotations, widely used as automated proxy.
- **ImageReward:** 137k prompt-image preference pairs, MIT license.
- **PickScore:** Trained on 2.6M preferences from Pick-a-Pic.
- **Chatbot-Arena-style image arenas:** https://imagearena.ai/ and others.

Failure modes:
- **Judge variance.** Non-expert vs expert preferences differ. Use both.
- **Prompt distribution.** Cherry-picked prompts bias toward a family. Always document.
- **LLM judge reward hacking.** GPT-4 judges can be fooled by "pretty but wrong" outputs. Triangulate with humans.

## Putting Them Together

A production evaluation report should include:

1. FID on 10–30k samples relative to a held-out real distribution (sample quality).
2. CLIP score / CMMD on the same samples relative to their prompts (adherence).
3. Win rate relative to the previous model in a blind arena (overall preference).
4. Failure mode analysis: randomly sample 50 outputs and tag by known issues (hand anatomy, text rendering, object count consistency).

Any single metric is a lie. Three mutually reinforcing metrics + qualitative review is a claim.

## Build It

`code/main.py` implements FID, a CLIP-like score, and Elo aggregation on synthetic "feature vectors" (we use 4D vectors in place of Inception features). You see:

- FID computation at small N and large N—the bias.
- "CLIP score" as cosine similarity between feature pools.
- Elo update rule from a synthetic preference stream.

### Step 1: FID in Four Lines

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Step 2: CLIP-Style Cosine Similarity

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Step 3: Elo Aggregation

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Pitfalls

- **FID at N=1000.** The heuristic is unreliable below N=10k. Papers reporting low-N FID are gaming.
- **Comparing FID across resolutions.** Inception's 299×299 rescaling changes the feature distribution. Only compare at matched resolutions.
- **Reporting a single seed.** Run at least 3 seeds. Report standard deviation.
- **Inflating CLIP score through negative prompts.** Some pipelines inflate CLIP by overfitting prompts. Check visual saturation.
- **Elo bias from prompt overlap.** If two models both saw a benchmark prompt during training, Elo is meaningless. Use held-out prompt sets.
- **Paid crowd-eval skew.** Prolific, MTurk annotators skew young / tech-friendly. Mix with recruited art/design experts.

## Use It

Production evaluation workflow in 2026:

| Pillar | Minimum | Recommended |
|--------|---------|-------------|
| Sample quality | FID on 10k relative to held-out real set | + CMMD on 5k + per-category FID subsets |
| Prompt adherence | CLIP score on 30k | + HPSv2 + ImageReward + VQA-style probes |
| Preference | 200-pair blind test vs baseline | + 2000-pair human eval + LLM judge + Chatbot Arena |
| Failure analysis | 50 hand-tagged | 500 hand-tagged + automated safety classifiers |

Four pillars in one report = a claim. Any single one = marketing.

## Ship It

Save as `outputs/skill-eval-report.md`. The skill accepts a new model checkpoint + baseline and outputs a full evaluation plan: sample sizes, metrics, failure-mode probes, release gates.

## Exercises

1. **Easy.** Run `code/main.py`. Compare FID at N=100 vs N=1000 on the same synthetic distributions. Report bias magnitude.
2. **Medium.** Implement CMMD from synthetic CLIP-style features (formula in Jayasumana et al., 2024). Compare its sensitivity to quality differences vs FID.
3. **Hard.** Reproduce the HPSv2 setup: take 1000 image-prompt pairs from a Pick-a-Pic subset, fine-tune a small CLIP-based scorer on these preferences, measure agreement with a held-out set.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | Fréchet distance between Gaussian fits of real vs generated Inception features. |
| CLIP score | "image-text similarity" | Cosine similarity between CLIP image and text embeddings. |
| CMMD | "FID replacement" | MMD of CLIP features; lower bias, no Gaussian assumption. |
| IS | "inception score" | Exp KL(p(y|x) \|\| p(y)); correlates poorly on modern models, retired. |
| HPSv2 / ImageReward / PickScore | "learned preference proxies" | Small models trained on human preferences; used as automated judges. |
| Elo | "chess rating" | Bradley-Terry aggregation of pairwise wins. |
| PartiPrompts | "the benchmark prompt set" | Google's curated 1,600 prompts across 12 categories. |
| FD-DINO | "self-supervised alternative" | FD using DINOv2 features; better for non-ImageNet domains. |

## Production Notes: Evaluation Is Also an Inference Workload

Running FID on 10k samples means generating 10k images. For 50-step SDXL base at 1024² on a single L4, that is ~11 hours of single-request inference. Evaluation budgets are real money, and the workload is exactly the offline inference scenario (maximize throughput, ignore TTFT):

- **Batch aggressively, forget latency.** Offline eval = static batching at the largest size that fits memory. Running `pipe(...).images` with `num_images_per_prompt=8` on 80GB H100 wallclocks 4–6× faster than single-request.
- **Cache real features.** Inception (FID) or CLIP (CLIP score, CMMD) feature extraction on the real reference set runs *once*, saved as a `.npz`. Don't recompute per eval.

CI / regression gating: each PR runs FID + CLIP score on a 500-sample subset (~30 min); nightly runs the full 10k FID + HPSv2 + Elo.

## Further Reading

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — the FID paper.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — failure mode survey.
