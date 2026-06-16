# Proximal Policy Optimization (PPO)

> A2C updates once per rollout and throws the data away. PPO wraps the policy gradient in a clipped importance ratio, letting you do 10+ epochs on the same data without blowing up the policy. Schulman et al. (2017). Still the default policy gradient algorithm in 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## The Problem

A2C (lesson 07) is on-policy: the gradient `E_{π_θ}[A · ∇ log π_θ]` requires data sampled from the *current* `π_θ`. One update and `π_θ` changes; the data you used is now off-policy. Reuse it and the gradient is biased.

Rollouts are expensive. On Atari, a single rollout across 8 envs × 128 steps = 1024 transitions plus tens of seconds of environment time. Throwing it away after one gradient step is wasteful.

Trust Region Policy Optimization (TRPO, Schulman 2015) was the first fix: constrain each update so the KL divergence between old and new policy stays below `δ`. Theoretically clean, but requires solving a conjugate gradient at every update. Nobody runs TRPO in 2026.

PPO (Schulman et al. 2017) replaces the hard trust-region constraint with a simple clipped objective. One extra line of code. Ten epochs per rollout. No conjugate gradient. Good-enough theoretical guarantees. Nine years later it remains the default policy gradient algorithm, from MuJoCo to RLHF.

## The Concept

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**Importance ratio.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

This is the likelihood ratio of the new policy relative to the one that collected the data. `r_t = 1` means no change. `r_t = 2` means the new policy is twice as likely to take `a_t` as the old one.

**Clipped surrogate objective.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Two terms:

- If advantage `A_t > 0` and the ratio tries to grow past `1 + ε`, the clip flattens the gradient — don't push a good action more than `+ε` above its old probability.
- If advantage `A_t < 0` and the ratio tries to grow past `1 - ε` (meaning we'd make a bad action more likely than its clipped reduction), the clip caps the gradient — don't push a bad action below `-ε`.

The `min` handles the other direction: if the ratio moved in the *favorable* direction, you still get gradient (no clipping on the side that helps you).

Typical `ε = 0.2`. Plot the objective as a function of `r_t`: a piecewise-linear function with a flat top on the "good side" and a flat bottom on the "bad side."

**Full PPO loss.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Same actor-critic structure as A2C. Three coefficients, typically `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**Training loop.**

1. Run `T` steps across `N` parallel environments, collecting `N × T` transitions.
2. Compute advantages (GAE), freeze as constants.
3. Freeze `π_{θ_old}` as a snapshot of the current `π_θ`.
4. Run `K` epochs, each with minibatches of `(s, a, A, V_target, log π_old(a|s))`:
   - Compute `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Apply `L^{CLIP}` + value loss + entropy.
   - Take one gradient step.
5. Discard the rollout. Go back to step 1.

`K = 10`, minibatch size 64 is a standard set of hyperparameters. PPO is robust: exact numbers rarely matter within ±50%.

**KL penalty variant.** The original paper proposes an alternative with an adaptive KL penalty: `L = L^{PG} - β · KL(π_θ || π_old)`, where `β` adjusts based on observed KL. The clipped version dominates; the KL variant survives in RLHF (where KL to a reference policy is a separate constraint you always want).

## Build It

### Step 1: Record `log π_old(a | s)` at rollout time

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

The snapshot is taken once at rollout time. It does not change during update epochs.

### Step 2: Compute GAE advantages (lesson 07)

Same as A2C. Normalize within the batch.

### Step 3: Clipped surrogate update

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

The "clipped → zero gradient" pattern is the heart of PPO. If the new policy has already drifted too far in the favorable direction, the update stops.

### Step 4: Value and entropy

Add standard MSE for the critic target, entropy bonus for the actor, same as A2C.

### Step 5: Diagnostics

Monitor three things every update:

- **Mean KL** `E[log π_old - log π_θ]`. Should stay in `[0, 0.02]`. If it spikes past `0.1`, reduce `K_EPOCHS` or `LR`.
- **Clip fraction** — the share of samples where the ratio falls outside `[1-ε, 1+ε]`. Should be `~0.1-0.3`. If `~0`, clipping never triggers → increase `LR` or `K_EPOCHS`. If `~0.5+`, you're overfitting the rollout → decrease them.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`. Critic quality metric. Should climb toward 1 as the critic learns.

## Pitfalls

- **Clip coefficient not tuned.** `ε = 0.2` is the de facto standard. Dropping to `0.1` makes updates too timid; `0.3+` invites instability.
- **Too many epochs.** `K > 20` often destabilizes because the policy drifts too far from `π_old`. Cap epochs, especially for large networks.
- **No reward normalization.** Large reward scales eat into the clip interval. Normalize rewards (running std) before computing advantages.
- **Forgetting advantage normalization.** Per-batch zero-mean/unit-std normalization is standard practice. Skipping it ruins PPO on most benchmarks.
- **No learning rate decay.** PPO benefits from linearly decaying LR to zero. Constant LR tends to underperform.
- **Importance ratio computed wrong.** Always use `exp(log_new - log_old)` for numerical stability, not `new / old`.
- **Gradient sign flipped.** Maximizing the surrogate = *minimizing* `-L^{CLIP}`. Flipping the sign is the most common PPO bug.

## Use It

PPO is the default RL algorithm in 2026, covering a surprisingly broad range:

| Use case | PPO variant |
|----------|-------------|
| MuJoCo / robotic control | PPO with Gaussian policy, GAE(0.95) |
| Atari / discrete games | PPO with categorical policy, rolling 128-step rollouts |
| RLHF for LLMs | PPO with KL penalty to reference model, reward from RM at end of response |
| Large-scale game agents | IMPALA + PPO (AlphaStar, OpenAI Five) |
| Reasoning LLMs | GRPO (lesson 12) — PPO variant without the critic |
| Preference-only data | DPO — closed-form collapse of PPO+KL with no online sampling |

PPO's *loss shape* — clipped surrogate + value + entropy — is the scaffolding beneath DPO, GRPO, and nearly every RLHF pipeline.

## Ship It

Save as `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Exercises

1. **Easy.** Run PPO on 4×4 GridWorld with `ε=0.2, K=4`. Compare sample efficiency against A2C (one epoch per rollout) with matched environment steps.
2. **Medium.** Sweep `K ∈ {1, 4, 10, 30}`. Plot return vs environment steps and track mean KL per update. At what `K` does the KL explode on this task?
3. **Hard.** Replace the clipped surrogate with an adaptive KL penalty (double `β` if `KL > 2·target`, halve if `KL < target/2`). Compare final return, stability, and degree of clip-free behavior.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`; deviation from the data-collecting policy. |
| Clipped surrogate | "PPO's main trick" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; gradient goes flat past the clip on the favorable side. |
| Trust region | "TRPO / PPO's intent" | Limit per-update KL to guarantee monotonic improvement. |
| KL penalty | "soft trust region" | Alternative PPO: `L - β · KL(π_θ || π_old)`. Adaptive `β`. |
| Clip fraction | "how often clipping triggers" | Diagnostic — should be 0.1–0.3; outside means mis-tuned. |
| Multiple epochs | "data reuse" | K passes per rollout; trades variance cost for sample efficiency. |
| Approximately on-policy | "basically on-policy" | PPO is nominally on-policy but K>1 epochs safely use slightly off-policy data. |
| PPO-KL | "the other PPO" | KL penalty variant; used in RLHF where KL to reference is already a constraint. |

## Further Reading

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — The paper.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, PPO's predecessor.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — Ablates every PPO hyperparameter.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; PPO recipe in RLHF.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — Clear modern walkthrough with PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — Reference single-file PPO used by many papers.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — Production recipe for running PPO on language models; read alongside lesson 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "The 37 code-level optimizations" paper; which PPO tricks are load-bearing vs folklore.
