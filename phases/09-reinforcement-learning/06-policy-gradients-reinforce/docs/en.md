# Policy Gradients — REINFORCE from Scratch

> Stop estimating values. Parameterize the policy directly, compute the gradient of expected return, and ascend. Williams (1992) wrote it in one theorem. PPO, GRPO, and every LLM RL loop exist because of it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## The Problem

Q-learning and DQN parameterize a *value* function. You pick actions via `argmax Q`. For discrete actions and discrete states, that works. But when actions are continuous (which 10-dimensional torque do you `argmax`?) or when you want a stochastic policy (`argmax` is inherently deterministic), it breaks down.

Policy gradients instead parameterize the *policy* itself. `π_θ(a | s)` is a neural network that outputs an action distribution. Sample from it to act. Compute the gradient of expected return with respect to `θ`. Ascend. No `argmax`. No Bellman recursion. Just gradient ascent on `J(θ) = E_{π_θ}[G]`.

The REINFORCE theorem (Williams 1992) tells you this gradient is computable: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Run an episode. Compute returns. Multiply each step by `∇ log π_θ(a | s)`. Average. Gradient ascent. Done.

Every LLM-RL algorithm in 2026 — PPO, DPO, GRPO — is a refinement of REINFORCE. Getting it under your fingers is the prerequisite for the rest of this phase, and for Phase 10 · 07 (RLHF implementation) and Phase 10 · 08 (DPO).

## The Concept

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Policy gradient theorem.** For any policy `π_θ` parameterized by `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

where `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` is the discounted return from step `t`. The expectation is over complete trajectories `τ` sampled from `π_θ`.

**The proof is short.** Differentiate `J(θ) = Σ_τ P(τ; θ) G(τ)` under the expectation. Use `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (log-derivative trick). Factor `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms independent of θ`. Environment terms vanish. Two lines of algebra yield the theorem.

**Variance reduction tricks.** Vanilla REINFORCE has enormous variance — returns are noisy, `∇ log π` is noisy, and their product is extremely noisy. Two standard fixes:

1. **Subtract a baseline.** Replace `G_t` with `G_t - b(s_t)`, where `b(s_t)` is any baseline independent of `a_t`. Unbiased because `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Typical choice: `b(s_t) = V̂(s_t)` learned by a critic → actor-critic (lesson 07).
2. **Reward-to-go.** Replace `Σ_t G_t · ∇ log π_θ(a_t | s_t)` with `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. For a given action, only future returns matter — past rewards contribute zero-mean noise.

Together, you get:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

This is REINFORCE with baseline — the direct ancestor of A2C (lesson 07) and PPO (lesson 08).

**Softmax policy parameterization.** For discrete actions, the standard choice:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

where `f_θ` is any neural network outputting a score per action. The gradient has a clean form:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

That is: the score of the taken action minus its expected value under the policy.

**Gaussian policy for continuous actions.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` has closed form. Phase 9 · 07's SAC needs exactly this.

## Build It

### Step 1: Softmax policy network

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Linear policy (one weight vector per action) for tabular environments. For Atari, swap in a CNN and keep the softmax head.

### Step 2: Sampling and log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Step 3: Rollout and record log-probs

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Step 4: REINFORCE update

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

The gradient `∇ log π(a|s) = e_a - π(·|s)` (one-hot of `a` minus probabilities) is the heart of the softmax policy gradient. Commit it to muscle memory.

### Step 5: Baseline

A running mean of `G` over recent episodes is enough variance reduction to make a 4×4 GridWorld work; converges in ~500 episodes. Upgrade the baseline to a learned `V̂(s)` and you have actor-critic.

## Pitfalls

- **Gradient explosion.** Returns can be large. Always normalize `G` within the batch to `~N(0, 1)` before multiplying by `∇ log π`.
- **Entropy collapse.** Policy converges too early to a near-deterministic action, stops exploring, gets stuck. Fix: add entropy bonus `β · H(π(·|s))` to the objective.
- **High variance.** Vanilla REINFORCE needs thousands of episodes. Critic baseline (lesson 07) or TRPO/PPO trust regions (lesson 08) are the standard fixes.
- **Sample inefficiency.** On-policy means each transition is used once then discarded. Off-policy correction via importance sampling recovers data at the cost of variance (PPO's ratio is a clipped IS weight).
- **Non-stationary gradients.** The same gradient from 100 episodes ago used an old `π`. This is why on-policy methods update every few rollouts.
- **Credit assignment.** Without reward-to-go, past rewards contribute noise. Always use reward-to-go.

## Use It

By 2026, REINFORCE is rarely run directly, but its gradient formula is everywhere:

| Use case | Descendant |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete-action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — rewrites REINFORCE into preference likelihood loss without sampling |

When you read `loss = -advantage * log_prob` in a 2026 training script, that's REINFORCE with baseline. Entire papers (DPO, GRPO, RLOO) are variance reduction tricks built on top of this single line.

## Ship It

Save as `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Exercises

1. **Easy.** Implement REINFORCE with a linear softmax policy on the 4×4 GridWorld. Train for 1000 episodes without baseline. Plot the learning curve; measure variance (standard deviation of returns).
2. **Medium.** Add a running-mean baseline. Train again. Compare sample efficiency and variance against the no-baseline run. How much does the baseline reduce convergence steps?
3. **Hard.** Add entropy bonus `β · H(π)`. Sweep `β ∈ {0, 0.01, 0.1, 1.0}`. Plot final return and policy entropy. Where is the sweet spot for this task?

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Policy gradient | "Train the policy directly" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`; derived via the log-derivative trick. |
| REINFORCE | "The original PG algorithm" | Williams (1992); Monte Carlo return times log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; makes gradient of an expectation tractable. |
| Baseline | "Variance reduction" | Any `b(s)` subtracted from `G`; unbiased because `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Only future returns count" | Use `G_t^{from t}` instead of full `G_0`; correct and lower variance. |
| Entropy bonus | "Encourage exploration" | `+β · H(π(·|s))` term preventing policy collapse. |
| On-policy | "Train on what you just saw" | Gradient expectation is over the current policy — can't directly reuse old data. |
| Advantage | "How much better than average" | `A(s, a) = G(s, a) - V(s)`; the signed quantity REINFORCE-with-baseline multiplies. |

## Further Reading

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — The original REINFORCE paper.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — The modern policy gradient theorem with function approximation.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Textbook presentation.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — Clear pedagogical exposition with PyTorch code.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — Variance reduction and the natural gradient perspective connecting REINFORCE to the trust-region family (TRPO, PPO).
