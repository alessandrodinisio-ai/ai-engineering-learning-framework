# Monte Carlo Methods — Learning from Complete Episodes

> Dynamic programming needs a model. Monte Carlo needs nothing but episodes. Run the policy, observe returns, average. It's the simplest idea in RL — and the one that unlocks everything after it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDP), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## The Problem

Dynamic programming is elegant, but it assumes you can query `P(s' | s, a)` for every state and action. Almost nothing in the real world works this way. A robot can't analytically compute the distribution of camera pixels after applying joint torques. A pricing algorithm can't integrate over every possible customer response. An LLM can't enumerate all possible continuations after a token.

You need a method that only requires being able to *sample* from the environment. Run the policy. Get a trajectory `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Use it to estimate values. That's Monte Carlo.

The shift from DP to MC is philosophically important: we move from *known model + exact backups* to *sampled rollouts + averaged returns*. Variance explodes, but applicability expands massively. Every RL algorithm after this lesson — TD, Q-learning, REINFORCE, PPO, GRPO — is essentially a Monte Carlo estimator, sometimes with a layer of bootstrapping on top.

## The Concept

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**Core idea in one line:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`, where `G^{(i)}(s)` is the observed return after visiting `s` under policy `π`.

**First-visit vs every-visit MC.** Given an episode that visits state `s` multiple times, first-visit MC only counts the return after the first visit; every-visit MC counts all visits. Both are unbiased in the limit. First-visit is easier to analyze (iid samples). Every-visit uses more data per episode and often converges faster in practice.

**Incremental mean.** Instead of storing all returns, update a running average:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Rearranged: `V_new = V_old + α · (target - V_old)` where `α = 1/n`. Replace `1/n` with a constant step size `α ∈ (0, 1)` and you get a non-stationary MC estimator that tracks changes in `π`. It's this single step that forms the entire bridge from MC to TD to every modern RL algorithm.

**Exploration is now a problem.** DP touches every state by enumeration. MC only sees states the policy visits. If `π` is deterministic, entire swaths of the state space are never sampled and their value estimates stay at zero forever. Three fixes, in historical order:

1. **Exploring starts.** Start each episode from a random (s, a) pair. Guarantees coverage; impractical in reality (you can't "reset" a robot to an arbitrary state).
2. **ε-greedy.** Act greedily on current Q, but with probability `ε` pick a random action. Asymptotically visits all state-action pairs.
3. **Off-policy MC.** Collect data under behavior policy `μ`, learn target policy `π` via importance sampling. High variance but it's the bridge to replay-buffer methods like DQN.

**Monte Carlo control.** Evaluate → improve → evaluate, same as policy iteration, but evaluation is sample-based:

1. Run `π`, get an episode.
2. Update `Q(s, a)` with observed returns.
3. Make `π` ε-greedy with respect to `Q`.
4. Repeat.

Converges to `Q*` and `π*` with probability 1 under mild conditions (every pair visited infinitely often, `α` satisfies Robbins-Monro).

## Build It

### Step 1: Rollout → (s, a, r) list

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

No model, just `env.reset()` and `env.step(s, a)`. Same interface as a gym environment, just stripped down.

### Step 2: Compute returns (backward sweep)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Single pass, `O(T)`. Backward recursion `G_t = r_{t+1} + γ G_{t+1}` avoids repeated summation.

### Step 3: First-visit MC evaluation

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Three lines do the work: mark state as seen on first visit, increment count, update running mean.

### Step 4: ε-greedy MC control (on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Step 5: Compare against the DP gold standard

As episodes → ∞, your MC estimate of `V^π` should match the DP result from lesson 02. In practice: 50000 episodes on the 4×4 GridWorld gets you within `~0.1` of the DP answer.

## Pitfalls

- **Infinite episodes.** MC requires episodes to *terminate*. If your policy can loop forever, cap `max_steps` and treat the cap as an implicit failure. GridWorld with random policy often times out — that's normal, just make sure you count correctly.
- **Variance.** MC uses full returns. Variance is huge on long episodes — one unlucky reward at the end shifts `V(s_0)` by the same amount. TD methods (lesson 04) cut this via bootstrapping.
- **State coverage.** Greedy MC on a fresh Q with ties only ever tries one action. You *must* explore (ε-greedy, exploring starts, UCB).
- **Non-stationary policy.** If `π` is changing (as in MC control), old returns come from a different policy. Constant-α MC handles this; sample-average MC does not.
- **Off-policy importance sampling.** Weights `π(a|s)/μ(a|s)` multiply along a trajectory. Variance explodes with horizon. Use per-decision weighted IS to cap, or switch to TD.

## Use It

Monte Carlo methods in 2026:

| Use case | Why MC |
|----------|--------|
| Short-horizon games (Blackjack, poker) | Episodes terminate naturally; returns are clean. |
| Offline evaluation of a logged policy | Average discounted returns over stored trajectories. |
| Monte Carlo tree search (AlphaZero) | MC rollouts from tree leaves guide selection. |
| LLM RL evaluation | Compute average reward of sampled completions for a given policy. |
| Baseline estimation in PPO | Advantage target `A_t = G_t - V(s_t)` uses an MC `G_t`. |
| Teaching RL | The simplest algorithm that actually works — strip away bootstrapping to see the core. |

Modern deep RL algorithms (PPO, SAC) interpolate between pure MC (full returns) and pure TD (single-step bootstrapping) via `n`-step returns or GAE. Both endpoints are instances of the same estimator.

## Ship It

Save as `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Exercises

1. **Easy.** Implement first-visit MC evaluation of the uniform random policy on the 4×4 GridWorld. Run 10000 episodes. Plot `V(0,0)` vs episode count and compare against the DP answer.
2. **Medium.** Implement ε-greedy MC control with `ε ∈ {0.01, 0.1, 0.3}`. Compare average returns after 20000 episodes. What do the curves look like? Where does the bias-variance trade-off land?
3. **Hard.** Implement *off-policy* MC with importance sampling: collect data under uniform random policy `μ`, estimate `V^π` for a deterministic optimal policy `π`. Compare naive IS, per-decision IS, and weighted IS. Which has the lowest variance?

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Monte Carlo | "Random sampling" | Estimating expectations by averaging iid samples from a distribution. |
| Return `G_t` | "Future rewards" | Discounted sum of rewards from step `t` to end of episode: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "Count each state once" | Only the first visit in an episode contributes to the value estimate. |
| Every-visit MC | "Use all visits" | Every visit contributes; slightly biased but more sample-efficient. |
| ε-greedy | "Exploration noise" | Select greedy action with probability `1-ε`; random action with probability `ε`. |
| Importance sampling | "Correcting for sampling from the wrong distribution" | Reweight returns by `π(a|s)/μ(a|s)` products to estimate `V^π` from `μ`'s data. |
| On-policy | "Learn from my own data" | Target policy = behavior policy. Vanilla MC, PPO, SARSA. |
| Off-policy | "Learn from someone else's data" | Target policy ≠ behavior policy. IS-based MC, Q-learning, DQN. |

## Further Reading

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Classic treatment.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — First-visit vs every-visit analysis.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — Off-policy MC and variance control.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — Modern low-variance IS estimators.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — First large-scale empirical demonstration of MC/TD self-play converging to superhuman level; conceptual precursor to every lesson in the second half of this phase.
