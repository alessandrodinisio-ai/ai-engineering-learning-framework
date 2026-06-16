# Temporal Difference — Q-Learning & SARSA

> Monte Carlo waits until the episode ends to update. TD bootstraps from the next step's value estimate after every single transition. Q-learning is off-policy, optimistic; SARSA is on-policy, cautious. Both are one line of code. Both are the foundation of every deep RL method in this phase.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDP), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 minutes

## The Problem

Monte Carlo works, but it has two expensive requirements. It needs episodes that terminate, and it only updates after receiving the final return. If your episode has 1000 steps, MC waits 1000 steps before updating anything. It's high-variance, low-bias, and slow in practice.

Dynamic programming's profile is the exact opposite — zero-variance bootstrapped backups — but requires a known model.

Temporal difference (TD) learning takes the middle ground. From a single transition `(s, a, r, s')`, construct a one-step target `r + γ V(s')` and nudge `V(s)` toward it. No model needed. No complete episode needed. Using the approximate `V` on the right-hand side introduces bias, but variance is far lower than MC, and updates happen online from the very first step.

This is the pivot on which all modern RL — DQN, A2C, PPO, SAC — turns. The rest of Phase 9 is layers of function approximation and tricks built on top of the single-step TD update you'll write in this lesson.

## The Concept

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**TD(0) update for V:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

The bracketed quantity is the TD error `δ = r + γ V(s') - V(s)`. It's the online version of MC's `G_t - V(s_t)`. Convergence requires `α` to satisfy Robbins-Monro (`Σ α = ∞`, `Σ α² < ∞`) and all states visited infinitely often.

**Q-learning.** An off-policy TD method for control:

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

The `max` assumes the *greedy* policy will be followed from `s'` onward, regardless of what the agent actually does. This decoupling lets Q-learning learn `Q*` while the agent explores with ε-greedy. Mnih et al. (2015) turned this into deep Q-learning on Atari (lesson 05).

**SARSA.** An on-policy TD method:

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

The name is the tuple `(s, a, r, s', a')`. SARSA uses the action `a'` the agent *actually* takes next, not the greedy `argmax`. It converges to `Q^π` for the running ε-greedy `π`, which becomes `Q*` in the limit as `ε → 0`.

**The cliff-walking difference.** On the classic cliff-walking task (falling off the cliff = reward -100), Q-learning learns the optimal path along the cliff edge but occasionally eats the penalty during exploration. SARSA learns a safer path one step away from the cliff because it accounts for exploration noise in its Q-values. Both reach optimality as `ε → 0` during training. But in practice this matters: SARSA behaves more conservatively when deployment does involve exploration.

**Expected SARSA.** Replace `Q(s', a')` with its expectation under `π`:

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

Lower variance than SARSA (doesn't sample `a'`), same on-policy target. Often the default in modern textbooks.

**n-step TD and TD(λ).** Interpolate between TD(0) and MC by waiting `n` steps before bootstrapping. `n=1` is TD, `n=∞` is MC. TD(λ) averages over all `n` with geometric weights `(1-λ)λ^{n-1}`. Most deep RL uses `n` between 3 and 20.

## Build It

### Step 1: SARSA on an ε-greedy policy

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

Eight lines. The *only* difference from Q-learning is the target line.

### Step 2: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

The `max` decouples target from behavior. That single symbol is the entire difference between on-policy and off-policy.

### Step 3: Learning curves

Track average return every 100 episodes. Q-learning converges faster on simple deterministic GridWorld; SARSA is more conservative on cliff-walking. On the 4×4 GridWorld in `code/main.py`, with `α=0.1, ε=0.1`, both approach optimality after ~2000 episodes.

### Step 4: Compare against DP ground truth

Run value iteration (lesson 02) for `Q*`. Check `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`. A healthy tabular TD agent gets within `~0.5` after 10000 episodes on the 4×4 GridWorld.

## Pitfalls

- **Initial Q values matter.** Optimistic initialization (setting `Q = 0` for a negative-reward task) encourages exploration. Pessimistic initialization can trap the greedy policy forever.
- **α schedule.** Constant `α` is fine for non-stationary problems. Decaying `α_n = 1/n` converges in theory but is too slow in practice — pin `α` in `[0.05, 0.3]` and watch the learning curve.
- **ε schedule.** Start high (`ε=1.0`), decay to `ε=0.05`. "GLIE" (greedy in the limit with infinite exploration) is the convergence condition.
- **Q-learning's max bias.** When `Q` is noisy, the `max` operator biases upward. Causes overestimation — Hasselt's Double Q-learning (used in DDQN in lesson 05) fixes it with two Q-tables.
- **Non-terminating episodes.** TD can learn without terminal states, but you must either cap steps or correctly handle bootstrapping at the cap. Standard practice: treat the cap as non-terminal and continue bootstrapping.
- **State hashing.** If states are tuples/tensors, use a hashable key (tuples not lists; rounded float tuples, not raw values).

## Use It

The TD landscape in 2026:

| Task | Method | Reason |
|------|--------|--------|
| Small tabular environments | Q-learning | Learns optimal policy directly. |
| On-policy, safety-critical | SARSA / Expected SARSA | Conservative during exploration. |
| High-dimensional states | DQN (Phase 9 · 05) | Neural Q-function with replay and target net. |
| Continuous actions | SAC / TD3 (Phase 9 · 07) | TD updates on Q-networks; policy network outputs actions. |
| LLM RL (reward model-based) | PPO / GRPO (Phase 9 · 08, 12) | Actor-critic with TD-style advantage via GAE. |
| Offline RL | CQL / IQL (Phase 9 · 08) | Q-learning with conservative regularization. |

90% of "RL" you read in 2026 papers is some refined version of Q-learning or SARSA. Get the tabular update under your fingers before going deeper.

## Ship It

Save as `outputs/skill-td-agent.md`:

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## Exercises

1. **Easy.** Implement Q-learning and SARSA on the 4×4 GridWorld. Plot learning curves (average return per 100 episodes) over 2000 episodes. Which converges faster?
2. **Medium.** Build a cliff-walking environment (4×12, bottom row is cliff, reward -100 and reset to start). Compare final policies of Q-learning and SARSA. Screenshot the paths each takes. Which is closer to the cliff?
3. **Hard.** Implement Double Q-learning. On a noisy-reward GridWorld (Gaussian noise σ=5 on each step reward), show that Q-learning overestimates `V*(0,0)` by a notable margin while Double Q-learning does not.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| TD error | "The update signal" | `δ = r + γ V(s') - V(s)`, the residual after bootstrapping. |
| TD(0) | "Single-step TD" | Update using only the next state's estimate after each transition. |
| Q-learning | "Off-policy RL 101" | TD update that takes `max` over next-state actions; learns `Q*` regardless of behavior policy. |
| SARSA | "On-policy Q-learning" | TD update using actual next action; learns `Q^π` for current ε-greedy π. |
| Expected SARSA | "Lower-variance SARSA" | Replaces sampled `a'` with its expectation under π. |
| GLIE | "The right exploration schedule" | Greedy in the limit with infinite exploration; needed for Q-learning convergence. |
| Bootstrapping | "Using current estimates in the target" | What distinguishes TD from MC. Source of bias but massive variance reduction. |
| Maximization bias | "Q-learning overestimates" | Taking `max` of noisy estimates biases upward; fixed by Double Q-learning. |

## Further Reading

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — Original paper and convergence proof.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — The fix for maximization bias.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Motivation for Expected SARSA.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — The paper that introduced SARSA (then called "modified connectionist Q-learning").
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — Generalizes TD(0) to TD(n), the path from Q-learning to eligibility traces to GAE in PPO.
