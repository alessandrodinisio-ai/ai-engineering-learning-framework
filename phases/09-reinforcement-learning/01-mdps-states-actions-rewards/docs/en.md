# MDPs, States, Actions & Rewards

> A Markov Decision Process is five things: states, actions, transitions, rewards, discount. Everything in RL — Q-learning, PPO, DPO, GRPO — optimizes over this structure. Learn it once; the rest of reinforcement learning is free.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Classification)
**Time:** ~45 minutes

## The Problem

You're writing a chess bot. Or an inventory planner. Or a trading agent. Or the PPO loop that trains a reasoning model. Four completely different domains, yet a surprising fact: all four collapse into the same mathematical object.

Supervised learning gives you `(x, y)` pairs and asks you to fit a function. Reinforcement learning gives you no labels — only a stream of states, actions you took, and a scalar reward. Did this move win? Did this restocking decision save money? Did this trade profit? Did the token the LLM just emitted make the judge give a higher reward?

Before you formalize this stream of data, you can't learn anything from it. "What I saw," "what I did," "what happened next," "how good was that" — each must become an object you can reason about. That formalization is the Markov Decision Process. Every RL algorithm in this phase, including the final RLHF and GRPO loops, optimizes over this structure.

## The Concept

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**Five objects.**

- **States** `S`. Everything the agent needs to make a decision. In GridWorld it's a cell, in chess it's the board, in an LLM it's the context window plus any memory.
- **Actions** `A`. Available choices. Move up/down/left/right. Make a chess move. Emit a token.
- **Transitions** `P(s' | s, a)`. The distribution over next states given state `s` and action `a`. Deterministic in chess, stochastic in inventory, near-deterministic in LLM decoding.
- **Rewards** `R(s, a, s')`. A scalar signal. Win = +1, lose = -1. Revenue minus cost. The log-likelihood ratio term in GRPO.
- **Discount** `γ ∈ [0, 1)`. How much future rewards are worth relative to the present. `γ = 0.99` buys ~100 steps of horizon; `γ = 0.9` buys ~10.

**Markov property** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. The future depends only on the current state. If this doesn't hold, the state representation is incomplete — that's not a failure of the method, it's a failure of the state.

**Policy and return.** A policy `π(a | s)` maps states to action distributions. The return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` is the discounted sum of future rewards. The value `V^π(s) = E[G_t | s_t = s]` is the expected return from `s` under policy `π`. The Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` is the expected return from a specific action. Every RL algorithm estimates one of these two, then improves `π` accordingly.

**Bellman equations.** The fixed-point equations everything in this phase relies on:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

These equations decompose expected return into "this step's reward" plus "discounted value of where you land." Recursive. Every algorithm in Phase 9 either iterates this equation to convergence (dynamic programming), samples from it (Monte Carlo), or bootstraps one step forward (temporal difference).

## Build It

### Step 1: A minimal deterministic MDP

A 4×4 GridWorld. Agent starts top-left, goal is bottom-right, reward -1 per step, action set `{up, down, left, right}`. See `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Five lines. That's the entire environment. Deterministic transitions, constant step penalty, absorbing terminal state.

### Step 2: Roll out a policy

A policy is a function from state to action distribution. The simplest: uniform random.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Run the random policy 1000 times. Average return on this 4×4 board is between -60 and -80. Optimal return is -6 (straight diagonal to bottom-right). The entire content of Phase 9 is closing this gap.

### Step 3: Compute `V^π` exactly with the Bellman equation

For small MDPs, the Bellman equation is a linear system. Enumerate states, apply the expectation, iterate until values stop changing.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

This is iterative policy evaluation. It's the first algorithm in Sutton & Barto and the theoretical bedrock of every RL method that follows.

### Step 4: `γ` is a hyperparameter with physical meaning

Effective horizon is roughly `1 / (1 - γ)`. `γ = 0.9` → 10 steps. `γ = 0.99` → 100 steps. `γ = 0.999` → 1000 steps.

Too low and the agent becomes myopic. Too high and credit assignment becomes noisy because many early steps share responsibility for distant rewards. LLM RLHF typically uses `γ = 1` because episodes are short and bounded. Control tasks use `0.95–0.99`. Long-horizon strategy games use `0.999`.

## Pitfalls

- **Non-Markovian states.** If you need the last three observations to decide, then "state" is not just the current observation. Fix: frame stacking (DQN on Atari stacks 4 frames), or recurrent state (LSTM/GRU over observation sequences).
- **Sparse rewards.** Giving reward only on wins makes learning nearly impossible in large state spaces. Shape rewards (give intermediate signals), or use imitation learning to bootstrap (Phase 9 · 09).
- **Reward hacking.** Optimizing proxy rewards often produces pathological behavior. OpenAI's boat-racing agent didn't finish the race — it circled collecting power-ups indefinitely. Always define rewards from target outcomes, not proxies.
- **Discount set wrong.** Using `γ = 1` on an infinite-horizon task makes every value infinite. Always use finite horizon or `γ < 1` to cap.
- **Reward scale.** {+100, -100} and {+1, -1} yield the same optimal policy, but gradient magnitudes differ wildly. Normalize to roughly `[-1, 1]` before feeding into PPO/DQN.

## Use It

The 2026 stack reduces every RL pipeline to an MDP before writing code:

| Scenario | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control (locomotion, manipulation) | Joint angles + velocities | Continuous torques | Task-specific shaping | 0.99 |
| Games (chess, Go, poker) | Board + history | Legal moves | Win=+1 / Lose=-1 | 1.0 (finite) |
| Inventory / pricing | Stock + demand | Order quantities | Revenue - cost | 0.95 |
| LLM RLHF | Context tokens | Next token | Terminal reward-model score | 1.0 (episode ~200 tokens) |
| Reasoning GRPO | Prompt + partial response | Next token | Terminal verifier 0/1 | 1.0 |

Write out the five-tuple before writing any training loop. The vast majority of "RL doesn't work" bug reports trace back to an MDP formulation that was already broken on paper.

## Ship It

Save as `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Exercises

1. **Easy.** Implement the 4×4 GridWorld and random-policy rollout in `code/main.py`. Run 10000 episodes. Report mean and standard deviation of returns. Compare against the optimal return (-6).
2. **Medium.** Run `policy_evaluation` for the uniform random policy with `γ ∈ {0.5, 0.9, 0.99}`. Print `V` as a 4×4 grid for each. Explain why higher `γ` increases state values faster near the goal.
3. **Hard.** Make the GridWorld stochastic: each action has `p = 0.1` probability of slipping to an adjacent direction. Re-evaluate the uniform policy. Does `V[start]` get better or worse? Why?

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| MDP | "The RL setting" | A tuple `(S, A, P, R, γ)` satisfying the Markov property. |
| State | "What the agent sees" | A sufficient statistic for future dynamics under the chosen policy class. |
| Policy | "The agent's behavior" | A conditional distribution `π(a | s)` or deterministic mapping `s → a`. |
| Return | "Total reward" | Discounted sum from current step: `Σ γ^t r_t`. |
| Value | "How good a state is" | Expected return from `s` under `π`. |
| Q-value | "How good an action is" | Expected return from `s` with first action `a`, under `π`. |
| Bellman equation | "The DP recursion" | Fixed-point decomposition of value/Q into one-step reward plus discounted successor value. |
| Discount `γ` | "Future vs present" | Geometric weight on future rewards; effective horizon `~1/(1-γ)`. |

## Further Reading

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — The textbook. Chapter 3 covers MDPs and Bellman equations; Chapter 1 motivates the reward hypothesis that every subsequent lesson builds on.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — The origin of the Bellman equation.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — A concise MDP introduction from a deep RL perspective.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — Operations research reference for MDPs and their exact solution methods.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — The cleanest derivation of MDPs as a special case of dynamic programming.
