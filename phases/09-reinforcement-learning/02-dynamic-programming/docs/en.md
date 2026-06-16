# Dynamic Programming — Policy Iteration & Value Iteration

> Dynamic programming is RL with cheat codes. You already know the transition function and the reward function; all you do is iterate the Bellman equation until `V` or `π` stops changing. It's the baseline every sampling-based method tries to approximate.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDP)
**Time:** ~75 minutes

## The Problem

You have a model-known MDP: for any state-action pair, you can query `P(s' | s, a)` and `R(s, a, s')`. The inventory planner knows the demand distribution. Chess transitions are deterministic. A gridworld is four lines of Python. You have a *model*.

Model-free RL (Q-learning, PPO, REINFORCE) was invented for when you don't have a model — you can only sample from the environment. But when you do have a model, there are faster, better methods: dynamic programming. Bellman designed them in 1957. To this day, they define "correct": when people say "the optimal policy for this MDP," they mean the one DP would return.

By 2026, you need them for three reasons. First, every tabular environment in RL research (GridWorld, FrozenLake, CliffWalking) is solved with DP to produce the gold-standard policy. Second, exact values let you *debug* sampling methods: if Q-learning's estimate of `V*(s_0)` is 30% off the DP answer, your Q-learning has a bug. Third, modern offline RL and planning methods (MCTS, AlphaZero's search, Phase 9 · 10's model-based RL) iterate Bellman backups on a learned or given model.

## The Concept

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**Two algorithms, both fixed-point iterations on the Bellman equation.**

**Policy iteration.** Alternate two steps until the policy stops changing.

1. *Evaluation:* Given policy `π`, repeatedly apply `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` until convergence to compute `V^π`.
2. *Improvement:* Given `V^π`, make `π` greedy with respect to `V^π`: `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

Convergence is guaranteed because (a) each improvement step either leaves `π` unchanged or strictly raises `V^π` for some state; (b) the space of deterministic policies is finite. Typically converges in ~5–20 outer iterations even for large state spaces.

**Value iteration.** Collapse evaluation and improvement into a single sweep. Apply the Bellman *optimality* equation:

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

Repeat until `max_s |V_{new}(s) - V(s)| < ε`. Extract the policy by taking greedy actions at the end. Each iteration is strictly faster — no inner evaluation loop — but typically requires more iterations to converge.

**Generalized Policy Iteration (GPI).** The unifying framework. Value function and policy are locked in a mutual improvement loop; any method that drives both toward mutual consistency (asynchronous value iteration, modified policy iteration, Q-learning, actor-critic, PPO) is an instance of GPI.

**Why `γ < 1` matters.** The Bellman operator is a `γ`-contraction in the sup-norm: `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. Contraction means unique fixed point and geometric convergence. Remove `γ < 1` and you lose this guarantee — you need finite horizon or an absorbing terminal state.

## Build It

### Step 1: Build the MDP model for GridWorld

Reuse the 4×4 GridWorld from lesson 01. We add a stochastic variant: the agent has `0.1` probability of slipping to a random perpendicular direction.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` returns a list of `(s', r, p)`. That's the entire model.

### Step 2: Policy evaluation

Given a policy `π(s) = {action: prob}`, iterate the Bellman equation until `V` stops changing:

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### Step 3: Policy improvement

Replace `π` with the greedy policy with respect to `V`. If `π` hasn't changed, return — we've reached optimality.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### Step 4: Put them together

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

Typical convergence on the 4×4: 4–6 outer iterations. Output `V*(0,0) ≈ -6` and a policy that strictly minimizes steps.

### Step 5: Value iteration (single-loop version)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

Same fixed point, fewer lines of code.

## Pitfalls

- **Forgetting terminal states.** If you apply Bellman to an absorbing state, it still picks a "best action" that changes nothing. Guard with `if s == terminal: V[s] = 0`.
- **Sup-norm vs L2 convergence.** Use `max |V_new - V|`, not the average. Theoretical guarantees are on the sup-norm.
- **In-place vs synchronous updates.** Updating `V[s]` in-place (Gauss-Seidel) converges faster than using a separate `V_new` dict (Jacobi). Production code uses in-place.
- **Policy ties.** If two actions have equal Q-values, `argmax` may break ties differently each iteration, causing the "policy stable" check to oscillate. Use a stable tie-breaking rule (first action in a fixed order).
- **State-space explosion.** DP sweeps are `O(|S| · |A|)`. Can handle up to ~10⁷ states. Beyond that, you need function approximation (from Phase 9 · 05 onward).

## Use It

By 2026, DP is the correctness baseline and the inner loop of planners:

| Use case | Method |
|----------|--------|
| Solve a small tabular MDP exactly | Value iteration (simpler) or policy iteration (fewer outer steps) |
| Validate a Q-learning / PPO implementation | Compare against DP's optimal V* on toy environments |
| Model-based RL (Phase 9 · 10) | Bellman backups on a learned transition model |
| Planning in AlphaZero / MuZero | Monte Carlo tree search = asynchronous Bellman backups |
| Offline RL (CQL, IQL) | Conservative Q-iteration — DP with penalties on OOD actions |

Whenever someone says "optimal value function," they mean "the fixed point of DP." When you see `V*` or `Q*` in a paper, this loop is what should come to mind.

## Ship It

Save as `outputs/skill-dp-solver.md`:

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## Exercises

1. **Easy.** Run value iteration on the 4×4 GridWorld with `γ ∈ {0.9, 0.99}`. How many sweeps to reach `max |ΔV| < 1e-6`? Print `V*` as a 4×4 grid.
2. **Medium.** Compare policy iteration and value iteration on the *stochastic* GridWorld (slip probability `0.1`). Report: number of sweeps, wall-clock time, final `V*(0,0)`. Which converges faster in iteration count? In wall-clock time?
3. **Hard.** Implement modified policy iteration: run only `k` sweeps of evaluation instead of running to convergence. Plot `V*(0,0)` error vs `k` for `k ∈ {1, 2, 5, 10, 50}`. What does the curve tell you about the evaluation/improvement trade-off?

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Policy iteration | "The DP algorithm" | Alternate evaluation (`V^π`) and improvement (greedy `π` w.r.t. `V^π`) until policy stops changing. |
| Value iteration | "Faster DP" | Apply the Bellman optimality backup in a single sweep; geometric convergence to `V*`. |
| Bellman operator | "The recursion" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; a `γ`-contraction in sup-norm. |
| Contraction | "Why DP converges" | Any operator `T` with `||T x - T y|| ≤ γ ||x - y||` has a unique fixed point. |
| GPI | "Everything is DP" | Generalized policy iteration: any method driving `V` and `π` toward mutual consistency. |
| Synchronous update | "Jacobi-style" | Entire sweep uses old `V`; cleaner to analyze but slower. |
| In-place update | "Gauss-Seidel-style" | Updates `V` as you go; converges faster in practice. |

## Further Reading

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — Classic presentation of policy iteration and value iteration.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — Rigorous treatment of the contraction mapping argument.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — Modified policy iteration and its convergence analysis.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — The original policy iteration paper.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — The bridge from DP to approximate DP / deep RL, used in every subsequent lesson.
