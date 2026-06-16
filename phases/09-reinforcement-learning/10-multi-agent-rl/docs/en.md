# Multi-Agent RL

> Single-agent RL assumes the environment is stationary. Put two learning agents in the same world and that assumption breaks: each agent is part of the other's environment, and both are changing. Multi-agent RL is the set of tricks that make learning converge anyway when the Markov assumption no longer holds.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 minutes

## The Problem

A robot learning to navigate a room is a single-agent RL problem. A soccer team is not. AlphaStar playing against StarCraft opponents is not. A bidding agent marketplace is not. Two cars gaming a four-way stop is not. Real-world multi-vs-multi problems are not.

In every multi-agent scenario, from any one agent's perspective, the other agents *are* part of the environment. When they learn and change behavior, the environment becomes non-stationary. Markovianity — "the next state depends only on the current state and my action" — is violated because the next state also depends on what the *other* agents chose, and their policies are moving targets.

This breaks tabular convergence proofs (Q-learning's guarantees assume a stationary environment). It also breaks naive deep RL: agents chase each other in circles, never converging to stable policies. You need multi-agent-specific techniques: centralized training / decentralized execution, counterfactual baselines, league play, self-play.

2026 applications: robot swarms, traffic routing, autonomous vehicle fleets, market simulators, multi-agent LLM systems (Phase 16), and any game with more than one intelligent player.

## The Concept

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**Formalism: Markov Games.** Generalization of MDP: states `S`, joint actions `a = (a_1, …, a_n)`, transition `P(s' | s, a)`, and per-agent rewards `R_i(s, a, s')`. Each agent `i` maximizes its own return under its own policy `π_i`. If rewards are identical, it's **fully cooperative**. If zero-sum, it's **adversarial**. If mixed, it's **general-sum**.

**Core challenges:**

- **Non-stationarity.** From agent `i`'s view, `P(s' | s, a_i)` depends on `π_{-i}`, which is changing.
- **Credit assignment.** When reward is shared, which agent caused it?
- **Exploration coordination.** Agents must explore complementary strategies, not redundantly explore the same states.
- **Scalability.** Joint action space grows exponentially with `n`.
- **Partial observability.** Each agent sees only its own observation; the global state is hidden.

**Four dominant paradigms:**

**1. Independent Q-learning / Independent PPO (IQL, IPPO).** Each agent learns its own Q or policy, treating others as part of the environment. Simple, sometimes works (especially when experience replay acts as a smoothing agent-modeling trick). Theoretical convergence: none. In practice: fine for loosely-coupled tasks, terrible for tightly-coupled ones.

**2. Centralized Training, Decentralized Execution (CTDE).** The most common modern paradigm. Each agent has its own *policy* `π_i` conditioned on local observation `o_i` — standard decentralized execution at deploy time. During *training*, a centralized critic `Q(s, a_1, …, a_n)` conditions on the full global state and joint actions. Examples:
- **MADDPG** (Lowe et al. 2017): DDPG with a centralized critic per agent.
- **COMA** (Foerster et al. 2017): Counterfactual baseline — ask "what would my reward be if I had taken action `a'` instead?" — isolates my contribution.
- **MAPPO** / **IPPO** with shared critic (Yu et al. 2022): PPO with a centralized value function. Dominates cooperative MARL in 2026.
- **QMIX** (Rashid et al. 2018): Value decomposition — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` with monotonic mixing.

**3. Self-play.** Two copies of the same agent play against each other. The opponent's policy *is* my past snapshot's policy. AlphaGo / AlphaZero / MuZero. OpenAI Five. Works best for zero-sum games; the training signal is symmetric.

**4. League play.** Extension of self-play for general-sum / adversarial environments: maintain a population of past and current policies, sample an opponent from the league, train against it. Add exploiters (specialized at beating the current best) and main exploiters (specialized at beating exploiters). AlphaStar (StarCraft II). Needed when the game has "rock-paper-scissors" style strategy cycles.

**Communication.** Allow agents to send learned messages `m_i` to each other. Effective in cooperative settings. Foerster et al. (2016) showed differentiable inter-agent communication can be trained end-to-end. Today's LLM-based multi-agent systems (Phase 16) are essentially communicating with natural language.

## Build It

This lesson uses a 6×6 GridWorld with two cooperative agents. They start from opposite corners and must reach a shared goal. Shared reward: `-1` per step while either agent is still moving, `+10` when both reach the goal. See `code/main.py`.

### Step 1: Multi-agent environment

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

The *joint* action space is `|A|² = 16`. The global state is both positions.

### Step 2: Independent Q-learning

Each agent runs its own Q-table keyed on the joint state. Each step: both pick ε-greedy actions, collect the joint transition, each updates its own Q with the shared reward.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Works on this task because the reward is dense and aligned. Fails on tightly-coupled tasks (e.g., where one agent must *wait* for the other).

### Step 3: Centralized Q with factored value update

Use a single Q over joint actions `Q(s, a_1, a_2)`. Update with the shared reward. Decentralize at execution via marginalization: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Trades exponential joint action space for a *correct* global view.

### Step 4: Simple self-play (adversarial 2-agent)

Same agent, two roles. Train agent A against agent B; after `K` episodes, copy A's weights into B. Symmetric training, consistent progress. Miniature version of the AlphaZero recipe.

## Pitfalls

- **Non-stationary replay.** Independent agents with experience replay are worse than single-agent because old transitions were generated by now-outdated opponents. Fix: re-label, or weight by recency.
- **Credit assignment ambiguity.** Shared reward after a long episode; no clear way to say which agent contributed. Fix: counterfactual baselines (COMA), or reward shaping per agent.
- **Policy drift / mutual chasing.** Each agent's best response changes as the other updates. Fix: centralized critic, slow learning rates, or freeze one at a time.
- **Reward hacking by coordination.** Agents find coordinated exploits the designer didn't anticipate. Auction agents converge to zero bids. Fix: careful reward design, behavior constraints.
- **Exploration redundancy.** Two agents exploring the same state-action pairs. Fix: per-agent entropy bonuses, or role conditioning.
- **League cycles.** Pure self-play may get stuck in a dominance cycle. Fix: league play with diverse opponents.
- **Sample explosion.** `n` agents × state space × joint actions. Use function approximation to approximate; factorize action spaces (one policy output head per agent).

## Use It

2026 MARL application map:

| Domain | Method | Notes |
|--------|--------|-------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE; shared critic + decentralized actors. |
| Two-player zero-sum board games (chess, Go, poker) | Self-play with MCTS (AlphaZero) | Zero-sum; symmetric training. |
| Complex multiplayer (Dota, StarCraft) | League play + imitation pretraining | OpenAI Five, AlphaStar. |
| Autonomous vehicle fleets | CTDE MAPPO / PPO with attention | Partial observability; variable team size. |
| Auction markets | Game-theoretic equilibria + RL | Mean-field RL when `n` → ∞. |
| LLM multi-agent systems (Phase 16) | Natural-language communication + role conditioning | RL loop is at the agent-planning layer. |

By 2026, the fastest-growing MARL domain is LLM-based: swarms of language model agents negotiating, debating, building software. RL manifests as preference optimization over *trajectory-level* outputs (not token-level) (Phase 16 · 03).

## Ship It

Save as `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Exercises

1. **Easy.** Train independent Q-learning on the 2-agent cooperative GridWorld. After how many episodes does mean return exceed 0? Plot the joint learning curve.
2. **Medium.** Add a "coordination" task: the goal is only reached if both agents step on it on the same turn. Can independent Q still converge? Where does it break?
3. **Hard.** Implement a centralized critic for MAPPO-style training and compare convergence speed against independent PPO on the coordination task.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| Markov Game | "multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; each agent has its own reward. |
| CTDE | "centralized training, decentralized execution" | Joint critic at train time; each agent's policy uses local obs only. |
| IPPO | "independent PPO" | Each agent runs its own PPO. Simple baseline; often underestimated. |
| MAPPO | "multi-agent PPO" | PPO with a centralized value function conditioned on global state. |
| QMIX | "monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)`, allows decentralized argmax. |
| COMA | "counterfactual multi-agent" | Advantage = my Q minus expected Q marginalized over my action. |
| Self-play | "agent vs past self" | Single agent, two roles; standard for zero-sum games. |
| League play | "population training" | Cache past policies, sample opponents from the pool; handles strategy cycles. |

## Further Reading

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE with centralized critic.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — Counterfactual baseline for credit assignment.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — Value decomposition with monotonicity.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO is surprisingly strong in MARL.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — Large-scale league play.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — Pure self-play in zero-sum games.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — Contains the textbook's brief treatment of multi-agent settings and the non-stationarity problem CTDE solves.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — Survey covering cooperative, competitive, and mixed MARL with convergence results.
