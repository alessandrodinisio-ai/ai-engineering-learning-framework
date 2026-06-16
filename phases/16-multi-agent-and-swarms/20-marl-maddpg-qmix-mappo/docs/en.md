# MARL — MADDPG, QMIX, MAPPO

> The reinforcement learning bloodline of multi-agent coordination, still influencing LLM-agent systems in 2026. **MADDPG** (Lowe et al., NeurIPS 2017, arXiv:1706.02275) introduced Centralized Training, Decentralized Execution (CTDE): each critic sees all agents' states and actions during training; only local actors run at test time. Works for cooperative, competitive, and mixed scenarios. **QMIX** (Rashid et al., ICML 2018, arXiv:1803.11485) is value decomposition with a monotonic mixing network; each agent's Q combines into a joint Q so `argmax` decomposes cleanly—dominates StarCraft Multi-Agent Challenge (SMAC). **MAPPO** (Yu et al., NeurIPS 2022, arXiv:2103.01955) is PPO with a centralized value function; "surprisingly effective" on particle-world, SMAC, Google Research Football, and Hanabi with nearly zero hyperparameter tuning. These underpin policy training for "teams of agents that must act decentrally." MAPPO is the **2026 default cooperative MARL baseline**. This lesson builds each from a small grid-world toy so all three ideas are in muscle memory before you touch LLM-agent training.

**Type:** Learn
**Languages:** Python (standard library, small implementations without NumPy)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 minutes

## The Problem

LLM-agent systems increasingly train policies for inter-agent coordination: when to defer, when to act, which peer to call. The literature that tells you how to train such policies is Multi-Agent Reinforcement Learning (MARL), which predates the LLM wave with a small set of dominant algorithms.

Reading MARL papers without this pattern vocabulary is painful. Centralized Training Decentralized Execution (CTDE), value decomposition, centralized critics are not buzzwords—they are specific answers to specific problems:

- Independent RL (each agent learns alone) is non-stationary from each agent's perspective. Bad.
- Centralized RL (one agent controls all) doesn't scale and violates execution constraints.
- CTDE takes the best of both: train with global information, deploy with local policies.

## The Concept

### Three environments the papers use

- **Particle World (Multi-Agent Particle Environment).** Simple 2D physics with cooperative/competitive tasks. MADDPG's original testbed.
- **StarCraft Multi-Agent Challenge (SMAC).** Cooperative micromanagement, partially observable. QMIX's testbed. Discrete actions, continuous state.
- **Google Research Football, Hanabi, MPE.** MAPPO's baselines.

Different environments have different action/observation types. Algorithms are chosen accordingly.

### MADDPG (2017) — the CTDE pattern

Each agent `i` has an actor `mu_i(o_i)` mapping its own observation to actions. Each agent also has a critic `Q_i(x, a_1, ..., a_n)` that sees all observations and all actions during training. The actor is updated using a policy gradient evaluated against the critic.

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimates
```

Why CTDE: during training we know everyone's actions; use that to reduce variance for each critic. At deployment, each agent sees only `o_i` and calls `mu_i(o_i)`.

Failure mode: critic grows with agent count N (input includes all actions). Doesn't scale past ~10 agents without approximations.

### QMIX (2018) — value decomposition

Cooperative only. The global reward is a monotonic function of individual agent Q values:

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

Monotonicity guarantees that `argmax_a Q_tot` can be computed by each agent independently choosing `argmax_{a_i} Q_i`. This is **exactly** the decentralized execution property you need. During training, a mixing network produces `Q_tot` from individual agent Qs.

Why QMIX wins on SMAC: cooperative StarCraft micromanagement has homogeneous agents, local observations, global reward—perfect fit for value decomposition.

Failure mode: monotonicity constraint is restrictive; some tasks have reward structures that aren't monotonically decomposable (one agent sacrificing for the team). Extensions (QTRAN, QPLEX) relax this.

### MAPPO (2022) — the overlooked default

Multi-Agent PPO: PPO with a centralized value function. Each agent has its own policy; all share (or each has) a value function that sees full state. Yu et al. 2022 compared MAPPO against MADDPG, QMIX, and their extensions across five benchmarks and found:

- MAPPO matches or beats off-policy MARL methods on particle-world, SMAC, Google Research Football, Hanabi, MPE.
- Nearly zero hyperparameter tuning needed.
- Training is stable; reproducible across seeds.

Before this paper, the community underestimated on-policy MARL. In 2026, MAPPO is the default cooperative MARL baseline; any new method must beat it.

### Why LLM-agent engineers should care

Three direct uses:

1. **Router training.** A meta-agent picks which sub-agent handles a task. This is a MARL problem: N decentralized sub-agents plus a centralized router. MAPPO fits.
2. **Role emergence.** In generative-agent simulations, training agents to adopt complementary roles over time is a disguised MARL problem. QMIX-style value decomposition forces complementarity by construction.
3. **Multi-agent tool use.** When agents share tools and compete for budget, training them with CTDE produces deployable local policies that respect resource constraints.

Practical caveat: in 2026, most production LLM-agent systems set their policies via prompts rather than training. MARL enters when you have (a) extensive interaction data, (b) clear reward signals, and (c) willingness to invest in training infrastructure.

### CTDE as a design pattern beyond RL

Even without training, CTDE is a useful architectural pattern:

- At *design* time, assume full-team visibility.
- At *runtime*, enforce decentralized execution: each agent sees only `o_i`.

This pattern forces you to keep each agent's state explicit and think about partial observability from the start. Many production multi-agent systems silently assume shared state everywhere—CTDE discipline prevents that.

### The non-stationarity problem

When multiple agents learn simultaneously, each agent's environment (which includes everyone else's policies) is non-stationary. Classical single-agent RL proofs collapse. This lesson's MARL algorithms all address this:

- MADDPG: global critic sees all actions, so its value estimate is stationary.
- QMIX: value decomposition moves learning into a joint Q space where optimality is well-defined.
- MAPPO: centralized value function dampens variance from others' policy changes.

In LLM-agent systems, non-stationarity manifests as "my agent worked fine last month, now the upstream agent changed and mine breaks." CTDE-trained MARL is the principled fix; prompt-level fixes are faster but less durable.

### What this lesson does not cover

Training real networks is Phase 09 material. This lesson builds scripted-policy versions that demonstrate the CTDE, value decomposition, and centralized value patterns without gradient updates. The goal is to internalize the patterns before you pick up a full MARL library (PyMARL, MARLlib, RLlib multi-agent).

## Build It

`code/main.py` implements three pattern demos, all on a tiny 2-agent cooperative grid world:

- Environment: 2 agents on a 4x4 grid, one reward pellet. Reward = 1 if either agent reaches the pellet; episode ends.
- `IndependentAgents` — each agent treats the other as environment. Baseline.
- `MADDPGStyle` — centralized critic computes a joint value; actor policies update accordingly. Scripted policy improvement.
- `QMIXStyle` — value decomposition with monotonic mixer.
- `MAPPOStyle` — centralized value function; policies update against shared baseline.

All four run the same episodes and report average steps to reach goal. CTDE variants converge to shorter paths than the independent baseline.

Run:

```
python3 code/main.py
```

Expected output: independent agents average ~6 steps; CTDE variants converge to ~3.5 steps (optimal for 4x4 grid is 3). Despite scripted policies, the pattern differences are visible.

## Use It

`outputs/skill-marl-picker.md` is a skill that picks a MARL algorithm for a given multi-agent task: cooperative vs competitive, homogeneous vs heterogeneous, action space type, scale, reward signal.

## Ship It

MARL is rare in production. When you do use it:

- **Start with MAPPO.** The 2022 paper established it as baseline; reproducing it first saves weeks chasing fancy methods.
- **Log per-agent observation and action streams.** Debugging MARL without per-agent traces is hopeless.
- **Separate training code from execution code.** CTDE is a discipline; make the execution path truly see only `o_i`.
- **Reward shaping caveat.** MARL is extremely sensitive to reward design. One coordination bug in shaping and agents learn to exploit it. Run adversarial tests.
- **For LLM agents**, consider prompt-level policies first. Invest in MARL training only when interaction data + reward signal + infrastructure are all in place.

## Exercises

1. Run `code/main.py`. Measure the steps-to-goal gap between independent and MAPPO-style agents. Does the gap grow or shrink on a 6x6 grid?
2. Implement a competitive variant: two agents, one pellet, only the first to arrive gets reward. Which pattern handles competition cleanly? Historically it's MADDPG.
3. Read MADDPG (arXiv:1706.02275) Section 3. Implement the exact critic update rule in symbolic pseudocode in your own words.
4. Read MAPPO (arXiv:2103.01955). Why do the authors argue "centralized value + PPO" beats off-policy MARL on their benchmarks? List three strongest claims.
5. Apply CTDE as a design pattern to a hypothetical LLM-agent system (e.g., research agent + summarizer + coder). What is the joint information available at design time but unavailable at runtime?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MARL | "Multi-agent RL" | Reinforcement learning for multi-agent systems. |
| CTDE | "Centralized Training, Decentralized Execution" | Train with global info; deploy with local policies. |
| MADDPG | "Multi-agent DDPG" | CTDE with "each agent's critic sees all observations + actions." |
| QMIX | "Value decomposition" | Monotonic mixing of individual agent Qs. Cooperative. |
| MAPPO | "Multi-agent PPO" | PPO with centralized value function. The 2026 default baseline. |
| Value decomposition | "Sum of individual Qs" | Joint Q represented as a monotonic function of agent Qs. |
| Non-stationarity | "Moving target" | Each agent's environment changes as others learn. The core MARL problem. |
| On-policy / off-policy | "Learn from current / learn from replay" | PPO is on-policy (MAPPO); DDPG and Q-learning are off-policy. |
| SMAC | "StarCraft Multi-Agent Challenge" | Cooperative micromanagement benchmark; QMIX's home turf. |

## Further Reading

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG; NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX; ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO; NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — Accessible summary of MAPPO results
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
