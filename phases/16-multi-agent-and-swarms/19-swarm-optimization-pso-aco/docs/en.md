# Swarm Optimization for LLMs (PSO, ACO)

> Bio-inspired optimization is making a comeback on LLMs. **LMPSO** (arXiv:2504.09247) uses PSO where each particle's velocity is a prompt and the LLM generates the next candidate; works well on structured sequential outputs (math expressions, programs). **Model Swarms** (arXiv:2410.11163) treats each LLM expert as a PSO particle on a model-weight manifold, reporting **13.3% average improvement** over 12 baselines across 9 datasets using only 200 instances. **SwarmPrompt** (ICAART 2025) hybridizes PSO + Grey Wolf for prompt optimization. **AMRO-S** (arXiv:2603.12933) is an ACO-inspired pheromone-expert approach to multi-agent LLM routing—**4.7x speedup**, interpretable routing evidence, and quality-gated asynchronous updates that decouple inference from learning. This lesson implements PSO on prompt parameter space and ACO on agent routing, measuring why these classical algorithms fit the LLM era and when they don't.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus & BFT)
**Time:** ~75 minutes

## The Problem

You have a prompt scoring 62% on a task eval. You want to improve it. The naive approach is gradient-free manual tweaking, which doesn't scale. RL needs a reward signal and enough rollouts to train. Backpropagating through a prompt isn't actually possible—a prompt is a discrete string, not a differentiable parameter.

Classical bio-inspired optimization—PSO for continuous search spaces, ACO for path selection—is designed exactly for this scenario: gradient-free, population-based, cheap per evaluation. Pair them with LLMs for gradient-free search steps and you get a surprisingly practical optimizer.

The same pattern applies to agent *routing* in multi-agent systems. An ACO-style pheromone trail records which agent performs best on which task type, the router exploits that trail, and pheromone decays so routes can be rediscovered.

## The Concept

### PSO refresher (Kennedy & Eberhart 1995)

Particle Swarm Optimization: a swarm of particles in a continuous search space. Each particle has position `x_i` and velocity `v_i`. Each iteration:

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

where `p_best` is the particle's own best, `g_best` is the swarm's best, `w, c1, c2` are inertia + cognitive + social weights, and `r1, r2` are random factors.

### PSO on LLM outputs — LMPSO

arXiv:2504.09247 adapts PSO to LLM-generated structured outputs (math expressions, programs). Each particle is a candidate output. The velocity is a *prompt describing how to modify the current output toward the individual/global best*. The LLM generates the new output from the velocity prompt. The "inertia" of the velocity is a prompt like "make small incremental changes."

It works well when:
- Outputs are structured (parseable, evaluable).
- Fitness is automatic (run tests, arithmetic evaluation).
- Population is small (~10-30 particles), keeping total LLM calls manageable.

It works poorly when fitness requires human review—the cost per iteration becomes prohibitive.

### Model Swarms

arXiv:2410.11163 moves PSO from the output layer into the *model* layer. Each "particle" is an expert LLM (parameters). The swarm moves parameters toward collective optima via gradient-free updates. Reports: 13.3% average improvement over 12 baselines across 9 datasets, only 200 instances per iteration.

The key insight is that LLM expert models are already close to each other on a shared parameter manifold (adapter weights, LoRA deltas). PSO on this low-dimensional subspace is both cheap and effective.

### ACO refresher (Dorigo 1992)

Ant Colony Optimization: ants traverse a graph; each path has a pheromone trail. Ants' movement probabilities are weighted by pheromone strength. Ants that complete a task deposit pheromone proportional to solution quality. Pheromone decays over time.

### AMRO-S — ACO for agent routing

arXiv:2603.12933 uses ACO for multi-agent routing. Each task type is a "destination"; each agent is a possible route. Routes that produce good outputs get pheromone reinforced. Key contributions:

- **Interpretable routing evidence.** Pheromone strength is a human-readable signal.
- **Quality-gated asynchronous updates.** Pheromone is only updated after quality checks pass, decoupling inference from learning.
- **4.7x speedup** on multi-agent routing benchmarks.

The quality gate matters: without it, "fast but wrong" agents accumulate pheromone and the system locks onto bad routes.

### When to use PSO / ACO for LLMs

**Use PSO when:**
- The search space is continuous, or maps to continuous parameters (prompt embeddings, LoRA weights, numeric generation parameters).
- Fitness is cheap and automatic.
- Population can stay small (10-30).

**Use ACO when:**
- You have a routing or path-selection problem.
- Decisions reinforce over time (the same task types recur).
- You need interpretable evidence for routing decisions.

**Use neither when:**
- Fitness requires human review (too expensive per iteration).
- The search space is discrete and combinatorial in ways PSO can't cover (use genetic algorithms instead).
- Real-time decisions require strict latency (PSO/ACO converge slowly relative to single-pass heuristics).

### Why bio-inspired still wins

Gradient-based methods require differentiable signals. LLM outputs and routing decisions are not naturally differentiable. Pseudo-gradient methods (RL-trained routers, DPO-style prompt tuners) work but require expensive training.

PSO and ACO only need an *evaluator* function. As long as you can score a candidate output or a routing decision, you can optimize over that space. This lowers the barrier to entry significantly.

### Practical limits

- **Population budget.** N particles × T iterations × cost per evaluation. LLM evals at ~$0.02/call means a 20-particle PSO running 50 iterations costs ~$20. Plan accordingly.
- **Exploration vs exploitation.** Pheromone decay rate and PSO inertia are a tradeoff; decay too fast → forget solutions; too slow → stuck in early local optima.
- **Catastrophic drift.** If the fitness landscape changes (new data distribution), both algorithms may converge then diverge. Monitor best-fitness stability.

## Build It

`code/main.py` implements:

- `LMPSO` — PSO on numeric prompt parameters (temperature, top_k weight). Each particle's "LLM generation" is simulated by a scripted fitness function. Runs the algorithm for 30 iterations, showing g_best convergence.
- `AMRO_S` — ACO-style routing. 3 agents, 4 task types, pheromone matrix, 100 routing tasks. Prints the (task_type → agent choices) distribution over time, showing trail formation.
- Comparison: random routing vs ACO routing on the same task stream. Measures quality and latency.

Run:

```
python3 code/main.py
```

Expected output:
- LMPSO: g_best fitness improves from random to near-optimal over 30 iterations.
- AMRO-S: pheromone table stabilizes to the correct agent for each task type; ACO routing outperforms random by ~30-40% on quality and reduces latency (fewer retries).

## Use It

`outputs/skill-swarm-optimizer.md` helps you choose between PSO, ACO, genetic algorithms, and gradient-based optimizers for LLM/agent optimization problems.

## Ship It

- **Start small.** 10-20 particles, 20-50 iterations. Scale up only when convergence curves show clear gains.
- **Log pheromone or g_best per iteration.** Debugging a swarm optimizer without traces is painful.
- **Quality-gate updates.** Especially for ACO routing: "fast and wrong" agents must not accumulate pheromone.
- **Reset decay on distribution shift.** When your eval distribution changes, stale pheromone is expired; temporarily reset or double decay rate.
- **Cap cost per iteration.** Emit a "cost-per-iteration" metric. A PSO that costs $500 per iteration for 0.5% gain is not shippable.

## Exercises

1. Run `code/main.py`. Observe LMPSO convergence. Try population sizes of 5, 10, 20, 50. At which size does "time to convergence" saturate?
2. Implement a "catastrophic drift" experiment: change the fitness function after iteration 30. How fast does PSO adapt? Does resetting `p_best` help?
3. Add a quality gate to AMRO-S: deposit pheromone only on runs with eval score > 0.7. How does this change convergence compared to the ungated version?
4. Read LMPSO (arXiv:2504.09247). Map the paper's "velocity as prompt" back to your numeric velocity. What is lost and what is preserved in the simulation?
5. Read AMRO-S (arXiv:2603.12933). Implement the decoupled "inference fast-path" plus asynchronous pheromone update. How does this change system latency under sustained load?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| PSO | "Particle Swarm Optimization" | Kennedy-Eberhart 1995. Population-based gradient-free optimizer. |
| ACO | "Ant Colony Optimization" | Dorigo 1992. Path/route optimization via pheromone trails. |
| LMPSO | "PSO with LLM generation" | arXiv:2504.09247. Velocity is a prompt; LLM produces candidates. |
| Model Swarms | "PSO on expert weights" | arXiv:2410.11163. Gradient-free updates on model parameter subspace. |
| AMRO-S | "ACO for agent routing" | arXiv:2603.12933. Pheromone matrix on task-type × agent. |
| p_best / g_best | "personal / global best" | Per-particle and swarm-wide best solutions found so far. |
| Pheromone | "routing memory" | Strength on an edge; decays over time; deposited proportional to quality. |
| Quality-gated update | "learn only from good runs" | Pheromone deposit conditioned on quality check. |
| Catastrophic drift | "distribution shift" | Fitness landscape changes; old p_best and pheromone become stale. |

## Further Reading

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — The 1995 PSO paper
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — The 1992 ACO foundation
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — PSO for structured LLM outputs
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — PSO on model weight subspace
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — Pheromone-driven routing with quality gates
