# Sim-to-Real Transfer

> A policy trained in a simulator that fails on hardware is a policy that memorized the simulator. Domain randomization, domain adaptation, and system identification are the three tools that get learned controllers across the reality gap.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 minutes

## The Problem

Training a real robot is slow, dangerous, and expensive. A bipedal robot needs millions of training episodes to learn to walk; a real biped breaks hardware on the first fall. Simulation gives you unlimited resets, deterministic reproducibility, parallel environments, and no physical damage.

But simulators are wrong. Bearing friction is higher than the MuJoCo model assumes. Cameras have lens distortion the simulator doesn't include. Motors have latency, backlash, and saturation that 99% of sim models skip. Wind, dust, and varying lighting break a policy trained on sterile renders. The **reality gap** — the systematic difference between the sim distribution and the real distribution — is the core problem of deploying RL to robots.

You need a policy that is *robust to the sim-to-real distribution shift*. Three historical approaches: randomize the simulator (domain randomization), adapt the policy with a small amount of real data (domain adaptation / fine-tuning), or identify the real system's parameters and match them (system identification). By 2026, the dominant recipe combines all three with massive parallel simulation (Isaac Sim, Isaac Lab, Mujoco MJX on GPU).

## The Concept

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR).** Tobin et al. 2017, Peng et al. 2018. During training, randomize every simulation parameter that might differ on the real robot: masses, friction coefficients, motor PD gains, sensor noise, camera positions, lighting, textures, contact models. The policy learns a conditional distribution over "which sim it's in today" and generalizes across the span. If the real robot falls inside the training envelope, the policy works.

- **Pro:** No real data needed. One recipe, many robots.
- **Con:** Over-randomized training produces a "jack of all trades" policy that is overly conservative. Too much noise ≈ too much regularization.

**System Identification (SI).** Fit the simulator's parameters to real-world data before training. If you can measure the arm's joint friction on the real robot, plug it into the sim. Then train a policy that expects those values. Requires access to the real system, but directly narrows the reality gap.

- **Pro:** Precise, low-noise training target.
- **Con:** Residual model errors are invisible to the policy; small unidentified effects (e.g., motor dead-zones) still break deployment.

**Domain Adaptation.** Train in sim, fine-tune with a small amount of real data. Two flavors:

- **Real2Sim2Real:** Learn a residual simulator `f(s, a, z) - f_sim(s, a)` from real rollouts, train in the corrected sim. Narrows the gap without much real data.
- **Observation adaptation:** Train a policy via a learned feature extractor (e.g., GAN pixel-to-pixel) that maps real observations → sim-like observations. The controller stays in sim.

**Privileged Learning / Teacher-Student.** Miki et al. 2022 (ANYmal quadruped). Train a *teacher* in simulation with access to privileged information (ground-truth friction, terrain height, IMU drift). Distill into a *student* that sees only real sensor observations. The student learns to infer privileged features from history, robust across physics parameters.

**Massive Parallel Simulation.** 2024–2026. Isaac Lab, Mujoco MJX, Brax all run thousands of parallel robots on a single GPU. PPO with 4096 parallel humanoids collects years of experience in hours. The wider the training distribution, the smaller the "reality gap"; when those 4096 envs each have different randomization parameters, DR is nearly free.

**2026 real-world recipe (quadruped locomotion example):**

1. Massively parallel sim, randomizing gravity, friction, motor gains, payload.
2. Train a teacher policy with privileged info (terrain map, body velocity ground truth).
3. Distill a student policy using only proprioception (leg joint encoders).
4. Optional: observation adaptation with an autoencoder on real IMU.
5. Deploy. Zero-shot on 10+ environments. If it fails, a few minutes of real-world fine-tuning with safety-constrained PPO.

## Build It

This lesson's code is a miniature demonstration of domain randomization on a GridWorld with *noisy* transitions. We train a policy that sees randomized slip probability in "sim" and evaluate on a "real" slip level it never saw during training. The shape maps directly to MuJoCo-to-hardware transfer.

### Step 1: Parameterized simulation

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` is a parameter the simulator exposes. On a real robot it could be friction, mass, motor gain — anything that shifts between sim and real.

### Step 2: Train with DR

Each episode start, sample `slip ~ Uniform[0.0, 0.4]`. Train PPO / Q-learning / whatever. Run many episodes this way.

### Step 3: Zero-shot evaluation on "real" slip values

Evaluate on `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. The first four are within the training support; `0.5` and `0.7` are out. A DR-trained policy should be near-optimal within the support and degrade gracefully outside it. A policy trained on a fixed slip value will be brittle outside its training slip.

### Step 4: Compare against narrow training

Train a second policy with `slip = 0.0` only. Evaluate on the same `slip` sweep. You should see catastrophic degradation as soon as real slip > 0.

## Pitfalls

- **Too much randomization.** Training on `slip ∈ [0, 0.9]` and your policy becomes so risk-averse it never attempts the optimal path. Match the *expected* real-world distribution, not "anything goes."
- **Too little randomization.** Training on a thin slice, the policy can't generalize at all. Use adaptive curriculum (automatic domain randomization) that widens the distribution as the policy improves.
- **Wrong parameter space identified.** Randomizing the wrong thing (the reality gap is motor latency, you randomized camera hue), and DR won't help. Profile the real robot first.
- **Privileged information leakage.** A teacher that makes decisions from global state instead of only observations may produce a policy the student can never match. Ensure the teacher's policy is achievable given observation history for the student.
- **Sim-to-sim transfer failure.** If your policy isn't robust to a harder sim variant, it won't be robust to the real world. Always test on held-out sim variants before deployment.
- **No real-world safety envelope.** A policy that works in sim and "also works" in real but without an underlying safety shield can still damage hardware. Add rate limits, torque limits, joint limits in a non-learned controller.

## Use It

2026 sim-to-real stack:

| Domain | Stack |
|--------|-------|
| Legged locomotion (ANYmal, Spot, humanoids) | Isaac Lab + DR + privileged teacher/student |
| Manipulation (dexterous hands, pick-and-place) | Isaac Lab + DR + DR-GAN for vision |
| Autonomous driving | CARLA / NVIDIA DRIVE Sim + DR + real fine-tuning |
| Drone racing | RotorS / Flightmare + DR + online adaptation |
| Finger/in-hand manipulation | OpenAI Dactyl (DR at unprecedented scale) |
| Industrial arms | MuJoCo-Warp + SI + minimal real fine-tuning |

The workflow is consistent across all scales of control: fit sim as well as you can, randomize what you can't fit, train massive policies, distill, deploy with a safety shield.

## Ship It

Save as `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Exercises

1. **Easy.** Train a Q-learning agent on a fixed-slip GridWorld (slip=0.0). Evaluate on slip ∈ {0.0, 0.1, 0.3, 0.5}. Plot return vs slip.
2. **Medium.** Train a DR Q-learning agent sampling `slip ~ Uniform[0, 0.3]`. Evaluate on the same sweep. How much does DR help at slip=0.5 (out of distribution)?
3. **Hard.** Implement a curriculum: start at slip=0.0, widen the DR range each time the policy achieves 90% of optimal. Measure total environment steps to zero-shot reach slip=0.3 compared to the fixed DR baseline.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| Reality gap | "sim-to-real mismatch" | Distribution shift between training and deployment physics/sensing. |
| Domain Randomization (DR) | "train across random sims" | Randomize sim parameters during training so the policy generalizes. |
| System Identification (SI) | "measure real, fit sim" | Estimate real physics parameters; set sim to match. |
| Domain Adaptation | "fine-tune on real data" | Small amount of real-world fine-tuning after sim training; may adapt obs or dynamics. |
| Privileged information | "ground truth for the teacher" | Info only sim has; the student must infer it from observation history. |
| Teacher-student | "distill privileged into observable" | Teacher trains with shortcuts; student learns to mimic without them. |
| ADR | "automatic domain randomization" | Curriculum that widens DR ranges as the policy improves. |
| Real2Sim | "use real data to narrow the gap" | Learn a residual so sim mimics real rollouts. |

## Further Reading

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — The original DR paper (robotic vision).
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR for dynamics, quadruped locomotion.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, massive-scale ADR.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal teacher-student.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — Massive parallel sim powering 2025–2026 deployments.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR curriculum approach.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna framework (planning with model + rollouts), the foundation of modern sim-to-real pipelines.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — Taxonomy of sim-to-real methods and benchmark results.
