# The Direct Preference Optimization Family

> Rafailov et al. (2023) proved that the optimal solution to RLHF can be written in closed form using preference data, so you can skip the explicit reward model and optimize the policy directly. This insight spawned a family — IPO, KTO, SimPO, ORPO, BPO — each fixing a different failure mode of DPO. By 2026, direct alignment algorithms power more frontier post-training runs than PPO. But the overoptimization curve from Lesson 2 still applies: DAAs haven't escaped Goodhart — they've just moved where it bites.

**Type:** Learn
**Languages:** Python (standard library, six-variant preference loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward Hacking), Phase 10 · 08 (DPO Basics)
**Time:** ~75 minutes

## Learning Objectives

- Derive the DPO closed form from the optimal solution to "RLHF with KL."
- Name which failure mode of DPO each of IPO, KTO, SimPO, ORPO, and BPO fixes.
- Distinguish "implicit reward gap" from "preference strength" and explain why IPO's identity mapping matters.
- Explain why Rafailov et al. (NeurIPS 2024) proved that DAAs overoptimize even without an explicit RM.

## The Problem

The RLHF objective (Lesson 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

Has a known optimal solution:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

So the reward is implicitly defined as the ratio between the optimal policy and the reference policy:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Substitute this into the Bradley-Terry preference likelihood. The partition function `Z(x)` cancels because it depends only on `x`. What remains is a loss containing only policy parameters — no reward model needed. This is DPO.

The wrinkle: this derivation assumes the optimal solution is reachable, the preference data is in-distribution, and the reference policy is a true mode anchor. None of these hold exactly. Each family member fixes a different violated assumption.

## The Concept

### DPO (Rafailov et al., 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

What goes wrong:

- The implicit reward gap `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` is unbounded. A small preference can create an arbitrarily large gap.
- The loss pushes chosen and rejected log-probabilities in opposite directions. As long as rejected drops faster, it can push the chosen absolute log-probability down too. This is the "Degraded Chosen Response" phenomenon.
- Out-of-distribution preferences (rare-vs-rare pairs) produce arbitrary implicit rewards.

### IPO (Azar et al., 2024)

Identity Preference Optimization replaces the log-sigmoid with an identity mapping over preference probabilities. The loss becomes a squared error against a bounded target:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

The margin is clamped by `1/(2 beta)`. Preference strength is proportional to the implicit reward gap. No explosion.

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky Optimization drops the pairwise structure entirely. Given a single labeled output and a binary "desirable" or "undesirable" signal, it maps to a prospect-theoretic utility:

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

Gains and losses are weighted differently (loss aversion). Benefit: you can use unpaired data, which is far more abundant.

### SimPO (Meng et al., 2024)

Simple Preference Optimization aligns the training signal with generation. It drops the reference policy entirely and normalizes log-likelihood by length:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

With a margin `gamma` for stability. Length normalization removes the incentive to exploit DPO's length-bias failure mode (by construction, a longer `y_w` yields a larger log-probability gap).

### ORPO (Hong et al., 2024)

Odds-Ratio Preference Optimization adds a preference term on top of the standard SFT negative log-likelihood:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

No reference policy — the SFT term is the regularizer. Single-stage training from a base model to an aligned model. No separate SFT checkpoint.

### BPO (ICLR 2026 submission, OpenReview id=b97EwMUWu7)

It identifies the degraded chosen response problem: DPO preserves the ranking `y_w > y_l`, but the absolute log-probability of `y_w` can drop. BPO adds a correction term that penalizes downward movement on the chosen response. Reports +10.1% accuracy over DPO on math reasoning tasks with Llama-3.1-8B-Instruct.

### Universal Conclusion: DAAs Overoptimize Too

Rafailov et al. "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024) trained policies with DPO, IPO, SLiC across multiple datasets and KL budgets. The gold-reward-vs-KL curves have the same "peak then collapse" shape as Gao et al. The implicit reward queries out-of-distribution samples during training; KL regularization doesn't stabilize this.

DAAs haven't escaped Goodhart. They've shifted the surface it bites from "reward model being overoptimized" to "reference policy ratio being overoptimized." The universal fix — better data, ensembling, early stopping — applies to both.

### Choosing Between Them (2026)

- If you have abundant paired preference data: DPO with a conservative beta; SimPO if length bias is obvious.
- If you have unpaired binary feedback: KTO.
- If you want a single-stage pipeline from a base model: ORPO.
- If you see chosen log-probability degradation in DPO logs: BPO.
- If preference strengths vary widely and DPO saturates: IPO.

Every lab runs all five on a test split and picks the winner per task. There's no reason to expect the same optimum for math reasoning and safety tasks.

## Build It

`code/main.py` compares six losses (DPO, IPO, KTO, SimPO, ORPO, BPO) on a toy preference dataset where ground-truth preference strength varies per pair. Each loss optimizes a small softmax policy on the same 500 pairs. It plots final win rate, chosen log-probability drift, and implicit reward dispersion for each method.

## Use It

This lesson produces `outputs/skill-preference-loss-selector.md`. Given dataset statistics (paired vs unpaired, variable vs uniform preference strength, length distribution) and a target (single-stage or SFT-then-preference), it recommends a preference loss and reports which failure mode it guards against.

## Exercises

1. Run `code/main.py`. Report the final chosen log-probability drop for DPO and BPO. BPO should preserve higher chosen absolute probability — verify this.

2. Modify the preference data so all pairs have equal strength. Which of the six methods is most robust? Which degrades? Explain IPO's advantage here.

3. Make rejected responses on average 2x longer than chosen. With everything else fixed, demonstrate DPO's length exploit numerically and SimPO's fix.

4. Rafailov et al. (NeurIPS 2024) claim DAAs overoptimize. Reproduce a single-point version: plot the KL divergence of "chosen minus rejected" and observe DPO's overoptimization at large beta.

5. Read the BPO paper abstract (OpenReview b97EwMUWu7). Write down the single correction line BPO adds to DPO. Confirm against the implementation in `code/main.py`.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| DPO | "RLHF without a reward model" | A loss derived from the RLHF closed-form optimal; contains only policy parameters |
| Implicit reward | "The log ratio" | `beta * log(pi(y|x) / pi_ref(y|x))` — the reward implied by DPO |
| IPO | "Bounded DPO" | Replaces log-sigmoid with identity mapping; implicit reward gap capped at `1/(2 beta)` |
| KTO | "Unpaired DPO" | Prospect-theoretic utility with loss aversion on single labels |
| SimPO | "Reference-free DPO" | Length-normalized log-likelihood + margin; no reference policy |
| ORPO | "Single-stage DPO" | NLL + odds-ratio preference term; one-pass training from base model |
| BPO | "Chosen-preserving DPO" | DPO plus a penalty term preventing absolute log-probability drop on chosen responses |
| Chosen degradation | "Chosen dropping" | DPO pushes chosen log-probability down as long as rejected drops faster |
| DAA | "Direct alignment algorithm" | Any preference loss method that skips the explicit RM |

## Further Reading

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
