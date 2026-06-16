# Instruction Following as the Alignment Signal

> Every critique of RLHF that follows in this phase targets this pipeline. Before you study how optimization pressure distorts a proxy metric, you need to see what that proxy metric looks like. InstructGPT (Ouyang et al., 2022) defined the reference architecture: supervised fine-tuning on instruction-response pairs, training a reward model on pairwise preference rankings, then optimizing against the reward model with PPO while adding a KL penalty toward the SFT policy. A 1.3B InstructGPT beat 175B GPT-3 on human preference. That single result is why every frontier lab in 2026 still ships RLHF-shaped post-training pipelines.

**Type:** Learn
**Languages:** Python (standard library, toy three-stage pipeline)
**Prerequisites:** Phase 10 - 06 (SFT), Phase 10 - 07 (RLHF), Phase 10 - 08 (DPO)
**Time:** ~45 minutes

## Learning Objectives

- Name the three stages of the InstructGPT pipeline and the loss used at each stage.
- Explain why a 1.3B instruction-tuned model beats the raw 175B GPT-3 on human preference evaluations.
- Articulate what the Stage 3 KL penalty prevents and why removing it collapses into mode-seeking behavior.
- Describe the alignment tax and the PPO-ptx mitigation that Ouyang et al. used to address it.

## The Problem

Pretrained language models only continue text. They do not answer questions. Ask GPT-3 to "write a Python function that reverses a list" and you often get back another prompt - because the training distribution is mostly web text, and web text is followed by more web text. The model is doing its job - it is just that the job itself is wrong.

The proxy metric every serious lab uses to fix this is human preference. Two completions go to an annotator; the annotator picks the better one; a reward model learns from that annotator. Then an RL loop pushes the policy toward outputs the reward model scores highly. Three sentences and you have the entire core argument of InstructGPT. The rest of the paper is engineering.

## The Concept

### Stage 1: Supervised Fine-Tuning (SFT)

Collect prompt-response pairs where the responses are what a well-intentioned human would write. Ouyang et al. used 13k prompts from annotators and the OpenAI API. Fine-tune the base model on this data with standard cross-entropy loss.

What SFT gives you: the model now answers questions instead of continuing them. What it does not give you: any signal about which answer an annotator prefers when multiple answers are plausible.

### Stage 2: Reward Model (RM)

For each prompt, sample K completions from the SFT model. An annotator ranks them. Train a reward model that scores any (prompt, response) pair such that for pairs where `y_w` is preferred over `y_l`:

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

This is the Bradley-Terry pairwise preference loss. The RM is typically initialized from the SFT model with the LM head replaced by a scalar head.

Reward models are small: 6B suffices for 175B InstructGPT. They are also fragile - most of Section 5 of the paper discusses reward-hacking behaviors that emerge at small scale.

### Stage 3: PPO with KL Penalty

Define the objective:

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

Maximize with PPO. The KL term keeps `pi` from drifting too far from the SFT policy. Without it, the optimizer finds adversarial examples - strings that score high under the RM because the RM has never seen them, not because humans actually prefer them.

The KL coefficient `beta` is the single most important hyperparameter in RLHF. Too low: reward hacking. Too high: no improvement over SFT.

### Alignment Tax

After RLHF, the model is more preferred by humans but regresses on standard benchmarks (SQuAD, HellaSwag, DROP). Ouyang et al. call this the alignment tax and fix it with PPO-ptx: mixing pretraining gradients into the RL objective so the model does not forget downstream tasks it was never rewarded for.

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx became standard. Anthropic, DeepMind, and Meta all use some variant.

### Results

A 1.3B InstructGPT (SFT + RM + PPO-ptx) is preferred over 175B base GPT-3 by annotators approximately 70% of the time. On held-out test prompts from production traffic, the gap widens. Two things to read from this number:

1. Alignment is a different axis from capability. The 175B model is more capable; the 1.3B model is better aligned; annotators chose the aligned one.
2. The capability floor is set by the base model. You cannot RLHF a base model into facts it has never seen.

### Why This Is the Reference Point for Phase 18

Every subsequent critique in this phase - reward hacking (Lesson 2), DPO (Lesson 3), sycophancy (Lesson 4), CAI (Lesson 5), sleeper agents (Lesson 7), alignment faking (Lesson 9) - targets a specific part of this pipeline. Reward hacking attacks Stage 2. DPO collapses Stages 2 and 3. CAI replaces the human annotator. Sycophancy shows the annotator itself is a biased signal. Alignment faking shows the policy can bypass Stage 3 entirely. Without this pipeline in your head, none of those critiques are legible.

## Use It

`code/main.py` simulates the three stages on toy preference data. The base "policy" is a biased coin over actions {A, B, C}. Stage 1 SFT imitates an annotator's actions on 200 prompts. Stage 2 fits a Bradley-Terry reward model from 500 pairwise rankings. Stage 3 runs a simplified PPO update with a KL penalty toward the SFT policy. You can watch reward climb, KL divergence grow, and policy drift - and you can turn off the KL term and watch reward hacking emerge within 50 update steps.

Things to look at:

- Reward trajectories under `beta = 0.1` vs `beta = 0.0`.
- `KL(pi || pi_SFT)` as training steps progress.
- Final action distribution compared to annotator preferences.

## Ship It

This lesson produces `outputs/skill-instructgpt-explainer.md`. Given an RLHF pipeline description or a paper abstract, it identifies which of the three stages was modified, what loss is used at each stage, and whether a KL penalty or equivalent regularization is present.

## Exercises

1. Run `code/main.py`. Set `beta = 0.0` and report the action distribution after 200 PPO steps. Explain the mode-seeking behavior in one paragraph.

2. Modify the reward model to add a +0.5 bias to action B (simulating a reward bug). Run PPO with `beta = 0.1`. Does the KL penalty prevent the policy from exploiting this bias? At what value of `beta` does exploitation become visible?

3. Read Figure 1 of Ouyang et al. (arXiv:2203.02155). Reproduce the annotator preference curve by running PPO for 1, 5, 20, 100 steps and measuring preference relative to the SFT model.

4. Section 4.3 of the paper reports that 1.3B InstructGPT beats 175B GPT-3 approximately 70% of the time. Why is this ratio higher on held-out production prompts than on annotator-written prompts?

5. On the same preference data, replace the PPO loss with DPO (Phase 10 - 08). Compare final policy drift (KL to SFT) and final reward. Which method drifts further for comparable reward?

## Key Terms

| Term | Colloquial Usage | What It Actually Is |
|------|------------------|---------------------|
| SFT | "Instruction tuning" | Stage 1: cross-entropy fine-tuning on prompt-response pairs |
| Reward Model | "The RM" | A scalar regressor that scores (prompt, response) pairs, trained with Bradley-Terry on pairwise annotations |
| Bradley-Terry | "Pairwise preference loss" | -log sigmoid(r_w - r_l); reduces pairwise ranking to binary classification |
| KL Penalty | "The regularizer" | `beta * KL(pi || pi_SFT)` - anchors the RL policy near the SFT checkpoint |
| PPO-ptx | "PPO with pretraining mix" | Adds a fraction of pretraining log-likelihood to the PPO objective to offset the alignment tax |
| Alignment Tax | "RLHF regression" | Performance drops on standard benchmarks that RLHF did not target |
| Annotator Preference | "The ground truth" | A sample of human rankings; the RM is a statistical proxy of it, not a proxy of "human values" |

## Further Reading

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — The InstructGPT paper, foundation of every RLHF pipeline since
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — Predecessor work on RLHF for summarization
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — The earliest formalization of preference-based RL
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — Anthropic's HH extension of the InstructGPT pipeline
