# Reward Modeling & RLHF

> Humans can't write a reward function for "good assistant response," but they can compare two responses and pick the better one. Fit a reward model to those comparisons, then RL the language model against it. Christiano 2017. InstructGPT 2022. The recipe that turned GPT-3 into ChatGPT. By 2026 it's largely replaced by DPO — but the mental model persists.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## The Problem

You trained a language model with a next-token prediction objective. It writes grammatical English. It also lies, rambles, and fails to refuse when it should. You can't fix this with more pretraining — web text is the disease, not the cure.

You want a *scalar reward* that says "for instruction X, response A is better than response B." Hand-writing that reward function is impossible. "Helpfulness" isn't a closed-form expression over tokens. But humans can compare two outputs and mark a preference. That's cheap to collect at scale.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) turns preferences into a reward model, then optimizes the LM against that reward via PPO. Three stages: SFT → RM → PPO. This is the recipe that shipped ChatGPT, Claude, Gemini, and every other aligned LLM from 2023–2025.

By 2026, the PPO stage is largely replaced by DPO (Phase 10 · 08) because it's cheaper and nearly as good for alignment tuning. But the *reward model* piece remains the foundation of every Best-of-N sampler, every RL-from-verifiable-rewards pipeline, and every reasoning model using process reward models. Understand RLHF and you understand the entire alignment stack.

## The Concept

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**Stage 1: Supervised Fine-Tuning (SFT).** Start with a pretrained base model. Fine-tune on human-written demonstrations of the target behavior (instruction-following responses, helpful answers, etc.). Result: a model `π_SFT` that is *biased toward* good behavior but whose action space is still unbounded.

**Stage 2: Reward Model Training.**

- For a prompt `x`, collect paired responses `(y_+, y_-)` where humans label "y_+ is better than y_-."
- Train a reward model `R_φ(x, y)` that scores `y_+` higher.
- Loss: **Bradley-Terry pairwise logistic**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ is sigmoid. The reward difference encodes the log-odds of the preference. BT has been standard since 1952 (Bradley-Terry) and is the dominant choice in modern RLHF.

- `R_φ` is typically initialized from the SFT model with a scalar head on top. Same transformer backbone; one linear layer outputs the reward.

**Stage 3: PPO against the RM with KL penalty.**

- Initialize trainable policy `π_θ` from `π_SFT`. Keep a frozen *reference* `π_ref = π_SFT`.
- Reward at end of response `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  The KL penalty prevents `π_θ` from drifting arbitrarily from `π_SFT` — it's the *regularizer*, not a hard trust region. `β` is typically `0.01`–`0.05`.
- Run PPO (lesson 08) with this reward. Advantages are computed over token-level trajectories, but the RM only scores complete responses.

**Why KL?** Without it, PPO will happily find reward-hacking strategies — the RM was only trained on in-distribution completions. An out-of-distribution response might score higher than anything a human wrote. KL keeps `π_θ` near the manifold the RM was trained on. It's the single most important knob in RLHF.

**2026 landscape:**

- **DPO** (Rafailov 2023): Closed-form algebra collapses stages 2 and 3 into a single supervised loss on preference data. No RM, no PPO. Comparable quality on alignment benchmarks at a fraction of compute. Covered in Phase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO with critic replaced by group-relative baseline, reward from a *verifier* (code runs / math answer matches) instead of a human-trained RM. Dominates for reasoning models. Covered in Phase 9 · 12.
- **Process Reward Models (PRM):** Score partial solutions (each reasoning step), useful in both RLHF and GRPO variants for reasoning.
- **Constitutional AI / RLAIF:** Use an aligned LLM instead of humans to generate preferences. Scales the preference budget.

## Build It

This lesson uses a minimal synthetic "prompt" and "response" represented as strings. The RM is a linear scorer on bag-of-tokens representations. No real LLM — what matters is the *shape* of the pipeline, not the scale. See `code/main.py`.

### Step 1: Synthetic preference data

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

In real RLHF this is replaced by human annotators. The shape — `(prompt, preferred_response, rejected_response)` — is identical.

### Step 2: Bradley-Terry reward model

Linear scorer: `R(x, y) = w · bag(y)`. Train to minimize BT pairwise log-loss:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

After a few hundred updates, `w` assigns positive weight to good-word tokens and negative weight to bad ones.

### Step 3: PPO-style policy on top of the RM

Our toy policy produces a single token from a vocabulary. We score that token under the RM, compute `log π_θ(token | prompt)`, add a KL penalty to the reference, and apply the clipped PPO surrogate.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Step 4: Monitor KL

Track mean `KL(π_θ || π_ref)` every update. If it climbs past `~5-10`, the policy has drifted too far from `π_SFT` — either `β` is too low or reward hacking has begun. This is the #1 diagnostic in real RLHF.

### Step 5: Production recipe with TRL

After understanding the toy pipeline, here is the same loop as a real library user would write it. Hugging Face's [TRL](https://huggingface.co/docs/trl) is the reference implementation — `RewardTrainer` for stage 2, `PPOTrainer` for stage 3 (with built-in KL to reference).

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

The library handles three things for you. `adap_kl_ctrl=True` implements an adaptive β schedule: if observed KL exceeds `target_kl`, β doubles; if below half, β halves. The reference model is frozen by convention — you must never accidentally share parameters with `policy`. The value head lives on the same backbone as the policy (`AutoModelForCausalLMWithValueHead` attaches a scalar MLP head), which is why TRL reports `policy/kl` and `value/loss` separately.

## Pitfalls

- **Over-optimization / reward hacking.** The RM is imperfect; `π_θ` finds adversarial completions that score high but are terrible. Symptoms: reward climbs unboundedly while human eval plateaus or drops. Fix: early stopping, higher `β`, broader RM training data.
- **Length hacking.** RMs trained on helpful responses often implicitly reward length. The policy learns to pad responses. Remedy: length-normalized rewards, or RLAIF with a length-aware RM.
- **RM too small.** The RM should be at least as large as the policy. A small RM can't faithfully score the policy's outputs.
- **KL tuning.** β too low → drift and reward hacking. β too high → policy barely changes. Standard trick is *adaptive* β that pins per-step KL to a fixed value.
- **Preference data noise.** ~30% of human labels are noisy or ambiguous. Train the RM with consistency-filtered data to calibrate, or use a temperature on BT.
- **Off-policy issues.** After the first epoch PPO data is slightly off-policy. Monitor clip fraction as in lesson 08.

## Use It

RLHF in 2026 is layered:

| Layer | Goal | Method |
|-------|------|--------|
| Instruction following, helpfulness, harmlessness | Alignment | DPO (Phase 10 · 08) preferred over RLHF-PPO. |
| Reasoning correctness (math, code) | Capability | GRPO with verifier rewards (Phase 9 · 12). |
| Long-horizon multi-step tasks | Agentic | PPO / GRPO with process reward models over steps. |
| Safety / refusal behavior | Safety | RLHF-PPO with a separate safety RM, or Constitutional AI. |
| Inference-time Best-of-N | Fast alignment | RM used at decoding time; no policy training required. |
| Reward distillation | Inference compute | Train a small "reward head" on a frozen LM. |

RLHF is *the* method from 2022–2024. By 2026, production alignment pipelines prefer DPO and reserve PPO only for steps where the RM is dense or safety-critical.

## Ship It

Save as `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Exercises

1. **Easy.** Train a Bradley-Terry reward model on 500 synthetic preference pairs in `code/main.py`. Test pairwise accuracy on a held-out 100 pairs. Should exceed 90%.
2. **Medium.** Run the toy PPO-RLHF loop with `β ∈ {0.0, 0.1, 1.0}`. For each, plot RM score vs KL to reference over updates. Which one exhibits reward hacking?
3. **Hard.** Implement DPO (closed-form preference likelihood loss) on the same preference data and compare compute used and final RM score achieved against the RLHF-PPO pipeline.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| RLHF | "alignment RL" | Three-stage SFT + RM + PPO pipeline (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "scoring network" | Learned scalar function fit to pairwise preferences via Bradley-Terry. |
| Bradley-Terry | "pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; standard RM objective. |
| KL penalty | "stay near reference" | `β · KL(π_θ || π_ref)` in the reward; regularizer against reward hacking. |
| Reward hacking | "Goodhart's law" | Policy exploiting RM flaws; symptoms: reward rises, human eval plateaus. |
| RLAIF | "AI-labeled preferences" | RLHF where labels come from another LM instead of humans. |
| PRM | "process reward model" | Scores partial reasoning steps; used in reasoning pipelines. |
| Constitutional AI | "Anthropic's method" | AI-generated preferences guided by explicit rules. |

## Further Reading

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — The paper that started RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — The recipe behind ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — Earlier RLHF for summarization.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; the post-RLHF default by 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF and self-critique loops.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — The HH paper.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — Production-grade `RewardTrainer` and `PPOTrainer`. Read the trainer source for adaptive KL and value head details.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — Classic illustrated walkthrough of the three-stage pipeline.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — The library; `examples/` has end-to-end RLHF scripts for Llama, Mistral, Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — Reward hypothesis perspective; prerequisite for thinking about reward hacking.
