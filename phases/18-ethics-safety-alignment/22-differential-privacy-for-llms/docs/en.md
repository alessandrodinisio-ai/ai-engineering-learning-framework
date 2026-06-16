# Differential Privacy for LLMs

> DP-SGD remains the standard — noise-injected gradient updates provide formal (epsilon, delta) guarantees. The overhead in compute, memory, and utility is substantial; parameter-efficient DP fine-tuning (LoRA + DP-SGD) is the common 2025 configuration (ACM 2025). Two lines of evidence are in tension: canary-based membership inference (Duan et al., 2024) reports limited success on language models; training data extraction (Carlini et al., 2021; Nasr et al., 2025) recovers substantial verbatim memorization. Resolution (arXiv:2503.06808, March 2025): the gap lies in what is being measured — inserted canaries vs "most extractable" data. New canary designs enable loss-based MIA without shadow models, and produce the first non-trivial DP audit of an LLM trained on real data with realistic DP guarantees. Alternatives: PMixED (arXiv:2403.15638) — inference-time private prediction via a mixture of experts over the next-token distribution; DP synthetic data generation (Google Research 2024). Emerging attack: differential privacy reversal via LLM feedback — confidence scores leak.

**Type:** Build
**Languages:** Python (standard library, DP-SGD noise injection and ε-δ accountant demo)
**Prerequisites:** Phase 01 · 09 (information theory), Phase 10 · 01 (LLM training)
**Time:** ~60 minutes

## Learning Objectives

- Define (epsilon, delta)-differential privacy and state the DP-SGD recipe.
- Explain the 2024–2025 tension: canary MIA vs training data extraction give different pictures.
- Describe PMixED and why inference-time private prediction is an alternative to DP training.
- Describe the differential privacy reversal attack via LLM feedback.

## The Problem

LLMs memorize. Carlini et al. 2021 demonstrated that production language models can verbatim reproduce training text on demand. DP is the formal defense: make the output provably insensitive to any single training sample during training. 2024–2025 evidence shows DP-SGD is necessary, but deployed ε values may not match the threat model.

## The Concept

### (ε, δ)-Differential Privacy

A randomized algorithm M is (ε, δ)-DP if for any two datasets differing in one sample and any event S:
P(M(D) in S) <= e^ε * P(M(D') in S) + δ.

Interpretation: the output distribution is close enough (parameterized by ε) that any single individual's contribution cannot be reliably inferred, except with probability δ.

### DP-SGD

Abadi et al. 2016. Standard recipe:
1. Sample a mini-batch.
2. Compute per-sample gradients.
3. Clip each per-sample gradient to a threshold C.
4. Sum the clipped gradients and add Gaussian noise with standard deviation σ * C.
5. Update parameters with the noised sum.

Privacy cost is tracked by an accountant (moments accountant, Rényi DP accountant). Reported ε values in the LLM literature vary widely depending on threat model, data sensitivity, and utility targets; there is no universally "safe" default ε. Published examples range roughly across ε ≈ 1–10 in some LLM training settings, but these are illustrative — not recommended defaults. Lower ε generally requires more noise and may increase utility loss.

### LoRA + DP-SGD

Full DP-SGD on a frontier model is prohibitively expensive. LoRA (Hu et al. 2022) restricts gradient updates to a small adapter, reducing per-sample gradient storage. LoRA + DP-SGD is the common 2025 configuration. The DP guarantee applies to the adapter; the base model remains frozen.

### The 2024–2025 Tension

Two lines of evidence:

- **Canary MIA (Duan et al. 2024).** Insert unique canaries into training data, measure whether a membership inference attacker can identify them. Reports limited success on language models. Suggests MIA is hard.
- **Training data extraction (Carlini 2021, Nasr et al. 2025).** Prompt the model with a prefix; measure whether it can recover verbatim text from training. Reports substantial memorization. Suggests MIA is easy in a relevant sense.

March 2025 resolution (arXiv:2503.06808): the two measure different things. MIA asks "is sample e in D?" on inserted canaries. Extraction asks "how much of D can I recover?" What matters for privacy is the "most extractable" samples; canaries underestimate this because they are not optimized to be extractable.

New canary designs. Loss-based MIA without shadow models. The first non-trivial DP audit of an LLM trained on real data with realistic DP guarantees.

### Alternatives to DP Training

- **PMixED (arXiv:2403.15638).** Inference-time private prediction. A mixture of experts over the next-token distribution; each expert sees a shard of training data; aggregation adds noise for DP. Avoids DP training entirely.
- **DP synthetic data generation (Google Research 2024).** LoRA fine-tune with DP-SGD, sample synthetic data, train a downstream classifier on synthetic data.

Both sidestep the utility cost of full DP training at the price of a different threat model.

### Differential Privacy Reversal via LLM Feedback

Emerging 2025 attack. Use the confidence scores of a DP-trained model as an oracle to re-identify individuals. Even if outputs do not leak, the confidence distribution may.

Defense: do not expose confidence scores, or truncate/quantize them before exposure. This is an additional requirement beyond (ε, δ)-DP training.

### Position in Phase 18

Lessons 20–21 are bias/fairness. Lesson 22 is privacy. Lesson 23 is provenance via watermarking. Lesson 27 covers regulatory-level data provenance.

## Build It

`code/main.py` simulates DP-SGD on a toy binary classification dataset. You can sweep noise multiplier σ and clipping norm C, tracking the (ε, δ) budget and accuracy cost. A "canary attack" inserts a unique training sample and measures whether a log-loss test can detect it before and after DP.

## Use It

This lesson produces `outputs/skill-dp-audit.md`. Given a DP claim on a language model deployment, it audits: the (ε, δ) values, the accountant used, the MIA evaluation protocol, and whether the confidence-exposure vector has been assessed.

## Exercises

1. Run `code/main.py`. Sweep σ over {0.5, 1.0, 2.0} and report the (ε, δ)-accuracy tradeoff. Identify the point where utility collapses.

2. Implement a canary insertion and a log-loss test. Measure detection rate before and after DP-SGD at σ = 1.0.

3. Read Nasr et al. 2025 on training data extraction. Why doesn't extraction success collapse at moderate ε? What does this imply for "MIA as evaluation"?

4. Design a deployment that operates entirely at inference time using PMixED (arXiv:2403.15638). What threat model does PMixED handle that DP-SGD does not?

5. Sketch the DP reversal attack via LLM feedback. Design a countermeasure that limits confidence score leakage, and estimate its deployment cost.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| DP | "(ε, δ)-differential privacy" | Formal privacy: output distributions close under adjacent dataset changes |
| DP-SGD | "Noise-injected SGD" | Gradient clipping + Gaussian noise; standard DP training |
| LoRA + DP-SGD | "Efficient private fine-tuning" | DP-SGD on low-rank adapters; 2025 standard configuration |
| MIA | "Membership inference" | Attack to determine whether a sample was in training data |
| Canary | "Inserted watermark sample" | Unique training sample used to measure DP leakage |
| PMixED | "Private inference mixture" | Inference-time DP via mixture of experts over next-token distribution |
| DP reversal | "Confidence leakage attack" | Attack using model confidence as a re-identification oracle |

## Further Reading

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133) — standard DP training algorithm
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805) — foundational extraction paper
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841) — limited-success MIA
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808) — resolving the tension
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638) — inference-time private prediction
