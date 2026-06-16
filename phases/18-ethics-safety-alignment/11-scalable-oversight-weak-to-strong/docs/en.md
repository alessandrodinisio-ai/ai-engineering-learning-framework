# Scalable Oversight and Weak-to-Strong Generalization

> Burns et al. (OpenAI Superalignment, "Weak-to-Strong Generalization", 2023) proposed a proxy for the superalignment problem: fine-tune a strong model on labels produced by a weaker model. If the strong model can generalize correctly from imperfect weak supervision, then current human-scale alignment methods may extend to superhuman systems. Scalable oversight and W2SG are complementary. Scalable oversight (debate, recursive reward modeling, task decomposition) raises the supervisor's effective capability so it keeps pace with the supervised model. W2SG ensures the strong model generalizes correctly from whatever imperfect supervision the supervisor provides. Debate Helps W2SG (arXiv:2501.13124, January 2025) combines both.

**Type:** Learn
**Languages:** Python (standard library, W2SG gap simulator)
**Prerequisites:** Phase 18 - 01 (Instruction Following), Phase 18 - 10 (AI Control), Phase 09 (RL Basics)
**Time:** ~60 minutes

## Learning Objectives

- Define scalable oversight and weak-to-strong generalization, and explain how they complement each other.
- Describe the Burns et al. 2023 experimental setup: fine-tuning GPT-4 on labels from GPT-2.
- Explain the Performance Gap Recovered (PGR) metric and what it measures.
- Name three main scalable oversight mechanisms (debate, recursive reward modeling, task decomposition) and one advantage of each.

## The Problem

So far, every alignment technique in Phase 18 assumes the supervisor can evaluate the model's behavior. When the model is superhuman, the supervisor is the weak link. The superalignment problem: can a weaker supervisor reliably produce a stronger, aligned model?

Burns et al. reduce it to an operationalized empirical setting: supervise strong with weak, measure how much of the strong model's capability survives weak supervision. This is not a solution to superalignment - it is a proxy that lets us measure progress.

## The Concept

### W2SG: The Burns et al. Setup

- Weak model: GPT-2 level.
- Strong model: GPT-4 level.
- Target: the strong GPT-4's ceiling on the task.

Procedure:
1. Obtain the weak model's zero-shot predictions on a task.
2. Fine-tune the strong model on weak-labeled data.
3. Measure the fine-tuned strong model's accuracy.

Baselines: the weak model's own accuracy, the strong model's supervised ceiling on gold-standard labels.

Gap metric: Performance Gap Recovered (PGR) = (fine-tuned - weak) / (ceiling - weak). PGR of 1.0 means weak supervision fully closes the gap; PGR of 0 means weak supervision did not help.

### Empirical Findings from Burns et al.

On NLP, chess puzzles, and reward modeling tasks, PGR is consistently positive: the strong model generalizes beyond the weak supervisor's errors. Task-specific PGR ranges from approximately 20% to approximately 80%. The strong model "understands" the intended task from imperfect labels and uses its pretrained prior to fill in.

Limitations noted by Burns et al.:
- The weak-to-strong gap here is capability, not alignment. The weak model is aligned; a truly superhuman model might be human-aligned but with a much larger capability gap.
- The strong model's generalization may be more about task priors than recovering ground-truth intent.
- Eliciting the strong model's latent knowledge is the real problem; PGR measures one particular operationalization.

### Scalable Oversight: Three Mechanisms

- **Debate (Irving et al. 2018).** Two U instances argue opposing sides; a weak judge adjudicates. The assumption is: finding a persuasive true answer is easier than finding a persuasive false answer. Recent work (2024-2025, Khan et al., Michael et al.) shows debate sometimes helps and sometimes does not, depending on task structure.
- **Recursive Reward Modeling (Leike et al. 2018).** U helps a human train the reward model for U+1. The supervisor's effective capability grows with the model.
- **Task Decomposition (Christiano, Shlegeris, Amodei 2018).** Recursively break a hard task into sub-tasks a human can check. Assumes decomposability.

Each mechanism makes assumptions about task structure or alignment of intermediate components.

### Why Scalable Oversight and W2SG Are Complementary

Scalable oversight raises the quality of the effective signal from the supervisor.
W2SG closes the gap from whatever imperfect signal the supervisor can provide.

Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124) combines both: a debate protocol provides better weak labels, and the strong model trains on those labels. Reports PGR gains on NLP tasks.

### Organizational Drama

OpenAI's Superalignment team disbanded in May 2024 after Jan Leike left for Anthropic. The agenda (scalable oversight, W2SG, automated alignment research) continues at Anthropic and academic labs - MATS (Lesson 28), Redwood (Lesson 10), Apollo (Lesson 8), METR (Lesson 28). The organizational structure changed; the research question did not.

### Where This Fits in Phase 18

Lessons 6-10 describe threats and defense paradigms under the assumption "U is not trustworthy." Lesson 11 is the offensive paradigm: make the supervisor strong enough to verify U's alignment. Lessons 12-16 then turn to practical tools for adversarial evaluation.

## Use It

`code/main.py` simulates W2SG fine-tuning on a synthetic task. The weak labeler has 70% accuracy with structured errors; the strong model has a 95% ceiling on gold-standard labels. You fine-tune the strong model on weak labels, measure PGR, and compare against "strong-on-gold" and "weak-alone."

## Ship It

This lesson produces `outputs/skill-w2sg-pgr.md`. Given a supervision setup description, it identifies the weak supervisor, strong model, supervision quality, and computes (or requests) PGR. It flags whether the claim is "weak can supervise strong" or "weak + oversight mechanism can supervise strong."

## Exercises

1. Run `code/main.py`. Report PGR at weak_accuracy = 0.60, 0.70, 0.80. Explain the shape of the PGR curve.

2. Modify the weak labeler to have structured errors (e.g., always wrong on a specific input class). Does PGR go up, down, or stay the same? Explain.

3. Read Burns et al. 2023 Section 4.3 (NLP tasks). Reproduce the intuition behind the "confidence auxiliary loss": when the strong model is more confident than the weak label, who wins?

4. Design a scalable oversight protocol combining debate and task decomposition for a software engineering task. Name one failure mode of each component and explain how the combination handles or fails to handle each.

5. Articulate what would falsify the claim "weak-to-strong generalization is a viable path to superalignment." Specify the empirical signature you would need to see.

## Key Terms

| Term | Colloquial Usage | What It Actually Is |
|------|------------------|---------------------|
| Scalable Oversight | "Making the supervisor stronger" | Mechanisms that raise the supervisor's ability to evaluate a more-capable model |
| W2SG | "Weak supervises strong" | Fine-tuning a strong model on weak labels and measuring recovered capability |
| PGR | "Performance Gap Recovered" | (fine-tuned - weak) / (ceiling - weak); 1.0 = fully closed, 0 = no help |
| Debate | "Two U instances argue" | A scalable oversight mechanism where a weak judge chooses between two U debaters |
| RRM | "Recursive Reward Modeling" | U helps train the reward model for U+1; supervisor capability tracks U |
| Task Decomposition | "Human-checkable sub-tasks" | Recursively breaking hard tasks into sub-tasks a human can verify |
| Superalignment | "Aligning superhuman AI" | The research agenda focused on aligning models humans cannot directly evaluate |

## Further Reading

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — The W2SG paper
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — The debate mechanism
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — Recursive reward modeling
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — 2024 empirical study of debate with stronger debaters
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — 2025 combination of debate + W2SG
