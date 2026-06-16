# Red Teaming: PAIR and Automated Attacks

> Chao, Robey, Dobriban, Hassani, Pappas, Wong (NeurIPS 2023, arXiv:2310.08419). PAIR — Prompt Automatic Iterative Refinement — is the canonical automated black-box jailbreak. An attacker LLM with a red-team system prompt iteratively proposes jailbreak candidates for a target LLM, accumulating attempts and responses into its own conversation history as in-context feedback. PAIR typically succeeds within 20 queries, orders of magnitude more efficient than GCG (Zou et al.'s token-level gradient search), and requires no white-box access. PAIR is now a standard baseline in JailbreakBench (arXiv:2404.01318) and HarmBench, alongside GCG, AutoDAN, TAP, and Persuasive Adversarial Prompts.

**Type:** Build
**Languages:** Python (standard library, simulated PAIR loop against a toy target)
**Prerequisites:** Phase 18 · 01 (Instruction Following), Phase 14 (Agent Engineering)
**Time:** ~75 min

## Learning Objectives

- Describe the PAIR algorithm: attacker system prompt, iterative refinement, in-context feedback.
- Explain why PAIR is strictly more efficient than GCG when the target is black-box.
- Name four other automated attack baselines (GCG, AutoDAN, TAP, PAP) and state one distinguishing feature of each.
- Describe the JailbreakBench and HarmBench evaluation protocols and what "attack success rate" means under each definition.

## The Problem

Red teaming used to be a manual activity. A small number of expert testers crafted adversarial prompts and tracked what worked. This doesn't scale: attack success rates need statistical samples, and the target moves with every model release. PAIR operationalizes red teaming as an optimization problem with a black-box target.

## The Concept

### The PAIR Algorithm

Inputs:
- Target LLM T (the model we want to attack).
- Judge LLM J (evaluates whether a response constitutes a jailbreak).
- Attacker LLM A (the red-team optimizer).
- Goal string G: "Respond with [harmful instruction]."
- Budget K (typically 20 queries).

Loop for k = 1..K:
1. Prompt A with goal G and the history of (prompt, response) pairs so far.
2. A produces a new prompt p_k.
3. Submit p_k to T; receive response r_k.
4. J scores (p_k, r_k) against the goal.
5. If score >= threshold, stop — jailbreak found.
6. Otherwise, append (p_k, r_k) to A's history; continue.

Empirical results (NeurIPS 2023): ASR >50% against GPT-3.5-turbo, Llama-2-7B-chat; mean queries to success in the 10–20 range.

### Why PAIR Is Efficient

GCG (Zou et al. 2023) searches over adversarial token suffixes via gradients; it requires white-box model access and produces unreadable suffixes. PAIR is black-box and produces natural-language attacks that transfer across models. PAIR's in-context feedback lets the attacker learn from each refusal; GCG has no equivalent (each new token update must re-discover prior progress).

### Related Automated Attacks

- **GCG (Zou et al. 2023, arXiv:2307.15043).** Token-level gradient search over adversarial suffixes. White-box, transferable, produces unreadable strings.
- **AutoDAN (Liu et al. 2023).** Evolutionary search over prompts guided by a hierarchical objective.
- **TAP (Mehrotra et al. 2024).** Tree of Attack with Pruning — branches out multiple PAIR-style rollouts.
- **PAP (Zeng et al. 2024).** Persuasive Adversarial Prompts — encodes human persuasion techniques into prompt templates.

### JailbreakBench and HarmBench

Both (2024) standardize evaluation:

- JailbreakBench (arXiv:2404.01318). 100 harmful behaviors across 10 OpenAI policy categories. Primary metric is attack success rate (ASR). Requires a judge (GPT-4-turbo, Llama Guard, or StrongREJECT).
- HarmBench (Mazeika et al. 2024). 510 behaviors across 7 categories with both semantic and functional harm tests. Compares 18 attacks against 33 models.

ASR is typically reported under a fixed query budget. Comparing attacks requires matching budgets; 90% ASR at 200 queries is not comparable to 85% ASR at 20 queries.

### Why This Matters for 2026 Deployments

Today every frontier lab runs PAIR and TAP against production models before release. ASR trajectories appear in model cards (Lesson 26) and safety case appendices (Lesson 18). This attack is not exotic — it is standard infrastructure.

### Where This Fits in Phase 18

Lesson 12 is the foundation for automated attacks. Lesson 13 (Many-Shot Jailbreaking) is a complementary length exploitation. Lesson 14 (ASCII Art / Visual) is an encoding attack. Lesson 15 (Indirect Prompt Injection) is the 2026 production attack surface. Lesson 16 covers defensive tooling against these (Llama Guard, Garak, PyRIT).

## Use It

`code/main.py` builds a toy PAIR loop. The target is a simulated classifier that rejects "obvious" harmful prompts (keyword filtering). The attacker is a rule-based refiner that tries paraphrasing, role-play framing, and encoding. The judge scores responses. You will watch the attacker break through keyword filtering in ~5–15 iterations while failing against semantic filtering.

## Ship It

This lesson produces `outputs/skill-attack-audit.md`. Given a red-team evaluation report, it audits: which attacks were run (PAIR, GCG, TAP, AutoDAN, PAP), at what budget each, with which judge, against which harmful behavior set (JailbreakBench, HarmBench, internal set).

## Exercises

1. Run `code/main.py`. Measure the mean queries to success for the three built-in attacker strategies. Explain which target defense assumption each one exploits.

2. Implement a fourth attacker strategy (e.g., translation to another language, base64 encoding). Report the new mean queries to success against both the keyword-filter target and the semantic-filter target.

3. Read Chao et al. 2023 Figure 5 (PAIR vs GCG comparison). Describe two scenarios where GCG is still preferred despite PAIR's efficiency advantage.

4. JailbreakBench reports ASR against a fixed behavior set. Design an additional metric that measures attack diversity (variance of successful prompts). Explain why diversity matters for defense evaluation.

5. TAP (Mehrotra 2024) extends PAIR with branching + pruning. Sketch a TAP-style extension to `code/main.py` and describe the compute-cost vs success-rate tradeoff.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| PAIR | "automated jailbreak" | Prompt Automatic Iterative Refinement; attacker LLM + judge LLM loop |
| GCG | "gradient jailbreak" | White-box token-level gradient search over adversarial suffixes |
| Attack Success Rate (ASR) | "jailbreak percentage at k queries" | Primary metric; must be reported alongside query budget and judge identity |
| Judge LLM | "the scorer" | LLM that evaluates whether a response satisfies the harmful goal |
| JailbreakBench | "the eval" | Standardized harmful behavior set with labeled categories |
| HarmBench | "the broader benchmark" | 510 behaviors, functional + semantic harm tests |
| TAP | "attack tree" | PAIR with branching + pruning; trades more compute for better ASR |

## Further Reading

- [Chao et al. — Jailbreaking Black Box LLMs in Twenty Queries (arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — PAIR paper, NeurIPS 2023
- [Zou et al. — Universal and Transferable Adversarial Attacks on Aligned LLMs (arXiv:2307.15043)](https://arxiv.org/abs/2307.15043) — GCG paper
- [Chao et al. — JailbreakBench (arXiv:2404.01318)](https://arxiv.org/abs/2404.01318) — Standardized evaluation
- [Mazeika et al. — HarmBench (ICML 2024)](https://arxiv.org/abs/2402.04249) — Broader evaluation
