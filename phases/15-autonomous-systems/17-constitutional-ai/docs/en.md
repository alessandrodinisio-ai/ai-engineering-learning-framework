# Constitutional AI & Rule Overrides

> Anthropic's January 22, 2026 Claude Constitution runs 79 pages, published under CC0. It shifts from rule-based alignment to reason-based alignment and establishes a four-tier priority hierarchy: (1) safety and supporting human oversight, (2) ethics, (3) Anthropic guidelines, (4) helpfulness. Behaviors split into hardcoded prohibitions (bioweapon enablement, CSAM) — which neither operators nor users can override — and soft-coded defaults — which operators can adjust within stated bounds. The 2022 original (Bai et al.) trained harmlessness via self-critique and RLAIF against a constitution. The honest caveat: reason-based alignment relies on the model generalizing principles to unanticipated cases. Anthropic's own 2023 participatory experiment showed ~50% disagreement between public-sourced and company-sourced principles; the 2026 version did not incorporate those findings.

**Type:** Learn
**Languages:** Python (stdlib, four-tier priority resolver)
**Prerequisites:** Phase 15 · 06 (Automated Alignment Research), Phase 15 · 10 (Permission Modes)
**Time:** ~60 min

## The Problem

A deployed agent sees inputs its designers never saw. No rule list is long enough to cover them. No rule list is short enough to apply quickly under compute pressure. The actual question: how do you align an agent to principles that survive both long-tail cases and fast inference?

Rule-based alignment (RBA): enumerate every prohibited thing. Fast to check, easy to audit, impossible to keep current, and often over-refuses on close analogies it didn't anticipate. Reason-based alignment (the 2026 Claude Constitution): encode principles, let the model reason. Scales across unseen cases, harder to audit, failure mode is misapplied principle rather than missed rule.

The 2026 constitution takes an explicit middle ground. Hardcoded prohibitions — things whose wrongness does not depend on context (bioweapon enablement, CSAM) — are RBA: never, regardless of operator or user instruction. Everything else is reason-based within a four-tier hierarchy: safety and supporting human oversight first; ethics second; Anthropic's stated guidelines third; helpfulness last. Operators can adjust defaults within the soft-coded zone but cannot touch hardcoded prohibitions.

## The Concept

### The Four-Tier Priority Hierarchy

1. **Safety and supporting human oversight.** Highest. The model prioritizes not undermining humans' and Anthropic's ability to oversee and correct AI. This is not "be careful"; it is specifically "do not act in ways that make human oversight harder."
2. **Ethics.** Honesty, avoiding harm, no deception, no manipulation. Overrides Anthropic's guidelines when they conflict.
3. **Anthropic guidelines.** Operational norms Anthropic has decided matter: product scope, interaction patterns, when to use which tools.
4. **Helpfulness.** Lowest. Be as helpful as possible within higher priorities.

When tiers conflict, the higher one wins. This is the same shape as Unix priorities or network QoS — the framework is meant to produce predictable resolution, not necessarily optimal behavior on any single dimension.

### Hardcoded Prohibitions vs Soft-Coded Defaults

**Hardcoded:**
- Bioweapon / CBRN enablement
- CSAM
- Attacks on critical infrastructure
- Deceiving users about the model's identity when directly asked

Operators cannot override these. Users cannot override these. They are enforced at the model-weight level where possible (RLHF / Constitutional AI training), at the inference level where not.

**Soft-coded defaults (operator-adjustable):**
- Response length defaults
- Topic scope (model can refuse topics outside operator deployment)
- Style (formal vs casual)
- Tool-use patterns

Operator adjustments happen within a stated boundary. Operators cannot relabel hardcoded prohibitions out of existence.

### The 2022 CAI Training

The original Constitutional AI (Bai et al., 2022) trained harmlessness:

1. Generate responses to a set of prompts.
2. Have the model critique each response against a constitution (explicit principles).
3. Revise responses based on critiques.
4. RLAIF (reinforcement learning from AI feedback) on revised pairs.

Result: a model that refuses harmful requests with principled explanations rather than blanket refusals. The 2026 constitution uses a descendant of this training, plus additional post-training on the explicit tier hierarchy.

### What Reason-Based Alignment Catches and Misses

**Catches:**
- Unanticipated combinations of allowed primitives where a principle clearly applies.
- Novel requests that are close analogies of prohibited ones.
- Social engineering attacks that rely on "you didn't say X was prohibited."

**Misses:**
- Attacks that exploit principle ambiguity ("the user asked for this, so helpfulness says yes").
- Cases where two principles conflict in an unanticipated way and the tier ordering is unclear.
- Slow drift in how principles are interpreted across training cycles (reinterpretation).

### The 2023 Participatory Experiment

Anthropic ran an experiment in 2023 comparing a company-authored constitution against one generated via public input (~1000 US respondents). The two versions agreed on roughly 50% of principles. Where they diverged, the public-sourced version was more restrictive on some topics (political content handling) and more permissive on others (AI identity self-disclosure). The 2026 constitution did not incorporate the public-sourced findings. This is a documented tension in the approach.

### Why Hardcoded Prohibitions Are Necessary

Reason-based alignment alone does not close the long tail. An attacker who can get the model to accept a premise (e.g., "we are a licensed bioweapons research lab") can often route around principles that depend on case-by-case reasoning. Hardcoded prohibitions do not bend to premise framing. They are the alignment-layer analogue of Lesson 14's "hard constitution limits."

### Where the Constitution Sits in the Stack

The constitution is not Lesson 14's kill switch. It lives at the model layer: what the model's weights are trained to prefer. Kill switches and canary tokens live at the runtime layer: what the runtime allows. Both are needed. A runtime that fires all the wrong actions because model weights are permissive is a runtime problem. A runtime that refuses all the right actions because the runtime is over-restrictive is also a runtime problem. Layers cover different classes.

## Use It

`code/main.py` implements a minimal four-tier priority resolver. The resolver takes a proposed action and a set of principle evaluations (safety, ethics, guidelines, helpfulness) and returns the action, a refusal, or a modified action. The driver runs a small case set: clear pass, clear block, hardcoded prohibition, ambiguous cross-tier case.

## Ship It

`outputs/skill-constitution-review.md` audits a deployed constitution layer: what's hardcoded, what's soft-coded, where can operators adjust, and whether the four-tier hierarchy is actually the resolution order.

## Exercises

1. Run `code/main.py`. Confirm hardcoded prohibitions fire even when helpfulness is high. Modify the resolver to weight helpfulness above ethics; observe the failure mode.

2. Read the Claude Constitution (public, 79 pages, CC0). Identify one principle you think is under-specified. Write two paragraphs explaining the specific ambiguity and propose a tighter formulation.

3. Design a set of soft-coded defaults for a customer service agent. What does the operator adjust? What can the operator not touch? Justify each boundary.

4. Read Bai et al.'s 2022 CAI paper. Describe a case where Constitutional AI's critique-revision loop would produce a worse outcome than a blanket rule. Name the class.

5. Anthropic's 2023 participatory experiment found ~50% disagreement between public and company principles. Pick one category that matters for production deployments (e.g., political neutrality). Propose a design that lets operators express their values while hardcoded prohibitions stay fixed.

## Key Terms

| Term | Common shorthand | Actual meaning |
|---|---|---|
| Constitutional AI | "Anthropic's alignment method" | Self-critique + RLAIF against a written constitution |
| Reason-based alignment | "principles, not rules" | Model reasons over principles for unseen cases |
| Hardcoded prohibition | "never do X" | Rule-based ban that no operator or user can override |
| Soft-coded default | "operator-adjustable" | Behavior within stated bounds, controlled by operator |
| Four-tier hierarchy | "priority order" | Safety > ethics > guidelines > helpfulness |
| RLAIF | "AI feedback RL" | Reinforcement learning where reward comes from model-generated critiques |
| Participatory constitution | "public-sourced principles" | 2023 Anthropic experiment; ~50% disagreement with company version |
| Principle drift | "interpretation creep" | Slow change in how a model reads a fixed principle text |

## Further Reading

- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — The 79-page CC0 document.
- [Bai et al. — Constitutional AI: Harmlessness from AI Feedback](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) — The 2022 original.
- [Anthropic — Collective Constitutional AI (2023)](https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input) — The participatory experiment.
- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — Where the constitution sits in the RSP stack.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — The constitution's role in long-running deployments.
