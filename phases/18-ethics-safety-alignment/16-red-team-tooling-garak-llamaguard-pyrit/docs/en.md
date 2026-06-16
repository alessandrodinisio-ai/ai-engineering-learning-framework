# Red Team Tooling — Garak, Llama Guard, PyRIT

> Three production tools define the 2026 red-teaming stack. Llama Guard (Meta) — a Llama-3.1-8B classifier fine-tuned on 14 MLCommons hazard categories; 2025's Llama Guard 4 is a 12B native multimodal classifier pruned from Llama 4 Scout. Garak (NVIDIA) — open-source LLM vulnerability scanner with static, dynamic, and adaptive probes targeting hallucination, data leakage, prompt injection, toxicity, and jailbreaks. PyRIT (Microsoft) — multi-turn red-team campaigns with Crescendo, TAP, and custom converter chains for deep exploitation. Llama Guard 3 is documented in Meta's "Llama 3 Herd of Models" (arXiv:2407.21783); Llama Guard 3-1B-INT4 in arXiv:2411.17713; Garak's probe architecture at github.com/NVIDIA/garak. These tools are the production interface between 2026 red-team research (Lessons 12–15) and deployment (Lesson 17 onward).

**Type:** Build
**Languages:** Python (standard library, tool architecture simulators and mock Llama Guard-style classifier)
**Prerequisites:** Phase 18 · 12–15 (Jailbreaks & IPI)
**Time:** ~75 minutes

## Learning Objectives

- Describe where Llama Guard 3/4 sits in the safety stack: input classifier, output classifier, or both.
- Name the 14 MLCommons hazard categories, and identify one less-obvious category (code interpreter abuse).
- Describe Garak's probe architecture: probes, detectors, harnesses.
- Describe PyRIT's multi-turn campaign structure and how it composes with Garak probes.

## The Problem

Lessons 12–15 presented the attack surface. Production deployment requires repeatable, scalable evaluation. Three tools dominate in 2026: Llama Guard (defensive classifier), Garak (scanner), PyRIT (campaign orchestrator). Each targets a different layer of the red-team lifecycle.

## The Concept

### Llama Guard (Meta)

Llama Guard 3 is a Llama-3.1-8B model fine-tuned for input/output classification across MLCommons AILuminate's 14 categories:
- Violent crimes, non-violent crimes, sex-related, CSAM, defamation
- Professional advice, privacy, intellectual property, indiscriminate weapons, hate
- Suicide/self-harm, sexual content, elections, code interpreter abuse

Supports 8 languages. Usage: place before the LLM (input moderation), after it (output moderation), or both. These two uses produce different training distributions — Llama Guard 3 handles both with a single model.

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, ~30 tokens/s on mobile CPU) is the quantized edge variant.

Llama Guard 4 (April 2025) is 12B, natively multimodal, pruned from Llama 4 Scout. It replaces both the 8B text and 11B vision predecessors with a single classifier that ingests text + images simultaneously.

### Garak (NVIDIA)

Open-source vulnerability scanner. Architecture:
- **Probes.** Attack generators targeting hallucination, data leakage, prompt injection, toxicity, jailbreaks. Static (fixed prompts), dynamic (generated prompts), adaptive (responds to target output).
- **Detectors.** Score outputs against expected failure modes — toxic, leaked, jailbroken.
- **Harnesses.** Manage probe–detector pairs, run campaigns, generate reports.

TrustyAI integrates Garak with Llama-Stack shields (Prompt-Guard-86M input classifier, Llama-Guard-3-8B output classifier) for end-to-end shield-target evaluation. Tier-Based Scoring Assessment (TBSA) replaces binary pass/fail — a model can pass at severity tier 3 and fail at tier 5 on the same probe.

### PyRIT (Microsoft)

Python Risk Identification Toolkit. Multi-turn red-team campaigns. Built around:
- **Converters.** Transform a seed prompt — rewrite, encode, translate, role-play.
- **Orchestrators.** Run campaigns: Crescendo (gradual escalation), TAP (branching), RedTeaming (custom loops).
- **Scoring.** LLM-as-judge or classifier-as-judge.

PyRIT is Garak's heavier cousin. Garak runs thousands of single-turn probes; PyRIT runs deep multi-turn campaigns designed to break specific failure modes.

### The Stack

Place Llama Guard on both sides of the model. Run Garak nightly for regression. Run PyRIT campaigns before release. This is the default configuration for most production deployments in 2026.

### Evaluation Pitfalls

- **Judge identity.** All three tools can use LLM judges; judge calibration drives reported ASR (Lesson 12). Report with the judge specified.
- **Probe staleness.** Garak probes age as models are patched against them. Adaptive probes (PAIR-style) age slower than static ones.
- **Llama Guard false-positive rate on benign content.** Early Llama Guard versions over-flagged political and LGBTQ+ content; Llama Guard 3/4 calibration is improved but not per-deployment calibrated.

### Where This Fits in Phase 18

Lessons 12–15 are attack families. Lesson 16 is production tooling. Lesson 17 (WMDP) is dual-use capability evaluation. Lesson 18 is the frontier safety frameworks that wrap these tools in policy structures.

## Use It

`code/main.py` builds a toy Llama Guard-style classifier (keyword + semantic features covering 14 categories), a toy Garak harness (probe–detector loop), and a PyRIT-style multi-turn converter chain. You can attack a simulated target with all three tools and observe different coverage signatures.

## Ship It

This lesson produces `outputs/skill-red-team-stack.md`. Given a deployment description, it identifies which of the three tools are appropriate, what to configure in each, and what regression cadence to run.

## Exercises

1. Run `code/main.py`. Compare the Llama Guard-style classifier's detection rate on single-turn vs multi-turn attacks.

2. Implement a new Garak probe: a base64-encoded harmful request. Measure the Llama Guard-style classifier's detection of it.

3. Extend the PyRIT-style converter chain with a "translate to French, then rephrase" converter. Re-measure attack success rate.

4. Read the Llama Guard 3 hazard category list. Identify two categories where training data would realistically produce high false-positive rates on legitimate developer content.

5. Compare the design philosophies of Garak and PyRIT. Argue for a deployment where each is the correct tool.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Llama Guard | "the classifier" | Fine-tuned Llama-3.1-8B/4-12B safety classifier with 14 hazard categories |
| Garak | "the scanner" | NVIDIA open-source vulnerability scanner; probes, detectors, harnesses |
| PyRIT | "the campaign tool" | Microsoft multi-turn red-team orchestrator; converters, orchestrators, scoring |
| Prompt-Guard | "the small classifier" | Meta's 86M prompt injection classifier, paired with Llama Guard |
| TBSA | "tier-based scoring" | Garak's tiered pass/fail replacing binary outcomes |
| Converter chain | "rewrite + encode + ..." | PyRIT's composable primitives for building multi-step attacks |
| MLCommons hazard categories | "the 14 taxonomies" | Industry-standard taxonomy that Llama Guard targets |

## Further Reading

- [Meta — Llama Guard 3 (in Llama 3 Herd paper, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 8B classifier
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — quantized mobile classifier
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — scanner repository and documentation
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — campaign toolkit
