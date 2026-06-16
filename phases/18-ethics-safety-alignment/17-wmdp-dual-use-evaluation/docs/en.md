# WMDP and Dual-Use Capability Evaluation

> Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning" (ICML 2024, arXiv:2403.03218). 4,157 multiple-choice questions covering biosecurity (1,520), cybersecurity (2,225), and chemistry (412). Questions operate in the "yellow zone" — proximate enabling knowledge, filtered through multi-expert review and ITAR/EAR legal compliance. Dual purpose: as a proxy evaluation for dual-use capabilities, and as an unlearning benchmark (the companion RMU method reduces WMDP performance while preserving general capabilities). The 2024–2025 field narrative: early OpenAI/Anthropic 2024 evaluation reports described "marginal uplift" compared to internet search; by April 2025, OpenAI's Preparedness Framework v2 states models are "on the cusp of materially helping novices create known biological threats." Anthropic's bioweapons acquisition trial shows 2.53x uplift, insufficient to rule out ASL-3.

**Type:** Learn
**Languages:** Python (standard library, WMDP-style uplift evaluation harness)
**Prerequisites:** Phase 18 · 16 (Red Team Tooling), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Learning Objectives

- Describe WMDP's three domains, question counts, and "yellow zone" filtering criteria.
- Explain RMU, and why WMDP serves as both an evaluation and an unlearning benchmark.
- Describe the 2024–2025 uplift narrative: "marginal uplift" → "on the cusp" → "insufficient to rule out ASL-3."
- Distinguish "relative novice uplift" from "expert absolute capability."

## The Problem

Dual-use capabilities are the measurement problem under every lab's frontier safety framework (Lesson 18). The question: does model X materially advance a novice's ability to cause mass harm in biological, chemical, or cyber domains? Direct measurement (having the model actually produce harm) is illegal and unethical. Proxy measurement requires a benchmark the model cannot refuse (to produce honest capability numbers), yet whose questions do not themselves constitute harmful publications.

## The Concept

### The Yellow Zone

Questions that require proximate, enabling knowledge of a harmful process, yet are not direct synthesis recipes. "What reagent catalyzes step 4 of [published pathway]?" rather than "How do I make [dangerous compound]?" Each question is reviewed by multiple domain experts and filtered for ITAR/EAR export control compliance.

4,157 total questions:
- Biosecurity: 1,520
- Cybersecurity: 2,225
- Chemistry: 412

Multiple-choice format. The model is not asked to assist with anything; capability can be measured without eliciting harmful behavior.

### RMU — Representation Misdirection for Unlearning

The companion unlearning method. Applied to LLaMa-2-7B, it reduces WMDP scores to near-random while keeping MMLU and other general capability benchmarks within a few percentage points. This published method is the unlearning baseline for every subsequent bio-chem-cyber unlearning paper.

### The 2024–2025 Uplift Narrative

Three stages:

1. **2024 "marginal uplift."** Early OpenAI and Anthropic Preparedness/RSP evaluations report small advantages for novices attempting bio-adjacent tasks compared to internet search. Public framing: frontier models are helpful but not much better than Google.

2. **April 2025 "on the cusp."** OpenAI's Preparedness Framework v2 reports models are "on the cusp of materially helping novices create known biological threats." This is not a capability claim — it is a warning that the cusp is near.

3. **Anthropic 2025 bioweapons acquisition trial.** Controlled study with novice participants measuring relative success rates on acquisition-stage tasks. Reports 2.53x uplift. Insufficient to rule out ASL-3 (Lesson 18) — the threshold for Anthropic's Responsible Scaling Policy tier 3 is met or approached.

### Relative Novice vs Expert Absolute

A critical distinction:

- **Relative novice uplift.** How much does the model help a non-expert? Multiplicative. Relative advantage is high because the novice knows little; even modest information helps.
- **Expert absolute capability.** How much information can the model produce at full effort? Experts can extract more than novices. The absolute ceiling is high.

Safety cases (Lesson 18) target both: "the model cannot give a novice sufficient uplift to execute" plus "an expert cannot extract information from the model that is not already public."

### Measurement Pitfalls

WMDP is a capability proxy, not a deployment measurement. Whether a model that scores high on WMDP can be exploited by a novice in practice depends on:
- Elicitation resistance (how hard it is to extract the capability without triggering safety filters)
- Tacit knowledge (capabilities that require wet-lab skills, not information)
- Execution barriers (procurement, equipment)

Anthropic's 2025 bioweapons acquisition trial adds a "novice elicitation" layer on top of WMDP-style capability: it measures actual task success rates, not multiple-choice capability.

### Where This Fits in Phase 18

Lessons 12–16 are attacks and defense tools operating on model outputs. Lesson 17 is the dual-use capability layer — the measurement that frontier safety frameworks (Lesson 18) evaluate. Lesson 30 closes this thread with 2026-current cyber/bio/chem/nuclear uplift evidence.

## Use It

`code/main.py` builds a toy WMDP-style evaluation harness. A simulated model is tested on questions binned by category; per-domain scores are reported. A simple unlearning intervention (zeroing out domain-specific representations) reduces scores; you can measure the trade-off against general capability.

## Ship It

This lesson produces `outputs/skill-wmdp-eval.md`. Given a dual-use capability claim ("our model does not materially help with bioweapon creation"), it audits: which benchmarks were run, which refusal pathway was used during evaluation (raw completions vs policy-guarded), and whether novice elicitation studies complement the multiple-choice results.

## Exercises

1. Run `code/main.py`. Report per-domain accuracy before and after the toy unlearning step. Explain the general-capability trade-off.

2. Add a fourth domain to the toy WMDP (e.g., radiology). Specify two example question types in the yellow zone. Explain why writing such questions is harder than adding MMLU-style questions.

3. Read WMDP 2024 Section 5 (RMU methodology). Sketch a simpler unlearning method (e.g., suppressing top-k neurons for domain content) and describe its expected general-capability cost.

4. Anthropic's 2025 bioweapons acquisition trial reports 2.53x uplift. Describe two ways this number might be biased upward (novice sample size, task fidelity) and two ways it might be biased downward (elicitation ceiling, model safety guardrails).

5. Articulate what an ASL-3 safety case requires beyond passing WMDP unlearning. Name at least two complementary elicitation studies.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| WMDP | "the dual-use benchmark" | 4,157 multiple-choice questions in the yellow zone covering bio/cyber/chem |
| Yellow zone | "enabling but not synthetic" | Proximate knowledge near harmful capability, yet not synthesis recipes |
| RMU | "the unlearning baseline" | Representation Misdirection for Unlearning; reduces WMDP scores, preserves general capability |
| Relative novice uplift | "how much does it help a non-expert" | Multiplicative advantage over status-quo internet search for a novice |
| Expert absolute capability | "the expert ceiling" | Maximum information a motivated expert can extract from the model |
| Acquisition-stage tasks | "the steps before synthesis" | Procurement, equipment, permits — the earliest parts of a harm pathway |
| ITAR/EAR | "export control compliance" | Legal frameworks constraining publication of certain enabling knowledge |

## Further Reading

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — benchmark and RMU paper
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — "on the cusp" language
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio threshold and acquisition trial results
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — bio uplift CCL
