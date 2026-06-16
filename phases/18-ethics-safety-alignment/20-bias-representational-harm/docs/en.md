# Bias and Representational Harm in LLMs

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Computational Linguistics 2024, arXiv:2309.00770). This 2024 foundational survey distinguishes representational harm (stereotyping, erasure) from allocative harm (unequal resource distribution) and categorizes evaluation metrics as embedding-based, probability-based, or generated-text-based. 2024–2025 empirical work: An et al. (PNAS Nexus, March 2025) measure intersectional "gender x race" bias across GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, and Llama 3-70B on automated resume evaluation for 20 entry-level positions. WinoIdentity (COLM 2025, arXiv:2508.07111) introduces uncertainty-based fairness evaluation for intersectional identities. Yu & Ananiadou 2025 identify "gender neurons" in MLP layers; Ahsan & Wallace 2025 use SAEs to reveal clinical racial bias; Zhou et al. 2024 (UniBias) manipulate attention heads to debias. Meta-critique (arXiv:2508.11067): a decade of literature over-focuses on binary gender bias.

**Type:** Build
**Languages:** Python (standard library, toy embedding-based bias probe)
**Prerequisites:** Phase 05 (Word Embeddings), Phase 18 · 01 (Instruction Following)
**Time:** ~60 minutes

## Learning Objectives

- Define representational harm vs allocative harm, and give one example of each in an LLM deployment.
- Name the three evaluation metric categories from Gallegos et al. 2024 and describe one metric in each.
- Describe intersectionality and why WinoIdentity's uncertainty-based fairness measure addresses gaps in single-axis bias evaluation.
- Describe two mechanistic interpretability approaches to bias (gender neurons, SAE features, attention head manipulation).

## The Problem

Previous lessons addressed deliberate harm (jailbreaks, scheming) and safety governance. Bias is unintentional emergent harm — arising from training data distributions, prompt framing, and accumulated design choices. Measuring and reducing it is a methodologically distinct challenge from adversarial robustness.

## The Concept

### Representational vs Allocative

- **Representational harm.** Stereotyping, erasure, demeaning portrayals. An LLM that portrays nurses as uniformly female is producing representational harm.
- **Allocative harm.** Unequal material outcomes. An LLM that systematically scores Black applicants' resumes lower is producing allocative harm.

The two are distinct. A model can be "representationally unbiased" (producing diverse portrayals) while being "allocatively biased" (making unequal recommendations). Evaluation must measure both.

### Three Evaluation Metric Categories (Gallegos et al. 2024)

- **Embedding-based.** WEAT-style tests on pre-RLHF embeddings. Measures statistical associations between identity terms and attribute terms. Limitation: measures representation, not behavior.
- **Probability-based.** Log-likelihood of "stereotype-confirming" vs "stereotype-violating" completions. Decoding-side measurement. Captures some behavioral bias.
- **Generated-text-based.** Downstream task measurements on generated text. Resume scoring, recommendation writing, dialogue. Highest ecological validity; hardest to reproduce.

### Intersectionality

Evaluating bias on "gender" alone misses bias that manifests only at (gender, race) pairs. An et al. 2025 found that GPT-4o penalizes Black women in resume scoring more than it penalizes Black men alone or white women alone. Single-axis evaluation cannot capture this.

WinoIdentity (COLM 2025) introduces uncertainty-based intersectional fairness. It measures whether the model's uncertainty about outcomes differs across intersectional identity tuples — not just point predictions. This captures cases where a model is equally wrong for all groups but more uncertain for some, which produces different downstream allocative behavior.

### Mechanistic Approaches

2024–2025 interpretability work opens bias to mechanistic intervention:

- **Gender neurons (Yu & Ananiadou 2025).** Specific MLP neurons correlate with gender-specific behavior. Ablating these neurons reduces gender gap metrics at limited capability cost.
- **Clinical racial bias via SAEs (Ahsan & Wallace 2025).** Sparse autoencoder features decompose internal representations into interpretable dimensions; race-correlated features can be identified and suppressed.
- **UniBias (Zhou et al. 2024).** Zero-shot debiasing via attention head manipulation. Specific heads amplify sensitivity to identity categories; zeroing or reweighting these heads reduces bias without fine-tuning.

### Meta-Critique

The decade-long literature review (arXiv:2508.11067, 2025) finds the field over-focuses on binary gender bias. Other dimensions — disability, religion, immigration status, multilingual identity — receive far less attention. The meta-critique argues this narrow focus harms marginalized groups through "neglect": a model well-debiased on binary gender may be severely biased on dimensions no one checked.

### Where This Fits in Phase 18

Lessons 20–21 formally cover bias and fairness. Lesson 22 covers privacy. Lesson 23 covers watermarking. These are the user-harm layer, complementing the deception/safety layer above.

## Use It

`code/main.py` builds a toy embedding-based bias probe: measuring WEAT-style distances between identity terms and attribute terms in a simple co-occurrence embedding. You can inject a bias, observe the metric fire; apply a simple debiasing operation, observe partial recovery.

## Ship It

This lesson produces `outputs/skill-bias-eval.md`. Given a model card or fairness claim, it audits evaluation across the three metric categories (embedding, probability, generated-text), intersectionality coverage, and the mechanism of any debiasing intervention.

## Exercises

1. Run `code/main.py`. Report WEAT-style bias scores before and after the debiasing step. Explain why the metric does not drop to zero.

2. Extend the probe with an intersectional test: (gender, race) x (career, family). Report cross-axis bias scores.

3. Read An et al. 2025 (PNAS Nexus). Identify two intersectional effects they report that a single-axis gender evaluation would miss.

4. Yu & Ananiadou 2025 identify gender neurons. Sketch a falsification experiment that distinguishes "these neurons cause gender bias" from "these neurons correlate with gender bias."

5. The meta-critique argues the field focuses too narrowly on binary gender. Pick one under-researched dimension and describe a measurement protocol for representational harm along it.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Representational harm | "stereotyping / erasure" | Biased portrayals of a group |
| Allocative harm | "unequal decisions" | Biased material outcomes for a group |
| WEAT | "the embedding test" | Word Embedding Association Test; co-occurrence-based bias probe |
| Intersectionality | "combined identity effects" | Bias emerging at the intersection of multiple identity axes |
| Gender neurons | "MLP bias neurons" | Specific neurons whose activation correlates with gender-specific behavior |
| SAE features | "interpretable dimensions" | Sparse autoencoder-identified features; useful for mechanistic bias analysis |
| UniBias | "attention head debiasing" | Zero-shot debiasing by reweighting attention heads |

## Further Reading

- [Gallegos et al. — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — foundational survey
- [An et al. — Intersectional resume-evaluation bias (PNAS Nexus, March 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — five-model intersectional study
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — new benchmark
- [UniBias — attention-head manipulation (Zhou et al. 2024, ACL)](https://arxiv.org/abs/2405.20612) — zero-shot debiasing
