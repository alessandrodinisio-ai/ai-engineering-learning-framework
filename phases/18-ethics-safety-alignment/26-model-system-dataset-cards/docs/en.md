# Model Cards, System Cards & Dataset Cards

> Three documentation formats form the backbone of AI transparency. Model cards (Mitchell et al. 2019) — the nutrition label for a model: training data, disaggregated quantitative analysis, ethical considerations, caveats; yet only 0.3% of model cards on Hugging Face document ethical considerations (Oreamuno et al. 2023). Datasheets for Datasets (Gebru et al. 2018, CACM) — motivation, composition, collection process, labeling, distribution, maintenance; analogous to electronic component datasheets. Data Cards (Pushkarna et al., Google 2022) — modular layered detail (telescopic, periscopic, microscopic), serving as boundary objects for diverse audiences. 2024-2025 advances: LLM-automated generation (CardGen, Liu et al. 2024); model card detail level correlates with up to 29% more downloads on HF (Liang et al. 2024); verifiable claims (Laminator, Duddu et al. 2024); sustainability reporting supplements for carbon/water (Jouneaux et al. July 2025); EU/ISO regulatory cards emerging. System Cards (Sidhpurwala 2024; Meta system-level transparency; "Blueprints of Trust" arXiv:2509.20394) — end-to-end AI system documentation covering safety capabilities, prompt injection defenses, data exfiltration detection, and alignment with human values.

**Type:** Build
**Languages:** Python (standard library, model card + datasheet + system card generator)
**Prerequisites:** Phase 18 Lesson 18 (safety frameworks), Phase 18 Lesson 24 (regulation)
**Time:** ~60 minutes

## Learning Objectives

- Describe the original Mitchell et al. 2019 model card and Gebru et al. 2018 datasheet.
- Describe the telescopic/periscopic/microscopic layering of data cards.
- Describe system cards and their end-to-end coverage.
- Name three 2024-2025 advances (automated generation, verifiable claims, sustainability reporting).

## The Problem

Regulatory frameworks (Lesson 24) and lab safety policies (Lesson 18) both require documentation. Documentation formats evolved from "model-specific" (model cards) to "dataset-specific" (datasheets) to "system-specific" (system cards). Each addresses a different scope of transparency. The 2024-2025 automation and verifiable claims work addresses the long-standing adoption problem.

## The Concept

### Model Cards (Mitchell et al. 2019)

Sections:
- Model details.
- Intended use.
- Factors (demographic or environmental factors relevant to evaluation).
- Metrics.
- Evaluation data.
- Training data.
- Quantitative analyses (disaggregated by factors).
- Ethical considerations.
- Caveats and recommendations.

Adoption problem: Oreamuno et al. 2023 audit of Hugging Face model cards found only 0.3% document ethical considerations.

### Datasheets for Datasets (Gebru et al. 2018)

Analogous to electronic component datasheets. Sections:
- Motivation (why the dataset was created).
- Composition (what is in it).
- Collection process (how it was assembled).
- Labeling (if applicable).
- Uses (intended, prohibited, risky).
- Distribution.
- Maintenance.

Published in CACM 2021. Datasheets are upstream documentation; model cards depend on accurate datasheets.

### Data Cards (Pushkarna et al., Google 2022)

Modular layered detail. Three zoom levels:
- **Telescopic.** High-level summary for non-experts.
- **Periscopic.** Mid-level overview for ML practitioners.
- **Microscopic.** Feature-level detailed documentation for auditors.

Boundary object framework: different readers extract different information from the same document.

### System Cards

Scope: end-to-end AI systems including model + safety stack + deployment context. Sections typically include:
- Safety capabilities.
- Prompt injection defenses.
- Data exfiltration detection.
- Alignment with stated human values.
- Incident response.

Sidhpurwala 2024 and Meta's system-level transparency work. "Blueprints of Trust" (arXiv:2509.20394) formalizes system cards as the deployment-layer complement to model cards.

### 2024-2025 Advances

- **CardGen (Liu et al. 2024).** LLM-automated model card generation; reports higher objectivity than many human-written cards on standardized Mitchell 2019 fields.
- **Download correlation (Liang et al. 2024).** Detailed model cards correlate with up to 29% higher download rates on HF — adoption pressure is now market-driven, not just compliance-driven.
- **Laminator (Duddu et al. 2024).** Verifiable claims via hardware TEE / cryptographic signatures — lets model cards carry a "proof of claim" rather than just a claim.
- **Sustainability (Jouneaux et al. July 2025).** Supplements for carbon, water, and compute footprint; emerging ISO standards.
- **Regulatory cards.** EU AI Act (Lesson 24) GPAI Code of Practice transparency chapter requires model cards as a compliance artifact.

### Position Within Phase 18

Lessons 24-25 are the regulatory and CVE layers. Lesson 26 is the documentation layer. Lesson 27 is training data governance, which is upstream of datasheets. Lesson 28 is the research ecosystem that produces the evaluations referenced in cards.

## Use It

`code/main.py` generates a minimal model card, datasheet, and system card for a toy deployment. Each follows the canonical section structure. You can inspect the formats and compare the three scopes.

## Ship It

This lesson produces `outputs/skill-card-audit.md`. Given a model card, datasheet, or system card, it audits section coverage, numerical disaggregation, and presence of verifiable claims.

## Exercises

1. Run `code/main.py`. Inspect the generated cards. Identify weak (placeholder-only) sections and explain what evidence would strengthen them.

2. Extend the model card with a disaggregated quantitative analysis across two demographic groups (Lesson 20).

3. Read Oreamuno et al. 2023 on the 0.3% adoption rate. Propose a structural change to the model card specification that would increase adoption of ethical considerations.

4. Laminator (Duddu et al. 2024) uses TEEs for verifiable claims. Design a model card field that carries a "cryptographic proof of evaluation results" and describe the verifier's role.

5. Write a system card (not a model card) for one of your past projects or a hypothetical deployment. Identify the section most valuable to a third-party auditor.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Model card | "the Mitchell card" | Mitchell et al. 2019 standard documentation for ML models |
| Datasheet | "the Gebru datasheet" | Gebru et al. 2018 standard documentation for datasets |
| Data card | "the Pushkarna card" | Google 2022 modular layered data documentation |
| System card | "the deployment card" | End-to-end AI system documentation including safety stack |
| Boundary object | "different readers, same doc" | Data card framework: one document serving diverse audiences |
| Verifiable claim | "the Laminator proof" | Cryptographic or TEE-backed proof attached to documentation claims |
| Sustainability field | "carbon / water footprint" | Emerging 2025 environmental accounting supplements |

## Further Reading

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — the canonical model card
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — the datasheet paper
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — layered data documentation
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — system card formalization
