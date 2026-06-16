# Data Provenance & Training Data Governance

> The EU AI Act requires GPAI to support machine-readable opt-out standards by August 2025 (via the EU Copyright Directive TDM exception). California AB 2013 (signed 2024) — generative AI training data transparency requiring developers to publish a dataset summary with 12 mandatory fields. In 2025, data protection authorities (DPAs) converged on "legitimate interest": the Irish DPC (May 21, 2025), following EDPB opinion, accepted Meta's plan to train LLMs on first-party public EU/EEA adult content with safeguards; the Cologne Higher Regional Court (May 23, 2025) dismissed an injunction; the Hamburg DPA withdrew emergency proceedings; the UK ICO (September 23, 2025) issued a positive regulatory response to LinkedIn's AI training safeguards (transparency, simplified opt-out, extended objection window) with ongoing monitoring — this is not a formal clearance. Brazil's ANPD (July 2, 2024) suspended Meta's processing on informational transparency grounds; the preventive measure was lifted on August 30, 2024 after Meta submitted a compliance plan. The key irreversibility problem: cookie consent frameworks were designed for real-time, reversible tracking; once data enters model weights, surgical erasure is impossible — there is no practically feasible GDPR right to erasure for trained neural networks. The compliance window is at collection time. Data Provenance Initiative (dataprovenance.org, Longpre, Mahari, Lee et al., "Consent in Crisis", July 2024): large-scale audit showing the AI data commons is rapidly shrinking as publishers adopt robots.txt restrictions.

**Type:** Learn
**Languages:** Python (standard library, California AB 2013 12-field scaffold generator)
**Prerequisites:** Phase 18 Lesson 24 (regulation), Phase 18 Lesson 26 (cards)
**Time:** ~60 minutes

## Learning Objectives

- Describe the 12 mandatory fields of California AB 2013 for generative AI training data transparency.
- Name the 2025 DPA positions on "legitimate interest LLM training" (Irish DPC, UK ICO, Hamburg, Cologne).
- Describe the irreversibility problem: why the GDPR right to erasure has no practically feasible counterpart for trained neural networks.
- Name the Data Provenance Initiative "Consent in Crisis" findings.

## The Problem

Training data governance is upstream of every model card (Lesson 26) and regulatory obligation (Lesson 24). In 2024-2025, the regulatory landscape converged on three principles: opt-out infrastructure, per-dataset disclosure, and legitimate interest accommodation for publicly available data. Providers who are non-compliant at collection time cannot remediate downstream.

## The Concept

### California AB 2013

Signed 2024. For systems released on or after January 1, 2022, documentation must be published on or before January 1, 2026. Section 3111(a) requires developers to publish a high-level summary of datasets used for training, with 12 statutory items:
1. The sources or owners of the datasets.
2. A description of how those datasets further the intended purpose of the AI system.
3. The number of data points in the datasets (approximate ranges accepted; dynamic datasets may estimate).
4. A description of the types of data points (label types for labeled datasets; general characteristics for unlabeled).
5. Whether the datasets contain data protected by copyright, trademark, or patent, or are entirely in the public domain.
6. Whether the datasets were purchased or licensed.
7. Whether the datasets contain personal information (per Cal. Civ. Code Section 1798.140(v)).
8. Whether the datasets contain aggregate consumer information (per Cal. Civ. Code Section 1798.140(b)).
9. Any cleaning, processing, or other modifications made by the developer, and their intended purpose.
10. The time period of data collection, and whether collection is ongoing.
11. The date the dataset was first used in development.
12. Whether the system uses or continues to use synthetic data generation.

Item 12 (synthetic data) is new relative to Gebru et al. 2018 datasheets. Item 7 (personal information) triggers California Privacy Rights Act (CPRA) obligations. The statute exempts safety/integrity, aircraft operations, and federal-only national security systems (Section 3111(b)).

### EU AI Act (Lesson 24) & TDM Opt-Out

The EU Copyright Directive's text and data mining exception allows training on publicly available content unless rights holders opt out. The EU AI Act GPAI Code of Practice copyright chapter requires GPAI providers to respect machine-readable opt-out signals (robots.txt, C2PA "No AI Training" declarations, etc.).

### 2025 DPA Convergence on Legitimate Interest

Irish DPC (May 21, 2025): following EDPB opinion, accepted Meta's plan to train on first-party public EU/EEA adult user content with safeguards. Cologne Higher Regional Court (May 23, 2025) dismissed an injunction against Meta: opt-out was sufficient. Hamburg DPA withdrew emergency proceedings for EU-wide consistency. UK ICO (September 23, 2025) issued a positive regulatory response to LinkedIn resuming AI training under similar safeguards with ongoing monitoring — not a formal clearance.

Convergence principle: legitimate interest can justify "training on public first-party content with opt-out." Consent is not required.

### Brazil's ANPD (June 2024)

Suspended Meta's processing of Brazilian user data for AI training on informational transparency grounds. Outcome differs from EU DPAs — ANPD placed transparency above the admissibility of legitimate interest.

### The Irreversibility Problem

Cookie consent was designed for real-time, reversible tracking. Training data is different: once data enters model weights, surgical erasure is impossible. Full retraining from scratch is the only complete remedy, and it is prohibitively expensive.

Partial remedies:
- **Unlearning.** Approximate removal; measured with MIA (Lesson 22).
- **Influence-function-based localization.** Identify weights most affected by the data; selective update.
- **Fine-tuning suppression.** Train the model to refuse outputs derived from the data.

None fully solves the problem. The compliance window is at collection time.

### Data Provenance Initiative

dataprovenance.org. Longpre, Mahari, Lee et al. "Consent in Crisis" (July 2024): large-scale audit of the AI training data commons. Finding: publishers are adopting robots.txt restrictions at an accelerating rate. The publicly trainable commons is rapidly shrinking. From 2023 to 2024, approximately 25% of top training sources adopted some form of restriction. Implication: future training data availability depends on new acquisition paradigms (licensing, synthetic generation, incentivized participation).

### Position Within Phase 18

Lesson 26 is model-level documentation. Lesson 27 is dataset-level governance. Together they define the transparency layer. Lesson 28 maps the research ecosystem investigating these issues.

## Use It

`code/main.py` generates a California AB 2013-compliant 12-field dataset summary scaffold for a toy dataset. You can fill in the fields and observe which ones trigger privacy or copyright follow-on obligations.

## Ship It

This lesson produces `outputs/skill-provenance-check.md`. Given a dataset used for training, it checks AB 2013 12-field coverage, opt-out infrastructure compliance, DPA consistency, and irreversibility risk assessment.

## Exercises

1. Run `code/main.py`. Produce a 12-field summary for a toy dataset and identify which fields are insufficiently described.

2. The EU Copyright Directive's TDM opt-out is machine-readable. Propose a standard format for opt-out signals and compare it to robots.txt and C2PA "No AI Training."

3. Read the Data Provenance Initiative's "Consent in Crisis" (July 2024). Describe the three content categories restricting fastest and argue one economic consequence.

4. The 2025 DPA convergence accepts legitimate interest for public content training. Construct a scenario where legitimate interest is insufficient and identify the alternative legal basis the provider needs.

5. Sketch a training data provenance checklist that combines AB 2013 fields with a C2PA-signed provenance chain for each dataset. Identify one technical barrier and one legal barrier.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| AB 2013 | "that California law" | Generative AI training data transparency; 12 mandatory fields |
| TDM exception | "text and data mining" | EU Copyright Directive training data exception with opt-out |
| Legitimate interest | "that EU basis" | GDPR Article 6 basis that can justify "training on public content" |
| Opt-out signal | "machine-readable do-not-train" | robots.txt, C2PA "No AI Training", TDM.Reservation |
| Irreversibility | "can't un-train" | Data in model weights cannot be surgically removed |
| Unlearning | "approximate removal" | Post-training intervention to reduce model reliance on specific data |
| Consent in Crisis | "the DPI audit" | July 2024 finding that robots.txt restrictions are accelerating |

## Further Reading

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — generative AI training data transparency law
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — copyright chapter
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — DPI audit
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — regulatory context
