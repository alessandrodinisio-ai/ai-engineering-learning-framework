# Regulatory Frameworks — EU, US, UK, Korea

> Four major regulatory regimes define the AI governance landscape in 2026. EU AI Act (entered into force 1 August 2024) — prohibited practices and AI literacy apply from 2 February 2025; GPAI obligations from 2 August 2025; full application and Article 50 transparency from 2 August 2026; legacy GPAI and embedded high-risk systems from 2 August 2027; fines up to €15M or 3% of global turnover. GPAI Code of Practice (10 July 2025): three pillars — transparency, copyright, safety & security — with 12 commitments total; enforcement from August 2026. UK AISI -> AI Security Institute (February 2025): the rename signals scope narrowing. US AISI -> CAISI (June 2025): Center for AI Standards and Innovation under NIST; pivot to pro-growth posture. South Korea AI Framework Act (passed December 2024, effective January 2026): Article 12 establishes an AISI under MSIT; requires foreign AI companies to appoint local representatives, conduct risk assessments, and adopt safety measures for high-impact and generative AI.

**Type:** Learn
**Languages:** None
**Prerequisites:** Phase 18 · 18 (frontier frameworks), Phase 18 · 27 (data governance)
**Time:** ~75 minutes

## Learning Objectives

- Describe the EU AI Act's risk tiers (prohibited, high-risk, general-purpose, limited-risk) and the 2025 Aug / 2026 Aug / 2027 Aug timeline.
- Describe the GPAI Code of Practice's three pillars and which providers each pillar binds.
- Describe the 2025 renames: UK AISI -> AI Security Institute; US AISI -> CAISI; the policy direction each rename implies.
- State the core provisions of South Korea's AI Framework Act.

## The Problem

Lab frameworks (Lesson 18) are voluntary. Regulatory frameworks are mandatory. The 2024–2026 period is when the first wave of comprehensive AI regulation takes effect. Deployers must map technical controls to regulatory obligations; that mapping differs by jurisdiction.

## The Concept

### EU AI Act

**Entered into force 1 August 2024.** Risk-tiered structure:

- **Prohibited practices** (Article 5). Social scoring, real-time remote biometric identification in public spaces (with law enforcement exceptions), exploitative manipulation of vulnerable groups. Applies from 2 February 2025.
- **High-risk systems** (Annex III). Employment, education, credit, law enforcement, justice, immigration. Require conformity assessments, risk management, logging, transparency.
- **General-Purpose AI (GPAI) models**. Applies from 2 August 2025. All GPAI providers have obligations; systemic-risk GPAI (training compute >1e25 FLOP) have additional obligations.
- **Limited-risk systems**. Transparency obligations under Article 50 (AI-generated content labeling). Applies from 2 August 2026.

Timeline:
- 2 February 2025: Prohibited practices + AI literacy.
- 2 August 2025: GPAI + governance.
- 2 August 2026: Full application + Article 50 transparency + fines up to €15M / 3% global turnover.
- 2 August 2027: Legacy GPAI + embedded high-risk.

The Commission proposed adjusting the high-risk timeline to 16 months in late 2025.

### GPAI Code of Practice

Published 10 July 2025. Three pillars:

- **Transparency.** All GPAI providers.
- **Copyright.** All GPAI providers.
- **Safety & Security.** Systemic-risk GPAI providers (estimated 5–15 companies).

12 commitments total. A signatory working group chaired by the AI Office manages implementation. Enforcement from 2 August 2026; good-faith compliance accepted before then.

### Article 50 Transparency Code of Practice

First draft 17 December 2025. Second draft March 2026. Finalization June 2026. Covers AI-generated content labeling including deepfakes — this is the regulatory layer requiring the watermarking technology of Lesson 23.

### UK AI Security Institute (February 2025)

Renamed from AI Safety Institute. The rename narrows scope: drops algorithmic bias and freedom-of-expression framing; focuses on frontier capability safety. Open-sourced the Inspect evaluation tool (May 2024). Partnered with Redwood (Lesson 10) on control safety cases.

### US CAISI (June 2025)

The Trump administration converted NIST's AI Safety Institute into the Center for AI Standards and Innovation. Per VP Vance's remarks at the Paris AI Action Summit, the pivot is toward "pro-growth AI policy." De-emphasizes pre-deployment evaluation; emphasizes standards and innovation support. A domestic counterweight to the EU AI Act's regulatory posture.

### South Korea AI Framework Act

Passed December 2024. Promulgated January 2025. Effective January 2026. Consolidates 19 separate AI bills.

Article 12 establishes an AISI under the Ministry of Science and ICT (MSIT). Requirements:
- Foreign AI companies operating in Korea appoint local representatives.
- Risk assessments for "high-impact" AI systems.
- Safety measures for generative AI and high-impact AI.

First jurisdiction in Asia with comprehensive horizontal AI regulation.

### Cross-Jurisdictional Dynamics

- EU: Strict, risk-tiered, heavy fines. Benchmark for privacy-adjacent regulation.
- US: Innovation-favoring, decentralized, with states (e.g., California AB 2013 — Lesson 27) filling the federal gap.
- UK: Narrow safety focus, strong evaluation infrastructure.
- Korea: MSIT-led, focused on foreign providers.

Competing regulatory philosophies. Deployers in multiple jurisdictions must comply with the strictest, which in 2026 is typically the EU AI Act.

### Position in Phase 18

Lesson 18 is voluntary lab governance; Lesson 24 is regulation; Lesson 25 is emerging CVEs for AI systems; Lessons 26–27 cover documentation (cards) and training data governance.

## Build It

No code. Read the EU AI Act primary sources: regulation text, GPAI Code of Practice, UK AISI Inspect framework. Map your deployment to the applicable obligations in each jurisdiction.

## Use It

This lesson produces `outputs/skill-regulatory-map.md`. Given a deployment description, it maps: applicable jurisdictions, tier classification in each, per-jurisdiction obligations, and deadline structure.

## Exercises

1. Read the EU AI Act (Regulation 2024/1689) and the GPAI Code of Practice (10 July 2025). Identify three obligations that apply to every GPAI provider and three that apply only to systemic-risk GPAI.

2. A deployment is made by a US company, runs on EU infrastructure, and serves Korean users. Which three jurisdictions' rules apply, and which rule governs on each substantive issue?

3. The UK AI Security Institute's rename narrows scope. Argue for and against the narrower framing. Identify the policy assumptions each position depends on.

4. CAISI's "pro-growth" framing is a departure from the 2022–2024 AI safety institute model. Identify two measurable policy shifts that would follow from this framing.

5. South Korea's AI Framework Act requires foreign providers to appoint local representatives. Describe the operational impact for a Bay Area company serving Korean users.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| EU AI Act | "The regulation" | Risk-tiered horizontal AI regulation; effective August 2024 |
| GPAI | "General-purpose AI" | Large foundation models; systemic-risk subset has additional obligations |
| Article 50 | "Transparency obligations" | AI-generated content labeling; applies August 2026 |
| UK AISI | "AI Security Institute" | Renamed February 2025; narrower frontier safety focus |
| CAISI | "US AI standards center" | Renamed from AI Safety Institute June 2025; pro-growth posture |
| Korea AI Framework Act | "MSIT horizontal regulation" | Asia's first comprehensive AI law; effective January 2026 |
| Systemic-risk GPAI | "The 1e25 FLOP threshold" | Additional obligations tier; estimated to bind 5–15 companies |

## Further Reading

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — regulation and timeline
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice) — three-pillar code
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute) — official page
- [CSET — South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) — Korea framework analysis
