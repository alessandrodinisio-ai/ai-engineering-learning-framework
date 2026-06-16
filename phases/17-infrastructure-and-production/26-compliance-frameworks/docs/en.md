# Compliance — SOC 2, HIPAA, GDPR, PCI-DSS, EU AI Act, ISO 42001

> Multi-framework coverage is the table stakes for enterprise deals in 2026. **EU AI Act**: in force since August 1, 2024. Most high-risk requirements become enforceable on August 2, 2026. Fines for violating high-risk system obligations (Article 99(4)) up to EUR 15 million or 3% of global annual turnover; prohibited AI practices (Article 99(3)) up to EUR 35 million or 7%. Applies globally as long as you serve EU users. **Colorado AI Act**: effective June 30, 2026 (delayed from February 2026 by SB25B-004) — impact assessments for high-risk systems, right to appeal AI decisions. Virginia is similar for credit/employment/housing/education. **SOC 2 Type II**: the de facto B2B AI requirement (fintech demands Type II, not Type I). **GDPR**: the largest recorded AI-specific fine is EUR 30.5 million against Clearview AI (Dutch DPA, September 2024); Italy's Garante issued EUR 15 million against OpenAI in December 2024 (later overturned on appeal in March 2026). Real-time PII redaction at inference is the defensible standard; post-processing cleanup is insufficient. **HIPAA**: healthcare constraint — you cannot send PHI to an external AI service without a BAA. **PCI-DSS**: AI interaction layer coverage requires configuration + contractual agreement, it is not automatic. **ISO 42001**: emerging AI governance standard, paired with ISO 27001, increasingly required in procurement. Reference profile: OpenAI maintains SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, and PCI-DSS for ChatGPT payment components. Cross-framework mapping reduces audit fatigue: access control spans ISO 27001 A.5.15-5.18, GDPR Article 32, HIPAA §164.312(a).

**Type:** Learn
**Languages:** (Python optional — compliance is policy + process, not code)
**Prerequisites:** Phase 17 · 25 (Security), Phase 17 · 13 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- List the seven 2026 frameworks relevant to LLM products and map each to a customer segment.
- Cite the EU AI Act enforcement timeline (in force August 2024; high-risk enforcement August 2026) and the two fine tiers (high-risk obligations EUR 15M / 3%, prohibited practices EUR 35M / 7%).
- Explain why post-processing PII cleanup is insufficient for GDPR and name real-time inference-layer redaction as the defensible standard.
- Describe cross-framework control mapping (e.g., access control maps to ISO 27001 A.5.15-5.18 + GDPR Article 32 + HIPAA §164.312(a)).

## The Problem

An enterprise customer's procurement requires SOC 2 Type II, GDPR, HIPAA BAA, ISO 27001, and an "EU AI Act compliance statement." Your team has SOC 2 Type I. You are six months away from Type II, and GDPR Article 30 records have not even started.

Multi-framework coverage is not an LLM problem — it is an enterprise SaaS problem with LLM-specific overlays. Procurement teams in 2026 want a matrix with one row per framework and one column per control, not a PDF.

## The Concept

### Seven Frameworks

| Framework | Scope | LLM-Specific Requirement |
|-----------|-------|--------------------------|
| SOC 2 Type II | B2B SaaS baseline | Process controls audited over 6-12 months |
| HIPAA | US healthcare | BAA required; PHI cannot leave infrastructure without a signed agreement |
| GDPR | EU users | Real-time PII redaction; data subject rights; Article 30 records |
| PCI-DSS | Payment data | AI touching payments requires configuration + contract |
| EU AI Act | Serving EU users | Risk classification; high-risk systems: conformity assessment, documentation, logging |
| Colorado AI Act | Serving Colorado residents | Impact assessments; right to appeal |
| ISO 42001 | AI governance | Emerging; paired with ISO 27001 |

### EU AI Act Timeline

- August 1, 2024: Enters into force.
- February 2, 2025: Prohibited AI practices enforcement.
- August 2, 2026: High-risk system enforcement (conformity assessment, documentation, logging).
- August 2027: High-risk systems in products governed by Union harmonization legislation.

Risk classification: Unacceptable (prohibited), High-risk (conformity assessment + logging), Limited risk (transparency), Minimal risk (no constraints). Most B2B LLM SaaS falls under limited risk; high-risk triggers for employment, credit, education, law enforcement, immigration, and essential services.

Fines (Article 99): Violation of high-risk system obligations up to EUR 15 million or 3% of global annual turnover (Article 99(4)); prohibited AI practices up to EUR 35 million or 7% (Article 99(3)); whichever is higher applies.

### GDPR — Real-Time Redaction Is the Standard

Post-processing cleanup (redacting PII after the LLM has already seen the data) is not a defensible posture — the model has already seen that data. Real-time inference-layer redaction is the 2026 standard:

- Entity recognition before the LLM call.
- Consistent tokenization (mesh approach) to preserve semantics.
- Store only redacted prompts + opt-in raw data with consent.

Recent enforcement: The EUR 30.5 million fine against Clearview AI (Dutch DPA, September 2024) is the largest recorded AI-specific GDPR fine to date; the EUR 15 million fine against OpenAI (Italy's Garante, December 2024) is the largest LLM-specific fine, although it was overturned on appeal in March 2026 and the ruling remains under further review. Post-processing arguments have not held up in audits.

### HIPAA — BAA Is Not Optional

Without a signed Business Associate Agreement (BAA), you cannot send PHI to an external AI service. All three hyperscaler LLM platforms (Bedrock, Azure OpenAI, Vertex) offer BAAs. OpenAI's direct API offers a BAA. Anthropic's direct API offers a BAA. Confirm before sending PHI.

### SOC 2 Type II

Type I: Controls are designed and documented.
Type II: Controls have operated effectively over 6-12 months.

B2B procurement in 2026 defaults to Type II. Type I is the entry point; Type II is the gate.

Common audit drivers: access logs (who viewed what), change management (how deployments happen), risk assessment (quarterly), incident response (has it been tested?). The audit logs from Phase 17 · 25 can be directly reused.

### Cross-Framework Mapping

A single access control policy satisfies multiple framework controls:

| Control | Frameworks |
|---------|-----------|
| Access logs | ISO 27001 A.5.15-5.18, GDPR Article 32, HIPAA §164.312(a) |
| Change management | ISO 27001 A.8.32, PCI DSS Req. 6, HIPAA breach notification scope |
| Encryption in transit | ISO 27001 A.8.24, GDPR Article 32, HIPAA §164.312(e) |
| Key management | ISO 27001 A.8.19, PCI DSS Req. 8, SOC 2 CC6.1 |

Compliance tools (Drata, Vanta, Secureframe) automate this mapping. Worth the cost at scale.

### ISO 42001 — Emerging

Published late 2023. Paired with ISO 27001, increasingly required in procurement. An AI governance framework covering risk management, data quality, transparency, and human oversight.

### OpenAI's Reference Profile

OpenAI maintains SOC 2 Type 2, ISO/IEC 27001:2022, ISO/IEC 27701:2019, GDPR/CCPA/HIPAA (BAA)/FERPA, and PCI-DSS for ChatGPT payment components. This is roughly the enterprise table stakes for 2026.

### Numbers to Remember

- EU AI Act fines: up to EUR 15M / 3% (high-risk obligations, Article 99(4)); up to EUR 35M / 7% (prohibited practices, Article 99(3)).
- EU AI Act high-risk enforcement: August 2, 2026.
- Largest recorded AI-specific GDPR fine: EUR 30.5 million, Clearview AI (Dutch DPA, September 2024).
- Largest LLM-specific GDPR fine: EUR 15 million, OpenAI (Italy's Garante, December 2024; overturned on appeal March 2026).
- SOC 2 Type II window: 6-12 months of control operation.
- Colorado AI Act effective date: June 30, 2026 (delayed from February 2026 by SB25B-004).

## Use It

`code/main.py` is a compliance mapping table in Python — given a control, list the frameworks it satisfies.

## Ship It

This lesson produces `outputs/skill-compliance-matrix.md`. Given a customer segment and geography, it identifies the required frameworks and controls.

## Exercises

1. Your first enterprise customer requires SOC 2 Type II, HIPAA BAA, and an EU AI Act statement. What is the minimum viable compliance posture to win the deal?
2. Classify three hypothetical LLM products by EU AI Act risk tier. What changes when a product is high-risk?
3. You accidentally sent PHI to a vendor without a BAA. Walk through the incident response.
4. Argue whether ISO 42001 is "necessary" for a mid-market AI vendor in 2026.
5. Map your LLM audit log fields (Phase 17 · 25) to at least three framework controls.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| SOC 2 Type II | "Audited controls" | Controls operating 6-12 months, attested by an independent party |
| HIPAA BAA | "Healthcare contract" | Business Associate Agreement; required for PHI |
| GDPR | "EU privacy" | Real-time PII redaction is the defensible 2026 standard |
| EU AI Act | "EU AI rules" | High-risk enforcement August 2026; EUR 15M / 3% (high-risk obligations) — EUR 35M / 7% (prohibited practices) |
| Colorado AI Act | "US state AI law" | Effective June 30, 2026 (delayed by SB25B-004); impact assessments |
| ISO 42001 | "AI governance" | Emerging AI risk + transparency framework |
| ISO 27001 | "Security ISMS" | Information security management system baseline |
| Conformity assessment | "EU AI documentation package" | High-risk requirement: documentation, testing, logging |
| Cross-framework mapping | "One control, multiple frameworks" | A single policy satisfies multiple framework controls |

## Further Reading

- [OpenAI Security and Privacy](https://openai.com/security-and-privacy/) — reference compliance profile.
- [GuardionAI — LLM Compliance 2026: ISO 42001, EU AI Act, SOC 2, GDPR](https://guardion.ai/blog/llm-compliance-guide-iso-42001-eu-ai-act-soc2-gdpr-2026)
- [Dsalta — SOC 2 Type 2 Audit Guide 2026: 10 AI Controls](https://www.dsalta.com/resources/ai-compliance/soc-2-type-2-audit-guide-2026-10-ai-powered-controls-every-saas-team-needs)
- [EU AI Act official text](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) — primary source.
- [Colorado AI Act](https://leg.colorado.gov/bills/sb24-205) — primary source.
- [ISO/IEC 42001:2023](https://www.iso.org/standard/81230.html) — AI management system standard.
