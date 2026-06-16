# Frontier Safety Frameworks — RSP, PF, FSF

> Three major lab frameworks define industry governance of frontier capabilities in 2026. Anthropic's Responsible Scaling Policy v3.0 (February 2026) introduces tiered AI Safety Levels (ASL-1 through ASL-5+), modeled on biosafety levels, with ASL-3 activated for CBRN-relevant models in May 2025. OpenAI's Preparedness Framework v2 (April 2025) defines five criteria for tracked capabilities and separates Capabilities Reports from Safeguards Reports. DeepMind's Frontier Safety Framework v3.0 (September 2025) introduces Critical Capability Levels including a new "harmful manipulation" CCL. All three now include "competitor adjustment" clauses allowing deferred enforcement when peer labs ship without comparable safeguards. Cross-lab alignment remains structural rather than terminological: "capability thresholds," "high capability thresholds," and "critical capability levels" refer to similar constructs.

**Type:** Learn
**Languages:** None
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07–09 (Deception failures)
**Time:** ~75 minutes

## Learning Objectives

- Describe Anthropic's ASL tier structure and what activated ASL-3.
- Name the five criteria OpenAI's Preparedness Framework v2 defines for tracked capabilities.
- Describe DeepMind's Critical Capability Level structure and the "harmful manipulation" CCL.
- Explain competitor adjustment clauses and why they matter for race dynamics.
- Define safety cases and describe the three-pillar structure (monitoring, illegibility, incapability).

## The Problem

Lessons 7–17 established: deception is possible, dual-use capabilities exist, evaluation has limits. A lab with a frontier-capable model needs an internal governance structure that:
- Defines thresholds at which new safeguards are required.
- Defines evaluations required before scaling.
- Describes what a safety case looks like.
- Handles race dynamics (what do you do if a competitor ships without safeguards?).

These three 2025–2026 frameworks are the current state of the art — imperfect, evolving, and cross-lab aligned enough that the governance question is now "are the frameworks sufficient," not "do frameworks exist."

## The Concept

### Anthropic Responsible Scaling Policy v3.0 (February 2026)

ASL structure:
- ASL-1: Not a frontier model (covered by "weaker-than-frontier" baseline).
- ASL-2: Current frontier baseline; deploy with routine safeguards.
- ASL-3: Meaningfully higher catastrophic misuse risk; CBRN-relevant capabilities. Activated May 2025.
- ASL-4: Crosses the AI R&D-2 threshold; models that can automate entry-level AI research.
- ASL-5+: Advanced AI R&D; models that can substantially accelerate effective scaling.

v3.0 additions:
- Frontier Safety Roadmap (published in redacted form).
- Risk Reports (quarterly, partially externally reviewed).
- AI R&D split into AI R&D-2 and AI R&D-4.
- An affirmative safety case is required once the AI R&D-4 threshold is crossed, identifying misalignment risk from "models pursuing misaligned goals."

### OpenAI Preparedness Framework v2 (April 15, 2025)

Five criteria for tracked capabilities:
- **Plausible.** A credible threat model exists.
- **Measurable.** Empirical evaluation is possible.
- **Severe.** The harm is large.
- **Net-new.** Not an amplification of existing risk.
- **Instantaneous-or-irremediable.** The harm happens fast or cannot be undone.

Capabilities meeting all five criteria are tracked. Others are not.

Other PF v2 structures:
- Separates Capabilities Reports (what the model can do) from Safeguards Reports (what controls exist).
- Safety Advisory Group reviews.
- Leadership sign-off; Board Safety and Security Committee oversight.
- "Adjustment clause": if another lab ships without comparable safeguards, OpenAI may lower requirements.

### DeepMind Frontier Safety Framework v3.0 (September 2025)

Critical Capability Levels (CCL) by domain:
- Bioweapons uplift
- Cyber uplift
- ML R&D acceleration
- Harmful manipulation (new in v3.0): models that can materially shift beliefs/behaviors in high-stakes contexts.

v2.0 (February 2025) added a "deceptive alignment" section and higher security levels for the ML R&D CCL.

### Cross-Lab Alignment

- Anthropic "capability thresholds."
- DeepMind "critical capability levels."
- OpenAI "high capability thresholds."

No industry-standard terminology. Structurally aligned: three tiers of frontier capability with published evaluation criteria. All three include competitor adjustment clauses (since 2025).

### Safety Cases

A safety case is a written argument claiming that a deployment is acceptably safe under worst-case assumptions. The standard structure targets three pillars:

- **Monitoring.** If bad behavior occurs, can we detect it?
- **Illegibility.** Does the model lack the ability to execute a coherent plan to cause harm?
- **Incapability.** Does the model lack the capability to cause the harm in question?

Different safety cases target different pillars. For an ASL-3 CBRN case, incapability (via unlearning) is the primary target. For deceptive alignment, monitoring and illegibility are the targets. For cyber uplift, all three are relevant.

### The Race Dynamics Problem

Competitor adjustment clauses are controversial. Critics argue they create a race to the bottom: if all three labs will lower requirements when a competitor defects, the equilibrium shifts toward defection. Defenders argue that if the defecting lab is less safety-conscious, the alternative (unilateral safeguards) produces worse outcomes.

UK AISI, US CAISI, EU AI Office (Lesson 24) are external governance counterparts. Lab frameworks are voluntary; regulatory frameworks are forming.

### Where This Fits in Phase 18

Lessons 17–18 are the "measurement and governance" layer stacked on top of deception and red-team analysis. Lessons 19–24 cover welfare, bias, privacy, watermarking, and regulatory structures. Lesson 28 sketches the research ecosystem (MATS, Redwood, Apollo, METR) that operationalizes these evaluations.

## Use It

No code for this lesson. Read the three primary sources: RSP v3.0, PF v2, FSF v3.0. Map each lab's tier structure to the other two, and identify one threshold each lab defines that the others do not.

## Ship It

This lesson produces `outputs/skill-framework-diff.md`. Given a safety framework or release note, it compares the framework's threshold definitions, required evaluations, and safety case structure against RSP v3.0, PF v2, and FSF v3.0, and flags cross-lab gaps.

## Exercises

1. Read RSP v3.0, PF v2, and FSF v3.0. Compile a table listing each lab's CBRN threshold, their respective AI R&D thresholds, and their required pre-deployment evaluations.

2. Competitor adjustment clauses exist in all three frameworks (since 2025). Write one paragraph defending them; write one paragraph opposing them. Identify the assumptions each position depends on.

3. Design a safety case for a model that crosses Anthropic's AI R&D-4 threshold. Specify the evidence needed for each of the three pillars (monitoring, illegibility, incapability).

4. DeepMind's FSF v3.0 introduces a "harmful manipulation" CCL. Propose three empirical measurements that would indicate a model has crossed this threshold.

5. Read METR's "Common Elements of Frontier AI Safety Policies" (2025). Name three strongest cross-lab convergence points and two largest divergence points.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| RSP | "Anthropic's framework" | Responsible Scaling Policy; ASL tiers; v3.0 February 2026 |
| PF | "OpenAI's framework" | Preparedness Framework; five criteria; v2 April 2025 |
| FSF | "DeepMind's framework" | Frontier Safety Framework; CCLs; v3.0 September 2025 |
| ASL-3 | "biosafety level 3 analogy" | Anthropic's tier for CBRN-relevant capabilities; activated May 2025 |
| CCL | "critical capability level" | DeepMind's threshold construct; domain-specific |
| Safety case | "the formal argument" | Written argument claiming "deployment is acceptably safe under worst-case U" |
| Adjustment clause | "allows competitor defection" | Framework clause allowing lowered requirements when competitors ship without comparable safeguards |

## Further Reading

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL tiers, roadmap, AI R&D split
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — five criteria, adjustment clause
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0, harmful manipulation
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — cross-lab comparison
