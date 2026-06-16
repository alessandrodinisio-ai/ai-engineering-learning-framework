# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 took effect February 24, 2026, replacing the 2023 policy. Two-tier mitigation: things Anthropic will do unilaterally vs things positioned as industry-wide recommendations (including RAND SL-4 security standards). The Frontier Safety Roadmap and Risk Report are promoted to standing documents rather than one-off deliverables. The 2023 pause commitment is removed. An AI R&D-4 threshold is introduced: once crossed, Anthropic must publish an affirmative case naming misalignment risks and mitigations. Claude Opus 4.6 has not crossed it. Anthropic's v3.0 announcement states "ruling this out with confidence is becoming difficult." SaferAI rated the 2023 RSP at 2.2; they downgraded v3.0 to 1.9, placing Anthropic alongside OpenAI and DeepMind in the "weak" tier. Qualitative thresholds replace 2023's quantitative commitments; removing the pause clause is the sharpest rollback.

**Type:** Learn
**Languages:** Python (stdlib, RSP threshold decision engine)
**Prerequisites:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Time:** ~45 min

## The Problem

Scaling policies published by frontier labs are part technical document, part governance document, part signal to regulators. RSP v3.0 is Anthropic's current document. Reading it closely matters not because compliance with it is binding (it is not), but because the framework shapes how a lab conceptualizes catastrophic risk and communicates tradeoffs to the public.

The diff of v3.0 vs v2.0 is the useful unit. What was added: the Frontier Safety Roadmap, the Risk Report, the AI R&D-4 threshold. What was removed: the 2023 pause commitment. What was repositioned: a mitigation timeline split into Anthropic-unilateral and industry-recommendation tiers. External review — SaferAI — downgraded the score from 2.2 (v2) to 1.9 (v3.0). This is how a scaling policy can become less rigorous while appearing more polished.

## The Concept

### Two-Tier Mitigation Timeline

- **Anthropic unilateral**: things Anthropic will do regardless of what other labs do. Pause training above a threshold, specific safety measures, specific deployment gates.
- **Industry-wide recommendation**: things Anthropic thinks the industry should do collectively. Includes RAND SL-4 security standards. These are not Anthropic commitments; they are Anthropic's policy advocacy.

The two-tier structure did not exist in v2. It means the reader must check which column each commitment lives in. A safety measure in the "industry-wide recommendation" column is not an Anthropic commitment; it is an Anthropic aspiration.

### AI R&D-4 Threshold

This is the capability level RSP v3.0 names as the important next threshold. Specifically: a model that can automate a substantial fraction of AI research at competitive cost. Once Anthropic believes a model has crossed it, they must publish an affirmative case naming misalignment risks and mitigations before continuing to scale.

Per the v3.0 announcement, Claude Opus 4.6 has not crossed it. The document adds: "ruling this out with confidence is becoming difficult." The phrasing matters; it acknowledges the threshold is close enough to be a practical concern rather than a hypothetical ceiling.

Lesson 6 (Automated Alignment Research) and Lesson 7 (Recursive Self-Improvement) feed directly into this threshold. Autonomous alignment researchers crossing a research-quality gate is evidence that the AI R&D-4 threshold is approaching.

### Frontier Safety Roadmap & Risk Report

v3.0 promotes two artifact types to standing documents:

- **Frontier Safety Roadmap**: a forward-looking document describing planned safety work, expected capabilities, and mitigation research.
- **Risk Report**: a backward-looking document for a specific model after release, describing observed capabilities and residual risks.

Both are public. Both update on a stated cadence. The utility: a reader can track what Anthropic says they will do in a roadmap against what they report in a risk report.

### Removal of the Pause Clause

The 2023 RSP contained an explicit pause commitment: if a model crosses specific capability thresholds, training pauses until mitigations are in place. v3.0 replaces this explicit pause with a softer formulation (publish an affirmative case; continue if mitigations are sufficient). SaferAI and other analysts called this out directly as the strongest rollback in the new document.

The policy argument supporting the change: the 2023 quantitative thresholds turned out to be unreachable by 2026-era capability benchmarks because the benchmarks themselves were rescaled. The counterargument: a pause clause in a scaling policy is a commitment device; removing it removes the policy's credibility.

### SaferAI Downgrade

SaferAI is an independent organization that rates RSP-style documents. Their public ratings: the 2023 Anthropic RSP scored 2.2 (on a scale where 4.0 is current best-in-class RSP and 1.0 is nominal floor). v3.0 scored 1.9. This moved Anthropic from "moderate" to "weak," alongside OpenAI and DeepMind in the weak tier.

Per SaferAI, factors in the downgrade:
- Qualitative thresholds replaced quantitative thresholds.
- Pause commitment removed.
- AI R&D-4 threshold mitigation described as "affirmative case" rather than concrete measures.
- Review mechanism relies on Anthropic's Safety Advisory Group, with limited independent oversight.

### What This Lesson Is Not

This is not a lesson about compliance. RSP v3.0 is not a regulation; nothing forces Anthropic to follow it. This lesson is about reading the document with the specificity and skepticism it deserves. Scaling policies are the primary public signal a frontier lab sends about its catastrophic-risk posture. Reading them well is a practical skill for anyone whose work depends on frontier capabilities.

## Use It

`code/main.py` implements a small decision engine that mirrors the shape of RSP threshold evaluation: given a candidate model and a set of capability measurements, return whether the AI R&D-4 threshold is crossed, the required affirmative case sections, and whether deployment can proceed. It is deliberately simple; the point is to make the document's logic explicit.

## Ship It

`outputs/skill-scaling-policy-review.md` reviews a scaling policy (Anthropic, OpenAI, DeepMind, or internal) against the v3.0 reference: two-tier structure, thresholds, pause commitments, independent review.

## Exercises

1. Run `code/main.py`. Feed three synthetic models at different capability levels. Confirm the threshold evaluator behaves as expected and produces the correct affirmative-case template.

2. Read RSP v3.0 in full (32 pages). Identify every commitment that lives in the "industry-wide recommendation" tier. Which of those would have been "Anthropic unilateral" in v2?

3. Read SaferAI's RSP rating methodology. Apply their rubric to this document and reproduce their 1.9 score for v3.0. Which rubric row most drives the downgrade?

4. The 2023 pause commitment was removed. Propose an alternative commitment that preserves policy credibility while acknowledging the 2026 benchmark-rescaling problem.

5. Compare RSP v3.0 against the OpenAI Preparedness Framework v2 (Lesson 20). Pick one area where v3.0 is stronger. Pick one area where the Preparedness Framework is stronger.

## Key Terms

| Term | Common shorthand | Actual meaning |
|---|---|---|
| RSP | "Anthropic's scaling policy" | Responsible Scaling Policy; v3.0 effective February 24, 2026 |
| AI R&D-4 | "research automation threshold" | Capability to automate a substantial fraction of AI research at competitive cost |
| Affirmative case | "safety argument" | Published argument claiming risks are named and mitigations sufficient |
| Frontier Safety Roadmap | "forward-looking plan" | Standing document on planned safety work and expected capabilities |
| Risk Report | "model retrospective" | Standing document on observed capabilities and residual risks post-release |
| Two-tier mitigation | "unilateral vs industry" | Anthropic commitments vs industry recommendations, separated |
| Pause commitment | "the 2023 clause" | Explicit commitment to pause training; removed in v3.0 |
| SaferAI rating | "independent RSP score" | Third-party rubric; v3.0 scored 1.9 (v2 was 2.2) |

## Further Reading

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — The full 32-page policy.
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — Summary of changes relative to v2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — Standing document linked from RSP v3.0.
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — Retrospective on the current frontier model.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Connects AI R&D-4 to measured autonomy.
