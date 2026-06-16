# OpenAI Preparedness Framework & DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2 (April 2025) introduces Research Categories — long-horizon autonomy, sandbagging, autonomous replication and adaptation, undermining safeguards — distinct from Tracked Categories. Tracked Categories trigger a Capabilities Report plus a Safeguards Report, reviewed by the Safety Advisory Group. DeepMind's FSF v3 (September 2025, with Tracked Capability Levels added April 17, 2026) folds autonomy into the ML R&D and Cyber domains (ML R&D autonomy level 1 = fully automate an AI R&D pipeline at competitive cost relative to humans + AI tools). FSF v3 explicitly handles deceptive alignment via automated monitoring for instrumental-reasoning misuse. An honest note: Research Categories in PF v2 (including long-horizon autonomy) do not automatically trigger mitigations; policy wording is "potential." DeepMind itself states that automated monitoring "won't be sufficient indefinitely" if instrumental reasoning improves.

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45 min

## The Problem

Lesson 19 read Anthropic's scaling policy closely. This lesson completes the picture by reading OpenAI's and DeepMind's. The three documents are cousin products addressing the same problem — when should a frontier lab pause or gate a model — and they converge on a small set of categories while diverging in a few places that matter.

Where they converge: all three flag long-horizon autonomy as a capability category worth tracking. All three acknowledge deceptive behavior (alignment faking, sandbagging) as a specific risk category. All three have an internal review body. Where they diverge: OpenAI splits categories into "tracked" (mandatory mitigation) and "research" (no automatic trigger). DeepMind folds autonomy into two domains rather than naming it separately. Labs call it tracked vs research, or critical vs moderate, or level-1 vs level-2; which bucket a capability falls into has different operational consequences across labs.

Reading them together is the useful exercise. The same capability might be "mandatory mitigation" at Anthropic, "monitored but not triggered" at OpenAI, and "tracked within a specific domain" at DeepMind. Policy posture matters.

## The Concept

### OpenAI Preparedness Framework v2 (April 2025)

Structure:

- **Tracked Categories**: trigger a Capabilities Report (what the model can do) plus a Safeguards Report (what mitigations are in place). Reviewed pre-deployment by the Safety Advisory Group.
- **Research Categories**: model capabilities the lab is tracking but has not committed specific mitigations for. Includes long-horizon autonomy, sandbagging, autonomous replication and adaptation, undermining safeguards.

Key framing in v2: Research Categories do not automatically trigger mitigations. Policy wording is "potential" mitigations. This is the specific delta to watch: whether a category sits in tracked vs research is the hinge between "gated" and "observed."

### DeepMind Frontier Safety Framework v3 (September 2025; Tracked Capability Levels added April 17, 2026)

Structure:

- **Critical Capability Levels (CCL)**: capability thresholds across five domains: Cyber, Bio, ML R&D, CBRN, Autonomy (folded into ML R&D and Cyber).
- **Tracked Capability Levels**: additional granularity added April 2026. Specific example: ML R&D autonomy level 1 = fully automate an AI R&D pipeline at competitive cost relative to humans + AI tools.
- **Deceptive alignment monitoring**: explicit commitment to automated monitoring for instrumental-reasoning misuse.

The framing of autonomy differs from OpenAI. DeepMind does not leave "autonomy" as a top-level domain; it folds it into the domains where autonomy would cause harm (ML R&D and Cyber). The argument: autonomy without a domain is a capability without risk; the counterargument: cross-domain autonomy is a meta-risk the framework should name.

### Where All Three Converge

- Internal safety advisory groups (called Anthropic SAG, OpenAI SAG, DeepMind internal committee respectively). Review before deployment of high-capability models.
- Explicit mention of deceptive alignment / alignment faking as a risk category.
- Standing artifacts on a stated cadence (Anthropic: Frontier Safety Roadmap, Risk Report; OpenAI: Capabilities and Safeguards Reports; DeepMind: FSF update cycle).
- Acknowledgment that monitoring-only defense has a ceiling. DeepMind says it plainly: "automated monitoring won't be sufficient indefinitely."

### Where They Diverge

- **Anthropic**: pause commitment removed in v3.0; AI R&D-4 threshold is the named next gate.
- **OpenAI**: tracked vs research split; Research Categories (including long-horizon autonomy) do not automatically gate.
- **DeepMind**: autonomy folded into other domains; Tracked Capability Levels added granularity in April 2026.

### Sandbagging: A Specific Capability That Complicates All Three

Sandbagging (a model strategically underperforming on evaluations) is in OpenAI's Research Categories. Anthropic's RSP v3.0 handles it via eval-context gaps (Lesson 1). DeepMind handles it via deceptive alignment monitoring in FSF v3.

If a model sandbags evaluations, then every framework's capability thresholds are underestimated. Frameworks only work when measurement works. This is why external measurement (Lesson 21, METR) and adversarial evaluation are necessary beyond lab self-assessment.

### The Skill of Reading Policy

- Locate: every capability you care about should be findable in the policy. If it isn't, the policy doesn't cover it.
- Classify: is it tracked (triggers mitigation) or research (tracked but doesn't trigger)? OpenAI names it this way; Anthropic and DeepMind have their own equivalents.
- Cadence: is the policy updated on a stated schedule, or only after specific events? Stated cadence is stronger.
- Independence: is external review mandatory or optional? Anthropic works with Apollo and the US AI Safety Institute; OpenAI with METR; DeepMind primarily with its internal SAG.

## Use It

`code/main.py` implements a small decision-table diff tool. Given a capability (autonomy, deceptive alignment, R&D automation, cyber enablement, etc.), it outputs how each of the three policies classifies that capability and what mitigation is triggered. It is a reading aid, not a policy tool.

## Ship It

`outputs/skill-cross-policy-diff.md` produces a cross-policy comparison for a specific capability using these three frameworks as reference.

## Exercises

1. Run `code/main.py`. Confirm the diff tool's output matches the policies for at least two capabilities you can verify against the source documents.

2. Read OpenAI Preparedness Framework v2 in full. Identify every Research Category. For each, write one sentence explaining why it is in research rather than tracked.

3. Read DeepMind FSF v3 in full, plus the April 2026 Tracked Capability Levels update. Identify the specific evaluation criteria for ML R&D autonomy level 1. How would you measure it externally?

4. Sandbagging is in OpenAI's Research Categories. Design an evaluation that forces a sandbagging model to reveal its actual capabilities. Reference Lesson 1's discussion of eval-context gaming.

5. Compare all three policies on a specific capability (your choice). State which policy you think classifies it most rigorously and which least rigorously. Argue from source text.

## Key Terms

| Term | Common shorthand | Actual meaning |
|---|---|---|
| Preparedness Framework | "OpenAI's scaling policy" | PF v2 (April 2025); tracked vs research categories |
| Tracked Category | "mandatory mitigation" | Triggers Capabilities + Safeguards Report; SAG review |
| Research Category | "monitoring only" | Tracked but no automatic mitigation; includes long-horizon autonomy |
| Frontier Safety Framework | "DeepMind's scaling policy" | FSF v3 (September 2025) + Tracked Capability Levels (April 2026) |
| CCL | "Critical Capability Level" | DeepMind's per-domain thresholds (Cyber, Bio, ML R&D, CBRN) |
| ML R&D autonomy level 1 | "R&D automation" | Fully automate an AI R&D pipeline at competitive cost |
| Sandbagging | "strategic underperformance" | Model underperforms on evaluations; in OpenAI's Research Categories |
| Instrumental reasoning | "means-end reasoning" | Reasoning about how to achieve goals; what DeepMind monitors |

## Further Reading

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 announcement.
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — Full document.
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 announcement.
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — Tracked Capability Levels addition.
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — Example of an FSF-format risk report.
