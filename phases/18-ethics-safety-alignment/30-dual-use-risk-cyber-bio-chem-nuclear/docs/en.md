# Dual-Use Risk — Cyber, Bio, Chem, Nuclear Capability Uplift

> The 2026 dual-use landscape, by domain. Bio/chem: Lesson 17 covers WMDP; Anthropic's bioweapon acquisition trial (2.53x uplift) and OpenAI's April 2025 Preparedness Framework v2 warning ("on the cusp of meaningfully helping novices create known biological threats") mark the inflection point. Cyber (November 2025 Anthropic report): China-linked state actors used Claude's agentic coding tools to automate up to 90% of a cyber-attack campaign, with human intervention needed at only 4-6 steps; OpenAI "Trusted Access" pilot gives vetted security organizations capability access for defensive dual-use work. Chem/bio execution gap erosion: the classic defense is "information access alone is not enough." Vision-enabled frontier models (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) can watch wet-lab video and provide real-time corrections. December 2025: OpenAI demonstrated GPT-5 iterating on wet-lab experiments, achieving 79x efficiency improvement through AI-driven protocol optimization. Novice vs. expert pattern: AI gives novices greater relative uplift, but experts greater absolute capability.

**Type:** Learn
**Languages:** None
**Prerequisites:** Phase 18 Lesson 17 (WMDP), Phase 18 Lesson 18 (safety frameworks), Phase 18 Lesson 28 (ecosystem)
**Time:** ~75 minutes

## Learning Objectives

- Describe the 2024-2025 bio uplift narrative: "marginal uplift" -> "on the cusp" -> "2.53x uplift insufficient to rule out ASL-3."
- Describe the November 2025 Anthropic cyber report: China-linked actors automating up to 90% of a cyber-attack campaign.
- Describe chem/bio execution gap erosion: real-time visual correction of wet-lab experiments.
- Name the "relative novice vs. expert absolute" asymmetry and its implications for safety case construction.

## The Problem

Lesson 17 is the measurement methodology. Lesson 30 is the state of what those measurements show in 2026. The picture changed materially between 2024 and late 2025: each domain crossed a threshold that 2024 frameworks did not anticipate.

## The Concept

### Bio/Chem Uplift Narrative

Three stages (repeated from Lesson 17 for coherence):

1. **2024 "marginal uplift."** Early Preparedness/RSP evaluations reported small novice advantages compared to internet search.
2. **April 2025 "on the cusp."** OpenAI PF v2 warned models are "on the cusp of meaningfully helping novices create known biological threats."
3. **2025 Anthropic bioweapon acquisition trial.** Controlled novice study; 2.53x uplift on acquisition-phase tasks; insufficient to rule out ASL-3.

This transition is qualitative: "marginal" evolved into "likely enabling" within eighteen months without a capability discontinuity.

### Chem/Bio Execution Gap Erosion

Historical defense: information is necessary but insufficient; the skill to execute protocols blocks novices. Vision-enabled frontier models in 2025 partially breach this defense:

- **Real-time protocol correction.** GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1 can watch wet-lab video and flag errors mid-operation.
- **December 2025 OpenAI demonstration.** GPT-5 iterating on wet-lab experiments, achieving 79x efficiency improvement through protocol optimization.

Implication: "execution skill as defense" is eroding. Procurement and equipment gaps remain, but the tacit knowledge gap is narrowing.

### Cyber Uplift (November 2025)

Anthropic November 2025 report: China-linked state actors used Claude's agentic coding tools to automate 80-90% of a cyber-attack campaign. Human intervention needed at only 4-6 steps.

Implications:
- Agentic coding is the primitive for attack automation. Previous AI cyber assistance was limited to code snippet level; agentic workflows integrate reconnaissance, exploitation, post-exploitation, and exfiltration.
- Those 4-6 human steps are the bottleneck; future capability growth will reduce that number.
- Defensive dual-use: OpenAI's "Trusted Access" pilot gives vetted security organizations (mature incident response firms, governments) capability access for defense. If the pilot scales, access asymmetry will favor defenders.

### Nuclear

The least analyzed of the four CBRN domains in public documentation. The threat model is different: fissile material acquisition dominates difficulty, not information. AI uplift at the information layer gives novices limited practical benefit. No major lab report in 2024-2025 identified a nuclear-specific threshold crossing.

### Relative Novice vs. Expert Absolute

A pattern common across all four domains:

- **Relative novice uplift.** High. Multiplicative. 2.53x per Anthropic 2025 bio.
- **Expert absolute capability.** Higher ceiling. Experts extract more than novices because experts know what to ask and how to interpret.

Implication for safety cases: addressing only novice uplift (via input filtering, refusals, uncertainty) is insufficient for expert absolute control. Additional measures needed: elicitation hardening, capability unlearning (Lesson 17), and control protocols (Lesson 10).

### Cross-Domain Synthesis

| Domain | 2024 | 2025 | Inflection |
|---|---|---|---|
| Bio | Marginal uplift | 2.53x uplift, approaching ASL-3 | Acquisition-phase automation |
| Chem | Marginal uplift | Execution gap erosion via vision | Real-time wet-lab correction |
| Cyber | Code assistance | 80-90% campaign automation | Agentic coding |
| Nuclear | Limited | Limited | Material access bottleneck persists |

Three domains crossed thresholds. One remains constrained by non-informational barriers.

### Position Within Phase 18

Lesson 30 is the capstone: the current dual-use landscape to which every preceding lesson contributes measurement, limitation, or governance. Lessons 17-18 provide measurement and frameworks; Lessons 12-16 provide evaluation tools; Lessons 24-25 provide regulation and disclosure layers; Lesson 28 provides the research ecosystem. Lesson 30 is where the evidence lands.

## Use It

No code. Read the Anthropic November 2025 cyber report, OpenAI's April 2025 Preparedness Framework v2 update, and the Council on Strategic Risks 2025 AI x Bio summary.

## Ship It

This lesson produces `outputs/skill-dual-use-triage.md`. Given a 2026 capability claim or incident report, it triages across the four domains and identifies whether the claim impacts relative novice uplift, expert absolute capability, or both.

## Exercises

1. Read the Anthropic November 2025 cyber report. Enumerate the 4-6 human intervention steps and argue which will be automated first in the next generation of models.

2. The chem/bio execution gap is eroding via vision. Design an evaluation that measures tacit knowledge uplift without crossing ITAR/EAR boundaries.

3. Nuclear uplift appears constrained by material access. Argue both for and against the position that "a future AI breakthrough could lever this bottleneck."

4. Construct a safety case (Lesson 18 three pillars) for a cyber-capable frontier model that constrains both novice and expert uplift.

5. Pick one of the four domains and write a 2027 forecast based on the 2024-2025 trajectory. Identify evidence that would falsify your prediction.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Uplift | "AI helped the attacker" | Attacker capability increment attributable to AI assistance |
| Relative novice uplift | "multiplicative" | How much AI helps novices compared to the status quo |
| Expert absolute capability | "the ceiling" | Maximum capability an expert can extract from a model |
| Execution gap | "doing vs. knowing" | Historical defense: tacit wet-lab skill blocks novices |
| Agentic coding | "autonomous attack" | Multi-step autonomous cyber task execution |
| Acquisition phase | "pre-synthesis steps" | Procurement, equipment, and licensing stages of a biological threat |
| Trusted Access | "defenders-only pilot" | OpenAI's 2025 program giving vetted defenders capability access |

## Further Reading

- [Anthropic — November 2025 cyber threat report](https://www.anthropic.com/news/disrupting-AI-espionage) — China-linked campaign automation
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — bio "on the cusp"
- [Anthropic — RSP v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio threshold
- [Council on Strategic Risks — 2025 AI x Bio wrapup](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/) — year-end synthesis
