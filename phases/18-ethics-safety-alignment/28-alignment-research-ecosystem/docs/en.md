# Alignment Research Ecosystem — MATS, Redwood, Apollo, METR

> Five organizations define the non-lab alignment research layer as of 2026. MATS (ML Alignment & Theory Scholars): 527+ researchers since late 2021, 180+ papers, 10K+ citations, h-index 47; 2024 summer cohort incorporated as 501(c)(3), approximately 90 scholars and 40 mentors; ~80% of pre-2025 alumni work in safety/security, with 200+ at Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo. Redwood Research: applied alignment lab founded by Buck Shlegeris; proposed AI control (Lesson 10); partners with UK AISI on control safety cases. Apollo Research: pre-deployment scheming evaluations for frontier labs; authored in-context scheming (Lesson 8) and Towards Safety Cases for AI Scheming. METR (Model Evaluation and Threat Research): task-based capability evaluations, autonomous task horizon studies; "Common Elements of Frontier AI Safety Policies" compares lab frameworks. Eleos AI Research: model welfare pre-deployment evaluations (Lesson 19); conducted the Claude Opus 4 welfare evaluation.

**Type:** Learn
**Languages:** None
**Prerequisites:** Phase 18 Lessons 01-27 (preceding lessons in Phase 18)
**Time:** ~45 minutes

## Learning Objectives

- Identify the five organizations in the non-lab alignment research ecosystem and their core outputs.
- Describe MATS's scale (scholars, papers, h-index) and its role as a talent pipeline.
- Describe Redwood's AI control agenda and its partnership with UK AISI.
- Describe METR's task-based evaluation methodology.

## The Problem

Frontier labs (Lesson 18) produce safety evaluations internally and publish selected results. The non-lab ecosystem is where evaluations get validated, new failure modes get discovered first, and talent gets trained. Understanding this ecosystem helps interpret which research findings are trusted by whom.

## The Concept

### MATS (ML Alignment & Theory Scholars)

Launched late 2021. Research mentorship program; scholars spend 10-12 weeks working with a senior researcher on a specific alignment problem.

Scale (2026):
- 527+ researchers since inception.
- 180+ published papers.
- 10K+ citations.
- h-index 47.
- 2024 summer: 90 scholars + 40 mentors; incorporated as 501(c)(3).

Career placement: ~80% of pre-2025 alumni work in safety/security. 200+ at Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo.

### Redwood Research

Applied alignment lab. Founded by Buck Shlegeris. Proposed the AI control agenda (Lesson 10). Partners with UK AISI on control safety cases. Provides evaluation design advice to DeepMind and Anthropic.

Key papers: Greenblatt, Shlegeris et al., "AI Control" (arXiv:2312.06942, ICML 2024); Alignment faking (Greenblatt, Denison, Wright et al., arXiv:2412.14093, with Anthropic).

Style: concrete threat models, worst-case adversaries, specific protocols that can be stress-tested.

### Apollo Research

Pre-deployment scheming evaluations for frontier labs. Authored in-context scheming (Lesson 8, arXiv:2412.04984). Partner in OpenAI's 2025 anti-scheming training collaboration. Produced Towards Safety Cases for AI Scheming (2024).

Style: evaluations of agentic settings where deception may emerge; three-pillar decomposition (misalignment, goal-directedness, situational awareness).

### METR (Model Evaluation and Threat Research)

Task-based capability evaluations. Autonomous task completion horizon studies. "Common Elements of Frontier AI Safety Policies" (metr.org/common-elements, 2025) compares lab frameworks.

Co-authored AI scheming safety case sketches with Apollo.

Style: long-horizon task evaluations, empirical capability measurement, framework synthesis.

### Eleos AI Research

Model welfare pre-deployment evaluations. Conducted the Claude Opus 4 welfare evaluation documented in system card Section 5.3. Provides external methodological review for welfare-related claims in Lesson 19.

### The Pipeline

MATS trains researchers. Graduates go to Anthropic, DeepMind, OpenAI (lab safety teams), or to Redwood, Apollo, METR, Eleos (external evaluation). External evaluators partner with labs and with UK AISI / CAISI. Publications feed back into the next MATS cohort.

### Why This Layer Matters

Single-source evaluations are unreliable: labs evaluating their own models have structural conflicts of interest. External evaluators can propose and validate failure modes that labs might underreport. The 2024 sleeper agents paper (Lesson 7) was Anthropic + Redwood; alignment faking was Anthropic + Redwood; in-context scheming was Apollo; anti-scheming was Apollo + OpenAI. This multi-organization structure is the quality control.

### Position Within Phase 18

Lessons 7-11 cite Redwood and Apollo work; Lesson 18 cites METR's framework comparison; Lesson 19 cites Eleos. Lesson 28 is the explicit organizational map of the ecosystem that the rest of the phase relies on.

## Use It

No code. Read METR's "Common Elements of Frontier AI Safety Policies" as an example of how external synthesis adds value to lab-internal policy work.

## Ship It

This lesson produces `outputs/skill-ecosystem-map.md`. Given an alignment claim or evaluation, it identifies the organization, publication venue, methodological style, and cross-checks against known counterpart organizations.

## Exercises

1. Pick a paper from Lessons 7-15 and identify the organizations involved. Cross-check authors against MATS alumni and current ecosystem affiliations.

2. Read METR's "Common Elements of Frontier AI Safety Policies." Identify three cross-lab convergence points they highlight and two biggest divergences.

3. MATS career placement is ~80% safety/security. Argue whether this selection pressure is adaptive (builds the field) or biased (filters out heterodox positions).

4. Redwood and Apollo both do control/scheming work but with different styles. Pick a failure mode and describe how each would investigate it.

5. Eleos AI is the only purely model welfare organization. Design a hypothetical second organization focused on a different welfare-adjacent problem (cognitive liberty, robotic embodiment, etc.) and articulate its methodology.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| MATS | "the mentorship program" | ML Alignment & Theory Scholars; 527+ researchers since 2021 |
| Redwood Research | "the control lab" | Applied alignment; AI control authors; UK AISI partner |
| Apollo Research | "the scheming evaluators" | Pre-deployment scheming evaluations for frontier labs |
| METR | "the task horizon evaluators" | Task-based capability evaluations; framework synthesis |
| Eleos AI | "the welfare lab" | Model welfare pre-deployment evaluations |
| Talent pipeline | "MATS -> labs" | MATS graduates flow to Anthropic, DM, OpenAI, Redwood, Apollo, METR |
| External evaluation | "non-lab review" | Evaluations not conducted by the model producer; adds credibility |

## Further Reading

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — mentorship program
- [Redwood Research](https://www.redwoodresearch.org/) — AI control papers
- [Apollo Research](https://www.apolloresearch.ai/) — scheming evaluations
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — framework comparison
- [Eleos AI Research](https://www.eleosai.org/research) — model welfare methodology
