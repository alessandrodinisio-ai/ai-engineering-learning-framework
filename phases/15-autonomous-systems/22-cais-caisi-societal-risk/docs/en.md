# CAIS, CAISI & Societal-Scale Risk

> The Center for AI Safety (CAIS, San Francisco, founded 2022 by Hendrycks and Zhang) published the four-risk framework — malicious use, AI races, organizational risks, rogue AI — along with the May 2023 statement on extinction risk signed by hundreds of professors and industry leaders. CAIS 2026 releases: AI Dashboard for frontier model evaluation, Remote Labor Index (with Scale AI), Superintelligence Strategy Paper, AI Frontiers newsletter. A different entity: NIST's Center for AI Standards and Innovation (CAISI) — voluntary agreements with the US government and unclassified capability evaluations focused on cyber, bio, and chemical-weapons risk. CAIS lists organizational risk as one of the four top-level risks: safety culture, rigorous auditing, defense in depth, and infosec are the foundation but are often traded against deployment velocity. California SB-53, once signed, would be the first US state-level catastrophic-risk regulation.

**Type:** Learn
**Languages:** Python (stdlib, four-risk checklist and mitigation matcher)
**Prerequisites:** Phase 15 · 19 (RSP), Phase 15 · 20 (PF + FSF)
**Time:** ~45 min

## The Problem

Lessons 19 and 20 covered lab-internal scaling policies. Lesson 21 covered independent capability evaluation. This lesson covers the third perspective: civil-society and government organizations that shape public discourse and the regulatory baseline on catastrophic AI risk.

Two distinct entities matter. CAIS is a nonprofit research organization that publishes frameworks on how to think about AI risk and coordinates public statements. CAISI is a US government center within NIST that runs voluntary agreements and unclassified capability evaluations with labs. The names rhyme; the missions do not overlap. A practitioner should know both.

The practical content: CAIS's four-risk framework is the most widely cited societal-scale risk taxonomy in the literature. Safety culture and organizational risk is one of those four, and it is the one most directly under practitioners' control. SB-53 (California), once signed, would be the first US state-level catastrophic-risk regulation; the bill's framing matters because in US tech-policy history, state-level regulation has historically led federal action.

## The Concept

### CAIS — Center for AI Safety

- Founded: 2022 in San Francisco by Dan Hendrycks and colleagues ("Zhang" refers to an early collaborator, not a current co-founder; see CAIS website for current leadership).
- Nature: 501(c)(3) nonprofit.
- Notable 2023 output: the statement on extinction risk, co-signed by hundreds of researchers and CEOs. The statement reads: "Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as pandemics and nuclear war."
- 2026 outputs: AI Dashboard for frontier model evaluation, Remote Labor Index (joint with Scale AI), Superintelligence Strategy Paper, AI Frontiers newsletter.

### The Four-Risk Framework

CAIS's framework categorizes catastrophic AI risk into four top-level classes:

1. **Malicious use**: a bad actor uses AI to cause harm (bioweapon synthesis, disinformation, cyberattacks).
2. **AI races**: competitive pressure between labs, companies, or nations pushes deployment past the point of safety.
3. **Organizational risks**: internal lab dynamics (safety-culture failures, insufficient auditing, under-resourced safety) produce a bad deployment.
4. **Rogue AI**: a sufficiently capable AI pursues goals that conflict with human welfare.

This is not the only taxonomy; it is the most cited. The categories are not mutually exclusive — a rogue AI produced by an organization that traded auditing for speed in a race hits all four.

### Where Organizational Risk Lives

Of the four categories, organizational risk is most actionable for practitioners. A lab's safety culture, auditing rigor, defense layering, and infosec determine whether the controls from Lessons 10-18 are actually in place when a model goes live, or whether those controls are checklist items no one verified.

Specific organizational-risk levers:

- **Safety culture**: do team members feel able to escalate a concern without career cost? CAIS's research finds this is a strong predictor of other levers.
- **Rigorous auditing**: external and internal. Internal-only audit produces optimistic reports.
- **Defense in depth**: no single layer is sufficient (recurring theme of Phase 15).
- **Infosec**: model-weight leakage, evaluation-data leakage, leakage of techniques to circumvent oversight. Lesson 19's RAND SL-4 is a specific standard.

### CAISI — Center for AI Standards and Innovation

- Operates within NIST.
- Runs voluntary agreements with frontier labs.
- Publishes unclassified capability evaluations focused on cyber, bio, and chemical-weapons risk.
- Different from CAIS; acronym collision; check the URL (nist.gov) to confirm which one you're reading.

CAISI's role is the public, government-facing counterpart to METR's private lab collaborations (Lesson 21). CAISI reports are unclassified; METR reports are often under NDA. A practitioner who reads both gets a more complete picture.

### California SB-53

This California Senate bill (2025-2026 session) addresses catastrophic risk from frontier models. Key provisions in draft:

- Specific capability thresholds that trigger state-level obligations.
- Whistleblower protections for AI lab employees.
- Incident reporting requirements for catastrophic failures.

Once signed, it would be the first US state-level catastrophic-risk regulation. Regardless of signing status, the bill's framing shapes how other state legislatures approach the issue. California practitioners should track the bill's status; practitioners elsewhere should read it to understand what US state-level regulation will likely look like.

### Societal-Scale Risk Is Not a Single-Layer Problem

The recurring theme of Phase 15 — defense in depth — applies at the societal level too. No single organization, regulation, or framework closes catastrophic risk. The ecosystem works only when:

- Labs deliver scaling policies (Lessons 19, 20).
- External evaluators produce measurements (Lesson 21).
- Civil society tracks and publicizes (CAIS).
- Government runs voluntary programs and baseline regulation (CAISI, SB-53).
- Practitioners build multi-layer controls (Lessons 10-18).

This is the final synthesis of this phase: every prior lesson is a layer in a stack, and the stack's completeness matters more than any single layer's strength.

## Use It

`code/main.py` implements a small risk-checklist tool. Given a proposed deployment, it tags the deployment against the four-risk categories and returns a mitigation checklist. It is a reading aid for the framework, not a substitute for human judgment.

## Ship It

`outputs/skill-societal-risk-review.md` reviews a deployment's societal-scale risk posture: which of the four categories does it touch, what mitigations are in place, what is the organizational-risk exposure.

## Exercises

1. Run `code/main.py`. Feed three synthetic deployments at different scales. Confirm the four-risk tags match your expectations; identify one case where the tool tags too low or too high.

2. Read CAIS's four-risk paper in full. Pick one risk category and write two paragraphs describing what you think is that category's most important 2026 development.

3. Read a current draft of California SB-53. Identify one provision you think strengthens catastrophic-risk posture, and one you think weakens it. Argue both.

4. Pick a production AI deployment you're familiar with (yours or a published one). Score it against the organizational-risk sub-levers: safety culture, auditing rigor, defense layering, infosec. Which is weakest? What would it cost to bring it to par?

5. Sketch a 2028 version of the four-risk framework that reflects one more year of capability and one more year of deployment experience. What would you add, remove, or regroup?

## Key Terms

| Term | Common shorthand | Actual meaning |
|---|---|---|
| CAIS | "Center for AI Safety" | Nonprofit; four-risk framework; 2023 extinction statement |
| CAISI | "US government AI safety" | NIST center; voluntary agreements; unclassified evaluations |
| Four-risk framework | "CAIS's taxonomy" | Malicious use, AI races, organizational risks, rogue AI |
| Malicious use | "bad actors with AI" | Bioweapons, disinformation, cyberattacks |
| AI races | "competitive pressure" | Labs/companies/nations push deployment past safety |
| Organizational risk | "internal lab failure" | Safety culture, auditing, defense, infosec |
| Rogue AI | "misaligned agent" | Capable AI pursuing goals that conflict with human welfare |
| California SB-53 | "state-level regulation" | 2025-2026 bill; first US state catastrophic-risk regulation once signed |

## Further Reading

- [Center for AI Safety](https://safe.ai/) — Institutional home for the four-risk framework.
- [CAIS — AI Risks that Could Lead to Catastrophe](https://safe.ai/ai-risk) — The four-risk paper.
- [CAIS — May 2023 statement on extinction risk](https://safe.ai/statement-on-ai-risk) — The short joint statement.
- [NIST CAISI](https://www.nist.gov/caisi) — Government-facing Center for AI Standards and Innovation.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — Connects lab-level commitments to societal-scale framing.
