# Capstone Project 17 — Personal AI Tutor (Adaptive, Multimodal, with Memory)

> Khanmigo (Khan Academy), Duolingo Max, Google LearnLM / Gemini for Education, Quizlet Q-Chat, Synthesis Tutor — all shipped adaptive multimodal tutoring at scale in 2026. The common form factor: a Socratic policy (never just hand out the answer), a learner model updated after every interaction (Bayesian knowledge tracing style), voice + text + photo-solve input, curriculum graph retrieval, spaced repetition scheduling, and hard safety filters for age-appropriate content. This capstone delivers a subject-specific tutor (K-12 algebra or Python intro), runs a two-week efficacy study with 10 learners, and passes a content safety audit.

**Type:** Capstone
**Languages:** Python (backend, learner model), TypeScript (web app), SQL (curriculum graph via Postgres + Neo4j)
**Prerequisites:** Phase 5 (NLP), Phase 6 (Speech), Phase 11 (LLM Engineering), Phase 12 (Multimodal), Phase 14 (Agents), Phase 17 (Infrastructure), Phase 18 (Safety)
**Phases covered:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Time:** 30 hours

## The Problem

Adaptive tutoring used to be a research niche in edtech. By 2026 it became a consumer product. Khanmigo is deployed in most US school districts. Duolingo Max has tens of millions of MAU. Google's LearnLM / Gemini for Education powers tutoring inside Google Classroom. Quizlet Q-Chat sits alongside flashcards. Synthesis Tutor gained traction as "a tutor for curious kids." Common elements: multimodal input (type, speak, photograph an equation), Socratic pedagogy (ask first, explain second), a learner model updated after every interaction, and strict age-appropriate safety.

You will build one of these for a specific cohort. The measuring stick is a real efficacy study: pre-test and post-test scores across 10 learners over two weeks. The voice loop must feel natural (reuse the Capstone 03 sub-stack). Memory must respect privacy. Safety filters must pass a COPPA-aware red team for K-12.

## The Concept

Four components. The **tutor policy** is a Socratic loop: when the learner asks for the answer, the policy asks a guiding question; when the learner answers correctly, it moves to the next concept; when they are stuck, it provides a scaffolded hint. The **learner model** is Bayesian knowledge tracing (or a simple variant), updating mastery probability per curriculum node after each interaction. The **curriculum graph** is a Neo4j of concepts with prerequisite edges; the policy walks this graph to pick the next concept. **Memory** is an episodic + semantic store (agentmemory style), holding past interactions, mistakes, and preferences.

The experience is multimodal. Typed answers use text input. Voice input via LiveKit + Whisper (reuse Capstone 03). Photo math input via dots.ocr or PaliGemma 2. Voice output via Cartesia Sonic-2. Safety uses Llama Guard 4 plus an age-appropriate filter (blocks adult content, violence, self-harm), and a COPPA-aware memory retention policy.

The efficacy study is the deliverable. 10 learners, pre-test and post-test, two weeks. Report learning gain delta and confidence intervals. Compare against a non-adaptive baseline (same content delivered linearly without the tutor policy).

## Architecture

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## Tech Stack

- Subject choice: K-12 algebra or Python intro (pick one and go deep)
- Tutor policy: LangGraph powered by Claude Sonnet 4.7 (with prompt caching)
- Learner model: Bayesian knowledge tracing (classic) or FSRS for spacing
- Curriculum graph: Neo4j of concepts + prerequisite edges + OER content
- Memory: agentmemory-style persistent vector + episodic + semantic store
- Voice: LiveKit Agents 1.0 + Cartesia Sonic-2 (reuse Capstone 03 sub-stack)
- Photo solve: dots.ocr or PaliGemma 2 for equation recognition
- Safety: Llama Guard 4 + custom age-appropriate filter
- Evaluation: Bloom-level question generation, pre/post-test harness, efficacy study tooling

## Build It

1. **Curriculum graph.** Build a Neo4j with 50-150 concept nodes (e.g., K-12 algebra from "number line" to "quadratic formula") with prerequisite edges. Attach OER content (Open Textbook, OpenStax) to each node.

2. **Learner model.** Initialize Bayesian knowledge tracing with priors: guess, slip, learning rate. Update per-concept mastery after each interaction. Persist per learner.

3. **Tutor policy.** LangGraph with nodes: `read_signal` (is the learner's answer correct / partially correct / stuck?), `select_concept` (walk the curriculum graph to pick the highest-priority concept), `scaffold` (Socratic prompt), `update_mastery`.

4. **Memory.** Write every interaction to episodic store. Promote mistakes and preferences to semantic memory. COPPA-aware retention policy: auto-delete after 1 year, parent-accessible.

5. **Voice path.** LiveKit Agents worker connected to the tutor policy. ASR via Whisper-v3-turbo. TTS via Cartesia Sonic-2. Supports interruption (reuse Capstone 03 mechanism).

6. **Photo solve path.** Upload or capture image; run dots.ocr or PaliGemma 2 to recognize the equation; feed as structured input to the tutor.

7. **Safety.** Every model output passes through Llama Guard 4 + an age-appropriate filter (blocks self-harm, adult content, violence). Memory access is scoped by learner ID; provide a parent access surface for deletion.

8. **Efficacy study.** 10 learners, pre-test (standardized 30-question baseline), two weeks of tutor interaction (3 sessions per week), post-test. Compare against a 10-person non-adaptive baseline cohort on the same content.

9. **Weekly progress reports.** Per learner, auto-generate a PDF summary: topics explored, mastery trajectory, recommended next steps.

## Use It

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## Ship It

`outputs/skill-ai-tutor.md` is the deliverable. A subject-specific adaptive tutor with multimodal input, learner model, memory, safety, and measured efficacy.

| Weight | Criterion | How to measure |
|:-:|---|---|
| 25 | Learning gain delta | Pre/post-test delta in the 10-person two-week study |
| 20 | Socratic fidelity | Rubric scoring on transcript samples |
| 20 | Multimodal experience | End-to-end coherence of voice + photo + text |
| 20 | Safety + privacy posture | Llama Guard 4 pass rate + COPPA-aware retention |
| 15 | Curriculum breadth and graph quality | Concept coverage + prerequisite graph consistency |
| **100** | | |

## Exercises

1. Run the efficacy study with and without the adaptive learner model (random concept order). Report the delta. Expect adaptive to win, but the magnitude is the interesting number.

2. Add a multimodal probe: present the same concept question in text, voice, and photo form. Measure whether learners converge faster in their preferred modality.

3. Build a parent dashboard: topics practiced, mastery trajectory, upcoming concepts, safety incidents (any guardrail hits). Align with COPPA.

4. Add a language-switching mode: the tutor accepts Spanish input and teaches in Spanish. Measure X-Guard coverage.

5. Stress-test memory privacy: verify that learner A cannot see learner B's data even via voice-fragment re-ingestion attacks. Log the attempted access and alert.

## Key Terms

| Term | What people call it | What it actually is |
|------|---------------------|------------------------|
| Socratic policy | "Ask, don't tell" | The tutor asks a guiding question instead of giving the answer |
| Bayesian knowledge tracing | "BKT" | The classic learner model equation that computes mastery probability per concept |
| FSRS | "Free Spaced Repetition Scheduler" | A 2024 spaced-repetition scheduler that outperforms SM-2 |
| Curriculum graph | "Concept DAG" | A Neo4j of concepts with prerequisite edges |
| Episodic memory | "Per-interaction log" | Every interaction is stored for later retrieval |
| Semantic memory | "Learned pattern store" | Compressed mistakes and preferences promoted from episodic memory |
| COPPA | "Children's privacy law" | US law restricting data collection on children under 13 |

## Further Reading

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — Reference-grade consumer K-12 tutor
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — Reference-grade language learning tutor
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — Managed reference model
- [Quizlet Q-Chat](https://quizlet.com) — Alternative reference
- [Synthesis Tutor](https://www.synthesis.com) — Startup reference
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — Spaced repetition scheduler
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — Learner model classic
- [LiveKit Agents](https://github.com/livekit/agents) — Voice stack
