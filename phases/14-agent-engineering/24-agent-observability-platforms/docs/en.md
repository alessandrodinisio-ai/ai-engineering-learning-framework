# Agent Observability: Langfuse, Phoenix, Opik

> Three open-source agent observability platforms dominate 2026. Langfuse (MIT) — 6M+ monthly installs, tracing + prompt management + evaluation + session replay. Arize Phoenix (Elastic 2.0) — deep agent-specific evaluation, RAG relevancy, OpenInference auto-instrumentation. Comet Opik (Apache 2.0) — automated prompt optimization, guardrails, LLM-as-judge hallucination detection.

**Type:** Learn
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 min

## Learning Objectives

- Name the three top open-source agent observability platforms and their licenses.
- Distinguish what each excels at: Langfuse (prompt management + sessions), Phoenix (RAG + auto-instrumentation), Opik (optimization + guardrails).
- Explain why by 2026 89% of organizations report having deployed agent observability.
- Implement a "trace-to-dashboard" pipeline with LLM-as-judge evaluation using the standard library.

## The Problem

OTel GenAI (Lesson 23) gives you the schema. You still need the platform that ingests spans, runs evaluations, stores prompt versions, and surfaces regressions. Three competitors each emphasize different parts of the lifecycle.

## The Concept

### Langfuse (MIT)

- 6M+ monthly SDK installs, 19k+ GitHub stars.
- Features: tracing, prompt management with versioning + playground, evaluation (LLM-as-judge, user feedback, custom), session replay.
- June 2025: previously commercial modules (LLM-as-a-judge, annotation queues, prompt experiments, Playground) open-sourced under MIT.
- Strongest at: end-to-end observability with a tight prompt management loop.

### Arize Phoenix (Elastic License 2.0)

- Deeper agent-specific evaluation: trace clustering, anomaly detection, retrieval relevancy for RAG.
- Native OpenInference auto-instrumentation.
- Paired with managed Arize AX for production.
- No prompt versioning — positioned as a drift/behavior regression tool used alongside a broader platform.
- Strongest at: RAG relevancy, behavioral drift, anomaly detection.

### Comet Opik (Apache 2.0)

- Automated prompt optimization via A/B experiments.
- Guardrails (PII redaction, topic constraints).
- LLM-as-judge hallucination detection.
- Benchmarks from Comet's own measurements: Opik logging + evaluation in 23.44 s vs Langfuse 327.15 s (~14x gap) — vendor benchmarks are directional only.
- Strongest at: optimization loops, automated experiments, guardrail enforcement.

### Industry data

Per Maxim (2026 field analysis): 89% of organizations have deployed agent observability; quality issues are the #1 production barrier (32% of respondents cite them).

### Picking one

| Need | Pick |
|------|------|
| All-in-one with prompt management | Langfuse |
| Deep RAG evaluation + drift | Phoenix |
| Automated optimization + guardrails | Opik |
| Open license, no ELv2 | Langfuse (MIT) or Opik (Apache 2.0) |
| Datadog / New Relic integration | Any — they all export OTel |

### Where this pattern breaks

- **No evaluation strategy.** Tracing without evaluation is just expensive logging.
- **Rolling your own LLM judge without grounding.** The CRITIC pattern (Lesson 05) applies — judges need external tools for fact verification.
- **Prompt versions not tied to traces.** When production regresses, you cannot bisect to the prompt that caused it.

## Build It

`code/main.py` implements a trace collector + LLM-as-judge evaluator using the standard library:

- Ingests GenAI-shaped spans.
- Groups by session, flags failed runs (guardrail triggers, low-confidence evaluations).
- A scripted LLM judge that scores agent responses against a rubric.
- A dashboard-style summary: failure rate, top failure reasons, evaluation score distribution.

Run it:

```
python3 code/main.py
```

Output: per-session evaluation scores and failure classification, matching what Langfuse/Phoenix/Opik would surface.

## Use It

- **Langfuse** self-hosted or cloud; wire in via OTel or their SDK.
- **Arize Phoenix** self-hosted; auto-instrument OpenInference.
- **Comet Opik** self-hosted or cloud; automated optimization loops.
- **Datadog LLM Observability** for hybrid ops+ML teams already running Datadog.

## Ship It

`outputs/skill-obs-platform-wiring.md` picks a platform and wires traces + evaluation + prompt versions into an existing agent.

## Exercises

1. Export a week of OTel traces to Langfuse cloud (free tier). Which sessions failed? Why?
2. Write an LLM judge rubric for your domain (factual correctness, tone, scope adherence). Test on 50 traces.
3. Compare Langfuse's prompt versioning against Phoenix's trace clustering. Which tells you faster what broke?
4. Read Opik's guardrail documentation. Wire a PII redaction guardrail into one of your agent runs.
5. Benchmark all three on your corpus. Ignore vendor-published numbers; measure your own.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Tracing | "Span collector" | Ingests OTel / SDK spans; indexes by session |
| Prompt management | "Prompt CMS" | Versioned prompts tied to traces |
| LLM-as-judge | "Automated evaluation" | A separate LLM scores agent output against a rubric |
| Session replay | "Trace playback" | Stepping through past runs for debugging |
| RAG relevancy | "Retrieval quality" | Whether retrieved context matches the query |
| Trace clustering | "Behavior grouping" | Grouping similar runs for drift detection |
| Guardrail enforcement | "Policy at logging time" | PII/toxicity/scope checks on what gets logged |

## Further Reading

- [Langfuse docs](https://langfuse.com/) — tracing, evaluation, prompt management
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation, drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the schema all three consume
