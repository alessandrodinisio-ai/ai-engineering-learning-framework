# Capstone Project 11 — LLM Observability & Evaluation Dashboard

> Langfuse went open-core. Arize Phoenix shipped its 2026 GenAI semconv mapping. Helicone and Braintrust are both doubling down on per-user cost attribution. Traceloop's OpenLLMetry became the de facto SDK instrumentation layer. The production shape is traces in ClickHouse, metadata in Postgres, UI in Next.js, plus a small fleet of eval jobs running on sampled traces (DeepEval, RAGAS, LLM-judge). Build a self-hosted version that ingests from at least four SDK families and demonstrates catching an injected regression within five minutes.

**Type:** Capstone
**Languages:** TypeScript (UI), Python / TypeScript (ingestion + evals), SQL (ClickHouse)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools), Phase 17 (Infrastructure), Phase 18 (Safety)
**Phases Involved:** P11 · P13 · P17 · P18
**Time:** 25 hours

## The Problem

In 2026 every AI team running production traffic keeps an observability plane next to the model. Cost attribution. Hallucination detection. Drift monitoring. Jailbreak signals. SLO dashboards. PII leak alerts. The open-source references — Langfuse, Phoenix, OpenLLMetry — have converged on using OpenTelemetry GenAI semantic conventions as the ingestion schema. You can now instrument OpenAI, Anthropic, Google, LangChain, LlamaIndex, and vLLM with a single SDK and emit compatible spans.

You will build a self-hosted dashboard that ingests from at least four SDK families, runs a small set of eval jobs on sampled traces, detects drift, and alerts. The measuring stick: given a deliberately injected regression (a prompt that starts producing PII), the dashboard must catch it and fire an alert within five minutes.

## The Concept

Ingestion uses OTLP HTTP. SDKs produce GenAI-semconv spans: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. Spans land in ClickHouse for columnar analytics; metadata (users, sessions, apps) lands in Postgres.

Evals run as batch jobs on sampled traces. DeepEval scores faithfulness, toxicity, and answer relevance. When traces carry retrieval context, RAGAS scores retrieval metrics. A custom LLM-judge runs domain-specific checks (PII leakage, off-policy responses). Eval runs write back to the same ClickHouse as eval spans linked to the parent trace.

Drift detection watches embedding-space distribution over time (PSI or KL divergence on prompt embeddings) plus eval-score trends. Alerts feed into Prometheus Alertmanager and then to Slack / PagerDuty. The UI is Next.js 15 with Recharts.

## Architecture

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Tech Stack

- Ingestion: OpenTelemetry SDK + GenAI semantic conventions; OTLP HTTP transport
- Collector: OpenTelemetry Collector with tail-sampling processor (for cost control)
- Storage: ClickHouse for spans, Postgres for metadata, S3 for raw event archival
- Evals: DeepEval, RAGAS 0.2, Arize Phoenix evaluator package, custom LLM-judge
- Drift: Weekly PSI / KL on pooled prompt embeddings (sentence-transformers)
- Alerting: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- SDK support out of the box: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## Build It

1. **Collector configuration.** OpenTelemetry Collector with an OTLP HTTP receiver, a tail-sampler that retains 100% of errored traces and 10% of successful ones, and exporters to ClickHouse and S3.

2. **ClickHouse schema.** Table `spans` with columns mapping to GenAI semconv: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`, plus a JSON bag for large payloads. Add secondary indexes on user_id and app_id.

3. **SDK coverage test.** Write a small client app with each SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) using OpenLLMetry auto-instrumentation. Verify each produces standard GenAI spans that land in ClickHouse.

4. **Eval jobs.** A scheduled job reads the last 15 minutes of sampled traces and runs DeepEval faithfulness, toxicity, and answer relevance. Outputs are eval spans linked to the parent trace.

5. **Custom LLM-judge.** A PII leakage judge: given a response, call a guard LLM to score the likelihood of PII leakage. High-scoring responses land in a triage queue.

6. **Drift detection.** A weekly job computes PSI between this week's pooled prompt embeddings and the baseline from the previous 4 weeks. If PSI exceeds a threshold, alert.

7. **Dashboard.** Next.js 15 with pages for: overview (spans/sec, cost/user, p95 latency), traces (search + waterfall), evals (faithfulness trend, toxicity), drift (PSI over time), alerts.

8. **Alert chain.** Prometheus exporter reads eval-score aggregates and latency percentiles; Alertmanager routes warnings to Slack and critical violations to PagerDuty.

9. **Regression probe.** Inject a bug: the chatbot under evaluation starts leaking fake SSNs at 1% probability. Measure MTTR: from bug deployment to Slack alert.

## Use It

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Ship It

`outputs/skill-llm-observability.md` is the deliverable. Given an LLM application, the dashboard ingests its traces, runs evals, alerts on drift, and renders cost/user breakdowns in Next.js.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | Trace schema coverage | Number of SDK families producing standard GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard experience | MTTR on injected regression (target under 5 minutes) |
| 20 | Cost / scale | Sustained ingestion at 1k spans/sec without backlog |
| 15 | Alerting + drift detection | Prometheus/Alertmanager chain running end-to-end |
| **100** | | |

## Exercises

1. Add custom instrumentation for the Haystack framework. Verify that standard spans with faithful `gen_ai.*` attributes land in ClickHouse.

2. Swap DeepEval for Phoenix evaluators on the same batch of traces. Measure score drift between the two eval engines.

3. Sharpen the drift detector: compute PSI per app-id instead of globally. Show per-application drift tracks.

4. Add a "user impact" page: per-user cost and per-user failure rate with mini sparklines.

5. Build a tail-sampling policy that retains 100% of traces with toxicity > 0.5 plus a 10% stratified sample of the rest. Measure the sampling bias introduced.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | The 2025 OpenTelemetry spec for LLM span attributes (system, model, tokens) |
| Tail sampling | "After-the-fact sampling" | The collector decides to keep or drop a trace after the trace completes (can peek at errors) |
| PSI | "Population Stability Index" | A drift metric comparing two distributions; > 0.2 typically signals meaningful drift |
| LLM-judge | "Model-as-evaluator" | An LLM scoring another LLM's output against criteria (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Retention rules" | Rules deciding which traces to keep and which to drop; errors + sample rate |
| Eval span | "Linked evaluation trace" | A child span carrying eval scores, linked to the original LLM call span |
| Cost per user | "Unit economics" | Dollar cost attributed to a user_id within a time window; a key product metric |

## Further Reading

- [Langfuse](https://github.com/langfuse/langfuse) — reference open-core observability platform
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — alternative reference with strong drift support
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — auto-instrumentation SDK family
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — ingestion schema
- [Helicone](https://www.helicone.ai) — alternative hosted observability
- [Braintrust](https://www.braintrust.dev) — alternative eval-first platform
- [ClickHouse documentation](https://clickhouse.com/docs) — columnar span storage
- [DeepEval](https://github.com/confident-ai/deepeval) — evaluator library
