# LLM Observability Stack Selection

> The 2026 observability market splits into two categories. Development platforms (LangSmith, Langfuse, Comet Opik) bundle monitoring with eval, prompt management, and session replay. Gateway/instrumentation tools (Helicone, SigNoz, OpenLLMetry, Phoenix) focus on telemetry. Langfuse core is MIT-licensed, with strong OSS balance (cloud free tier: 50K events/month). Phoenix is OpenTelemetry-native with Elastic License 2.0 — excellent for drift/RAG visualization but not a durable production backend. Arize AX uses zero-copy Iceberg/Parquet integration, claiming ~100x cheaper than monolithic observability at scale. LangSmith leads for LangChain/LangGraph, $39/user/month, self-host only on enterprise. Helicone is proxy-based, 15-30 minute setup, 100K free requests/month, but weaker on agent trace depth. Common production pattern: gateway (Helicone/Portkey) + eval platform (Phoenix/TruLens), glued together with OpenTelemetry.

**Type:** Learn
**Languages:** Python (standard library, a toy trace sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 minutes

## Learning Objectives

- Distinguish development platforms (bundled: eval + prompt + sessions) from gateway/telemetry tools (traces + metrics only).
- Map six major tools (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) to their license, pricing, and best-fit use case.
- Explain the OpenTelemetry glue pattern that lets you combine a gateway tool with a separate eval platform.
- State the 2026 cost differentiator (Arize AX's zero-copy approach vs monolithic ingestion) and give the rough 100x multiplier.

## The Problem

You shipped an LLM feature. It works. You have zero visibility into prompt failures, tool loops, latency regressions, cost spikes, or prompt cache hit rates. You Google "LLM observability" and get eight tools, all claiming to solve the same problem at three different price points.

They don't solve the same problem. LangSmith answers "why did this LangGraph run fail?" Phoenix answers "is my RAG pipeline drifting?" Helicone answers "which app is burning tokens?" Langfuse answers "can I self-host the whole thing?" Different tools, different audiences.

Selection involves four dimensions: stack (LangChain? bare SDK? multi-vendor?), license tolerance (MIT only? Elastic OK? Commercial fine?), budget (free tier? $100/month? $1000/month?), self-hosting (must? nice-to-have? never?).

## The Concept

### Two Categories

**Development platforms** bundle observability with eval, prompt management, dataset versioning, and session replay. You run experiments, see which prompt worked, run dataset regressions with new prompts against old winners. LangSmith, Langfuse, Comet Opik.

**Gateway/telemetry tools** instrument inference calls — prompt, response, tokens, latency, model, cost. Helicone, SigNoz, OpenLLMetry, Phoenix. Minimal. Can be combined with a separate eval tool via OpenTelemetry.

### Langfuse — OSS Balance

- Core Apache / MIT license; self-host via Docker.
- Cloud free tier: 50K events/month. Paid: Team at $29/month.
- Eval, prompt management, traces, datasets. All four development platform features covered reasonably well.
- Best-fit: you want LangSmith-level features but must self-host or stay on OSS license.

### Phoenix (Arize) — Telemetry-First, OpenTelemetry-Native

- Elastic License 2.0; easy to self-host.
- Excellent at RAG and drift visualization. Embedding space scatter plots as a first-class feature.
- Not designed as a durable production backend — primarily development-time observability.
- Best-fit: RAG pipeline development, drift debugging, paired with a separate gateway in production.

### Arize AX — The Scale Play

- Commercial. Zero-copy data lake integration via Iceberg/Parquet.
- Claims ~100x cheaper than monolithic observability (Datadog-tier) at scale. The math: you store traces in Parquet on your own S3; Arize reads directly.
- Best-fit: >10M traces/day, existing data lake, want LLM-specific dashboards without Datadog pricing.

### LangSmith — LangChain/LangGraph First

- Commercial, $39/user/month. Self-host only on enterprise.
- Best-in-class for LangChain and LangGraph stacks. Less compelling if you're not on either.
- Best-fit: teams committed to LangChain, willing to pay.

### Helicone — Proxy-Based Minimum Viable

- Swap your `OPENAI_API_BASE` to the Helicone proxy, 15-30 minute setup.
- MIT license; 100K free requests/month, paid from $20/month.
- Includes failover, caching, rate limiting — doubles as a gateway.
- Weaker on agent / multi-step trace depth.
- Best-fit: fast start, single-stack apps, want gateway + observability in one.

### Opik (Comet) — OSS Development Platform

- Apache 2.0, fully OSS.
- Feature set similar to Langfuse, with Comet lineage.
- Best-fit: ML teams already on Comet who want LLM observability in the same panel.

### SigNoz — OpenTelemetry-First Full APM

- Apache 2.0. Handles general APM plus LLM via OpenTelemetry.
- Best-fit: unified observability across services and LLM calls.

### The Glue: OpenTelemetry + GenAI Semantic Conventions

OpenTelemetry shipped GenAI semantic conventions in late 2025 (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). Tools that consume OTel can interoperate. The emerging production pattern:

1. Emit OTel with GenAI conventions from every LLM call.
2. Route to a gateway (Helicone / Portkey) for day-to-day.
3. Dual-ship to an eval platform (Phoenix / Langfuse) for regressions.
4. Archive to a data lake (Iceberg), query via Arize AX or DuckDB for long-term analysis.

### Pitfalls: Instrumenting at the Wrong Layer

Instrumenting inside your agent framework (e.g., adding LangSmith traces) couples you to that framework. Instrumenting at the HTTP/OpenAI-SDK layer (via OpenLLMetry or your gateway) is portable.

### Sampling — You Can't Keep Everything

At >1M requests/day, full trace retention costs more than the LLM calls. Sample by rule: 100% errors, 100% high-cost, 5% successes. Always retain aggregates; retain raw data for the long tail.

### Numbers You Should Remember

- Langfuse free cloud: 50K events/month.
- LangSmith: $39/user/month.
- Helicone free: 100K requests/month.
- Arize AX claim: ~100x cheaper than monolithic at scale.
- OpenTelemetry GenAI conventions: shipped 2025, broadly adopted 2026.

## Use It

`code/main.py` simulates a 1M-trace day across multiple retention strategies (100% ingestion, sampled, sampled + errors). Reports storage cost and what is lost under each strategy.

## Ship It

This lesson produces `outputs/skill-observability-stack.md`. Given stack, scale, budget, and license stance, picks the tools.

## Exercises

1. Your team on LangChain wants OSS self-hosted observability. Choose between Langfuse and Opik and argue.
2. At 5M traces/day with Datadog quoting $150K/month, calculate Arize AX's break-even.
3. Design a set of OpenTelemetry GenAI attributes your org's guidelines should mandate on every LLM call.
4. Argue whether Phoenix alone is sufficient for production. When does it fall short?
5. Helicone adds 20ms of proxy overhead. At P99 TTFT of 300 ms, is this acceptable? What if the SLA is 100 ms?

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| OpenLLMetry | "OTel for LLMs" | Open-source OpenTelemetry instrumentation for LLMs |
| GenAI conventions | "OTel attributes" | Standard OTel attribute names for LLM calls |
| LangSmith | "LangChain observability" | Commercial platform bundled with LangChain ecosystem |
| Langfuse | "OSS LangSmith" | MIT OSS with similar feature set |
| Phoenix | "Arize dev tool" | OpenTelemetry-native dev/eval platform |
| Arize AX | "Scale observability" | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | "Proxy observability" | HTTP proxy that collects LLM telemetry + gateway features |
| Opik | "Comet LLM" | Apache 2.0 OSS development platform from Comet |
| Session replay | "Trace replay" | Replaying a full agent session with tool calls |
| Eval | "Offline testing" | Running candidate models/prompts on labeled datasets |

## Further Reading

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
