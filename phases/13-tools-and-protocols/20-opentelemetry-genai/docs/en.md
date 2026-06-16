# OpenTelemetry GenAI — End-to-End Tracing of Tool Calls

> An agent calls five tools, three MCP servers, and two sub-agents. You need a single trace that spans all of it. OpenTelemetry GenAI semantic conventions (attributes stable since v1.37) are the 2026 standard, natively supported by Datadog, Langfuse, Arize Phoenix, OpenLLMetry, and AgentOps. This lesson names the required attributes, walks through the span hierarchy (agent → LLM → tool), and delivers a standard-library span emitter that can plug into any OTel exporter.

**Type:** Build
**Languages:** Python (standard library, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## Learning Objectives

- Name the required OTel GenAI attributes for an LLM span and a tool-execution span.
- Build a trace hierarchy covering the agent loop, LLM calls, tool calls, and MCP client dispatch.
- Decide what content to capture (opt-in) vs. what to redact (default).
- Emit spans to a local collector (Jaeger, Langfuse) without rewriting tool code.

## The Problem

A debugging session in February 2026: a user reports "my agent sometimes takes 30 seconds to respond; other times 3 seconds." No trace. Logs show LLM calls but no tool dispatch, no MCP server round-trips, no sub-agents. You guess. Eventually you track it down: one MCP server occasionally stalls on a cold start.

Without end-to-end tracing, you cannot find this. OTel GenAI fixes it.

These conventions were finalized in 2025–2026 under the OpenTelemetry semantic conventions group. They define stable attribute names so that Datadog, Langfuse, Phoenix, OpenLLMetry, and AgentOps all parse the same spans. Instrument once; ship to any backend.

## The Concept

### Span Hierarchy

```
agent.invoke_agent  (top-level, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

Everything nests under a single trace ID. Span IDs chain parent-child relationships.

### Required Attributes

Per the 2025–2026 semconv:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — the requested model string (e.g., `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — the actually served model.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — provider response ID for correlation.

For tool spans:

- `gen_ai.tool.name` — tool identifier.
- `gen_ai.tool.call.id` — that specific call ID.
- `gen_ai.tool.description` — tool description (optional).

For agent spans:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Span Kinds

- Calls that cross process boundaries (LLM provider, MCP server) use `SpanKind.CLIENT`.
- The agent's own loop steps and tool execution use `SpanKind.INTERNAL`.

### Opt-In Content Capture

By default, spans carry metrics and timing — not prompts or completions. Large payloads and PII are off by default. Set `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` and specific content-capture environment variables to include content. Review carefully before enabling in production.

### Events on Spans

Token-level events can be added as span events:

- `gen_ai.content.prompt` — input messages.
- `gen_ai.content.completion` — output messages.
- `gen_ai.content.tool_call` — recorded tool calls.

Events are time-ordered within a span for detailed replay.

### Exporters

OTel spans export to:

- **Jaeger / Tempo.** Open source, self-hosted.
- **Langfuse.** LLM observability purpose-built; visualizes token usage.
- **Arize Phoenix.** Evals + traces in one.
- **Datadog.** Commercial; natively parses `gen_ai.*` attributes.
- **Honeycomb.** Columnar; query-friendly.

All speak OTLP as the wire format. Your code doesn't care.

### Propagation Across MCP

When an MCP client calls a server, inject the W3C traceparent header into the request. Streamable HTTP supports standard headers. stdio does not natively carry HTTP headers; the spec's 2026 roadmap discusses adding a `_meta.traceparent` field on JSON-RPC calls.

Until that lands: manually place the traceparent in each request's `_meta`. The server records the trace ID.

### Metrics

Beyond spans, the GenAI semconv defines metrics:

- `gen_ai.client.token.usage` — histogram.
- `gen_ai.client.operation.duration` — histogram.
- `gen_ai.tool.execution.duration` — histogram.

Use these for dashboards that don't need per-call detail.

### The AgentOps Layer

AgentOps (founded 2024) focuses on GenAI observability. It wraps popular frameworks (LangGraph, Pydantic AI, CrewAI) and automatically emits OTel spans. Useful if your stack uses a supported framework; otherwise use manual instrumentation.

## Use It

`code/main.py` emits OTel-shaped spans (in an OTLP-JSON-like format) to stdout for an agent that calls one LLM, dispatches two tools, and makes one MCP round-trip. No real exporter — this lesson focuses on span shapes and attribute sets. Paste the output into an OTLP-compatible viewer, or just read it directly.

What to look at:

- The trace ID is shared across all spans.
- Parent-child links are encoded via `parentSpanId`.
- Required `gen_ai.*` attributes are all populated.
- Content capture is off by default; one scenario turns it on via environment variable.

## Ship It

This lesson produces `outputs/skill-otel-genai-instrumentation.md`. Given an agent codebase, this skill produces an instrumentation plan: where to add spans, which attributes to populate, and which exporters to target.

## Exercises

1. Run `code/main.py`. Count the spans and identify which are CLIENT and which are INTERNAL.

2. Turn on content capture (environment variable); confirm that `gen_ai.content.prompt` and `gen_ai.content.completion` events appear. Note the PII implications.

3. Add the tool execution metric `gen_ai.tool.execution.duration`, emitting a histogram sample per call.

4. Propagate a traceparent from the parent agent span into an MCP request's `_meta.traceparent` field. Verify the MCP server sees the same trace ID.

5. Read the OTel GenAI semconv spec. Find one attribute listed in the semconv that this lesson's code does not emit. Add it.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | Open standard for traces, metrics, and logs |
| GenAI semconv | "GenAI semantic conventions" | Stable attribute names for LLM / tool / agent spans |
| `gen_ai.*` | "Attribute namespace" | All GenAI attributes share this prefix |
| Span | "Timed operation" | A unit of work with start, end, and attributes |
| Trace | "Cross-span lineage" | A tree of spans sharing one trace ID |
| SpanKind | "CLIENT / SERVER / INTERNAL" | Hint about the span's direction |
| OTLP | "OpenTelemetry Line Protocol" | Wire format for exporters |
| Opt-in content | "Prompt / completion capture" | Off by default; enabled via environment variables |
| traceparent | "W3C header" | Propagates trace context across services |
| Exporter | "Backend-specific shipper" | Component that sends spans to Jaeger / Datadog etc. |

## Further Reading

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — Authoritative conventions for GenAI spans, metrics, and events
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — Attribute list for LLM and tool execution spans
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — Agent-level `invoke_agent` spans
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub-hosted source of truth
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — Production integration walkthrough
