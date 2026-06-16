# OpenTelemetry GenAI Semantic Conventions

> OpenTelemetry's GenAI SIG (formed April 2024) defines a standard schema for agent telemetry. Span names, attributes, and content capture rules converge across vendors so that agent traces mean the same thing in Datadog, Grafana, Jaeger, and Honeycomb.

**Type:** Learn + Build
**Languages:** Python (standard library)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 min

## Learning Objectives

- Name the GenAI span categories: model/client, agent, tool.
- Distinguish the `invoke_agent` CLIENT span from the INTERNAL span, and when each applies.
- List the top-level GenAI attributes: provider name, request model, data source ID.
- Explain the content capture contract: opt-in, `OTEL_SEMCONV_STABILITY_OPT_IN`, external reference recommendation.

## The Problem

Every vendor invents its own span names. Ops teams end up building separate dashboards for every framework. OpenTelemetry's GenAI SIG fixes this by defining a standard that the entire ecosystem aligns to.

## The Concept

### Span categories

1. **Model / client span.** Covers the raw LLM call. Emitted by provider SDKs (Anthropic, OpenAI, Bedrock) and framework model adapters.
2. **Agent span.** `create_agent` (when the agent is constructed) and `invoke_agent` (when it runs).
3. **Tool span.** One per tool call; linked to the agent span via parent-child relationship.

### Agent span naming

- Span name: `invoke_agent {gen_ai.agent.name}` if named; falls back to `invoke_agent`.
- Span kind:
  - **CLIENT** — for remote agent services (OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — for in-process agent frameworks (LangChain, CrewAI, local ReAct).

### Key attributes

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — model ID.
- `gen_ai.response.model` — resolved model (may differ from request due to routing).
- `gen_ai.agent.name` — agent identifier.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — for RAG: which corpus or store was consulted.

Technology-specific conventions exist for Anthropic, Azure AI Inference, AWS Bedrock, and OpenAI.

### Content capture

Default rule: instrumentation should not capture input/output by default. Capture is opted into via:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Recommended production pattern: store content externally (S3, your log store), record references (pointer IDs, not prose) on the span. This wires Lesson 27's content poisoning defenses into observability.

### Stability

As of March 2026, most conventions are still experimental. Opt into the stable preview with:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ maps GenAI attributes natively into its LLM Observability schema. Other backends (Grafana, Honeycomb, Jaeger) support raw attributes.

### Where this pattern breaks

- **Capturing full prompts in spans.** PII, secrets, customer data land in traces ops can read. Store externally.
- **Missing `gen_ai.provider.name`.** Multi-provider dashboards break when attribution is missing.
- **Spans without parent links.** Orphan tool spans. Always propagate context.
- **Not setting the stability opt-in.** Your attributes may get renamed on backend upgrades.

## Build It

`code/main.py` implements a span emitter matching GenAI conventions using the standard library:

- `Span` with GenAI attribute schema.
- `Tracer` with `start_span`, nested context.
- A scripted agent run emitting: `create_agent`, `invoke_agent` (INTERNAL), per-tool spans, `chat` span for LLM calls.
- A content capture mode that stores prompts externally and records IDs on the span.

Run it:

```
python3 code/main.py
```

Output: a span tree with all required GenAI attributes, and an "external store" showing opt-in content references.

## Use It

- **Datadog LLM Observability** (v1.37+) maps attributes natively.
- **Langfuse / Phoenix / Opik** (Lesson 24) — auto-instrument the ecosystem.
- **Jaeger / Honeycomb / Grafana Tempo** — raw OTel traces; build dashboards from GenAI attributes.
- **Self-hosted** — run OTel Collector with a GenAI processor.

## Ship It

`outputs/skill-otel-genai.md` wires OTel GenAI spans into an existing agent with content capture defaults and external reference storage.

## Exercises

1. Instrument your Lesson 01 ReAct loop with `invoke_agent` (INTERNAL) + per-tool spans. Ship to a Jaeger instance.
2. Add content capture with a "references only" mode: prompts go to SQLite, span attributes carry only the row ID.
3. Read the spec for `gen_ai.data_source.id`. Wire it into your Lesson 09 Mem0 search.
4. Set `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` and verify your attributes do not get renamed by the collector.
5. Build a dashboard: "which tool errors correlate with which models" using only GenAI attributes.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI group" | The OTel working group that defines the schema |
| invoke_agent | "Agent span" | Span name representing one agent run |
| CLIENT span | "Remote call" | Span for a call to a remote agent service |
| INTERNAL span | "In-process" | Span for an in-process agent run |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG source" | Which corpus/store a retrieval hit |
| Content capture | "Prompt logging" | Opt-in capture of messages; stored externally in production |
| Stability opt-in | "Preview mode" | Environment variable to pin experimental conventions |

## Further Reading

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — GenAI spans by default
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — built-in OTel spans
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation
