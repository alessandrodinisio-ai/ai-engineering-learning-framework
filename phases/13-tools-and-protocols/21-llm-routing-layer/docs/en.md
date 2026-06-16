# LLM Routing Layer — LiteLLM, OpenRouter, Portkey

> Provider lock-in is expensive. Different tool-call workloads suit different models. A routing gateway gives you a single API surface, retries, failover, cost tracking, and guardrails. Three archetypes dominate in 2026: LiteLLM (open-source self-hosted), OpenRouter (hosted SaaS), and Portkey (production-grade, open-sourced March 2026). This lesson names the decision criteria and walks through a standard-library routing gateway.

**Type:** Learn
**Languages:** Python (standard library, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## Learning Objectives

- Distinguish self-hosted, hosted, and production-grade routing options.
- Implement a fallback chain that retries across providers in a defined priority order on failure.
- Track per-request cost and token usage across providers.
- Choose between LiteLLM, OpenRouter, and Portkey for a given set of production constraints.

## The Problem

Scenarios where provider routing is useful:

1. **Cost.** Claude Sonnet costs 3x what Haiku does. For a triage task, Haiku suffices; for a synthesis task, Sonnet is worth the price. Route per request.

2. **Failover.** OpenAI has a bad hour. Every request fails. You want automatic fallback to Anthropic without redeploying.

3. **Latency.** A real-time chat UI needs fast time-to-first-token. A batch summarizer does not. Route by latency SLA.

4. **Compliance.** EU users must stay in EU regions. Route by region.

5. **Experimentation.** A/B test two models on the same workload. Route by test bucket.

Hand-rolling each integration is repetitive. A routing gateway gives you one OpenAI-compatible API and handles the rest.

## The Concept

### OpenAI-Compatible Proxy Shape

Everyone speaks OpenAI shape. The routing gateway exposes `/v1/chat/completions`, accepts the OpenAI schema, and internally proxies to Anthropic / Gemini / Cohere / Ollama / anything. The client doesn't care.

### Model Aliases

Your code doesn't say `claude-3-5-sonnet-20251022`; it says `our_smart_model`. The gateway maps aliases to real models. When Anthropic releases Claude 4, you change the alias server-side; your code doesn't move.

### Fallback Chain

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: reject
```

The gateway defines this in a config. Retries count against a budget to prevent fallback cascades from blowing up costs.

### Semantic Caching

Identical or near-identical prompts hit a cache instead of going to the provider. Can save 30% to 60% on repetitive agent loops. Keys are based on embeddings; near-identical prompts share a cache slot.

### Guardrails

At the gateway layer:

- **PII redaction.** A regex or ML scan before sending the prompt.
- **Policy violations.** Reject prompts with forbidden content.
- **Output filtering.** Sanitize leaks in completions.

Portkey and Kong both deliver opinionated guardrails. LiteLLM leaves them optional.

### Per-Key Rate Limiting

One API key = one team. Per-key budgets prevent one team from burning through shared quota. Most gateways support this.

### Self-Hosted vs. Hosted Trade-offs

| Factor | LiteLLM (self-hosted) | OpenRouter (hosted) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Code | Open source, Python | Hosted SaaS | Open source (March 2026) + hosted |
| Setup | Deploy a proxy | Sign up | Either |
| Providers | 100+ | 300+ | 100+ |
| Billing | Use your own keys | OpenRouter credits | Use your own keys |
| Observability | OpenTelemetry | Dashboard | Full OTel + PII redaction |
| Best for | Teams wanting full control | Quick prototyping | Production with compliance |

LiteLLM wins when you have an SRE team and want data sovereignty. OpenRouter wins when you want a single subscription and no infrastructure. Portkey wins when you need out-of-the-box guardrails and compliance.

### Cost Tracking

Every request carries `provider`, `model`, `input_tokens`, `output_tokens`. Multiply by per-model per-token price (pulled from a pricing table the gateway maintains). Aggregate by user / team / project.

### MCP Plus Routing

A gateway can route both LLM calls and MCP sampling requests simultaneously. When a sampling request's modelPreferences favor a specific model, the gateway translates to the correct backend. This is where the MCP gateway from Phase 13 · 17 and this lesson's routing gateway sometimes merge into a single service.

### Routing Strategies

- **Static priority.** First in the list; fallback on error.
- **Load balancing.** Round-robin or weighted.
- **Cost-aware.** Pick the cheapest model that meets latency / quality.
- **Latency-aware.** Pick the fastest model in the last N minutes.
- **Task-aware.** A prompt classifier routes coding to one model, summarization to another.

## Use It

`code/main.py` implements a routing gateway in ~150 lines: accepts OpenAI-shaped requests, translates to per-provider stubs, runs a priority fallback chain, tracks per-request cost, and applies a PII redaction pass on input. Run it with three scenarios: a normal request, a primary-provider outage triggering fallback, and a PII leak caught by redaction.

What to look at:

- The `ROUTES` dict: alias -> priority-ordered list of concrete providers.
- The fallback loop retries on 5xx.
- The cost tracker multiplies token usage by per-model rates.
- The PII redactor sanitizes SSN-shaped patterns before forwarding.

## Ship It

This lesson produces `outputs/skill-routing-config-designer.md`. Given a workload profile (latency, cost, compliance), this skill picks LiteLLM / OpenRouter / Portkey and produces a routing configuration.

## Exercises

1. Run `code/main.py`. Trigger the outage scenario; confirm the fallback lands on the second provider and cost is correctly attributed.

2. Add semantic caching: the SHA256 of the prompt is a lookup key; a cache hit returns immediately. Measure cost savings on a repeated call.

3. Add a prompt classifier that routes "code ..." prompts to a smart-preferred alias and "summarize ..." prompts to a speed-preferred alias.

4. Design per-team budgets: each team has a monthly spend cap; the gateway rejects requests once the cap is hit. Pick an enforcement granularity (per-request or windowed).

5. Read the documentation for LiteLLM, OpenRouter, and Portkey side by side. Name the one feature each delivers that the other two do not.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Routing gateway | "LLM proxy" | A single API surface layer in front of many providers |
| OpenAI-compatible | "Speaks OpenAI schema" | Accepts `/v1/chat/completions` shape, translates to any backend |
| Model alias | "our_smart_model" | A name in your code that the gateway maps to a concrete model |
| Fallback chain | "Retry list" | Ordered list of providers to try on failure |
| Semantic caching | "Prompt-embedding cache" | Key is the prompt's embedding; near-duplicates share a cache hit |
| Guardrails | "Input/output filters" | Redact PII, reject policy violations |
| Per-key rate limit | "Team budget" | Quota scoped to one API key |
| Cost tracking | "Per-request spend" | Aggregated token usage x per-model price |
| LiteLLM | "Open-source proxy" | Self-hostable open-source routing gateway |
| OpenRouter | "Hosted SaaS" | Hosted gateway with credit-based billing |
| Portkey | "Production option" | Open source + hosted with built-in guardrails |

## Further Reading

- [LiteLLM — docs](https://docs.litellm.ai/) — Self-hosted routing gateway
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — Hosted routing SaaS
- [Portkey — docs](https://portkey.ai/docs) — Production routing with guardrails
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — Decision guide
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — Vendor survey
