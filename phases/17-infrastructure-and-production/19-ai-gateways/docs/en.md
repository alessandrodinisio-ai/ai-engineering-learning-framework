# AI Gateways — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> A gateway sits between your application and model providers. Core features are provider routing, fallback, retry, rate limiting, key vaulting, observability, and guardrails. The 2026 market breaks down as follows: **LiteLLM** is MIT OSS, 100+ providers, OpenAI-compatible, but collapses at ~2000 RPS (8 GB memory, cascading failures in published benchmarks); best for Python, <500 RPS, dev/staging. **Portkey** positions as a control plane (guardrails, PII redaction, jailbreak detection, audit trails), went Apache 2.0 open-source in March 2026, 20-40 ms latency overhead, $49/month production tier. **Kong AI Gateway** is built on Kong Gateway — Kong's own benchmark on the same 12-CPU hardware: 228% faster than Portkey, 859% faster than LiteLLM; $100/model/month (Plus tier up to 5); fits if you already run Kong. **Bifrost** (Maxim AI) — auto-retry with configurable backoff, fallback to Anthropic on OpenAI 429. **Cloudflare / Vercel AI Gateway** — managed, zero-ops, basic retry. Data residency drives self-hosting decisions; Portkey and Kong sit in the middle offering OSS + optional managed.

**Type:** Learn
**Languages:** Python (stdlib, a toy gateway routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 min

## Learning Objectives

- List six core gateway features (routing, fallback, retry, rate limiting, keys, observability, guardrails).
- Map four 2026 gateways (LiteLLM, Portkey, Kong AI, Bifrost) to their scale ceilings and use cases.
- Cite Kong's benchmark (228% faster than Portkey, 859% faster than LiteLLM) and explain why it matters above 500 RPS.
- Choose self-hosted vs managed given data residency and operational budget.

## The Problem

Your product calls OpenAI, Anthropic, and a self-hosted Llama. Each provider has different SDKs, error models, rate limits, and auth schemes. You want failover (if OpenAI 429s, try Anthropic), a single credential vault, unified observability, and per-tenant rate limiting.

Rebuilding this in the application layer couples every service to every provider. A gateway layer consolidates it into a single process, a single API (usually OpenAI-compatible), that fans out to providers.

## The Concept

### Six Core Features

1. **Provider routing** — OpenAI, Anthropic, Gemini, self-hosted, etc. hidden behind one API.
2. **Fallback** — on 429, 5xx, or quality failure, retry elsewhere.
3. **Retry** — exponential backoff, bounded attempts.
4. **Rate limiting** — per-tenant, per-key, per-model.
5. **Key vaulting** — pull credentials from vault at runtime (never in the application).
6. **Observability** — OTel + GenAI attributes (Phase 17 · 13) + cost attribution.
7. **Guardrails** — PII redaction, jailbreak detection, allowed-topic filtering.

### LiteLLM — MIT OSS, Python

- 100+ providers, OpenAI-compatible, router config, fallback, basic observability.
- Collapses at ~2000 RPS in Kong's benchmark; 8 GB memory footprint, cascading failures under sustained load.
- Best for: Python apps, <500 RPS, dev/staging gateway, experimental routing.
- Cost: OSS free; cloud free tier exists.

### Portkey — Control-Plane Positioning

- Apache 2.0 OSS as of March 2026. Guardrails, PII redaction, jailbreak detection, audit trails.
- 20-40 ms latency overhead per request.
- Production tier $49/month with retention + SLA.
- Best for: regulated industries needing guardrails + observability bundled.

### Kong AI Gateway — Scale Play

- Built on Kong Gateway (mature API gateway product, Lua + OpenResty).
- Kong's own benchmark on equivalent 12-CPU hardware: 228% faster than Portkey, 859% faster than LiteLLM.
- Pricing: $100/model/month, Plus tier up to 5.
- Best for: already running Kong; >1000 RPS; willing to pay license.

### Bifrost (Maxim AI)

- Auto-retry with configurable backoff.
- Fallback to Anthropic on OpenAI 429 is a standard recipe.
- Newer entrant; commercial.

### Cloudflare AI Gateway / Vercel AI Gateway

- Managed, zero-ops. Basic retry and observability.
- Best for: JavaScript apps doing edge serving on Cloudflare/Vercel.
- Limited on guardrails and rate limiting compared to Kong/Portkey.

### Self-Hosted vs Managed

Data residency is the forcing function. Healthcare and finance default to self-hosted (LiteLLM or Portkey OSS or Kong). Consumer products default to managed (Cloudflare AI Gateway) or middle-ground (Portkey managed). Hybrid: self-host for regulated tenants, managed for everyone else.

### Latency Budget

- LiteLLM: typical 5-15 ms overhead.
- Portkey: 20-40 ms overhead.
- Kong: 3-8 ms overhead.
- Cloudflare/Vercel: 1-3 ms overhead (edge advantage).

Gateway latency adds directly to TTFT. For TTFT P99 < 100 ms SLAs, use Kong or Cloudflare. For P99 < 500 ms, any will do.

### Rate-Limiting Semantics Matter

Simple token bucket works at moderate scale. Multi-tenant needs sliding window + burst allowance + per-tenant tiers. LiteLLM uses token bucket; Kong uses sliding window; Portkey uses tiered.

### Gateway + Observability + Routing Compose Together

Phase 17 · 13 (observability) + 16 (model routing) + 19 (gateway) are the same layer in production. Pick one tool that covers all three, or wire them together carefully: most 2026 deployments combine Helicone (observability) or Portkey (guardrails) with Kong (scale) to share roles.

### Numbers You Should Remember

- LiteLLM: ~2000 RPS collapse, 8 GB memory.
- Portkey: 20-40 ms overhead; Apache 2.0 since March 2026.
- Kong: 228% faster than Portkey, 859% faster than LiteLLM.
- Kong pricing: $100/model/month, Plus tier up to 5.
- Cloudflare/Vercel: 1-3 ms overhead at edge.

## Use It

`code/main.py` simulates gateway routing with fallback across 3 providers under injected 429/5xx. Reports latency, retry rate, and fallback hit rate.

## Ship It

This lesson produces `outputs/skill-gateway-picker.md`. Given scale, operational posture, compliance, and latency budget, pick a gateway.

## Exercises

1. Run `code/main.py`. Configure fallback from OpenAI → Anthropic → self-hosted. What's the expected hit rate at 5% provider error rate?
2. Your SLA is TTFT P99 < 200 ms on a 300 ms baseline. Which gateways stay within budget?
3. A healthcare client requires self-hosted + PII redaction + audit. Choose between Portkey OSS and Kong.
4. Compare LiteLLM vs Kong: at what RPS ceiling should a team migrate?
5. Design a rate-limiting strategy for a multi-tenant SaaS: free tier, trial tier, paid tier. Token bucket or sliding window?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Gateway | "API intermediary" | Process sitting between application and providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, 2K RPS collapse |
| Portkey | "guardrails gateway" | Control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | Built on Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | Retry + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | Managed gateway deployed at edge, zero-ops |
| PII redaction | "data scrubbing" | Regex + NER masking before sending to model |
| Jailbreak detection | "prompt injection guard" | Classifier on user input |
| Audit trail | "regulated logging" | Immutable record of every LLM call |
| Token bucket | "simple rate limit" | Refill-based rate limiter |
| Sliding window | "precise rate limit" | Time-window-based rate limiter; better fairness |

## Further Reading

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
