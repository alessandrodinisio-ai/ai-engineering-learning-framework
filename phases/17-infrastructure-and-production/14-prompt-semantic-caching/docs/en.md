# Prompt Caching and Semantic Caching Economics

> **Pricing snapshot as of 2026-04.** The numbers below reflect vendor rate cards scraped at the time this lesson was published; verify against the linked documentation before citing downstream.

> Caching happens at two layers. L2 (vendor-layer) prompt/prefix caching reuses attention KV for repeated prefixes — Anthropic's prompt-caching docs claim up to 90% cost reduction and 85% latency reduction on long prompts; for Claude 3.5 Sonnet, cached reads are $0.30/M vs $3.00/M fresh, with a 5-minute TTL and a 1-hour TTL option at 2x write premium (docs.anthropic.com, 2026-04). OpenAI prompt caching activates automatically for prompts ≥1024 tokens, with cached input pricing at roughly 10% of fresh (platform.openai.com, 2026-04); exact per-model cached rates depend on the live rate card. L1 (application-layer) semantic caching skips the LLM entirely on an embedding-similarity hit. Vendor claims of "95% accuracy" refer to match correctness, not hit rate — reported production hit rates range from 10% (open-ended chat) to 70% (structured FAQ); neither vendor publishes an official baseline, so treat these as community telemetry, not guarantees. Production pitfalls: parallelization kills caching (N parallel requests fired before the first cache write inflates costs by a multiple), and dynamic content in the prefix prevents cache hits entirely. ProjectDiscovery reported lifting hit rate from 7% to 74% by moving dynamic text out of the cacheable prefix (2025-11).

**Type:** Learn
**Languages:** Python (standard library, a toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Learning Objectives

- Distinguish L2 prompt/prefix caching (KV reuse at the vendor) from L1 semantic caching (bypass LLM on similar prompts).
- Explain Anthropic's explicit `cache_control` markup and two TTL options (5-minute vs 1-hour) with their price multipliers.
- Given hit rate, prompt/response mix, and token pricing, calculate expected monthly savings.
- State the parallelization anti-pattern that inflates bills 5-10x, and the dynamic-content anti-pattern that collapses hit rate.

## The Problem

You added prompt caching to your RAG service. The bill doesn't budge. You measure hit rate; it's 7%. Your prompts look static but aren't — the system prompt contains a current date down to the minute, a request ID, and examples randomly reshuffled for diversity. Every request writes a new cache entry, reads zero.

On the other side, your agent fires ten parallel tool calls per user question. All ten hit the vendor before the first cache write completes. Ten writes, zero reads. Your bill is 5-10x what "with caching" should cost.

Caching is a protocol, not a flag. Two layers, two different failure modes.

## The Concept

### L2 — Vendor Prompt/Prefix Caching

The vendor stores attention KV for a cacheable prefix and reuses it on the next request matching that prefix. You pay the write cost once; reads are nearly free.

**Anthropic (Claude 3.5 / 3.7 / 4 family)**: explicit `cache_control` markup in requests. You mark which blocks are cacheable. TTL: 5-minute (write costs 1.25x base) or 1-hour (write costs 2x base). Cached reads: $0.30/M on Claude 3.5 Sonnet vs $3.00/M fresh — 10x cheaper (docs.anthropic.com, as of 2026-04). Rates vary by model (Opus/Haiku published separately); always cross-check the live pricing page.

**OpenAI**: automatic caching for prompts ≥1024 tokens (platform.openai.com, 2026-04). No explicit flag. On the current gpt-4o/gpt-5 rate card, cached input is roughly 10x cheaper than fresh. Documentation and release notes do not publish an official hit-rate baseline; community reports cluster around 30-60% with well-engineered prompts. Monitor `usage.cached_tokens` to measure your own.

**Google (Gemini)**: context caching via explicit API; 1M token context means caching pays off more.

**Self-hosted (vLLM, SGLang)**: Phase 17 · 06 covers RadixAttention — same pattern on your own compute.

### L1 — Application-Layer Semantic Caching

Before calling the LLM at all, hash the prompt, embed it, find a similar cached request (cosine similarity above threshold, typically 0.95+). Hit returns the cached response. Miss calls the LLM and caches the result.

Open-source: Redis Vector Similarity, GPTCache, Qdrant. Commercial: Portkey Cache, Helicone Cache.

Vendor accuracy claims refer to how often the returned cached response is semantically appropriate — not how often you hit. Production hit rates:

- Open-ended chat: 10-15%.
- Structured FAQ / support: 40-70%.
- Code questions: 20-30% (small variations kill hits).
- Voice agents with repetitive prompts: 50-80% (voice normalizes to a fixed set).

### The Parallelization Anti-Pattern

Your agent fires 10 parallel tool calls. All 10 have the same 4K token system prompt. Anthropic cache writes are per-request; the first cache write completes ~300 ms after the vendor sees the prompt. Requests 2-10 arrive in the same millisecond window, each seeing a cache miss. You pay 10 write premiums, 0 read discounts.

Fix: sequential-first batching — send request 1 alone, wait for 1's cache to populate, then release 2-10. Adds 300 ms to the first tool call; saves 5-10x on the bill.

### The Dynamic Content Anti-Pattern

Your system prompt looks like this:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Every request is unique. Every request writes. Zero hits.

Fix: move everything truly static into the cacheable prefix; append dynamic content after the cache boundary:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery used this approach to lift cache hit rate from 7% to 74% and published a breakdown.

### Stacking Batch + Cache for Overnight Workloads

Batch APIs (Phase 17 · 15) give 50% off with 24-hour turnaround. Stacking cached input on top gives you another ~10x. Overnight classification, labeling, and report generation workloads can drop to ~10% of synchronous uncached cost through stacking.

### Numbers You Should Remember

Pricing points as of 2026-04 scraped from linked vendor docs; they drift every few months — re-verify before relying on them.

- Anthropic cached reads: $0.30/M on Claude 3.5 Sonnet, roughly 10x cheaper than fresh input (docs.anthropic.com).
- Anthropic cache write premium: 1.25x (5-minute TTL) or 2x (1-hour TTL).
- OpenAI automatic caching: activates for prompts ≥1024 tokens; cached input pricing is ~10% of fresh on current rate card (platform.openai.com).
- Semantic cache hit rates (community reports): open chat ~10%; structured FAQ up to ~70%. Not vendor-documented baselines.
- ProjectDiscovery: moving dynamic out of prefix, hit rate 7% → 74% (project blog, 2025-11).
- Parallelization anti-pattern: typical reported bill inflation 5-10x when N parallel requests all miss the first cache write.

## Use It

`code/main.py` simulates L1 + L2 caching on a mixed workload. Reports hit rates, billing, and demonstrates the parallelization penalty.

## Ship It

This lesson produces `outputs/skill-cache-auditor.md`. Given a prompt template and traffic, audits cacheability and recommends restructuring.

## Exercises

1. Run `code/main.py`. Toggle the parallelization flag. How much does the bill change?
2. Your system prompt has a date in it. Move it out. Give the before/after hit rate math.
3. Given your request arrival rate, calculate the break-even of 1-hour TTL (2x write) vs 5-minute TTL (1.25x write).
4. Semantic cache at 0.95 threshold hits 20%. At 0.85 it hits 50% but you see wrong cached responses. Pick the right threshold and argue.
5. You fire 10 parallel sub-queries per user question as a batch. Rewrite to be cache-friendly without adding end-to-end latency.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| L2 prompt cache | "Prefix cache" | Vendor stores KV for repeated prefixes |
| `cache_control` | "Anthropic cache markup" | Explicit attribute marking cacheable blocks |
| Cache write premium | "Write tax" | Extra cost for first miss into cache (1.25x or 2x) |
| L1 semantic cache | "Embedding cache" | Application-layer hash + embedding before calling LLM |
| GPTCache | "LLM cache library" | Popular OSS L1 cache library |
| Cache hit rate | "Hits / total" | Fraction of requests served by cache |
| Parallelization anti-pattern | "N-write trap" | N parallel requests miss cache N times |
| Dynamic content pitfall | "Timestamp-in-prompt trap" | Dynamic bytes in prefix kill hit rate |
| RadixAttention | "Intra-replica cache" | SGLang's prefix caching implementation |

## Further Reading

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — Official `cache_control` semantics and TTL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — Automatic caching behavior and eligibility.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
