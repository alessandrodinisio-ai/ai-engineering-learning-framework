# Prompt Caching and Context Caching

> Your system prompt is 4,000 tokens. Your RAG context is 20,000 tokens. You send both with every request. You also pay for both — every single time. Prompt caching lets the provider keep that prefix warm on its side and charge you only 10% of the normal rate on reuse. Used correctly, it cuts inference cost by 50–90% and time-to-first-token latency by 40–85%.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching & Cost)
**Time:** ~60 min

## The Problem

A coding agent sends the same 15,000-token system prompt to Claude on every turn of a conversation. Twenty turns at $3/M input tokens is $0.90 in input cost alone — before the user's actual messages. Multiply by 10,000 conversations per day and the bill rockets to $9,000/day for text that never changes.

You can't shorten the prompt without hurting quality. You can't avoid sending it — the model needs it every turn. The only move is: stop paying full price for a prefix the provider has already seen.

That move is prompt caching. Anthropic launched it in August 2024 (with a 1-hour extended-TTL variant in 2025), OpenAI automated it later that year, and Google shipped explicit context caching with Gemini 1.5 — today all three offer it as a first-class feature on their frontier models.

## The Concept

![Prompt caching: write once, read cheaply](../assets/prompt-caching.svg)

**Mechanism.** When a request's prefix matches a recent prior request, the provider serves the KV-cache from last run instead of re-encoding those tokens. You pay a small write premium the first time and a large read discount every time after.

**Three provider flavors in 2026.**

| Provider | API Style | Hit Discount | Write Premium | Default TTL | Min Cacheable |
|----------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Explicit `cache_control` markers on content blocks | 90% off input | +25% | 5 min (extendable to 1 h) | 1,024 tokens (Sonnet/Opus), 2,048 (Haiku) |
| OpenAI | Automatic prefix detection | 50% off input | None | Up to 1 h (best-effort) | 1,024 tokens |
| Google (Gemini) | Explicit `CachedContent` API | Billed as storage; reads ~25% of normal | Storage fee per token·hour | User-set (default 1 h) | 4,096 tokens (Flash), 32,768 (Pro) |

**Invariant.** All three cache only the prefix. If any token differs between requests, everything after the first divergence is a miss. Put *stable* parts at the top and *variable* parts at the bottom.

### Cache-friendly layout

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, otherwise don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cached (changes every time)
```

Break this order — put the user message above the system prompt, interleave dynamic retrieval between few-shot examples — and the cache never hits.

### Break-even math

Anthropic's 25% write premium means a cached block must be read at least twice to net-save money. 1 write + 1 read averages 0.675× cost per request (32% savings); 1 write + 10 reads averages 0.205× (80% savings). Rule of thumb: cache anything you expect to reuse 3+ times within the TTL.

## Build It

### Step 1: Anthropic prompt caching with explicit markers

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

The `cache_control` marker tells Anthropic to store this block for 5 minutes. Reuse within that window hits; reuse after expiry writes again.

**Response usage fields:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # billed at 1.25×
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # billed at 0.1×
```

Assert on these two fields in CI — if `cache_read_input_tokens` stays zero across multiple requests, your cache key is drifting.

### Step 2: One-hour extended TTL

For long-running batch jobs, the 5-minute default expires between jobs. Set `ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

The 1-hour TTL carries a 2× write premium (50% above baseline instead of 25%), but pays for itself quickly on any batch that reuses the prefix more than 5 times.

### Step 3: OpenAI automatic caching

OpenAI has nothing for you to configure. Any prefix longer than 1,024 tokens that matches a recent request automatically gets the 50% discount.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the portion that got the discount
```

The same cache-friendly layout rules apply. Two things break OpenAI's cache that don't break Anthropic's: changing the `user` field (treated as part of the cache key) and reordering tools.

### Step 4: Gemini explicit context caching

Gemini treats the cache as a first-class object you create and name:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini charges storage per token·hour as long as the cache is alive, with reads at ~25% of the normal input rate. This is the right shape when you reuse the same massive prompt across many sessions for days at a time.

### Step 5: Measuring hit rate in production

`code/main.py` contains a simulated three-provider ledger that tracks write/read/miss counts and computes blended cost per 1K requests. Gate deployments on target hit rates — most production Anthropic setups should see >80% read share after warm-up.

## Pitfalls still appearing in 2026

- **Dynamic timestamps at the top.** `"Current time: 2026-04-22 15:30:02"` at the top of the system prompt. Every request misses. Move the timestamp below the cache break.
- **Tool reordering.** Serialize tools in stable order — dictionary reshuffling between deploys breaks every hit.
- **Free-text near-duplicates.** "You are helpful." vs "You are a helpful assistant." — one byte difference = total miss.
- **Blocks too small.** Anthropic enforces a 1,024-token floor (2,048 for Haiku). Smaller blocks silently don't cache.
- **Blind cost dashboards.** Split "input tokens" into cached vs uncached. Otherwise a traffic drop looks like a caching win.

## Use It

The 2026 caching stack:

| Scenario | Choice |
|----------|--------|
| Multi-turn agent with stable 10k+ system prompt | Anthropic `cache_control` with 5 min TTL |
| Batch jobs reusing the prefix for 30+ minutes | Anthropic with `ttl: "1h"` |
| Serverless endpoints on GPT-5, no custom infra | OpenAI automatic (just keep your prefix stable and long) |
| Reusing a massive code/docs corpus across days | Gemini explicit `CachedContent` |
| Cross-provider fallback | Keep the cacheable prefix layout consistent across providers so any one hits |

Pair with semantic caching (Phase 11 · 11) for the user-message layer: prompt caching handles *token-identical* reuse, semantic caching handles *semantically-identical* reuse.

## Ship It

Save `outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests/hour, required TTL, provider), output:

1. Layout. Reordered sections with a single cache break marked; state which sections are stable and which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Validation plan. CI asserting cache_read_input_tokens > 0 on a second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. Three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you'll guard against each.

Refuse to ship any caching scheme that places a dynamic field above the break. Refuse to enable 1h TTL without a reuse count that pays back the 2× write premium.
```

## Exercises

1. **Easy.** Take a 10-turn conversation with a 5,000-token system prompt and send it to Claude. Run once without `cache_control`, once with. Report the input-token bill for each.
2. **Medium.** Write a test harness that, given a prompt template and a request log, computes expected hit rate and dollar savings for each provider (Anthropic 5m, Anthropic 1h, OpenAI auto, Gemini explicit).
3. **Hard.** Build a layout optimizer: given a prompt and a list of fields tagged `stable=True/False`, rewrite the prompt to place the single cache break at the most cache-friendly point without losing information. Validate on a real Anthropic endpoint.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| Prompt caching | "make long prompts cheap" | Reusing a provider-side KV-cache for matching prefixes; repeat input tokens billed at 50–90% off. |
| `cache_control` | "Anthropic's marker" | A content-block attribute that declares "everything up to here is cacheable"; `{"type": "ephemeral"}`. |
| Cache write | "pay the premium" | The first request that populates the cache; Anthropic bills at ~1.25× input rate, OpenAI is free. |
| Cache read | "the discount" | Subsequent requests matching the prefix; billed at 10% (Anthropic), 50% (OpenAI), ~25% (Gemini). |
| TTL | "how long it lives" | Seconds the cache stays warm; Anthropic defaults 5m (extendable 1h), OpenAI best-effort up to 1h, Gemini user-set. |
| Extended TTL | "Anthropic's 1-hour cache" | `{"type": "ephemeral", "ttl": "1h"}`; 2× write premium but worthwhile for any batch reusing the prefix 5+ times. |
| Prefix matching | "why my cache misses" | The cache hits only if every token from the start to the break is byte-identical. |
| Context caching (Gemini) | "the explicit one" | Google's named, storage-billed cache object; best for multi-day reuse of large corpora. |

## Further Reading

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, 1h TTL, break-even tables.
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — Automatic prefix matching.
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API and storage pricing.
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — Original launch blog with latency numbers.
- Phase 11 · 05 (Context Engineering) — Where to cut the prompt so caching can land.
- Phase 11 · 11 (Caching & Cost) — Pairing prompt caching with semantic caching on the user-message layer.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — The KV-cache memory model that prompt caching exposes to users; explains why re-reading a cached prefix is ~10× cheaper than recomputing.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — The prefill phase is what prompt caching short-circuits; this paper explains why TTFT drops sharply on cache hits while TPOT is unaffected.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — Prompt caching, like speculative decoding, Flash Attention, and MQA/GQA, is one of several levers bending the inference cost curve; read this to understand the other three.
