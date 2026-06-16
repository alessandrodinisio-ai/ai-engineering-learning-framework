# Batch APIs — 50% Off Becomes Industry Standard

> Every major vendor ships an async batch API at 50% off with ~24-hour turnaround. OpenAI, Anthropic, Google, and most inference platforms (Fireworks batch tier, Together batch) all implement the same pattern. Stack batch with prompt caching and overnight pipelines drop to ~10% of synchronous uncached cost. The rule is brutally simple: if it's not interactive, it belongs in batch. Content generation pipelines, document classification, data extraction, report generation, bulk labeling, catalog tagging — anything that can tolerate 24-hour latency is money left on the table until moved to batch. The 2026 production pattern is triaging every new LLM workload into three lanes: interactive (synchronous with caching), semi-interactive (async queue with fallback), batch (overnight, stacked with cached input). Workloads that pretend to be interactive but can tolerate minutes of delay waste the most.

**Type:** Learn
**Languages:** Python (standard library, a toy batch vs synchronous cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt and Semantic Caching)
**Time:** ~45 minutes

## Learning Objectives

- Name three vendors' batch APIs (OpenAI, Anthropic, Google) and the common 50% off + 24h turnaround guarantee.
- Calculate the cost of stacking batch + cached input on an overnight classification workload and compare against the synchronous uncached baseline.
- Triage a workload into interactive / semi-interactive / batch and argue for the lane.
- State two pitfalls: partial interactivity (user expectation faster than 24h) and output schema drift (batch file formats differ across vendors).

## The Problem

Your team ships an overnight report generation pipeline. 50,000 documents, each summarized, summaries clustered, an executive brief drafted. Running synchronously takes 4 hours and costs $2,000/night. You hear about batch APIs.

Batch gives you 50% off. You also enable prompt caching on the system prompt (shared across all 50K calls). Stacked, the bill drops to $180/night — about 9% of baseline. Same pipeline, three config changes.

Batch is the cheapest lever in the LLM cost toolbox that nobody pulls. The reason is mostly organizational: teams think "real-time" when the SLA is actually "by morning." This lesson is about not leaving 90% of the bill on the table.

## The Concept

### Three Batch APIs

**OpenAI Batch API**: upload a JSONL file with a list of requests. Guaranteed 24-hour turnaround (in practice typically ~2-8 hours). Input and output tokens at 50% off. `/v1/batches` endpoint. Cacheable inputs can also stack cached input pricing on top.

**Anthropic Message Batches**: JSONL upload. 24-hour turnaround. 50% off. Supports `cache_control` — cache writes are explicit, reads happen automatically within the batch.

**Google Vertex AI Batch Prediction**: BigQuery or GCS input. Similar 50% off for Gemini. Integrates with Vertex Pipelines.

### Semantics: Async, Not Slow

Batch is "I guarantee a return within 24 hours" — not "this takes 24 hours." Typical P50 is 2-6 hours. Vendors schedule your batch into low-utilization GPU inventory windows.

### Stacking with Cache

A 50K-document summarization with the same 4K token system prompt:

- Synchronous uncached: 50000 × ($input × 4000 + $output × 200), at full price.
- Synchronous cached: system prompt is cached after first write; remaining 49999 get 10x cheaper input.
- Batch cached: all of the above, plus 50% off on both reads and writes.

The stack: batch + cache = ~10% of synchronous uncached bill. Any overnight workload with a shared system prompt should use this.

### Workload Triaging

**Interactive** — user waits for the response. TTFT matters. Synchronous calls with prompt caching. Cannot batch.

**Semi-interactive** — user submits a task, comes back in a few minutes. Async queue with fallback to synchronous (when batch is unavailable). Think moderate-volume RAG indexing.

**Batch** — user expects results "by morning" or "within the next hour." Content pipelines, classification at scale, offline analysis. Always batch, always stack cache.

Common mistake: classifying everything as interactive because the pipeline is production. Production is not a latency spec — SLA is.

### The Partial Interactivity Trap

Some features look interactive but can tolerate 5-10 minutes. Example: an overnight customer health report with a "refresh" button. User hits refresh; waiting 10 minutes is fine. Team built it synchronous. The cost of 50 concurrent refreshes is 10x "batch and deliver via email."

The question to ask: "what does 24 hours mean for this user?" If the answer is "they won't notice," batch it.

### The Output Schema Pitfall

Batch file formats differ across vendors:

- OpenAI: JSONL, one request per line.
- Anthropic: JSONL, one message per line; response format inline.
- Vertex: BigQuery table, or GCS prefix with TFRecord.

Writing "one batch client" across vendors means adapter code per vendor. Gateways that advertise multi-vendor batch (Portkey, some LiteLLM tiers) are still thin wrappers over the raw formats.

### Numbers You Should Remember

- Batch discount across vendors: 50% off input + output uniformly.
- Turnaround SLA: 24 hours guaranteed, typical P50 is 2-6 hours.
- Stacked batch + cached input: ~10% of synchronous uncached cost.
- Workload triaging rule: if 24h delay is acceptable, always batch.

## Use It

`code/main.py` calculates costs for a 50K-document workload across synchronous, synchronous+cached, batch, and batch+cached. Reports savings in $ and percentages.

## Ship It

This lesson produces `outputs/skill-batch-triager.md`. Given workload characteristics, triages into interactive/semi-interactive/batch and estimates savings.

## Exercises

1. Run `code/main.py`. For a 100K-document pipeline with a 3K token system prompt and 500 token output, calculate the full stack (batch + cache) savings vs synchronous baseline.
2. Pick three features in a real product you know. Triage each into interactive/semi-interactive/batch.
3. A user complains their report took 3 hours. Is this a batch triage mistake, or reasonable interactivity? Write the decision criteria.
4. Your batch API has a 24h return SLA but P99 is 20 hours. How do you communicate this to users — what is the behavior of downstream systems at the edge case?
5. Calculate the break-even: at what shared prefix length does batch + cache become cheaper than running overnight on your own reserved GPUs?

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| Batch API | "Async discount" | 50% off, 24h turnaround |
| JSONL | "Batch format" | One JSON request per line; OpenAI/Anthropic standard |
| Message Batches | "Anthropic batch" | Anthropic's batch API product name |
| Batch prediction | "Vertex batch" | Vertex AI's batch API product |
| Turnaround SLA | "24h promise" | A guarantee, not typical; typical is 2-6h |
| Workload triaging | "Interactivity decision" | Routing decision between interactive / semi-interactive / batch |
| Output schema | "Response format" | Per-vendor JSONL layout; not portable |
| Stacked discount | "Batch + cache" | ~10% of uncached synchronous bill when both apply |

## Further Reading

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — JSONL format and `/v1/batches` semantics.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — Batch format and `cache_control` interaction.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction) — Gemini batch semantics.
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
