# FinOps for LLMs — Unit Economics and Multi-Tenant Attribution

> Traditional FinOps breaks down for LLM spending. Cost is a token transaction, not resource uptime. Tags do not map — an API call is a transaction, not an asset. Engineering decisions (prompt design, context window, output length) are financial decisions. The 2026 playbook has three attribution dimensions to instrument from day one: per-user (`user_id`) for seat pricing and expansion, per-task (`task_id` + `route`) for product-facing cost and prioritization, per-tenant (`tenant_id`) for unit economics and renewals. Four token layers — prompt, tool, memory, response — a single bucket hides the spend. Enforcement ladder for multi-tenant products: per-tenant rate limiting (2-3x expected peak, clear 429 + retry-after); daily spend cap (1.5-3x contract ceiling; triggers tighter throttling + alert); kill switch when spend z-score > 4 (auto-suspend + page on-call). Attribution patterns: tag aggregation, telemetry stitching (trace-ID to bill; highest accuracy), sampling extrapolation, model-based apportionment, event sourcing, real-time streaming. Unit metrics: cost per resolved query, cost per generated artifact — not $/M tokens. Retroactive tagging always leaks; instrument at request creation time.

**Type:** Learn
**Languages:** Python (stdlib, a toy cost-attribution simulator with kill switch)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Time:** ~60 minutes

## Learning Objectives

- Explain why traditional FinOps (tags + tiering) breaks down for LLM spending and name three new attribution dimensions.
- List the four token layers (prompt, tool, memory, response) and why single-bucket billing hides cost.
- Design an enforcement ladder for a multi-tenant product (rate limiting -> spend cap -> kill switch).
- Choose a unit metric (cost per resolved query/artifact) instead of $/M tokens.

## The Problem

Your bill says $40,000. You do not know:
- Which tenant spent it.
- Which product feature drove it.
- Whether any individual user is abusing it.
- Whether the culprit is prompt bloat, tool calls, or memory amplification.

Vendor-side tag aggregation works for cloud resources (EC2, S3) because tags propagate to billing line items. LLM API calls do not auto-tag — you must stamp user/task/tenant at the call site and carry it all the way through. Retroactive attribution always misses edge cases.

## The Concept

### Three Attribution Dimensions

**Per-user** (`user_id`): Who is spending how much. Drives seat pricing, expansion conversations, identifies heavy users.

**Per-task** (`task_id` + `route`): Which product surface is spending how much. Drives feature prioritization, decisions to cut expensive features.

**Per-tenant** (`tenant_id`): Which customer is profitable. Drives unit economics, renewal pricing, tiering thresholds.

Instrument all three at the call site from day one. Retroactive is always worse.

### Four Token Layers

| Layer | Example | Typical share of total |
|-------|---------|---------------------|
| Prompt | System + user input | 40-60% |
| Tool | Tool call results fed back | 20-40% (agent workloads) |
| Memory | Prior conversation / retrieved documents | 10-30% |
| Response | Model output | 10-30% |

Bucketing all four together makes optimization blind. Split them in your attribution schema.

### Enforcement Ladder

1. **Rate limiting** per tenant. 2-3x expected peak. Return 429 with `Retry-After`. Tenant feels friction; no surprise bills.

2. **Daily spend cap** per tenant. 1.5-3x contract ceiling. Trigger: tighten throttling + alert customer success team.

3. **Kill switch** when spend z-score relative to tenant baseline > 4. Auto-suspend tenant; page on-call; escalate to ops + CS.

### Attribution Patterns

- **Tag aggregation**: Stamp metadata headers; aggregate later. Simple; coarse.
- **Telemetry stitching**: Join traces to billing via trace ID. Highest accuracy. What mature teams do.
- **Sampling + extrapolation**: Sample 5-10%, multiply up. Cost-effective for rough allocation; misses tails.
- **Model-based apportionment**: Use regression to infer cost drivers. For untagged legacy data.
- **Event sourcing**: Cost as events in a stream (Kafka / Kinesis). Real-time.
- **Real-time streaming**: Sub-second dashboard updates.

### Cost-per-X Is the Unit Metric

$/M tokens is vendor parlance. Product metrics:

- Cost per resolved support ticket.
- Cost per generated article.
- Cost per successfully completed agent task.
- Cost per user session-minute.

Tie cost to a product outcome. Otherwise optimization has no anchor.

### Shape of a Cost Attribution Trace

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

Emit on every call. Store in a data lake. Aggregate by dimension. The observability stack from Phase 17 · 13 is where this lives.

### Stacked Savings

Stack: caching + batch + routing + gateway. All four applied:
- L2 cache (Phase 17 · 14): input ~10x cheaper.
- Batch (Phase 17 · 15): 50% off.
- Route to cheaper model (Phase 17 · 16): 60% cost reduction.
- Gateway efficiency (Phase 17 · 19): redundancy + retries.

Best-case stacked: approximately 5-10% of naive baseline. Most teams apply 2-3 levers; few stack all four.

### Numbers to Remember

- Attribution dimensions: per-user, per-task, per-tenant.
- Four token layers: prompt, tool, memory, response.
- Kill switch: spend z-score > 4.
- Unit metric: cost per resolved query, not $/M tokens.
- Stacked optimization: can reach approximately 5-10% of baseline.

## Use It

`code/main.py` simulates a multi-tenant LLM service with a three-tier enforcement ladder. Injects an abusive tenant and demonstrates the kill switch trigger.

## Ship It

This lesson produces `outputs/skill-finops-plan.md`. Given a product and scale, design an attribution schema and enforcement ladder.

## Exercises

1. Run `code/main.py`. At what z-score does the kill switch trigger? How would you choose the threshold?
2. Design a per-tenant, per-task cost dashboard. What are the first 5 views you build?
3. Your largest tenant has negative unit economics. Propose three interventions, ranked by customer impact.
4. Calculate cost-per-resolved-ticket for a support product: 3 million tokens per ticket, ~800 tickets per day, GPT-5 cached rate.
5. Argue whether retroactive tagging can ever work. When is it acceptable?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Per-user attribution | "User-level cost" | Stamp `user_id` on every call |
| Per-task attribution | "Feature cost" | `task_id` + `route` identifies the product surface |
| Per-tenant attribution | "Customer cost" | `tenant_id`; drives unit economics |
| Four token layers | "Cost layers" | prompt + tool + memory + response |
| Rate limiting | "429 guard" | Per-tenant cap enforced at the gateway |
| Daily spend cap | "Daily ceiling" | Tenant-level budget with alerting |
| Kill switch | "Auto-suspend" | Spend z-score > 4 triggers automatic suspension |
| Cost-per-resolved | "Product unit metric" | Cost tied to a product outcome, not tokens |
| Telemetry stitching | "Trace to bill" | Highest accuracy attribution pattern |
| Stacked optimization | "Cache+batch+route+gateway" | Stacked savings to approximately 5-10% of baseline |

## Further Reading

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
