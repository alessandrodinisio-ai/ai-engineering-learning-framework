# Model Routing as a Cost Primitive

> A dynamic intermediary evaluates each request (task type, token length, embedding similarity, confidence) and routes simple queries to cheap models while escalating complex ones to frontier models. Also called model cascading. Production cases show 20-60% cost reduction at equal quality across US/UK/EU deployments; on high-volume SaaS, a 30% routing efficiency gain translates to six-figure annual savings. The 2026 context is that LLM inference prices drop roughly 10x per year — a GPT-4-class token went from ~$20/M in late 2022 to ~$0.40/M in 2026. Most of that drop comes from better serving stacks (Phase 17 · 04-09), not hardware. Routing is how you convert that price drop into margin without degrading the product. The failure mode is cheap-model creep: the router sends 40% to a weaker model, quality drops 3-5% on reasoning tasks, and nobody notices for an entire quarter. Gate your routes with online quality metrics, not just offline eval sets.

**Type:** Learn
**Languages:** Python (stdlib, a toy cascade router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 min

## Learning Objectives

- Explain model cascading: cheap-first with a confidence check, escalating on low confidence.
- List four routing signals (task classification, prompt length, embedding similarity to known-hard sets, first-pass self-confidence).
- Calculate expected blended cost given target routing splits and quality-loss tolerance.
- Name the drift-monitoring metric that catches cheap-model creep (online quality gate).

## The Problem

Your service spends $80k/month on GPT-5. Your analytics show 70% of queries are trivial: "What time is it in Paris?" "Rephrase this sentence." A Haiku-class model handles these perfectly at 3% of the cost. 30% need GPT-5 reasoning — code, math, multi-step planning.

If you route that 70% to cheap and 30% to expensive, your bill drops ~65% at the same product quality. That's routing. The trick is building that intermediary without letting quality degrade.

## The Concept

### Four Routing Signals

1. **Task classification**: simple / complex / code-gen / math / chat. Can be a rule-based classifier, a small LLM (Haiku-class at $0.25/M), or embedding similarity to labeled buckets. Output: route = cheap / balanced / frontier.

2. **Prompt length**: Prompts >4K tokens often need frontier models for coherence. <500 tokens typically don't.

3. **Embedding similarity to known-hard sets**: If the query is close (cosine > 0.88) to a known-hard bucket, escalate directly to frontier.

4. **First-pass self-confidence**: Send to cheap first; if the model's log-probs show low confidence, or it refuses, or the output contains hedging language, retry on frontier. Adds ~P95 latency to ~10% of traffic but saves 50%+ on the other 90%.

### Three Patterns

**Pre-routing** (upfront classifier): Adds ~5-10ms latency; fastest overall.

**Cascade** (cheap-first, escalate on low confidence): ~1.2x median latency (cheap pass + verification), escalated ~2x. Best quality floor.

**Ensemble routing** (run cheap and frontier in parallel on a sample, reward model picks): Highest quality, highest cost; use only for critical A/B.

### Implementation

AI gateways (Phase 17 · 19) expose routing. LiteLLM has a `router` config with fallbacks and cost routing. Portkey has guard + routing. Kong AI Gateway has plugin-based routing. OpenRouter's model marketplace exposes a recommendation API.

Open-source: RouteLLM (LMSYS), Not Diamond (commercial), Prompt Mule.

### 2026 Price Curve

| Model Tier | Late 2022 | 2026 | Change |
|------------|-----------|------|--------|
| GPT-4-class quality | ~$20/M | ~$0.40/M | 50x cheaper |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | New tier |

Most improvement is serving efficiency — the core lessons in Phase 17 · 04-09 became vendor-side cost reductions. Routing lets you capture those gains at the application layer without waiting for all users to migrate to the cheap tier.

### Drift Is the Real Risk

Your router sends 40% to cheap. Over six months, the task distribution shifts (users become more sophisticated, ask longer questions). The router doesn't notice because its classifier was trained on Q1 data. Quality degrades silently. Nobody complains loudly enough. You find out in a lost competitive benchmark.

Gate your routes with online quality metrics:

- Per-route user thumbs-up / thumbs-down.
- Automated LLM judge on a held-out sample (5%) per route.
- Escalation rate: if the cascade routes up >30%, the cheap model is over-routed.
- Per-route refusal rate.

### Numbers You Should Remember

- Routing savings at equal quality in 2026: cases show 20-60%.
- LLM price drop 2022-2026: ~10x per year in aggregate.
- GPT-4-class 2022 vs 2026: ~$20/M → ~$0.40/M.
- Cascade latency impact: ~1.2x median, escalated ~2x (~10% of traffic).

## Use It

`code/main.py` simulates pre-routing, cascade, and ensemble on a mixed workload. Reports blended cost, quality loss, and escalation rate.

## Ship It

This lesson produces `outputs/skill-router-plan.md`. Given a workload and quality budget, pick a routing pattern and signals.

## Exercises

1. Run `code/main.py`. At what accuracy floor does cascade beat pre-routing?
2. Your user base is 30% enterprise (complex queries), 70% free tier (simple). Design a routing split. Which online metric gates it?
3. A route loses 2% quality but saves 40%. Should it ship? Depends on the product — argue both sides.
4. Implement a confidence check using OpenAI / Anthropic API logprobs. What threshold do you start with?
5. Over six months, escalation rate climbs from 8% to 22%. Diagnose three causes and give a fix for each.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Model routing | "cost intermediary" | Per-request dynamic model selection |
| Model cascading | "cheap-first escalation" | Run cheap, fall back to frontier on low confidence |
| Pre-routing | "classify first" | Upfront classifier; no re-run |
| Ensemble routing | "parallel pick" | Run multiple, reward model picks the best |
| Escalation rate | "up-route %" | Fraction of cascade requests that escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS model routing product |
| Drift | "cheap creep" | Distribution shift the router doesn't notice |
| Online quality gate | "live check" | Automated LLM judge on sampled live traffic |

## Further Reading

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives.
