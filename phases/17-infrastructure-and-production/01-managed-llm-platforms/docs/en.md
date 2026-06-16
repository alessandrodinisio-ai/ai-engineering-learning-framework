# Managed LLM Platforms — Bedrock, Vertex AI, Azure OpenAI

> Three hyperscalers, three radically different strategies. AWS Bedrock is a model marketplace — Claude, Llama, Titan, Stability, Cohere all sit behind the same API. Azure OpenAI is an exclusive partnership with OpenAI, plus Provisioned Throughput Units (PTUs) for dedicated compute. Vertex AI leads with Gemini, telling the best story on long context and multimodality. In 2026, Artificial Analysis measured median latency on Llama 3.1 405B class models at approximately 50 ms for Azure OpenAI and approximately 75 ms for Bedrock — the gap is explained by PTUs, since dedicated compute inherently outperforms shared on-demand. The decision criterion is not "who's fastest" but "whose model catalog and FinOps view match my product." This lesson teaches you to write down the tradeoffs before choosing, not go by gut feeling.

**Type:** Learn
**Languages:** Python (standard library, a toy-level cost-vs-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## Learning Objectives

- Name the three platform strategies (marketplace vs exclusive vs Gemini-first) and map each to a product use case.
- Explain what Provisioned Throughput Units (PTUs) buy you in Azure OpenAI, and why on-demand Bedrock typically adds ~25 ms at the 405B scale.
- Diagram each platform's FinOps attribution view (Bedrock Application Inference Profiles vs Vertex per-team projects vs Azure scopes + PTU reservations).
- Write a "dual-vendor from day one" strategy and explain why single-vendor lock-in is the most expensive mistake of 2026.

## The Problem

You picked Claude 3.7 Sonnet for your product. Now you need to serve it. You can call the Anthropic API directly, go through AWS Bedrock, or route through a gateway. Direct API is simplest; Bedrock adds BAA, VPC endpoints, IAM, and CloudWatch attribution. A gateway provides failover across multiple vendors, unified billing, and rate limiting.

The deeper problem is the catalog. If your product needs Claude, Llama, and Gemini simultaneously, you cannot buy them all from one place — unless that "place" is Bedrock plus Vertex plus Azure OpenAI together. These hyperscalers are not interchangeable — each has bet on different players to control the model layer.

This lesson lays out the three bets, the latency gap, the FinOps gap, and the lock-in risk.

## The Concept

### Three Strategies

**AWS Bedrock** — the marketplace. Claude (Anthropic), Llama (Meta), Titan (AWS first-party), Stability (images), Cohere (embeddings), Mistral, plus sub-catalogs for images and embeddings. One API, one IAM view, one CloudWatch export. Bedrock's bet: customers want optionality, not any single model.

**Azure OpenAI** — the exclusive partnership. You get GPT-4 / 4o / 5 / o-series, DALL·E, Whisper, and the ability to fine-tune OpenAI models inside Azure data centers. The "Azure OpenAI Service" catalog contains zero non-OpenAI models — those go to Azure AI Foundry (a separate product). Azure's bet: OpenAI remains the frontier, and customers want enterprise controls on top of that specific relationship.

**Vertex AI** — Gemini first, everything else second. Gemini 1.5 / 2.0 / 2.5 Flash and Pro, plus Model Garden (third-party). Vertex's bet is multimodal long context — Gemini's 1M token context window is the differentiator.

### The Latency Gap at Scale

Artificial Analysis runs continuous benchmarks. On equivalent Llama 3.1 405B deployments (shared on-demand), Azure OpenAI's median time-to-first-token is approximately 50 ms; Bedrock's is approximately 75 ms. The gap is not AWS doing something wrong — it's a difference in compute models. Azure sells PTUs (Provisioned Throughput Units) that reserve GPU compute for your tenant. Bedrock's equivalent (Provisioned Throughput) also exists, but starts at approximately $21/unit/hour, and most customers stay on shared on-demand.

Shared on-demand compute competes with every other customer's traffic for resources. Dedicated compute does not compete. If your product SLA demands P99 TTFT < 100 ms, you either buy PTUs on Azure, buy Bedrock Provisioned Throughput, or accept the default variance.

### The Economics of Provisioned Throughput

Azure PTUs: a reserved block of inference compute. Saves up to approximately 70% versus on-demand for predictable workloads. Hourly cost is fixed regardless of traffic — you pay for the reservation even when idle. Break-even is typically at approximately 40-60% sustained utilization.

Bedrock Provisioned Throughput: $21-50/hour depending on model and region. Same math — break-even at approximately half of peak utilization. Requires a monthly commitment.

Vertex's reserved compute is sold by Gemini SKU; pricing varies by model and region and is less publicly advertised.

### FinOps Views — The Real Differentiator

**Bedrock Application Inference Profiles** are the cleanest attribution in the marketplace model. Tag a profile with `team`, `product`, `feature`; route all model calls through it; CloudWatch breaks out cost per profile without post-processing. Added in 2025, it remains the most granular native capability among hyperscalers.

**Vertex** attribution is per-team projects plus labels everywhere. You model each team as a GCP project, label every resource, and aggregate via BigQuery Billing Export + DataStudio. More work, but BigQuery lets you run arbitrary SQL against cost data.

**Azure** relies on subscription/resource-group scopes plus tags, with PTU reservations as first-class cost objects. Tags are inherited from resource groups, not from requests, so per-request attribution requires Application Insights custom metrics or a gateway that stamps headers.

The pattern: Bedrock is cleanest natively, Vertex is most flexible via BigQuery, Azure is least transparent — unless you instrument it yourself.

### Lock-in Is the Risk of 2026

When a single model dominates, betting on a single hyperscaler is fine. But in 2026, the frontier shifts monthly — this quarter it's Claude 3.7, next quarter Gemini 2.5, the one after that GPT-5. Locking into one platform cuts you off from two-thirds of the frontier.

The pattern adopted by working teams: dual-vendor from day one for any product-critical LLM call. Bedrock plus Azure OpenAI is the common pair — Claude from one, GPT from the other, failover between them, same gateway. Cost overhead is negligible because the gateway routes optimally; the availability gain during outages (e.g., the January 2025 Azure OpenAI incident, AWS us-east-1 outages) is decisive.

### Data Residency, BAA, and Regulated Industries

Bedrock: BAA available in most regions; VPC endpoints; guardrails. Common fintech default.
Azure OpenAI: HIPAA, SOC 2, ISO 27001; EU data residency; the enterprise regulated default.
Vertex: HIPAA, GDPR, per-region data residency; Google Cloud's compliance stack.

All three pass the basic checkbox. Differences lie in data retention policies, how logs are handled, and whether abuse monitoring reads your traffic (most default to opt-in; enterprise tiers allow opt-out).

### Numbers You Should Remember

- Azure OpenAI median TTFT on Llama 3.1 405B class: ~50 ms (with PTUs).
- Bedrock on-demand median TTFT: ~75 ms.
- Bedrock Provisioned Throughput: $21-50/unit/hour.
- Azure PTU break-even: ~40-60% sustained utilization.
- PTU savings vs on-demand at high utilization: up to 70%.

## Use It

`code/main.py` compares the three platforms on a synthetic workload — it models the economics of on-demand vs PTU, TTFT variance, and cost attribution granularity. Run it and see where PTUs pay off and where the marketplace model's breadth outweighs the TTFT gap.

## Ship It

This lesson produces `outputs/skill-managed-platform-picker.md`. Given a workload profile (which models needed, TTFT SLA, daily volume, compliance requirements), it recommends a primary platform, a fallback platform, and a FinOps instrumentation plan.

## Exercises

1. Run `code/main.py`. For a 70B-class model, at what sustained utilization does the Azure PTU beat on-demand? Compute the break-even and compare to the advertised 40-60% range.
2. Your product needs Claude 3.7 Sonnet and GPT-4o. Design a dual-vendor deployment — which model on which hyperscaler, what gateway in front, what failover strategy?
3. A regulated healthcare customer requires BAA, US-East data residency, and P99 TTFT under 100 ms. Choose a platform and justify with three specific features.
4. You discover your Bedrock bill quadrupled this month with no traffic increase. Without Application Inference Profiles, how do you find the culprit? With profiles, how long does it take?
5. Read through the Azure OpenAI and Bedrock pricing pages. For a 100M tokens/month Claude workload, which is cheaper — direct Anthropic API, Bedrock on-demand, or Bedrock Provisioned Throughput?

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| Bedrock | "AWS's LLM service" | Model marketplace spanning Claude, Llama, Titan, Mistral, Cohere |
| Azure OpenAI | "Azure's ChatGPT" | Exclusive OpenAI models in Azure data centers with enterprise controls |
| Vertex AI | "Google's LLM" | Gemini-first platform with Model Garden for third-party models |
| PTU | "Dedicated compute" | Provisioned Throughput Unit — reserved inference GPUs billed hourly |
| Application Inference Profile | "Bedrock tagging" | Tagged per-product cost/usage profile with native CloudWatch support |
| Model Garden | "Vertex catalog" | Vertex AI's third-party model section, separate from Gemini |
| Dual-vendor from day one | "LLM redundancy" | Strategy of running every critical LLM path across >=2 hyperscalers |
| BAA | "HIPAA paperwork" | Business Associate Agreement; required when handling PHI; all three offer it |
| Abuse monitoring | "The log reader" | Vendor-side safety scanning of prompts/outputs; enterprise tiers allow opt-out |

## Further Reading

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — Authoritative rate cards and Provisioned Throughput pricing.
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU economics and rate cards.
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini tiers and Model Garden markups.
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — Continuous cross-vendor latency and throughput benchmarks.
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — Enterprise decision framework.
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — Attribution mechanisms compared side by side.
