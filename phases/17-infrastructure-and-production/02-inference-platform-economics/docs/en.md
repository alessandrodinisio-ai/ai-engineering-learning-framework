# Inference Platform Economics — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> The 2026 inference market is no longer about renting GPU time. It has split into three segments: custom silicon (Groq, Cerebras, SambaNova), GPU platforms (Baseten, Together, Fireworks, Modal), and API-first marketplaces (Replicate, DeepInfra). Fireworks raised its per-GPU-hour price by $1 on May 1, 2026, while the $4B valuation and 10T+ tokens processed daily tell you: the volume play works. Baseten closed a $300M Series E at a $5B valuation in January 2026. The competitive positioning follows a simple pattern: Fireworks optimizes for latency, Together for catalog breadth, Baseten for enterprise polish, Modal for Python-native developer experience, Replicate for multimodal coverage, Anyscale for distributed Python. This lesson gives you a comparison matrix you can hand directly to a founder.

**Type:** Learn
**Languages:** Python (standard library, a toy-level per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## Learning Objectives

- Name the three market segments (custom silicon, GPU platforms, API-first) and map each vendor to a segment.
- Explain why per-token API pricing compresses toward the serving engine's cost curve rather than the hardware's cost curve.
- Compute effective per-request cost across at least three vendors and explain when per-minute billing (Baseten, Modal) beats per-token.
- For a given workload type (serverless bursty, steady high-throughput, fine-tuned variant, multimodal), identify which platform is the appropriate default.

## The Problem

You evaluated managed hyperscaler platforms. You decided you need a narrower, faster vendor — Fireworks for latency, Together for breadth, Baseten for a fine-tuned custom model. Now you have six real options in front of you, and the pricing pages do not align. Fireworks quotes $/M tokens; Baseten quotes $/minute; Modal quotes $/second; Replicate quotes $/prediction. Without modeling your workload, you cannot put them side by side for a fair comparison.

Worse, the business model behind each pricing page is different. Fireworks runs its custom engine (FireAttention) on shared GPUs; the per-token rate reflects their utilization curve. Baseten gives you Truss + dedicated GPUs; the per-minute rate reflects exclusivity. Modal is true Python serverless — billed per second with sub-second cold starts. The same output (an LLM response), three different cost functions.

This lesson models all six and tells you when each one wins.

## The Concept

### Three Segments

**Custom silicon** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). Decode is typically 5-10x faster than GPU-based clusters on the same model. Per-token price is higher (Groq was approximately $0.99/M for Llama-70B in late 2025), but unmatched for latency-sensitive scenarios. Groq is the production default for voice agents and real-time translation.

**GPU platforms** — Baseten, Together, Fireworks, Modal, Anyscale. Running on NVIDIA (H100, H200, B200 in 2026), sometimes AMD. This is the economic layer between "bare GPU rental" (RunPod, Lambda) and "hyperscaler managed services" (Bedrock).

**API-first marketplaces** — Replicate, DeepInfra, OpenRouter, Fal. Broad catalogs, per-prediction or per-second billing, emphasis on "time to first API call."

### Fireworks — Latency-Optimized GPU Platform

- FireAttention engine (custom); claims 4x lower latency than vLLM on equivalent configurations.
- Batch tier for non-interactive workloads at approximately 50% of serverless rates.
- Fine-tuned models served at the same rate as the base model — a genuine differentiator versus vendors that charge a premium for your LoRA.
- Mid-2026: on-demand GPU rental increased $1/hour effective May 1. Volume pricing negotiable at scale.
- Financial signal: $4B valuation, 10T+ tokens processed daily.

### Together — Breadth-Optimized

- 200+ models with open-source versions tracked within days of upstream release.
- 50-70% cheaper than Replicate on equivalent LLM models — the "AI Native Cloud" positioning is about volume and catalog.
- Inference + fine-tuning + training in one API.

### Baseten — Enterprise Polish-Optimized

- Truss framework: packages dependencies, secrets, and serving configuration into a single manifest.
- GPU range from T4 to B200. Billed per minute with reasonable cold-start mitigation.
- SOC 2 Type II, HIPAA-ready. Common fintech and healthcare default.
- $5B valuation, January 2026 Series E ($300M from CapitalG, IVP, NVIDIA).

### Modal — Python-Native Experience-Optimized

- Infrastructure-as-code in pure Python. Decorate a function with `@modal.function(gpu="A100")` and deploy with one command.
- Billed per second. Cold start 2-4 seconds with warm pools; <1 second for small models.
- $87M Series B, $1.1B valuation (2025). Highest developer experience scores in independent surveys.

### Replicate — Multimodal Breadth

- Per-prediction billing. The default platform for image, video, and audio models.
- Integration ecosystem (Zapier, Vercel, CMS plugins).
- Less competitive on LLM per-token rates, but wins on multimodal variety.

### Anyscale — Ray-Native

- Built on Ray; RayTurbo is Anyscale's proprietary inference engine (competing with vLLM).
- Best for distributed Python workloads — the inference step is just one node in a larger graph.
- Managed Ray clusters; tight integration with Ray AIR and Ray Serve.

### Per-Token vs Per-Minute — When Each Wins

Per-token makes sense when the workload is latency-insensitive and bursty — you only pay for what you consume. Per-minute makes sense when utilization is high and predictable — once you're saturating the GPU, per-minute wins.

Rule of thumb: for workloads sustaining above approximately 30% utilization on a dedicated GPU, per-minute (Baseten, Modal) begins to beat per-token (Fireworks, Together). Below that threshold, per-token wins because you avoid paying for idle time.

### Custom Engines Are the Real Moat

Every platform built on top of vLLM and SGLang claims a custom engine. FireAttention, RayTurbo, Baseten's inference stack. The custom engine claim is partly marketing — the honest statement is: vLLM + SGLang account for approximately 80% of production open-source inference, and platform-layer differentiation is in developer experience, attribution, and SLAs.

### Numbers You Should Remember

- Fireworks GPU rental: +$1/hour effective May 1, 2026.
- Fireworks claim: 4x lower latency than vLLM on equivalent configs.
- Together: 50-70% cheaper than Replicate on LLMs.
- Baseten valuation: $5B (Series E, January 2026, $300M round).
- Modal valuation: $1.1B (Series B, 2025).
- Per-minute beats per-token above approximately 30% sustained utilization.

## Use It

`code/main.py` compares all six vendors across pricing models on a synthetic workload. Reports $/day and effective $/M tokens. Run it and find the break-even between per-token and per-minute.

## Ship It

This lesson produces `outputs/skill-inference-platform-picker.md`. Given a workload profile, SLA, and budget, it picks a primary inference platform and names the runner-up.

## Exercises

1. Run `code/main.py`. For a 70B model on an H100, at what sustained utilization does Baseten (per-minute) beat Fireworks (per-token)? Derive the crossover point yourself and compare to the rule of thumb.
2. Your product serves image generation + chat + speech-to-text. Pick a platform for each modality and name the gateway pattern that unifies them.
3. Fireworks raises the price on your primary model by $1/hour. If you shift 40% of traffic to the batch tier (50% discount), model the blended cost impact.
4. A regulated customer requires SOC 2 Type II + HIPAA + dedicated GPUs. Which three platforms qualify and which one wins on FinOps?
5. Compare cost per 1,000 predictions of Llama 3.1 70B on Fireworks serverless, Together on-demand, Baseten dedicated, and Replicate API. Which is cheapest at 10 predictions/day? At 10,000/day?

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| Custom silicon | "Non-GPU chips" | Groq LPU, Cerebras WSE, SambaNova RDU — optimized for decode |
| FireAttention | "Fireworks engine" | Custom attention kernel; claims 4x lower latency than vLLM |
| Truss | "Baseten's format" | Model packaging manifest; dependencies + secrets + serving config |
| Per-token | "API pricing" | Charged by tokens consumed; no cost for idle |
| Per-minute | "Dedicated pricing" | Charged by GPU wall-clock time; wins at high utilization |
| Per-prediction | "Replicate pricing" | Charged per model invocation; common for image/video |
| RayTurbo | "Anyscale engine" | Proprietary inference on Ray; competes with vLLM on Ray clusters |
| Batch tier | "50% off" | Discounted non-interactive queue; common on Fireworks, OpenAI |
| Fine-tuned at base rate | "Fireworks LoRA" | LoRA serving requests charged at the base model rate (differentiator) |

## Further Reading

- [Fireworks Pricing](https://fireworks.ai/pricing) — Per-token rates, batch tiers, GPU rental.
- [Baseten Pricing](https://www.baseten.co/pricing/) — Per-minute rates, committed compute, enterprise tiers.
- [Modal Pricing](https://modal.com/pricing) — Per-second GPU rates and free tier.
- [Together AI Pricing](https://www.together.ai/pricing) — Model catalog and per-token rates.
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo and managed Ray pricing.
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — Comparative evaluation.
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — Vendor landscape.
