# Multi-Region LLM Serving and KV Cache Locality

> For LLM inference with caching, round-robin load balancing is actively harmful. A request that doesn't land on the node holding its prefix pays full prefill cost — roughly 800 ms P50 on long prompts, versus ~80 ms for a cache hit. The 2026 production pattern is a cache-aware router (vLLM Router written in Rust, llm-d router) that consumes KV-cache events and routes by prefix hash matching. Recent research (GORGO) turns cross-region network latency into an explicit term in the routing objective. Commercial "cross-region inference" products (Bedrock cross-region inference, GKE Multi-Cluster Gateway) treat inference as opaque — they manage availability, not TTFT. JPMorgan Chase and Mayo Clinic ran a us-east-1 failover drill in November 2024 at ~22 minutes. The DR reality: 32% of LLM DR failures happen because teams back up weights but forget the tokenizer file or quantization config.

**Type:** Learn
**Languages:** Python (standard library, a toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 minutes

## Learning Objectives

- Explain why round-robin load balancing breaks cached inference, and quantify the TTFT penalty.
- Diagram a cache-aware router: inputs (KV-cache events), algorithm (prefix hash matching), tiebreaker (GPU utilization).
- State the 32% LLM DR failure cause (missing tokenizer files / quantization configs) and provide a three-file DR checklist.
- Distinguish commercial cross-region products (Bedrock CRI, GKE Multi-Cluster Gateway) from KV-aware routing.

## The Problem

Your service runs in us-east-1, us-west-2, and eu-west-1. You put an ALB in front doing round-robin. In production, prefix cache hit rate drops to 8%. TTFT P50 triples. Your vLLM logs show every request paying full prefill cost.

Round-robin is optimal for stateless services. LLM inference is stateful by design — the KV cache encodes everything the model has seen. Blind routing is routing into the wrong cache.

On the other side, your team has a DR plan. You replicate model weights cross-region to S3. A regional outage hits; you attempt failover; the replica refuses to start. You forgot that `tokenizer.json`, the quantization config, and the RoPE scaling config live in another bucket you didn't sync.

Multi-region LLM serving is a cache problem, a routing problem, and a DR hygiene problem — not a load balancer problem.

## The Concept

### Cache-Aware Routing

A request arrives with a prompt. The router hashes the prefix (e.g., the first 512 tokens); it asks each replica "do you have this prefix cached?". Replicas publish KV-cache events to a pub/sub channel as they allocate and evict blocks. The router picks the matching replica, falling back to GPU-utilization-based tiebreaking when no one matches.

**vLLM Router** (Rust, 2026 production-stack): subscribes to `kv.cache.block_added` events, maintains a prefix-hash → replica index, routes with O(1) lookup. Falls back to shortest queue depth on miss.

**llm-d router**: same pattern, Kubernetes-native. Events published via ControlPlane API.

**SGLang RadixAttention** (Phase 17 · 06) is the intra-replica counterpart. Cross-replica routing is strictly upstream.

### The Numbers

2K token prompt, Llama 3.3 70B FP8, TTFT P50 on H100:
- Cache hit (same replica, prefix resident): ~80 ms.
- Cache miss (cold prefill): ~800 ms.

10x gap. If your router hits 60-80% prefix cache across replicas, you approach single-replica performance on N replicas of capacity. If it hits 10%, you approach naive scale-out.

### Cross-Region Adds a New Constraint — Network Latency

Cross-region RTT:
- us-east-1 ↔ us-west-2: ~65 ms.
- us-east-1 ↔ eu-west-1: ~75 ms.
- us-east-1 ↔ ap-southeast-1: ~220 ms.

If routing sends a us-east-1 request to a hot prefix in ap-southeast-1, the saved prefill (800 → 80 ms) is drowned by the 440 ms round trip. GORGO (2026 research) makes this explicit — jointly minimizes `prefill_time + network_latency`, not just prefill. The answer is often to stay intra-region unless on prefill-dominated giant multi-MB prefixes.

### Commercial "Cross-Region Inference" Doesn't Help Here

AWS Bedrock cross-region inference automatically routes requests to other regions under compute pressure. It optimizes availability, not TTFT, and treats inference as opaque. GKE Multi-Cluster Gateway is the same — service-level failover, KV-cache-unaware.

Even with these, you still need an application-layer cache-aware router. They handle the "us-east-1 is on fire" case. Cache-aware routing handles the TTFT case.

### DR Hygiene — The 32% Missing-File Problem

The widely cited 2026 statistic: 32% of LLM DR failures happen because teams back up weights but forget:

- `tokenizer.json` or `tokenizer.model`
- Quantization config (`quantize_config.json`, AWQ scales, GPTQ zero-points)
- Model-specific config (RoPE scaling, attention masks, chat templates)
- Engine config (`vllm_config.yaml`, sampling defaults, LoRA adapter manifest)

The fix is a three-file minimum DR manifest:

1. All files under the HF model repo (weights + config + tokenizer).
2. Engine-specific serving config.
3. Deployment manifest (K8s YAML, Dockerfile, dependency lock).

Plus: run a DR drill quarterly. JPMorgan's us-east-1 drill hit 22-minute recovery in November 2024 only because the runbook had been rehearsed.

### Data Residency Is Orthogonal

EU customer PHI cannot leave the EU. If your cache-aware router sends a Paris-originated request to us-east-1 for a prefix match, you violate GDPR regardless of TTFT gains. Partition the router by residency boundaries before optimizing for cache.

### Numbers You Should Remember

- Cache hit vs miss TTFT gap: ~10x (80 ms vs 800 ms on a 2K prompt).
- Cross-region RTT US-EU: ~75 ms.
- DR failures: 32% due to missing tokenizer/quantization config.
- JPMorgan us-east-1 failover November 2024: 22 minutes (30-minute SLA).

## Use It

`code/main.py` simulates three routing strategies (round-robin, cache-aware intra-region, cache-aware global) on a multi-region workload. Reports cache hit rate, TTFT P50/P99, and cross-region billing.

## Ship It

This lesson produces `outputs/skill-multi-region-router.md`. Given regions, residency constraints, and SLAs, design a routing scheme.

## Exercises

1. Run `code/main.py`. Given 75 ms RTT, at what prompt length does cross-region routing beat local-only?
2. Your cache hit rate drops from 70% to 12%. Diagnose three possible causes and the observable that would confirm each.
3. Design a DR manifest for a 70B AWQ-quantized model served with vLLM, with 5 LoRA adapters. List every file and config.
4. Argue whether Bedrock cross-region inference is "good enough" for a fintech company with strict TTFT SLOs. Cite specific behaviors.
5. A Paris-originated request matches a prefix in us-east-1. Do you route it? Write the policy.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| Cache-aware routing | "Smart LB" | Routes to the replica holding KV cache by prefix hash match |
| KV-cache events | "Cache pub-sub" | Replicas publish block add/remove; router builds index |
| Prefix hash | "Cache key" | Hash of first N tokens used as router lookup |
| GORGO | "Cross-region routing research" | arXiv 2602.11688; network latency as explicit term |
| Cross-region inference | "Bedrock CRI" | AWS product; availability failover, not TTFT-aware |
| DR manifest | "Backup checklist" | Every file needed for recovery — not just weights |
| Data residency | "GDPR boundary" | Legal constraints on which region can see user data |
| RTT | "Round-trip time" | Network latency; US-EU 75 ms, US-APAC 220 ms |
| LLM-aware LB | "Cache-hit LB" | Cache-aware router as a product category |

## Further Reading

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1) — Cross-region KV-cache reuse with network latency term.
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — Availability failover documentation.
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — Cache-aware router source code.
