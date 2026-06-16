# Disaggregated Prefill/Decode — NVIDIA Dynamo and llm-d

> Prefill is compute-bound; decode is memory-bound. Running both on the same GPU wastes one resource or the other. Disaggregation splits them onto separate pools and transfers the KV cache between them via NIXL (RDMA/InfiniBand or fallback to TCP). NVIDIA Dynamo (announced GTC 2025, 1.0 GA) sits above vLLM/SGLang/TRT-LLM — its Planner Profiler + SLA Planner automatically proportions the prefill:decode ratio to meet SLOs. NVIDIA's published throughput gains fall roughly in this range — developer.nvidia.com (2025-06) shows ~6x improvement for DeepSeek-R1 MoE on GB200 NVL72 + Dynamo under moderate latency regimes, while the Dynamo product page (developer.nvidia.com, undated) advertises up to 50x MoE throughput on GB300 NVL72 + Dynamo compared to Hopper. The "30x" figure is a community aggregate reported across the full Blackwell + Dynamo + DeepSeek-R1 stack; we found no single primary source stating exactly 30x, so treat it as directional. llm-d (Red Hat + AWS) is Kubernetes-native: prefill / decode / router as separate Services with per-role HPA. llm-d 0.5 adds tiered KV offloading, cache-aware LoRA routing, UCCL networking, and scale-to-zero. Economics: internal synthesis across multiple customer disclosures suggests that switching from colocated to disaggregated with Dynamo at constant SLA can save 30-40% on a $2M-class inference spend (i.e., $600-800K/year); that specific $2M→$600-800K figure is an internal composite, not a single published case — treat it as an order-of-magnitude anchor, not a citable source. Short prompts (<512 tokens, short outputs) aren't worth the transfer cost.

**Type:** Learn
**Languages:** Python (stdlib, a toy disaggregated vs colocated simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 min

## Learning Objectives

- Explain why prefill and decode have different optimal GPU allocations and quantify colocated waste.
- Draw the disaggregated architecture: prefill pool, decode pool, KV transfer via NIXL, router.
- State the conditions under which disaggregation is not worthwhile (short prompts, short outputs).
- Distinguish NVIDIA Dynamo (above-stack) from llm-d (Kubernetes-native) and map each to an operational scenario.

## The Problem

You're running Llama 3.3 70B on 8x H100. Under mixed workloads (long prompts + short outputs), GPUs idle during decode because most compute was spent in prefill. Under another workload (short prompts + long outputs), it's the reverse. Colocated prefill + decode means you over-provision for both.

Budget impact: 20-40% of GPU time is wasted on the wrong resource. You're buying H100 compute to run memory-bound decode, or buying H100 HBM bandwidth to run compute-bound prefill. Both are expensive waste.

Disaggregation splits prefill and decode onto separate pools sized to their own bottleneck. The KV cache transfers from the prefill pool to the decode pool via high-bandwidth interconnect.

## The Concept

### Why the Bottlenecks Differ

**Prefill** — runs the transformer over the entire input prompt in a single forward pass. Matrix multiplications dominate; compute-bound. H100 FP8 delivers ~2000 TFLOPS of useful throughput. Batch efficiency is good — a single forward pass processes many tokens.

**Decode** — generates one token at a time, reading the full weight set every iteration. Memory-bandwidth-bound. HBM3 delivers ~3 TB/s. Batch efficiency is only good at high concurrency — the cost of reading weights amortizes across the batch.

Colocated: you buy GPUs optimized for both. H100 is good at both, but either use case costs the same. At scale, you want the prefill pool on H100 / compute-heavy; the decode pool on H200 / memory-heavy, or with aggressive quantization.

### Architecture

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill Pool │ ─── NIXL ────► │  Decode Pool │
            │  (compute)   │                │   (memory)   │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL is NVIDIA's cross-node transfer. Uses RDMA/InfiniBand when available, falls back to TCP. Transfer latency is real — typically 20-80 ms for a 4K-token prompt's KV cache on 70B FP8. This is why short prompts aren't worth disaggregating: the transfer tax exceeds the savings.

### Dynamo vs llm-d

**NVIDIA Dynamo** (announced GTC 2025, 1.0 GA):
- Sits above vLLM, SGLang, TRT-LLM as an orchestrator.
- Planner Profiler measures workloads; SLA Planner auto-configures the prefill:decode ratio.
- Rust core, Python-extensible.
- Throughput gains: NVIDIA reports 6x for DeepSeek-R1 MoE on GB200 NVL72 + Dynamo under moderate latency (developer.nvidia.com, 2025-06); community reports of "up to 30x" across the full Blackwell + Dynamo + DeepSeek-R1 stack lack a single primary source and should be treated as directional.
- GB300 NVL72 + Dynamo: up to 50x MoE throughput compared to Hopper, per the Dynamo product page (developer.nvidia.com, undated).

**llm-d** (Red Hat + AWS, Kubernetes-native):
- Prefill / decode / router as separate Kubernetes Services.
- Per-role HPA using queue depth (prefill) / KV utilization (decode) signals.
- `topologyConstraint packDomain: rack` packs prefill+decode pods onto the same rack for high-bandwidth KV transfer.
- llm-d 0.5 (2026): tiered KV offloading, cache-aware LoRA routing, UCCL networking, scale-to-zero.

Use Dynamo if you want a managed above-stack orchestrator. Use llm-d if you want Kubernetes-native primitives and bet on the CNCF ecosystem.

### Economics

Internal synthesis (not a single published case — order-of-magnitude anchor):

- $2M/year inference spend on colocated deployment.
- Switch to disaggregated with Dynamo.
- Same request volume, same P99 latency SLA.
- Reported savings: $600-800K/year (30-40% reduction).
- No new hardware added.

We synthesized this figure from multiple customer disclosures rather than a single citable case; the closest published data points are Baseten achieving 2x faster TTFT / 61% higher throughput with Dynamo KV routing (baseten.co, 2025-10) and VAST + CoreWeave predicting 60-130% more tokens/$ at 40-60% KV hit rates (vastdata.com, 2025-12). Savings come from right-sizing each pool; prefill-heavy workloads (RAG with 8K+ prefixes) benefit more than balanced ones.

### When Not to Disaggregate

- Prompt < 512 tokens and output < 200 tokens: transfer tax exceeds the gain.
- Small clusters (< 4 GPUs): not enough pool diversity.
- Teams that can't operate two GPU pools with per-role scaling: Dynamo helps but isn't effortless.
- No RDMA fabric: TCP transfer tax is heavier.

### Router Integration with Phase 17 · 11

The disaggregated router is KV-cache-aware (Phase 17 · 11). A request lands on the decode pool holding its prefix — if no match, it flows through prefill → decode. Hit rates and disaggregation stack — the cache-aware router decides whether a fresh prefill is actually needed.

### MoE on Blackwell Is Where the Real Numbers Are

GB300 NVL72 + Dynamo shows 50x MoE throughput versus Hopper baselines. MoE expert routing is compute-heavy on prefill and memory-heavy on decode (expert caching), so disaggregation is a double win. 2026 frontier model serving is dominated by MoE (DeepSeek-V3, future GPT-5 variants).

### Numbers You Should Remember

Benchmark numbers drift — NVIDIA and inference stacks publish new results every quarter. Re-verify before citing.

- DeepSeek-R1 on GB200 NVL72 + Dynamo: ~6x throughput vs baseline under moderate latency (developer.nvidia.com, 2025-06); community claims of "up to 30x" across the full Blackwell + Dynamo stack are directional aggregates without a single primary source.
- GB300 NVL72 + Dynamo: up to 50x MoE throughput vs Hopper (developer.nvidia.com, undated).
- Savings anchor (internal synthesis, not a single case): $600-800K/year savings from $2M annual spend at constant SLA.
- Disaggregation threshold: prompt >512 tokens + output >200 tokens.
- KV transfer via NIXL: 20-80 ms for 4K prompt KV on 70B FP8.

## Use It

`code/main.py` simulates colocated vs disaggregated serving. Reports throughput, per-request cost, and the crossover point by prompt length.

## Ship It

This lesson produces `outputs/skill-disaggregation-decider.md`. Given a workload and cluster, decide whether to disaggregate.

## Exercises

1. Run `code/main.py`. At what prompt length does disaggregated beat colocated?
2. Design prefill and decode pools for a RAG service with P99 prefix length 8K and output 300.
3. Dynamo vs llm-d: pick one for a pure-Kubernetes team with no Python runtime preference.
4. Calculate KV transfer cost: 4K prefill on 70B FP8 = ~500 MB KV. At RDMA 100 GB/s, transfer = 5 ms. TCP 10 GB/s = 50 ms. Which matters for your SLA?
5. MoE expert routing changes KV access patterns. How does disaggregation perform for MoE where different experts activate per token?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Disaggregated serving | "split prefill/decode" | Separate GPU pools for each phase |
| NIXL | "NVIDIA transfer" | Dynamo's cross-node KV transfer (RDMA/TCP) |
| NVIDIA Dynamo | "orchestrator" | Above-stack coordinator for vLLM/SGLang/TRT-LLM |
| llm-d | "Kubernetes-native" | Red Hat + AWS K8s disaggregated stack |
| Planner Profiler | "Dynamo auto-config" | Measures workloads, configures pool ratios |
| SLA Planner | "Dynamo policy" | Auto-proportions prefill:decode to meet SLOs |
| `packDomain: rack` | "llm-d topology" | Packs prefill+decode onto same rack for fast KV |
| UCCL | "unified collective comms" | llm-d 0.5's networking layer for scale-to-zero |
| MoE expert routing | "per-token expert" | DeepSeek-V3 pattern; disaggregation helps |

## Further Reading

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
