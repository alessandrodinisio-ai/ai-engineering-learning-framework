# vLLM Production Stack with LMCache KV Offloading

> vLLM's production-stack is the reference Kubernetes deployment — router, engines, and observability wired together. LMCache is the KV-offloading layer that moves KV cache out of GPU memory and reuses it across queries and engines (CPU DRAM, then disk/Ceph). vLLM 0.11.0's KV Offloading Connector (January 2026) makes this asynchronous and pluggable via the Connector API (v0.9.0+). Offload latency is invisible to users. Even without shared prefixes, LMCache has value — when a GPU's KV slots are exhausted, preempted requests can resume from CPU rather than re-computing prefill. In published benchmarks on 16x H100 (80GB HBM) across 4x a3-highgpu-4g: when KV cache exceeds HBM, both native CPU offload and LMCache significantly improve throughput; at low KV occupancy, all configurations match baseline with minimal overhead.

**Type:** Learn
**Languages:** Python (stdlib, a toy KV spill simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 min

## Learning Objectives

- Draw the vLLM production-stack layers: router, engines, KV offload, observability.
- Explain the KV Offloading Connector API (v0.9.0+) and how 0.11.0's async path hides offload latency.
- Quantify when LMCache CPU-DRAM helps (KV > HBM) vs when it adds overhead (KV fits in HBM).
- Choose between native vLLM CPU offload and the LMCache connector given deployment constraints.

## The Problem

Your vLLM serving cluster shows GPU HBM at 100% with preemption events as soon as concurrency climbs. Requests are evicted, re-queued, and you re-prefill the same 2K-token prompt four times in one minute. GPU compute is spent on redundant prefills; goodput is far below raw throughput.

Adding more GPUs is linear cost. Adding more HBM isn't possible. But CPU DRAM is cheap — 512 GB+ per socket, orders of magnitude worse latency than HBM, but sufficient for "temporarily warm" KV cache.

LMCache moves KV cache to CPU DRAM, lets preempted requests resume quickly, and allows cross-engine reuse of repeated prefixes without each engine re-prefilling.

## The Concept

### vLLM production-stack

`github.com/vllm-project/production-stack` is the reference Kubernetes deployment:

- **Router** — cache-aware (Phase 17 · 11). Consumes KV events.
- **Engines** — vLLM workers. One per GPU or one per TP/PP group.
- **KV cache offload** — LMCache deployment or native connector.
- **Observability** — Prometheus scrape, Grafana dashboards, OTel traces.
- **Control plane** — service discovery, config, rolling updates.

Ships as Helm chart + operator.

### KV Offloading Connector API (v0.9.0+)

vLLM 0.9.0 introduced a Connector API for pluggable KV cache backends. Your engine offloads blocks to the connector; the connector stores them (RAM, disk, object storage, LMCache). When a request needs a block, the connector loads it back.

vLLM 0.11.0 (January 2026) added an async offload path — offloads can happen in the background so the engine doesn't block on them in the common case. End-to-end latency and throughput still depend on workload shape, KV cache hit rate, and system pressure; vLLM's own documentation notes that custom kernel offload may reduce throughput at low hit rates, and async scheduling has known interaction issues with speculative decoding.

### Native CPU Offload vs LMCache

**Native vLLM CPU offload**: engine-local. Stores KV blocks in host RAM. Fast to implement, zero network hops. Doesn't cross engines.

**LMCache connector**: cluster-scale. Stores blocks in a shared LMCache service (CPU DRAM + Ceph/S3 tier). Blocks are accessible to any engine. Published benchmarks on 16x H100.

Pick native when a single engine has HBM pressure. Pick LMCache when multiple engines share prefixes (RAG with common system prompts, multi-tenant with shared templates).

### Benchmark Behavior

Tests on 16x H100 (80 GB HBM) across 4x a3-highgpu-4g:

- Low KV occupancy (short prompts, low concurrency): all configurations match baseline; LMCache adds ~3-5% overhead.
- Medium occupancy: LMCache starts helping with cross-engine prefix reuse.
- KV exceeds HBM: both native CPU offload and LMCache significantly improve throughput; LMCache gains are larger due to cross-engine sharing.

### When LMCache Is Decisive

- Multi-tenant serving where system prompts are shared across tenants.
- RAG where document chunks repeat across queries.
- Fine-tuned variants (LoRA) on the same base — base model KV reuse reduces redundant work.
- Preemption-heavy workloads: resuming from CPU is cheaper than re-prefilling.

### When Not to Enable

- HBM pressure is low — you pay overhead for no benefit.
- Short contexts (<1K tokens) — transfer time > re-prefill.
- Single-tenant, single-prompt workloads — no reuse to capture.

### Integration with Disaggregated Serving

Phase 17 · 17's disaggregated serving + LMCache stack: KV transferred from the prefill pool to the decode pool that goes unused falls into LMCache; subsequent queries pull from LMCache. Phase 17 · 11's cache-aware router can route to the engine with a local cache or LMCache shared-cache match.

### Numbers You Should Remember

- vLLM 0.9.0: Connector API released.
- vLLM 0.11.0 (January 2026): async offload path; end-to-end latency impact depends on workload, KV hit rate, and system pressure (not an absolute guarantee).
- 16x H100 benchmark: LMCache helps when KV occupancy exceeds HBM.
- Low HBM pressure: 3-5% overhead for no benefit.

## Use It

`code/main.py` simulates a preemption-heavy workload with and without LMCache. Reports avoided re-prefills, throughput gain, and breakeven HBM utilization.

## Ship It

This lesson produces `outputs/skill-vllm-stack-decider.md`. Given workload shape and vLLM deployment, decide native vs LMCache vs neither.

## Exercises

1. Run `code/main.py`. At what HBM utilization does LMCache start paying off?
2. A tenant shares a 6K-token system prompt across 200 queries/hour. Calculate expected LMCache savings per tenant.
3. The LMCache service is a single point of failure. Design an HA strategy (replicas, fallback to native).
4. LMCache stores to Ceph on spinning disk. For a 4K-token KV on 70B FP8 (500 MB), what's the read time vs re-prefill?
5. Argue whether vLLM 0.11.0's async path is "free" — where is the overhead hidden?

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| Production-stack | "reference deployment" | vLLM's Kubernetes Helm chart + operator |
| Connector API | "KV backend interface" | vLLM 0.9.0+ pluggable KV storage interface |
| Native CPU offload | "engine-local spill" | Store KV in the same engine's host RAM |
| LMCache | "cluster KV cache" | Cross-engine KV cache service on CPU DRAM + disk |
| 0.11.0 async | "non-blocking offload" | Offload hidden behind the engine stream |
| Preemption | "evict to make room" | KV cache shuffling when HBM is full |
| Prefix reuse | "same system prompt" | Multiple queries share their beginning; cache hit |
| Ceph tier | "disk tier" | Persistent storage below DRAM in the cache hierarchy |

## Further Reading

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack) — Helm chart + operator.
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache) — Connector implementation.
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases) — async path details.
