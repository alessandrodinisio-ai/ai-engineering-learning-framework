# Cold Start Mitigation for Serverless LLMs

> A 20 GB model image takes 5-10 minutes (7B) to 20+ minutes (70B) from cold to serving. In a truly serverless world, that is not a warm-up — it is an outage. Mitigation operates at five layers: pre-seeded node images (Bottlerocket on AWS, dual-volume architecture), model streaming (NVIDIA Run:ai Model Streamer, native in vLLM), GPU memory snapshots (Modal checkpoint, up to 10x faster restarts), warm pools (`min_workers=1`), tiered loading (ServerlessLLM's NVMe→DRAM→HBM pipeline, 10-200x latency reduction), and live migration that moves input tokens (KB-scale) rather than KV cache (GB-scale). Modal publishes 2-4 second cold starts as the floor; Baseten defaults to 5-10 seconds, sub-second after warm-up. This lesson teaches you to measure, budget, and stack these five layers.

**Type:** Learn
**Languages:** Python (standard library, a toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 minutes

## Learning Objectives

- List the five layers of cold start mitigation and name one tool or pattern for each.
- Break down a 70B model's total cold start time as (node provisioning) + (weight download) + (weight load into HBM) + (engine initialization).
- Explain why live migration moves input tokens (KB) rather than KV cache (GB), and what the cost is (recomputation).
- State the warm pool tradeoff (paying for idle GPUs vs accepting cold start tails), and the SLA threshold at which `min_workers > 0` becomes mandatory.

## The Problem

Your serverless LLM endpoint scales to zero overnight. At 8 AM traffic spikes. The first request waits for:

1. Karpenter provisions a GPU node: 45-60 seconds.
2. Container pulls a 30 GB image with weights: 120-300 seconds.
3. Engine loads weights into HBM: 45-120 seconds depending on model size and storage speed.
4. vLLM or TRT-LLM initializes CUDA graphs, KV cache pools, tokenizer: 10-30 seconds.

Total: 220-510 seconds (roughly 3-8 minutes) before a single token comes back. Your SLA is 2 seconds. You add a warm pool (`min_workers=1`), the problem appears to vanish — but now you pay for one idle GPU 24/7. If your service has 5 products each with one warm replica, that is 5 × 24 × 30 = 3,600 GPU-hours/month regardless of whether a single user ever called.

Cold start mitigation is about approaching always-on latency while preserving serverless economics.

## The Concept

### Layer 1 — Pre-seeded node images (Bottlerocket)

On AWS, Bottlerocket's dual-volume architecture separates OS from data. Snapshot the data volume with container images pre-pulled; reference the snapshot ID in your `EC2NodeClass`. New nodes boot with weights already on local NVMe — steps 2 and part of 3 disappear. Native with Karpenter. Typical saving: 2-4 minutes per cold start for large models.

GCP equivalent: custom VM images with pre-baked container layers. Azure: managed disk snapshots with the same pattern.

### Layer 2 — Model streaming (Run:ai Model Streamer)

Instead of loading the full file before answering the first request, stream weights layer by layer into GPU memory and start processing as soon as the first transformer block is resident. NVIDIA Run:ai Model Streamer ships natively in vLLM 2026. Works with S3, GCS, and local NVMe. Roughly halves weight load time for large models by overlapping I/O with compute setup.

### Layer 3 — GPU memory snapshots (Modal)

Modal checkpoints GPU state (weights, CUDA graphs, KV cache regions) after the first load. Subsequent restarts deserialize directly into HBM — 10x faster than re-initialization. This is the closest thing to "2-second start for a hot GPU." Tradeoff: snapshots are topology-specific, so if Karpenter migrates you to a different SKU, the checkpoint must be redone.

### Layer 4 — Warm pool (min_workers=1)

Simplest mitigation: always keep one replica ready. Cost is one GPU's hourly rate 24/7. The math is punishing for small models (you pay $0.85-$1.50/hour to avoid a 30-second cold start) but friendly for large models (pay $4/hour to avoid a 5-minute cold start). The SLA threshold at which warm pool becomes mandatory: typically TTFT P99 < 60 seconds on 70B+ models.

### Layer 5 — Tiered loading (ServerlessLLM)

ServerlessLLM treats storage as a hierarchy: NVMe (fast but large), DRAM (medium but tiered), HBM (small but instant). Weights are pre-loaded into DRAM; loaded into HBM on demand. The paper reports 10-200x latency reduction compared to naive disk-to-HBM. Production adoption is early, but integration with vLLM exists.

### Layer 6 — Live migration (bonus mode)

When a node becomes unavailable (spot eviction, node drain), the traditional pattern is cold-starting another replica and draining the request queue. Live migration moves input tokens (kilobytes) to a target that already has the model loaded, then recomputes KV cache on the target. Recomputation is cheaper than transferring several GB of KV cache over the network. Applicable to disaggregated deployments.

### The warm pool math

For a service with a P99 TTFT SLA of 2 seconds, the question is not "should we warm pool" but "how many warm replicas, on which paths." 

- High-value interactive paths (live chat, voice agents): `min_workers=1-2`.
- Background batch paths (overnight classification): accept scale-to-zero, 5-10 minute cold start is tolerable.
- Premium tier: per-tenant `min_workers`, dedicated compute.

### Measure before optimizing

Anatomy of a 70B model cold start on a fresh node (illustrative):

| Stage | Time | Mitigation |
|-------|------|-----------|
| Node provisioning | 50s | Bottlerocket + pre-seeded image, warm pool |
| Image pull | 180s | Pre-seeded data volume (eliminated) |
| Weights to HBM | 75s | Model streaming (halved); GPU snapshot (eliminated) |
| Engine initialization | 20s | Persisted CUDA graph cache |
| First forward pass | 3s | Minimum inherent latency |
| **Cold start total** | **328s** | |
| **With mitigation total** | **~15s** | 22x reduction |

### Numbers to remember

- Modal cold start: 2-4 seconds (with GPU snapshot).
- Baseten default cold start: 5-10 seconds; sub-second after warm-up.
- Bare 70B cold start: 3-8 minutes.
- Run:ai Model Streamer: ~2x weight load speedup.
- ServerlessLLM tiered loading: 10-200x latency reduction (paper numbers).

## Use It

`code/main.py` models a cold start path with and without each mitigation layer. Reports total cold start time, warm pool cost, and the break-even request rate at which warm pool starts paying for itself.

## Ship It

This lesson produces `outputs/skill-cold-start-planner.md`. Given an SLA, model size, and traffic shape, it picks which mitigations to stack.

## Exercises

1. Run `code/main.py`. Calculate the break-even request rate — above which, one warm replica is cheaper than paying the "cold start tax" by missing more requests on SLO.
2. You deploy a 13B model with a P99 TTFT SLA of 3 seconds. Pick the smallest mitigation stack (fewest layers) that achieves it.
3. Bottlerocket pre-seeding eliminates image pull, but weights still load from snapshot to HBM. If the snapshot-backed NVMe reads at 7 GB/s, calculate the wall-clock time for a 70B model.
4. Your serverless vendor offers GPU snapshots (Modal), and your team rejects them citing "snapshots leak PII." Argue both sides — what is the real risk, and what is the mitigation (ephemeral snapshots, encryption, namespace isolation)?
5. Design a tiered warm pool strategy: how many warm replicas for paid users, trial users, and batch workloads? Show the math.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| Cold start | "the big stall" | Time from request to first token on a fresh replica |
| Warm pool | "always-on floor" | `min_workers >= 1`, keeping at least one replica ready |
| Pre-seeded image | "baked AMI" | Node image with container weights already resident |
| Bottlerocket | "AWS node OS" | AWS container-optimized OS with dual-volume snapshot support |
| Model streaming | "stream loading" | Overlapping weight I/O with compute setup |
| GPU snapshot | "checkpoint to HBM" | Serialized post-load GPU state; deserializes on restart |
| Tiered loading | "NVMe + DRAM + HBM" | Storage hierarchy; on-demand loading |
| Live migration | "move tokens" | Transfer inputs (KB), recompute KV on target |
| `min_workers` | "warm replicas" | Serverless minimum keepalive count |
| Scale to zero | "fully serverless" | Zero cost when idle; accepts full cold start tax |

## Further Reading

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start) — Modal's published benchmarks and checkpoint architecture.
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket) — pre-seeded data volume snapshot pattern.
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer) — overlapping weight load with compute setup.
- [Baseten — Cold-start mitigation](https://www.baseten.co/blog/cold-start-mitigation/) — warm-up playbook.
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu) — tiered loading design.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — live migration for disaggregated deployments.
