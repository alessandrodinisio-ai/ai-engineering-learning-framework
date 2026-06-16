# GPU Autoscaling on Kubernetes — Karpenter, KAI Scheduler, Gang Scheduling

> It's three layers, not one. Karpenter provisions nodes dynamically (under a minute, 40% faster than Cluster Autoscaler). KAI Scheduler handles gang scheduling, topology-awareness, and hierarchical queues — it prevents the "seven-out-of-eight" partial allocation trap: seven nodes sitting idle, burning money, because they are waiting for one missing GPU. Application-layer autoscalers (NVIDIA Dynamo Planner, llm-d Workload Variant Autoscaler) scale on inference-specific signals — queue depth, KV cache utilization — not CPU/DCGM duty cycle. The classic HPA trap is that `DCGM_FI_DEV_GPU_UTIL` is a duty-cycle measurement: 100% could be 10 requests or 100 requests. vLLM pre-allocates KV cache memory, so memory never triggers scale-down. This lesson teaches you to compose the three layers and avoid Karpenter's default `WhenEmptyOrUnderutilized` policy — which will kill running GPU workloads mid-inference.

**Type:** Learn
**Languages:** Python (standard library, a toy-level queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## Learning Objectives

- Diagram the three autoscaling layers (node provisioning, gang scheduling, application-layer) and name the tool used at each.
- Explain why `DCGM_FI_DEV_GPU_UTIL` is the wrong HPA signal for vLLM and name two alternatives (queue depth, KV cache utilization).
- Describe gang scheduling and the partial allocation failure mode that KAI Scheduler prevents (7 out of 8 GPUs sitting idle).
- Name the Karpenter consolidation policy that kills running GPU workloads (`WhenEmptyOrUnderutilized`) and give the safe 2026 alternative.

## The Problem

Your team is shipping an LLM service on Kubernetes. You configured HPA with `DCGM_FI_DEV_GPU_UTIL` as the signal. The service pins at 100% utilization during working hours. HPA never scales up — it already thinks you're at capacity. You manually add a replica; TTFT drops. HPA still doesn't scale. The signal is lying to you.

On another front, your nodes use Cluster Autoscaler. A 1M-token prompt arrives at 2 AM; the cluster spends 3 minutes provisioning a node and the request times out.

On yet another front, you deploy a 70B model requiring 8 GPUs across 2 nodes. The cluster has 7 free GPUs scattered across 3 nodes. Cluster Autoscaler provisions a node for the missing 1 GPU. Seven nodes sit idle for 4 minutes, burning money, while Kubernetes brings up the last GPU.

Three layers, three different failure modes. GPU-aware autoscaling in 2026 is not "turn on HPA." It is composing node provisioning, gang scheduling, and application-signal scaling.

## The Concept

### Layer 1 — Node Provisioning (Karpenter)

Karpenter watches pending pods and provisions nodes in approximately 45-60 seconds (Cluster Autoscaler on GPU nodes typically takes 90-120 seconds). It dynamically picks instance types based on `NodePool` constraints — if your pod needs 8x H100 and no matching node exists in the cluster, Karpenter provisions one directly instead of scaling an existing group.

**The consolidation trap**: Karpenter's default `consolidationPolicy: WhenEmptyOrUnderutilized` is dangerous for GPU pools. It will kill a running GPU node by migrating pods to cheaper, right-sized instances. For inference workloads, this means evicting in-flight requests and reloading a 70B model on a new node. The loss is minutes of compute plus request failures.

Safe setting for GPU pools:

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

This lets Karpenter consolidate truly empty nodes after one hour but never evicts running workloads.

### Layer 2 — Gang Scheduling (KAI Scheduler)

KAI Scheduler (originally named "Karp," later renamed) handles what the default kube-scheduler cannot:

**Gang scheduling** — all-or-nothing placement. A distributed inference pod requiring 8 GPUs either starts with all 8 or doesn't start at all. Without this, you fall into the partial allocation trap: 7 of 8 pods start, waiting indefinitely, burning money.

**Topology-awareness** — knows which GPUs share NVLink, which are in the same rack, which have InfiniBand between them. Places pods accordingly. A DeepSeek-V3 67B tensor-parallel workload must stay within one NVLink domain; KAI Scheduler respects this.

**Hierarchical queues** — multiple teams with priorities and quotas competing for the same GPU pool. Team A's production spike only preempts Team B's training job when priority rules allow it.

KAI deploys as a secondary scheduler alongside kube-scheduler; you annotate workloads to route them through KAI. Both Ray and the vLLM production-stack integrate with it.

### Layer 3 — Application-Layer Signals

**The HPA trap**: `DCGM_FI_DEV_GPU_UTIL` is a duty-cycle metric — it measures whether the GPU was active during each sampling interval. 100% utilization could be 10 concurrent requests or 100; the GPU is busy regardless. Scaling on duty cycle is scaling blind.

Worse, vLLM and similar engines pre-allocate KV cache memory (up to `--gpu-memory-utilization`). Even with a single request, memory usage stays around 90%. Memory-based HPA never scales down.

**2026 alternative signals**:

- Queue depth (number of requests waiting for prefill).
- KV cache utilization (fraction of blocks allocated to active sequences).
- Per-replica P99 TTFT (your SLA signal).
- Goodput (requests per second meeting all SLOs).

NVIDIA Dynamo Planner and llm-d Workload Variant Autoscaler consume these signals to scale replicas. For LLM serving, they completely replace HPA.

### When to Use What

| Scaling decision | Tool |
|------------------|------|
| Add/remove nodes | Karpenter |
| Schedule multi-GPU jobs | KAI Scheduler |
| Add/remove replicas | Dynamo Planner / llm-d WVA (or custom queue-depth HPA) |
| Choose GPU type | Karpenter NodePool |
| Preempt low priority | KAI Scheduler queues |

### Disaggregated Prefill/Decode Makes Everything Harder

If you run disaggregated prefill/decode (Phase 17 · 17), you have two classes of pods with different scaling triggers: prefill pods scale on queue depth, decode pods scale on KV cache pressure. llm-d exposes them as separate `Services` with HPA configured per role. Do not try to put a single HPA in front of both.

### Cold Start Matters Here Too

Cold-start mitigation (Phase 17 · 10) is precisely where node provisioning time becomes user-visible. Karpenter's 45-60 second warm-up plus 20 GB model loading plus engine initialization means a request from zero takes 2-5 minutes. Keep a warm pool (`min_workers=1`) for SLO-critical paths, or use Modal-style checkpointing at the application layer.

### Numbers You Should Remember

- Karpenter node provision: ~45-60s vs Cluster Autoscaler ~90-120s (GPU nodes).
- KAI Scheduler prevents partial allocation waste — the seven-out-of-eight trap.
- `DCGM_FI_DEV_GPU_UTIL` as HPA signal: bad; use queue depth or KV utilization.
- Karpenter's `WhenEmptyOrUnderutilized`: will kill running GPU workloads. Use `WhenEmpty + consolidateAfter: 1h` for inference.

## Use It

`code/main.py` simulates a three-layer autoscaler on a bursty GPU workload. Compares naive HPA (duty cycle), queue-depth HPA, and KAI gang scheduling. Reports unserved requests, GPU idle minutes, and a composite score.

## Ship It

This lesson produces `outputs/skill-gpu-autoscaler-plan.md`. Given cluster topology, workload shape, and SLOs, it designs a three-layer autoscaling plan.

## Exercises

1. Run `code/main.py`. Under bursty workloads, how many requests does the naive duty-cycle HPA drop that the queue-depth HPA catches? Where does the difference come from?
2. Design a Karpenter NodePool for a cluster serving Llama 3.3 70B FP8 on H100 SXM5. Specify `capacity-type`, `disruption.consolidationPolicy`, `consolidateAfter`, and a taint that keeps non-GPU workloads off these nodes.
3. Your team reports deployments stuck in Pending with "free GPUs exist but pods won't schedule." Diagnose — is this a Karpenter, kube-scheduler, or KAI Scheduler issue? Which metrics confirm it?
4. Pick a scaling signal for disaggregated prefill pods and a different one for decode pods. Justify both.
5. Calculate the cost of the `WhenEmptyOrUnderutilized` consolidation trap on a 24/7 production service averaging 60 "dropped request" events per day with P99 TTFT > 10s.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| Karpenter | "The node provisioner" | Kubernetes node autoscaler; sub-minute provisioning |
| Cluster Autoscaler | "The old scaler" | Legacy Kubernetes node autoscaler; slower, group-based |
| KAI Scheduler | "The GPU scheduler" | Secondary scheduler handling gang + topology + queues |
| Gang scheduling | "All-or-nothing" | Atomically schedules N pods or defers them all |
| Topology-awareness | "Rack-awareness" | Places pods by NVLink/IB/rack position |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU utilization" | Duty-cycle metric; not a scaling signal for LLMs |
| Queue depth | "Pending requests" | Correct HPA signal for prefill-bound scaling |
| KV cache utilization | "Memory pressure" | Correct HPA signal for decode-bound scaling |
| Consolidation | "Karpenter cleanup" | Killing nodes in favor of cheaper instance types |
| `WhenEmpty + 1h` | "Safe cleanup" | Policy that never evicts running GPU workloads |

## Further Reading

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — Design documents and configuration examples.
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/) — Consolidation policy semantics and GPU-safe defaults.
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner scaling signals.
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray integration patterns.
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — Managed Kubernetes-specific guidance.
- [llm-d GitHub](https://github.com/llm-d/llm-d) — Workload Variant Autoscaler design.
