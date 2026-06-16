# TensorRT-LLM on Blackwell with FP8 and NVFP4

> TensorRT-LLM only runs on NVIDIA, but it wins on Blackwell. On a GB200 NVL72 with Dynamo orchestration, SemiAnalysis InferenceX measured $0.012 per million tokens for a 120B model in Q1-Q2 2026, compared to $0.09/M on H100 + vLLM — a 7x cost gap. The stack is a superposition of three floating-point regimes: FP8 remains critical for KV cache and attention kernels because they need FP8's dynamic range; NVFP4 (4-bit microscaling) handles weights and activations; multi-token prediction (MTP) and disaggregated prefill/decode layer another 2-3x on top. Day-0 model support loads FP4 weights directly without post-training conversion. The 2026 engineering team pitfall: TRT-LLM is a closed NVIDIA stack, so adopting it trades portability for throughput. Run the numbers on your specific model-hardware combination before committing.

**Type:** Learn
**Languages:** Python (standard library, a toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 minutes

## Learning Objectives

- Explain why FP8 remains critical for KV cache and attention even when weights are NVFP4.
- Calculate a frontier model's HBM footprint under BF16, FP8, and NVFP4, and reason about where savings come from.
- Name the Blackwell-specific features TRT-LLM exploits (day-0 FP4, MTP, disaggregated serving, all-to-all primitives).
- Judge when TRT-LLM's NVIDIA lock-in is worth the 7x cost gap over vLLM on Hopper.

## The Problem

The 2026 inference economics frontier is "tokens per dollar." The answer depends on four stacked choices: hardware generation (Hopper H100/H200 vs Blackwell B200/GB200), precision (BF16 → FP8 → NVFP4), serving engine (vLLM vs SGLang vs TRT-LLM), and orchestration (naive vs disaggregated vs Dynamo).

On Hopper with vLLM, a 120B MoE runs at roughly $0.09 per million tokens. On Blackwell with TRT-LLM + Dynamo, the same model runs at roughly $0.012 — 7x cheaper. Part of the gap is hardware (Blackwell single-GPU LLM throughput is 11-15x Hopper). Part is the stack: FP4 weights, MTP drafts, disaggregated prefill/decode, and NVLink 5 all-to-all for MoE expert communication.

You cannot replicate this outside the NVIDIA stack. That is the tradeoff — portability for economics. Understanding which stack choices contribute what fraction of the gap is the point of this lesson.

## The Concept

### Why FP8 is still the floor for KV cache

A common 2026 mistake: assuming NVFP4 applies everywhere. It does not. KV cache requires FP8 (8-bit floating point) because the attention keys and values it stores span a wide dynamic range. Quantizing KV to FP4 causes catastrophic accuracy loss — the tails of the distribution vanish and attention scores collapse. FP8's exponent bits give KV cache the range it needs.

NVFP4 (2025-2026) is for weights and activations. Microscaling: each block of weights has its own scale factor, so small blocks can span different dynamic ranges without suffering per-tensor scaling losses. For activations, FP4 holds up because the range within a single layer is small.

Typical Blackwell configuration:

- Weights: NVFP4 (4-bit microscaling).
- Activations: NVFP4.
- KV cache: FP8.
- Attention accumulators: FP32 (softmax stability).

### Blackwell-specific primitives used by TRT-LLM

- **Day-0 FP4 weights**: Model providers ship FP4 weights directly; TRT-LLM loads them without a post-training conversion step. FP4 does not need the AWQ / GPTQ pass.
- **Multi-token prediction (MTP)**: Same idea as EAGLE (Phase 17 · 05), but integrated into TRT-LLM's build.
- **Disaggregated serving**: Prefill and decode run on separate GPU pools with KV cache transferred over NVLink or InfiniBand. Same idea as Dynamo (Phase 17 · 20).
- **All-to-all communication primitives**: NVLink 5 cuts MoE expert communication latency 3x compared to Hopper. TRT-LLM's MoE kernels are tuned for this.
- **NVFP4 + MXFP8 microscaling**: Hardware-accelerated scale factor handling on Blackwell Tensor Cores.

### Numbers to memorize

- HGX B200 via TRT-LLM on GPT-OSS-120B: $0.02 per million tokens.
- GB200 NVL72 via Dynamo (orchestrating TRT-LLM): $0.012 per million tokens.
- H100 + vLLM on comparable workload: ≈ $0.09 per million tokens.
- TRT-LLM three-month update yielded 2.8x throughput improvement (2026).
- Single-GPU LLM throughput, Blackwell vs Hopper: 11-15x.
- MLPerf Inference v6.0 (April 2026): Blackwell dominates every submitted task.

### The real quality cost of FP4

NVFP4 is aggressive. On reasoning-heavy workloads (chain-of-thought, math, long-context code generation), FP4 weights degrade visibly. Per-block calibration mitigates but does not eliminate the issue. Teams shipping reasoning models often use FP8 weights + FP4 activations as a compromise, or stay on H200 + FP8 entirely.

The rule: always validate task quality on your eval set before committing to NVFP4 weights.

### Why this is an NVIDIA lock-in decision

TRT-LLM is C++ + CUDA + closed-source kernels. Models must be compiled for specific GPU SKUs. No AMD, no Intel, no ARM. If your infrastructure strategy is multi-vendor, TRT-LLM is not viable for that tier of serving — you can still serve on mixed hardware with vLLM. If you are all-NVIDIA, the 7x gap pays for the lock-in.

### 2026 playbook

For inference bills above $100M per year running on Hopper + vLLM, you are leaving 7-10x on the table. Migrate cost-dominant workloads to Blackwell + TRT-LLM + Dynamo. Keep experiment tiers on H100 + vLLM for model iteration speed. Validate quality on every NVFP4-converted model before shipping.

### The disaggregated multiplier

TRT-LLM's disaggregated serving (separate prefill and decode pools) is covered in depth in Phase 17 · 20. On Blackwell, the multipliers stack: FP4 weights × MTP speedup × disaggregated placement × cache-aware routing. The 7x number assumes the full stack.

## Use It

`code/main.py` calculates HBM footprint, decode throughput (memory-bound regime), and $/M-token across three stacks for a model: H100 + BF16 + vLLM, H100 + FP8 + vLLM, B200 + NVFP4/FP8 + TRT-LLM. Run it to see the stacking effect and how much each change contributes to the gap.

## Ship It

This lesson produces `outputs/skill-trtllm-blackwell-advisor.md`. Given a workload, model size, and annual token volume, it judges whether the Blackwell + TRT-LLM stack is worth the NVIDIA lock-in.

## Exercises

1. Run `code/main.py`. For a 120B MoE with 30% active parameters, calculate memory-bandwidth-bound decode throughput on H100 BF16, H100 FP8, and B200 NVFP4/FP8. Where does the biggest jump come from?
2. A customer spends $2M per year on H100 + vLLM. Given the 7x economics gap, how many Blackwell GPUs do they need to break even on the migration to TRT-LLM within 12 months?
3. You observe accuracy dropping 3 points on MATH after NVFP4 weight conversion. Name two recovery paths: one quality-first (keep FP8 weights), one cost-first (calibrate with in-domain data).
4. Read the MLPerf v6.0 inference results. Which task shows the smallest Blackwell-vs-Hopper gap, and why?
5. Calculate the HBM required for a 405B model with NVFP4 weights + FP8 KV cache at 128k context. Does it fit in a single GB200 NVL72 node?

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| FP8 | "eight-bit float" | 8-bit floating point; used for KV cache and attention due to dynamic range |
| NVFP4 | "four-bit micro" | NVIDIA's 4-bit microscaling FP format; used for weights and activations on Blackwell |
| MXFP8 | "MX eight" | Microscaling FP8 variant; hardware-accelerated on Blackwell Tensor Cores |
| Day-0 FP4 | "ship FP4 weights directly" | Model providers publish weights already in FP4; no post-training conversion step |
| MTP | "multi-token prediction" | Speculative decoding drafts integrated into TRT-LLM (Phase 17 · 05) |
| Disaggregated serving | "split prefill/decode" | Prefill and decode on separate GPU pools; KV transferred over NVLink/IB |
| All-to-all | "MoE expert communication" | Communication pattern routing tokens to expert GPUs; NVLink 5 cuts latency 3x |
| InferenceX | "SemiAnalysis inference benchmark" | The 2026 industry-accepted per-token cost benchmark |

## Further Reading

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — April 2026 MLPerf results.
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 all-to-all and MoE kernels.
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html) — official engine documentation.
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — disaggregated orchestration on top of TRT-LLM.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — the benchmark suite publishing Blackwell numbers.
