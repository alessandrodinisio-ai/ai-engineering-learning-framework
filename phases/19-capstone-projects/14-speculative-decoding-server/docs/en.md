# Capstone Project 14 — Speculative Decoding Inference Server

> EAGLE-3 in vLLM 0.7 delivers 2.5-3x throughput on real traffic. P-EAGLE (AWS 2026) pushes parallel speculation further. SGLang's SpecForge trains draft heads at scale. Red Hat's Speculators hub publishes aligned drafts for common open-source models. TensorRT-LLM makes speculative decoding a first-class citizen on NVIDIA. The 2026 production serving stack is vLLM or SGLang with EAGLE-family drafts, FP8 or INT4 quantization, and HPA on queue-wait. This capstone serves two open-source models at 2.5x+ baseline throughput with a full tail-latency report.

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configuration)
**Prerequisites:** Phase 3 (Deep Learning), Phase 7 (Transformers), Phase 10 (LLMs from Scratch), Phase 17 (Infrastructure)
**Phases Involved:** P3 · P7 · P10 · P17
**Time:** 30 hours

## The Problem

Speculative decoding became commodity in 2026. EAGLE-3 draft heads train on the target model's hidden states and predict N tokens ahead; the target model verifies in one pass. A 60-80% acceptance rate yields 2-3x end-to-end throughput. vLLM 0.7 integrates it natively. SGLang + SpecForge gives you the training pipeline. Red Hat's Speculators publishes aligned drafts for Llama 3.3 70B, Qwen3-Coder-30B MoE, and GPT-OSS-120B.

The craft is in serving operations, not in the model. Acceptance rate drifts with traffic distribution (ShareGPT vs code vs domain data). Tail latency on rejection is worse than no speculation — you must report p99 across batch sizes, not just steady-state tokens/sec. Cost per 1M tokens vs Anthropic / OpenAI API is the credibility lever.

## The Concept

Speculative decoding has two layers. A **draft** model (EAGLE-3 head, ngram, or a smaller aligned model) proposes k candidate tokens per step. The **target** model verifies all k in one pass; any accepted prefix replaces the greedy path. Acceptance rate depends on draft-target alignment and input distribution.

EAGLE-3 outperforms ngram drafts on most traffic. P-EAGLE runs parallel speculation for deeper draft trees. The cost: higher P99 latency on rejection because the verification pass is larger. Serving configuration must report latency bucketed by batch size to expose this.

Deployment is Kubernetes. vLLM 0.7 runs one replica per GPU or tensor-parallel shard. HPA autoscales on queue-wait, not CPU. FP8 (Marlin) and INT4 (AWQ) quantization keeps GPU memory within H100 / H200 range. The end-to-end report is throughput, acceptance rate, p50/p99 at batch 1/8/32, and $/1M tokens.

## Architecture

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## Tech Stack

- Serving: vLLM 0.7 or SGLang 0.4
- Speculation method: EAGLE-3 draft heads, P-EAGLE parallel speculation, ngram fallback
- Draft training: SpecForge (SGLang) or Red Hat Speculators
- Target models: Llama 3.3 70B, Qwen3-Coder-30B MoE, GPT-OSS-120B
- Quantization: FP8 (Marlin), INT4 AWQ
- Deployment: Kubernetes + NVIDIA device plugin; HPA on queue-wait metric
- Evaluation: ShareGPT, MT-Bench-v2, GSM8K, HumanEval for cross-domain distribution acceptance rate measurement
- Reference: TensorRT-LLM speculative decoding as vendor baseline

## Build It

1. **Target model preparation.** Choose Llama 3.3 70B. Quantize to FP8 via Marlin. Deploy on vLLM 0.7 on 1x H100 (or 2x tensor-parallel).

2. **Draft sourcing.** Pull an aligned EAGLE-3 draft head from Red Hat Speculators (or train one via SpecForge). Load into vLLM's speculative decoding config.

3. **Baseline numbers.** Before speculation: tokens/sec at batch 1/8/32, p50/p99 latency, GPU utilization. Publish.

4. **Enable EAGLE-3.** Flip the config; re-run the same benchmark. Report speedup, acceptance rate, and p99 tail-latency delta.

5. **P-EAGLE.** Enable parallel speculation; measure deeper draft tree vs serial EAGLE-3. Report the inflection point where P-EAGLE helps vs hurts.

6. **Domain traffic.** Run ShareGPT vs HumanEval vs domain-specific traffic through the same server. Measure acceptance rate per distribution. Identify when the draft drifts.

7. **Second target model.** Run the same pipeline on Qwen3-Coder-30B MoE. Drafting is trickier (MoE routing noise). Report.

8. **K8s HPA.** Deploy under K8s with HPA tracking `queue_wait_ms`. Demonstrate horizontal scale-out when load triples.

9. **Cost comparison.** Compute $/1M tokens vs Anthropic Claude Sonnet 4.7 and OpenAI GPT-5.4 on the same eval. Publish.

## Use It

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## Ship It

`outputs/skill-inference-server.md` describes the deliverable. A speculative-decoding serving stack with measured benchmarks, a full benchmark report, and a K8s deployment.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | Measured speedup vs baseline | 2.5x+ throughput at quality parity on two models |
| 20 | Acceptance rate on real traffic | Per-distribution acceptance rate report |
| 20 | P99 tail-latency discipline | p99 at batch 1/8/32 with and without speculation |
| 20 | Operations | K8s deployment, HPA on queue-wait, smooth rollout |
| 15 | Write-up and methodology | Clear explanation of what changed and why |
| **100** | | |

## Exercises

1. Measure acceptance rate degradation when the draft is one version behind the target (e.g., Llama 3.3 -> 3.4 drift). Build a monitoring alert.

2. Implement ngram fallback: if EAGLE-3 acceptance rate drops below threshold, switch to ngram draft. Report reliability improvement.

3. Run a controlled MoE experiment: same Qwen3-Coder-30B with injected routing noise vs without. Measure draft acceptance rate sensitivity.

4. Scale to H200 (141 GB). Report the model-size headroom gained per replica and whether you can serve an unquantized Llama 3.3 70B.

5. Benchmark TensorRT-LLM speculative decoding on the same H100 hardware. Report where it outperforms vLLM.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Draft model | "Speculator" | A small model proposing N tokens for the target to verify |
| EAGLE-3 | "2026 draft architecture" | Draft heads trained on target hidden states; ~75% acceptance rate |
| P-EAGLE | "Parallel speculation" | A tree of draft branches verified in a single target pass |
| Acceptance rate | "Hit rate" | Fraction of draft tokens accepted without resampling |
| Quantization | "FP8 / INT4" | Lower-precision weights fitting more model into GPU memory |
| Queue wait | "HPA metric" | Time a request waits in the pending queue before inference starts |
| Speculators hub | "Aligned drafts" | Red Hat Neural Magic's hub of EAGLE drafts for common open-source models |

## Further Reading

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — reference serving stack
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — parallel speculative decoding paper + integration
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft head training pipeline
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — aligned draft hub
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — vendor alternative
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — commercial reference
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — methodology paper
- [vLLM repository](https://github.com/vllm-project/vllm) — code and benchmarks
