# Inference Metrics — TTFT, TPOT, ITL, Goodput, P99

> Four metrics determine whether an inference deployment is usable. TTFT is prefill plus queue plus network. TPOT (equivalent to ITL) is the memory-bound per-token decode cost. End-to-end latency is TTFT plus TPOT times output length. Throughput is aggregate tokens per second across the cluster. But what actually matters for the product is goodput — the fraction of requests that simultaneously meet every SLO constraint. High throughput with low goodput means you are processing tokens that never reach the user in time. Reference numbers for Llama-3.1-8B-Instruct on TRT-LLM in 2026: mean TTFT 162 ms, mean TPOT 7.33 ms, mean E2E 1,093 ms. Always report P50, P90, P99 — never just the mean. Also watch for measurement pitfalls: GenAI-Perf excludes TTFT from its ITL calculation while LLMPerf includes it; the two tools disagree on TPOT for the same run.

**Type:** Learn
**Languages:** Python (standard library, a toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~60 minutes

## Learning Objectives

- Precisely define TTFT, TPOT, ITL, E2E, throughput, and goodput, and state which component each measures.
- Explain why the mean is the wrong statistic for LLM serving, and how to read P50/P90/P99.
- Construct a multi-constraint SLO (e.g., TTFT<500 ms AND TPOT<15 ms AND E2E<2 s) and calculate goodput from it.
- Name two benchmark tools that disagree on TPOT for the same run, and explain why.

## The Problem

"Our throughput is 15,000 tokens per second." So what? If 40% of requests blow past 2 seconds end-to-end, users abandon the session. Throughput alone does not tell you whether the product is usable.

Inference has multiple latency dimensions, and each fails differently. Prefill is compute-bound and grows with prompt length. Decode is memory-bound and grows with batch size. Queue delay is an operations problem. Network is a physical distance problem. You need a different metric for each, you need percentiles, and you need a single composite value that answers "did the user get what they expected?" — that is goodput.

## The Concept

### TTFT — Time to First Token

`TTFT = queue_time + network_request + prefill_time`

Prefill dominates for long prompts. On Llama-3.3-70B FP8 on an H100, a 32k prompt takes roughly 800 ms of pure prefill. Queue time is scheduler behavior under load. Network request is wire time including TLS. TTFT is the latency the user sees before anything streams back.

### TPOT / ITL — Inter-Token Latency

One quantity with many names. `TPOT` (time per output token), `ITL` (inter-token latency), `per-token decode latency` — all the same thing. It is the time between consecutive streaming tokens after the first one.

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

On the same Llama-3.3-70B H100 stack with chunked prefill, mean TPOT is roughly 7 ms. Without chunked prefill, TPOT can spike to 50 ms during a long prefill of an adjacent sequence. Watch P99, not the mean.

### E2E Latency

`E2E = TTFT + TPOT * output_tokens + network_response`

For long outputs (>500 tokens), E2E is dominated by TPOT. For short outputs on long prompts, E2E is dominated by TTFT. Report E2E grouped by output length.

### Throughput

`throughput = total_output_tokens / elapsed_time`

Aggregate metric. Tells you cluster efficiency. Does not tell you individual request health.

### Goodput — The Metric You Actually Care About

`goodput = fraction of requests satisfying (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

The SLO is a multi-constraint. A request counts as "good" only when every constraint holds. Goodput is that fraction. High throughput at 60% goodput is failure. Lower throughput at 99% goodput is the goal.

In 2026, goodput is the metric used in MLPerf Inference v6.0 submissions and in internal SLA tracking at AI platform vendors.

### Why the mean is the wrong statistic

LLM latency distributions are right-skewed. A decode batch with one long-prefill neighbor may emit 500 tokens at ~7 ms TPOT, then 20 tokens at ~60 ms TPOT. Mean TPOT is 9 ms. P99 TPOT is 65 ms. Users hit P99 regularly — that is what makes them leave.

Always report the triple (P50, P90, P99). For user experience, P99 is the one to optimize.

### Reference numbers — Llama-3.1-8B-Instruct on TRT-LLM, 2026

- Mean TTFT: 162 ms
- Mean TPOT: 7.33 ms
- Mean E2E: 1,093 ms
- P99 TPOT: varies between 10-25 ms depending on chunked prefill configuration.

These are NVIDIA-published reference points. They shift with model size (70B shows 3-5x), hardware (H100 vs B200 ~3x), and load.

### Measurement pitfalls

The two most commonly used benchmark tools in 2026 disagree on TPOT for the same run:

- **NVIDIA GenAI-Perf**: Excludes TTFT from its ITL calculation. ITL starts from the 2nd token onward.
- **LLMPerf**: Includes TTFT. ITL starts from the 1st token.

For a request with TTFT 500 ms, 100 output tokens, and 700 ms total decode, GenAI-Perf reports `ITL = 700/99 = 7.07 ms`, while LLMPerf reports `ITL = 1200/100 = 12.00 ms`. Which tool you pick changes the number.

Always state which tool you used. Always publish the definition.

### Constructing an SLO

A reasonable consumer-facing SLO for a 70B chat model in 2026:

- TTFT P99 <= 800 ms.
- TPOT P99 <= 25 ms.
- E2E P99 <= 3 s for outputs <300 tokens.
- Goodput target >= 99%.

Enterprise SLOs tighten TTFT (200-400 ms) and relax E2E. The point is to write them down, measure all three, and track goodput as the single composite value.

### How to measure

- Run real traffic or realistic synthetic traffic (LLMPerf with `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`).
- Benchmark runs should target 2x peak concurrency.
- Run 30-50 iterations, take percentiles from the pooled sample.
- Publish with tool name, tool version, model, hardware, concurrency, and prompt distribution.

## Use It

`code/main.py` is a toy goodput calculator. It generates a synthetic latency distribution, applies an SLO, and computes goodput. It also demonstrates the GenAI-Perf vs LLMPerf TPOT discrepancy on the same trace.

## Ship It

This lesson produces `outputs/skill-slo-goodput-gate.md`. Given a workload and SLO, it produces a CI/CD-ready benchmark recipe that gates deployments on goodput rather than throughput.

## Exercises

1. Run `code/main.py`. Generate a distribution with a 1% tail spike. How does goodput change when you tighten P99 TPOT from 30 ms to 15 ms?
2. A vendor reports "15,000 tok/s on Llama 3.3 70B H100." Name three questions to ask before believing it.
3. Why does chunked prefill protect P99 TPOT but not mean TPOT?
4. Construct a consumer SLO for a voice assistant (the first token is heard, not read). Which metric is most visible to the user?
5. Read the LLMPerf README and GenAI-Perf docs. Find three other metrics the two tools disagree on.

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| TTFT | "time to first token" | Queue + network + prefill; dominated by prefill for long prompts |
| TPOT | "time per output token" | Memory-bound per-token decode cost after the first token |
| ITL | "inter-token latency" | Same as TPOT in most tools (not all — see GenAI-Perf) |
| E2E | "end-to-end" | TTFT + TPOT * output_len; plus response-side network |
| Throughput | "tok/s" | Cluster efficiency; useless without latency percentiles |
| Goodput | "SLO attainment rate" | Fraction of requests satisfying every SLO constraint simultaneously |
| P99 | "the tail" | One-in-a-hundred worst latency; the user experience metric |
| SLO multi-constraint | "the conjunction" | AND of three latency bounds; violation of any one fails the request |
| GenAI-Perf vs LLMPerf | "tool pitfall" | Tools disagree on whether ITL includes TTFT |

## Further Reading

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — standard definitions of TTFT, ITL, TPOT.
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — alternative definitions and measurement recipes.
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — measured on real deployments.
- [LLMPerf](https://github.com/ray-project/llmperf) — Ray-based open-source benchmark.
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA's benchmark tool.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — industry-accepted goodput-based benchmark.
