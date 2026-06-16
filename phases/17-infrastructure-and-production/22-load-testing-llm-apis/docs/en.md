# Load Testing LLM APIs — Why k6 and Locust Will Lie to You

> Traditional load-testing tools weren't built for streaming responses, variable output lengths, token-level metrics, or GPU saturation. Two traps bite most teams. The GIL trap: Locust's token-level measurement runs tokenization under the Python GIL, competing with request generation under heavy concurrency; tokenization backlog then inflates reported inter-token latency — the bottleneck is your client, not the server. The prompt-uniformity trap: sending the same prompt in a loop only tests one point on the token distribution; real traffic has variable lengths and diverse prefix matches. LLMPerf fixes this with `--mean-input-tokens` + `--stddev-input-tokens`. 2026 tool map: LLM-specific (GenAI-Perf, LLMPerf, LLM-Locust, guidellm) for token-level precision; **k6 v2026.1.0** + **k6 Operator 1.0 GA (September 2025)** — streaming-aware, Kubernetes-native distributed via TestRun/PrivateLoadZone CRDs, best for CI/CD gates; Vegeta for Go constant-rate saturation; Locust 2.43.3 only works for streaming with the LLM-Locust extension. Load patterns: steady-state, ramp, spike (autoscale test), soak (memory leaks).

**Type:** Build
**Languages:** Python (standard library, a toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 minutes

## Learning Objectives

- Explain the two anti-patterns that make general-purpose load-testing tools lie on LLM APIs (GIL trap, prompt-uniformity trap).
- Pick the right tool for a given purpose: LLMPerf (benchmark runs), k6 + streaming extension (CI gates), guidellm (large-scale synthetic), GenAI-Perf (NVIDIA reference).
- Design four load patterns (steady-state, ramp, spike, soak) and name the failure mode each catches.
- Build a realistic prompt distribution using mean + stddev of input tokens rather than a fixed length.

## The Problem

You load-tested an LLM endpoint with k6 at 500 concurrent users. It held up. You shipped. In production, the service collapsed at an actual 200 users — P99 TTFT exploded, GPU pegged.

Two things happened. First, k6 sent 500 identical prompts — your request coalescing and prefix cache made it look like it was handling 500 concurrent decodes when you were actually handling one. Second, k6 doesn't track inter-token latency on a streaming response the way eyes experience it; it sees one HTTP connection, not 500 tokens arriving at varying intervals.

Load testing LLMs is its own discipline.

## The Concept

### The GIL Trap (Locust)

Locust uses Python and runs tokenization on the client under the GIL. Under high concurrency, the tokenizer queues behind request generation. Reported inter-token latency includes client-side tokenization backlog. You think the server is slow; it's actually the testing tool.

Fix: the LLM-Locust extension moves tokenization to a separate process, or use compiled-language tools (k6, LLMPerf with tokenizers.rs).

### The Prompt-Uniformity Trap

All known load-testing tools let you configure a prompt. In a 10,000-iteration loop test, the same prompt is sent every time. The server sees the same prefix every time — prefix cache hits approach 100%, throughput looks great.

Fix: sample from a prompt distribution. LLMPerf uses `--mean-input-tokens 500 --stddev-input-tokens 150` — variable lengths, variable content.

### Four Load Patterns

1. **Steady-state** — constant RPS for 30-60 minutes. Catches: baseline performance regression.
2. **Ramp** — RPS linearly from 0 to target over 15 minutes. Catches: capacity breakpoints, warm-up anomalies.
3. **Spike** — sudden 3-10x RPS for 2 minutes then back down. Catches: autoscale latency, queue saturation, cold-start impact.
4. **Soak** — steady-state for 4-8 hours. Catches: memory leaks, connection-pool drift, observability overflow.

### 2026 Tool Map

**LLMPerf** (Anyscale) — Python but Rust-backed tokenization. Mean/stddev prompts. Streaming-aware. Best default for benchmark runs.

**NVIDIA GenAI-Perf** — NVIDIA's reference. Uses Triton client; comprehensive metric coverage. Note that its ITL excludes TTFT; LLMPerf's includes it. The two tools give different TPOT on the same server.

**LLM-Locust** (TrueFoundry) — Locust extension that fixes the GIL trap. Familiar Locust DSL + streaming metrics.

**guidellm** — large-scale synthetic benchmarking.

**k6 v2026.1.0** + **k6 Operator 1.0 GA (September 2025)**:
- k6 itself (Go, compiled, no GIL) added streaming-aware metrics.
- k6 Operator uses TestRun / PrivateLoadZone CRDs for Kubernetes-native distributed testing.
- Best for CI/CD gates and SLA testing.

**Vegeta** — Go, simpler than k6. Constant-rate HTTP saturation. Not LLM-aware, but suitable for gateway / rate-limit testing.

**Locust 2.43.3 vanilla** — has the GIL trap for LLMs. Only works with the LLM-Locust extension.

### SLA Gates in CI

Run k6 on PRs:

- 30-50 iterations each, at baseline RPS.
- Gates: P50/P95 TTFT, 5xx < 5%, TPOT below threshold.
- Fail the build if thresholds are breached.

### Realistic Prompt Distribution

Build from real traffic samples (if you have them) or from published distributions (e.g., ShareGPT prompts for chat, HumanEval for code). Feed mean + stddev into LLMPerf. Avoid "one prompt in a loop" at all costs.

### Numbers You Should Remember

- k6 Operator 1.0 GA: September 2025.
- k6 v2026.1.0: streaming-aware metrics.
- Typical LLMPerf run: 100-1000 requests at concurrency X.
- Typical CI gate: 30-50 iterations per PR.
- Four patterns: steady-state, ramp, spike, soak.

## Use It

`code/main.py` simulates a load test with a realistic prompt distribution, measures effective TPOT, and demonstrates the prompt-uniformity trap.

## Ship It

This lesson produces `outputs/skill-load-test-plan.md`. Given a workload and SLA, it picks the tool and designs the four load patterns.

## Exercises

1. Run `code/main.py`. Compare uniform vs realistic distribution — where's the gap?
2. Write a k6 script for a CI gate: TTFT P95 < 800 ms at 100 concurrency, 5-minute duration.
3. Your soak test shows memory growing 50 MB/hour. Name three causes and the instrumentation to distinguish between them.
4. Spike from 10 RPS to 100 RPS. If Karpenter + vLLM production-stack are both in place (Phase 17 · 03 + 18), what's the expected recovery time?
5. GenAI-Perf reports TPOT=6ms; LLMPerf reports TPOT=11ms on the same server. Explain.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| LLMPerf | "LLM load tester" | Anyscale benchmark tool, streaming-aware |
| GenAI-Perf | "NVIDIA tool" | NVIDIA reference load-testing tool |
| LLM-Locust | "Locust for LLMs" | Locust extension fixing the GIL trap |
| guidellm | "Synthetic benchmark" | Large-scale synthetic tool |
| k6 Operator | "K8s k6" | CRD-based distributed k6 |
| GIL trap | "Python client overhead" | Tokenization backlog inflates reported latency |
| Prompt-uniformity trap | "Single-prompt lie" | Looping the same prompt hits cache, inflates throughput |
| Steady-state | "Constant load" | Flat RPS for N minutes |
| Ramp | "Linear ramp-up" | From 0 to target over a duration |
| Spike | "Burst test" | Sudden multiplier then back down |
| Soak | "Long test" | Hours-long run for leak detection |

## Further Reading

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
