# Self-Hosted Serving Selection — llama.cpp, Ollama, TGI, vLLM, SGLang

> In 2026 four engines dominate self-hosted inference. Choose based on hardware, scale, and ecosystem. **llama.cpp** is fastest on CPU — broadest model support, full control over quantization and threading. **Ollama** is a one-command install on developer laptops, approximately 15-30% slower than llama.cpp (Go + CGo + HTTP serialization), 3x throughput gap under production-like load. **TGI entered maintenance mode on December 11, 2025** — bug fixes only, raw throughput approximately 10% behind vLLM, but historically best-in-class observability and HF ecosystem integration. That maintenance status makes it a risky long-term bet — SGLang or vLLM is a safer default for new projects. **vLLM** is the general-purpose production default — v0.15.1 (February 2026) added PyTorch 2.10, RTX Blackwell SM120, H200 optimizations. **SGLang** is the agentic multi-turn / prefix-heavy specialist — 400,000+ GPUs in production (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). Hardware constraints: CPU-only -> llama.cpp only. AMD / non-NVIDIA -> vLLM only (TRT-LLM is NVIDIA-locked). 2026 pipeline pattern: dev = Ollama, staging = llama.cpp, production = vLLM or SGLang. Same GGUF/HF weights throughout.

**Type:** Learn
**Languages:** Python (stdlib, engine decision tree traverser)
**Prerequisites:** All engine lessons in Phase 17 (04, 06, 07, 09, 18)
**Time:** ~45 minutes

## Learning Objectives

- Given hardware (CPU / AMD / NVIDIA Hopper / Blackwell), scale (1 user / 100 / 10,000), and workload (general chat / agentic / long context), choose an engine.
- State TGI's maintenance mode status in 2026 (December 11, 2025) and why it biases new projects toward vLLM or SGLang.
- Describe a dev/staging/production pipeline using the same GGUF or HF weights throughout.
- Explain why "CPU-only" forces llama.cpp and "AMD" excludes TRT-LLM.

## The Problem

Your team is starting a new self-hosted LLM project. One engineer says Ollama, another says vLLM, a third says "doesn't TGI work out of the box?" All three are right in different scenarios. None is right for all scenarios.

The 2026 decision tree matters: hardware first, scale second, workload third. There is also a specific 2025 event — TGI entering maintenance mode on December 11 — that changes the default for new projects.

## The Concept

### Five Engines

| Engine | Best for | Notes |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / fewest dependencies / broadest model support | Fastest on CPU, full control |
| **Ollama** | Developer laptops, single user, one-command install | 15-30% slower than llama.cpp; 3x throughput gap in production |
| **TGI** | HF ecosystem, regulated industries | **Maintenance mode since December 11, 2025** |
| **vLLM** | General-purpose production, 100+ users | Broad production default; v0.15.1 February 2026 |
| **SGLang** | Agentic multi-turn, prefix-heavy workloads | 400,000+ GPUs in production |

### Hardware-First Decision

**CPU-only** -> llama.cpp. Ollama also works but is slower. No other engine is competitive on CPU.

**AMD GPU** -> vLLM (AMD ROCm support). SGLang also works. TRT-LLM is NVIDIA-locked, so it is out.

**NVIDIA Hopper (H100 / H200)** -> vLLM or SGLang or TRT-LLM. All three are top-tier.

**NVIDIA Blackwell (B200 / GB200)** -> TRT-LLM is the throughput leader (Phase 17 · 07). vLLM and SGLang follow closely.

**Apple Silicon (M-series)** -> llama.cpp (Metal). Ollama is a wrapper around it.

### Scale-Second Decision

**1 user / local dev** -> Ollama. One command, first token in seconds.

**10-100 users / small team** -> vLLM single GPU.

**100-10k users / production** -> vLLM production-stack (Phase 17 · 18) or SGLang.

**10k+ users / enterprise** -> vLLM production-stack + disaggregated (Phase 17 · 17) + LMCache (Phase 17 · 18).

### Workload-Third Decision

**General chat / Q&A** -> vLLM wins on broad defaults.

**Agentic multi-turn (tools, planning, memory)** -> SGLang's RadixAttention (Phase 17 · 06) dominates.

**Prefix-heavy RAG** -> SGLang.

**Code generation** -> vLLM is sufficient; SGLang is slightly better on caching.

**Long context (128K+)** -> vLLM + chunked prefill; SGLang + hierarchical KV.

### TGI Maintenance Pitfall

Hugging Face TGI entered maintenance mode on December 11, 2025 — bug fixes only going forward. Historically: best-in-class observability, best HF ecosystem integration (model cards, safety tools), raw throughput slightly behind vLLM.

For new projects in 2026: default away from TGI. Existing TGI deployments can continue running, but should eventually migrate. SGLang and vLLM are safer defaults.

### Pipeline Pattern

Dev (Ollama) -> Staging (llama.cpp) -> Production (vLLM). Same GGUF or HF weights throughout. Engineers iterate quickly on laptops; staging mirrors production quantization; production is serving objectives.

### Ollama Caveats

Ollama is great for development. It is not great for shared production: Go HTTP serialization adds overhead, concurrency management is simpler than vLLM, OpenTelemetry support lags behind. Use it where it shines — single user, one command — and switch to vLLM for shared serving.

### Self-Hosted vs Managed Is a Separate Decision

Phase 17 · 01 (managed hyperscaler), · 02 (inference platforms) cover managed. This lesson assumes you have already decided to self-host. Reasons to self-host: data residency, custom fine-tuning, total cost of ownership at scale, domain models not available on managed.

### Numbers to Remember

- TGI maintenance mode: December 11, 2025.
- vLLM v0.15.1: February 2026; PyTorch 2.10; Blackwell SM120 support.
- SGLang production footprint: 400,000+ GPUs.
- Ollama throughput gap vs llama.cpp: 15-30% slower; 3x gap under production load.

## Use It

`code/main.py` is a decision tree traverser: given hardware + scale + workload, it picks an engine and explains why.

## Ship It

This lesson produces `outputs/skill-engine-picker.md`. Given constraints, it picks an engine and writes a migration plan.

## Exercises

1. Run `code/main.py` with your hardware / scale / workload. Does the output match your intuition?
2. Your infrastructure is 12x H100 and 8x MI300X AMD. Which engine do you use? Why is TRT-LLM not on the table?
3. A team wants to use TGI in 2026, arguing "we know it." Make the case for migration.
4. Ollama dev to vLLM production: what changes in quantization, configuration, and observability?
5. A RAG product with P99 prefix length 8K and high cross-tenant reuse. Pick an engine and stack it with Phase 17 · 11 + 18.

## Key Terms

| Term | What people say | What it actually is |
|------|----------------|------------------------|
| llama.cpp | "The CPU one" | Broadest model support, fastest on CPU |
| Ollama | "The laptop one" | One-command install, dev-level throughput |
| TGI | "HF's server" | Maintenance mode since December 2025 |
| vLLM | "The default one" | Broad production baseline in 2026 |
| SGLang | "The agentic one" | Prefix-heavy, RadixAttention |
| TRT-LLM | "NVIDIA-locked" | Blackwell throughput leader, NVIDIA only |
| GGUF | "llama.cpp format" | Packaged K-quant variants |
| Production-stack | "vLLM K8s" | Phase 17 · 18 reference deployment |
| Pipeline pattern | "Dev -> Staging -> Prod" | Ollama -> llama.cpp -> vLLM on the same weights |

## Further Reading

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
