# Production Quantization — AWQ, GPTQ, GGUF K-quant, FP8, MXFP4/NVFP4

> Quantization format is not a one-size-fits-all choice — it is a function of hardware, serving engine, and workload. GGUF Q4_K_M or Q5_K_M dominates CPU and edge, delivered via llama.cpp and Ollama. GPTQ wins in vLLM when you need multi-LoRA on the same base. AWQ with the Marlin-AWQ kernel achieves ~741 tok/s on a 7B model and delivers the best Pass@1 among INT4 formats — the 2026 data-center production default. FP8 remains the middle ground on Hopper, Ada, and Blackwell — near-lossless and widely supported. NVFP4 and MXFP4 (Blackwell microscaling) are aggressive and require per-block validation. Two pitfalls bite teams: calibration datasets must match the deployment domain, and KV cache and weight quantization are separate — the "my model is only 4 GB now" AWQ lesson forgets the 10-30 GB KV cache at production batch sizes.

**Type:** Learn
**Languages:** Python (standard library, toy cross-format memory and throughput comparison)
**Prerequisites:** Phase 10 · 13 (Quantization Basics), Phase 17 · 04 (vLLM Serving Internals)
**Time:** ~75 minutes

## Learning Objectives

- Name six production quantization formats and their 2026 sweet spots.
- Given hardware (CPU vs GPU, Hopper vs Blackwell), engine (vLLM, TRT-LLM, llama.cpp), and workload (casual chat, reasoning, multi-LoRA), pick a format.
- Calculate the weight memory saved by a chosen format, and the KV cache that remains untouched.
- State the calibration dataset pitfall that causes quantized models to degrade on domain traffic.

## The Problem

Quantization reduces memory and HBM bandwidth, which is exactly what decode needs. A 70B model in FP16 has 140 GB of weights. Quantize weights to INT4 (AWQ or GPTQ), and the model is 35 GB — fits on a single H100 with room for KV cache, which matters because at 128 concurrent sequences with 2k context, KV cache alone is 20-30 GB.

But quantization is not free. Aggressive quantization degrades quality, especially on reasoning-heavy tasks. Different formats pair with different engines. Different hardware natively supports different precisions. The 2026 format zoo is real, and you cannot copy someone else's choice — you must pick based on your own stack.

## The Concept

### Six formats

| Format | Bits | Sweet spot | Engine |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU, edge, laptop | llama.cpp, Ollama |
| GPTQ | 4-8 | Multi-LoRA on vLLM | vLLM, TGI |
| AWQ | 4 | Data-center GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell data center | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell multi-tenant | TRT-LLM |
| NVFP4 | 4 | Blackwell multi-tenant | TRT-LLM |

### GGUF — CPU/edge default

GGUF is a file format, not a quantization scheme per se — it packages multiple K-quant variants (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0) into a single container. Q4_K_M and Q5_K_M are the production defaults — near-BF16 quality at 4-5 bits. Best choice for CPU or edge serving because llama.cpp is the fastest CPU inference engine by far.

Throughput penalty in vLLM: ~93 tok/s on 7B — the format is not optimized for GPU kernels. Use GGUF when the deployment target is CPU/edge. Not otherwise.

### GPTQ — Multi-LoRA in vLLM

GPTQ is a post-training quantization algorithm with a calibration pass. The Marlin kernel makes it fast on GPU (2.6x speedup over non-Marlin GPTQ). ~712 tok/s on 7B.

Unique advantage: GPTQ-Int4 supports LoRA adapters in vLLM. If you are serving one base plus 10-50 fine-tuned variants (each as a LoRA), GPTQ is your path. NVFP4 does not support LoRA as of early 2026.

### AWQ — Data-center GPU default

Activation-aware Weight Quantization. Protects ~1% of the most salient weights during quantization. Marlin-AWQ kernel: 10.9x speedup over naive. ~741 tok/s on 7B, best Pass@1 among INT4 formats.

Pick AWQ for new GPU serving unless you need multi-LoRA (GPTQ) or aggressive Blackwell FP4 (NVFP4).

### FP8 — The reliable middle ground

8-bit floating point. Near-lossless. Widely supported. Hopper Tensor Cores natively accelerate FP8. Blackwell inherits. When quality cannot be compromised (reasoning, medical, code generation), FP8 is the 2026 safe default. Memory savings are half of INT4, but quality risk is much lower.

### MXFP4 / NVFP4 — Blackwell aggressive

Microscaling FP4. Each block of weights has its own scale factor. Aggressive, but hardware-accelerated on Blackwell Tensor Cores. Halves per-token bytes compared to FP8 — the economics advantage from Phase 17 · 07.

Caveats:
- No LoRA support yet (early 2026).
- Visible quality degradation on reasoning-heavy workloads.
- Validate each model on your eval set.

### The calibration pitfall

AWQ and GPTQ require a calibration dataset — typically C4 or WikiText. For domain models (code, medical, legal), calibrating on generic web text causes the algorithm to make wrong decisions about which weights to protect. Pass@1 on HumanEval can drop several points.

Fix: calibrate on in-domain data. A few hundred domain samples are usually enough. Test on the eval set before shipping.

### The KV cache pitfall

AWQ shrinks weights to 4 bits. KV cache is separate and remains FP16/FP8. For a 70B model with AWQ:

- Weights: ~35 GB (from 140 GB down to INT4).
- KV cache at 128 concurrent × 2k context: ~20 GB.
- Activations: ~5 GB.
- Total: ~60 GB — fits in an H100 80GB.

Naively thinking "I quantized the model to 4 GB" forgets the other 30-50 GB. Budget HBM holistically.

Additionally, KV cache quantization (FP8 KV or INT8 KV) is a separate choice with its own tradeoffs — it directly affects attention accuracy and is not free.

### AWQ INT4 is risky for reasoning

Chain-of-thought, math, long-context code generation — these degrade visibly under aggressive quantization. AWQ INT4 drops ~3-5 points on MATH. For reasoning-heavy workloads, use FP8 or BF16; accept the memory cost.

### 2026 selection guide

- CPU/edge serving: GGUF Q4_K_M. Done.
- GPU serving, casual chat, no LoRA: AWQ.
- GPU serving, multi-LoRA: GPTQ with Marlin.
- Reasoning workloads: FP8.
- Blackwell data center, quality validated: NVFP4 + FP8 KV.
- Unsure: run a 1,000-sample eval on each candidate format.

## Use It

`code/main.py` calculates memory footprint (weights + KV + activations) and relative throughput across six formats over a range of model sizes. It shows where KV cache dominates, where weight compression pays off, and where FP8 is the safe pick.

## Ship It

This lesson produces `outputs/skill-quantization-picker.md`. Given hardware, model size, workload type, and quality tolerance, it picks a format and produces a calibration/validation recipe.

## Exercises

1. Run `code/main.py`. For a 70B model at 128 concurrent sequences with 2k context, calculate total HBM per format. Which formats let you fit on a single H100 80GB?
2. You have a 7B code model. Pick a format and justify it. If you misjudge quality tolerance, what is the recovery path?
3. Calculate the calibration dataset size needed to calibrate AWQ for a medical domain model. Why is more data not always better?
4. Read the Marlin-AWQ kernel paper or release notes. Explain in three sentences why AWQ achieves 741 tok/s on 7B while bare GPTQ is ~712.
5. When does combining AWQ weights with FP8 KV cache make sense, versus leaving KV in BF16?

## Key Terms

| Term | How people say it | What it actually is |
|------|-------------------|---------------------|
| GGUF | "llama.cpp format" | File format packaging K-quant variants; CPU/edge default |
| Q4_K_M | "Q4 K M" | 4-bit K-quant medium; production GGUF default |
| GPTQ | "gee pee tee q" | Post-training INT4 with calibration; supports LoRA in vLLM |
| AWQ | "a w q" | Activation-aware INT4; Marlin kernel; best Pass@1 among INT4 |
| Marlin kernel | "fast INT4 kernel" | Custom CUDA kernel for INT4 on Hopper; 10x speedup |
| FP8 | "eight-bit float" | Safe precision default on Hopper/Ada/Blackwell |
| MXFP4 / NVFP4 | "microscaling four" | Blackwell 4-bit FP with per-block scale factors |
| Calibration dataset | "cal data" | Input text used to choose quantization parameters; must match domain |
| KV cache quantization | "KV INT8" | Separate choice from weights; affects attention accuracy |

## Further Reading

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/) — comparative benchmarks.
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks) — per-format throughput numbers.
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/) — per-format selection.
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html) — supported formats and flags.
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978) — original AWQ formulation.
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323) — original GPTQ formulation.
