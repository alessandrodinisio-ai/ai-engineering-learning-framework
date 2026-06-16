# Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> The core constraint at the edge is memory bandwidth, not compute. Mobile DRAM runs at 50-90 GB/s; data center HBM3 exceeds 2-3 TB/s — a 30-50x gap. Decode is memory-bound, so this gap is decisive. The 2026 landscape splits into four targets. Apple M4/A18 Neural Engine peaks at 38 TOPS with unified memory (no CPU↔NPU copy). Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon reaches 45 TOPS. WebGPU + WebLLM runs Llama 3.1 8B (Q4) at ~41 tok/s on M3 Max (roughly 70-80% of native); 17.6k GitHub stars, OpenAI-compatible API, ~70-75% mobile coverage. NVIDIA Jetson Orin Nano Super (8GB) fits Llama 3.2 3B / Phi-3; AGX Orin runs gpt-oss-20b at ~40 tok/s via vLLM; Jetson T4000 (JetPack 7.1) is 2x AGX Orin. TensorRT Edge-LLM supports EAGLE-3, NVFP4, chunked prefill — demonstrated by Bosch, ThunderSoft, MediaTek at CES 2026.

**Type:** Learn
**Languages:** Python (standard library, a toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 minutes

## Learning Objectives

- Explain why mobile LLM inference is memory-bandwidth-bound and compute is secondary.
- List four edge targets (Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson) and map each to a use case.
- State the 2026 WebGPU coverage gap (Firefox Android catching up) and Safari iOS 26 landing.
- Pick a quantization format for each target (Core ML INT4 + FP16 for ANE, QNN INT8/INT4 for Hexagon, WebGPU Q4 for browser, NVFP4 for Jetson Thor).

## The Problem

A client wants an on-device chatbot: voice-first, private by default, works offline. On a MacBook Pro M3 Max, Llama 3.1 8B Q4 runs at ~55 tok/s — acceptable. On iPhone 16 Pro, the same model runs at 3 tok/s — unusable. On a mid-range Android with Snapdragon 8 Gen 3, 7 tok/s. In the browser via WebGPU on Chrome Android v121+, 4-8 tok/s depending on device.

The variance in throughput is not a porting problem. It is the bandwidth gap multiplied by quantization format multiplied by whether the NPU is accessible from userspace. Edge inference in 2026 is four different problems with four different solutions.

## The Concept

### Bandwidth Is the Real Ceiling

Decode reads the full weight set for every token. A Q4 7B model is 3.5 GB. Reading 3.5 GB at 50 GB/s takes 70 ms — theoretical ceiling of ~14 tok/s. At 90 GB/s (high-end mobile DRAM) the ceiling moves to ~25 tok/s. Below this, no amount of extra compute helps.

Data center HBM3 reads the same 3.5 GB in 1.2 ms at 3 TB/s — ceiling is 830 tok/s. Same model, same weights. Different memory subsystem.

### Apple Neural Engine (M4 / A18)

- Up to 38 TOPS. Unified memory (CPU and ANE share the same pool) — no copy overhead.
- Accessed via Core ML + `.mlmodel` compiled models, or via Metal Performance Shaders (MPS) through PyTorch.
- Llama.cpp Metal backend uses MPS, not ANE directly; native ANE requires Core ML conversion.
- Best-practice path for 2026 iOS apps: Core ML with INT4 weights + FP16 activations.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- Up to 45 TOPS. Integrated with CPU and GPU in the SoC, but separate memory domains.
- QNN (Qualcomm Neural Network) SDK and AI Hub provide conversion from PyTorch/ONNX.
- Chat templates, Llama 3.2, Phi-3 are all published as first-class artifacts on AI Hub.

### Intel / AMD NPU (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS. Software lags behind Apple/Qualcomm; OpenVINO is improving but niche.
- Best suited for Windows ARM copilot apps; native support for local-first on AMD/Intel desktops.

### WebGPU + WebLLM

- Runs models in the browser via WebGPU compute shaders; no install required.
- Llama 3.1 8B Q4 at ~41 tok/s on M3 Max — roughly 70-80% of native via the same backend.
- WebLLM has 17.6k stars on GitHub; OpenAI-compatible JS API; Apache 2.0.
- 2026 coverage: Chrome Android v121+, Safari iOS 26 GA, Firefox Android still catching up. Overall ~70-75% mobile coverage.

### NVIDIA Jetson Family

- Orin Nano Super (8GB): fits Llama 3.2 3B, Phi-3, decent tok/s.
- AGX Orin: runs gpt-oss-20b at ~40 tok/s via vLLM.
- Thor / T4000 (JetPack 7.1): 2x AGX Orin performance, supports EAGLE-3 and NVFP4.
- TensorRT Edge-LLM (2026) supports EAGLE-3 speculative decoding, NVFP4 weights, chunked prefill — porting data center optimizations to the edge.

### Quantization Choices per Target

| Target | Format | Notes |
|--------|--------|-------|
| Apple ANE | INT4 weights + FP16 activations | Core ML conversion path |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub converter |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | Use `mlc_llm convert_weight` + compiled `.wasm`; GGUF not supported |
| Jetson Orin Nano | Q4 GGUF or TRT-LLM INT4 | Memory-constrained |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM path |

### Long-Context Pitfall on Edge

Llama 3.1's 128K context is a data center feature. On a phone with 8 GB RAM, a 4 GB model + 2 GB KV cache for 32K tokens + OS overhead = OOM. Edge deployments keep context at 4K-8K unless accepting aggressive KV quantization (Q4 KV).

### Voice Is the Killer App

Voice agents are latency-sensitive (first token < 500 ms). Local inference eliminates network latency entirely. Combined with speech-to-text (Whisper Turbo variants run on edge), edge inference becomes a production-quality voice loop.

### Numbers You Should Remember

- Apple M4 / A18 ANE: 38 TOPS.
- Qualcomm Hexagon SD X Elite: 45 TOPS.
- WebLLM M3 Max: Llama 3.1 8B Q4 ~41 tok/s.
- AGX Orin: gpt-oss-20b via vLLM ~40 tok/s.
- Data center vs edge bandwidth gap: 30-50x.
- WebGPU mobile coverage: ~70-75% (Firefox Android lags).

## Use It

`code/main.py` computes theoretical decode throughput ceilings across edge targets using bandwidth-bound math. Compares against observed benchmarks and identifies where bandwidth (not compute) is the bottleneck.

## Ship It

This lesson produces `outputs/skill-edge-target-picker.md`. Given a platform (iOS/Android/browser/Jetson), model, and latency/memory budget, picks a quantization format and conversion pipeline.

## Exercises

1. Run `code/main.py`. For a Q4 7B model on Snapdragon 8 Gen 3 (~77 GB/s bandwidth), compute the decode ceiling. Compare against observed 6-8 tok/s — is the runtime efficient?
2. WebGPU on Android requires Chrome v121+. Design a fallback for older browsers — server-side via the same OpenAI-compatible API.
3. Your iOS app needs 4K context streaming. Which model/format combination keeps active memory under 4 GB on iPhone 16?
4. Jetson AGX Orin runs gpt-oss-20b at 40 tok/s. Jetson Nano only fits a 3B. If your product targets both, how do you unify the inference stack?
5. Argue whether "WebLLM is production-ready in 2026." Cite coverage, performance, and the Firefox Android gap.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| ANE | "Apple Neural Engine" | On-device NPU in M-series and A-series; unified memory |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU; accessed via QNN SDK |
| WebGPU | "Browser GPU" | W3C standardized browser GPU API; Chrome/Safari 2026 |
| WebLLM | "Browser LLM runtime" | MLC-LLM project; Apache 2.0; OpenAI-compatible JS |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000 family |
| TRT Edge-LLM | "Edge TensorRT" | TensorRT-LLM's 2026 edge port; EAGLE-3 + NVFP4 |
| Unified memory | "Shared pool" | CPU and NPU see the same RAM; no copy overhead |
| Bandwidth-bound | "Memory-bound" | Decode is bottlenecked by bytes/s reading weights |
| Core ML | "Apple conversion" | Apple's framework for ANE-native models |
| QNN | "Qualcomm stack" | Qualcomm Neural Network SDK |

## Further Reading

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — Landscape and benchmarks.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 edge port release.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — Design and benchmarks.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE-native conversion.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — Pre-converted models for Hexagon.
