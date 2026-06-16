# Capstone Project 07 — End-to-End Fine-Tuning Pipeline (Data to SFT to DPO to Serving)

> An 8B model trained on your own data, DPO-aligned with your own preferences, quantized, speculatively decoded, and served at a measurable $/1M tokens. The 2026 open-source stack is Axolotl v0.8, TRL 0.15, Unsloth for iteration, GPTQ/AWQ/GGUF for quantization, and vLLM 0.7 with EAGLE-3 for serving. This capstone project runs the entire pipeline reproducibly—YAML in, serving endpoint out—and publishes a model card under the 2026 Model Openness Framework.

**Type:** Capstone
**Languages:** Python (pipeline), YAML (config), Bash (scripts)
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (Transformers), Phase 10 (LLMs from Scratch), Phase 11 (LLM Engineering), Phase 17 (Infrastructure), Phase 18 (Safety)
**Phases involved:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 hours

## The Problem

Every serious AI team in 2026 keeps a fine-tuning pipeline ready at all times. Not because they are shipping a frontier foundation model, but because downstream adaptation—domain SFT, DPO against labeled preferences, distilling a draft for speculative decoding, serving with EAGLE-3—is where the measurable gains live. Axolotl v0.8 handles multi-GPU SFT configuration. TRL 0.15 handles DPO and GRPO. Unsloth lets you iterate fast on a single GPU. vLLM 0.7 with EAGLE-3 pushes decoding throughput to 2-3x with no quality loss. The tools are mature; the craft lies in the YAML, data hygiene, and evaluation discipline.

You will take an 8B base (Llama 3.3, Qwen3, or Gemma 3), SFT then DPO it on task-specific data, quantize for serving, and measure the gains against lm-evaluation-harness, RewardBench-2, MT-Bench-v2, and MMLU-Pro. You will produce a model card under the 2026 Model Openness Framework. The emphasis is on reproducibility—one command reruns the entire pipeline end-to-end.

## The Concept

The pipeline has five stages. **Data**: deduplication (MinHash / Datatrove), quality filtering (Nemotron-CC-style classifier), PII scrubbing, split hygiene with contamination checks against public benchmarks. **SFT**: Axolotl YAML, ZeRO-3 on 8x H100, cosine schedule, sequence packing, 2-3 epochs. **DPO or GRPO**: TRL config, 1 epoch, preference pairs (human-labeled or model-judged), beta tuning. **Quantize**: GPTQ + AWQ + GGUF for deployment flexibility. **Serve**: vLLM 0.7 with EAGLE-3 speculative head (or SGLang with SpecForge), K8s deployment, HPA on queue-wait.

Ablations are the deliverable: SFT-only vs SFT+DPO vs SFT+GRPO on three task-specific benchmarks. Serving metrics: tokens/s at batch sizes 1 / 8 / 32, EAGLE-3 acceptance rate, $/1M tokens. Safety evaluation: Llama Guard 4 pass rate. Model card: bias assessment, reproducible seeds, data licensing.

## Architecture

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## Tech Stack

- Data: Datatrove for deduplication, Nemotron-CC classifier for quality, Presidio for PII
- Base model: Llama 3.3 8B, Qwen3 14B, or Gemma 3 12B
- SFT: Axolotl v0.8 with ZeRO-3, Flash Attention 3, sequence packing
- Preference tuning: TRL 0.15 for DPO or GRPO; Unsloth for single-GPU iteration
- Quantization: GPTQ (Marlin), AWQ, GGUF via llama.cpp
- Serving: vLLM 0.7 with EAGLE-3 speculative decoding (or SGLang 0.4 + SpecForge)
- Evaluation: lm-evaluation-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro
- Safety evaluation: Llama Guard 4, ShieldGemma-2
- Infrastructure: Kubernetes + NVIDIA device plugin, HPA on queue-wait metric
- Observability: W&B for training, Langfuse for inference

## Build It

1. **Data pipeline.** Run Datatrove deduplication on the raw corpus. Apply a Nemotron-CC-style quality classifier. Scrub PII with Presidio. Write out train/val splits with explicit random seeds.

2. **Contamination check.** For each validation split, compute MinHash against MMLU-Pro, MT-Bench-v2, and RewardBench-2 test sets. Reject any overlap.

3. **Axolotl SFT.** YAML with ZeRO-3, FA3, sequence packing. 2-3 epochs on 8x H100. Log to W&B.

4. **TRL DPO / GRPO.** Take the SFT checkpoint and run one epoch of DPO on preference pairs (or GRPO with verifiable rewards on math/code). Sweep beta.

5. **Quantization.** Produce three quantized variants: GPTQ-INT4-Marlin, AWQ-INT4, GGUF-Q4_K_M for llama.cpp. Record sizes and nominal throughput.

6. **Serve with speculative decoding.** vLLM 0.7 config with an EAGLE-3 draft head trained using Red Hat Speculators. Measure acceptance rate and tail latency at batch sizes 1 / 8 / 32. Report $/1M tokens vs Anthropic / OpenAI on the same evaluation.

7. **Evaluation matrix.** Run lm-eval-harness, RewardBench-2, MT-Bench-v2, MMLU-Pro on base, SFT-only, SFT+DPO, SFT+GRPO. Produce a table.

8. **Safety evaluation.** Llama Guard 4 pass rate on a development set. ShieldGemma-2 output filter.

9. **Model card.** MOF 2026 template: data, training, evaluation, safety, licensing, reproducibility section with YAML and commit SHA.

## Use It

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## Ship It

`outputs/skill-finetuning-pipeline.md` describes the deliverable. One command takes data through SFT, DPO, quantization, serving, and evaluation, producing a model card plus the serving endpoint.

| Weight | Criterion | How to Measure |
|:-:|---|---|
| 25 | Eval delta vs base | Measured gains on target task (MMLU-Pro, MT-Bench-v2, task-specific) |
| 20 | Pipeline reproducibility | One command reruns end-to-end with same random seeds |
| 20 | Data hygiene | Dedup rate, PII scrub coverage, contamination check green |
| 20 | Serving efficiency | tokens/s at bs=1/8/32, EAGLE-3 acceptance rate, $/1M tokens |
| 15 | Model card + safety eval | 2026 MOF completeness + Llama Guard 4 pass rate |
| **100** | | |

## Exercises

1. Run SFT-only vs SFT+DPO vs SFT+GRPO on the same task-specific benchmark. Report which preference method wins and by how much.

2. Swap Llama 3.3 8B for Qwen3 14B. Measure $/1M tokens at quality parity.

3. Measure EAGLE-3 acceptance rate on domain data vs generic ShareGPT. Report the delta and what it means for latency budgets.

4. Inject 1% contamination (leak MMLU-Pro answers into training data) and rerun evaluation. Watch MMLU-Pro accuracy jump unrealistically. Build a contamination-check CI gate that catches it.

5. Add LoRA SFT as an alternative to full fine-tuning. Measure the quality gap at 10x lower memory.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|------------------------|
| Axolotl | "the SFT trainer" | A unified, YAML-driven trainer supporting SFT, DPO, and distillation |
| TRL | "the preference tuner" | Hugging Face's library for DPO, GRPO, and PPO on LLMs |
| GRPO | "Group Relative Policy Optimization" | DeepSeek R1's RL recipe with verifiable rewards |
| EAGLE-3 | "speculative decoding draft" | A draft head predicting N tokens ahead; vLLM verifies with the target model |
| MOF | "Model Openness Framework" | The 2026 standard for rating model releases on data, code, and licensing |
| Contamination check | "split hygiene" | MinHash-based detection of test-set leakage into training |
| Acceptance rate | "EAGLE / MTP metric" | The fraction of drafted tokens accepted by the target model |

## Further Reading

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — reference SFT / DPO trainer
- [TRL documentation](https://huggingface.co/docs/trl) — reference implementation for DPO and GRPO
- [Unsloth](https://github.com/unslothai/unsloth) — single-GPU iteration reference
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO methodology
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — reference serving stack
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — alternative speculative decoding trainer
- [Model Openness Framework 2026](https://isocpp.org/) — open-source release rating standard
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — standard evaluation runner
