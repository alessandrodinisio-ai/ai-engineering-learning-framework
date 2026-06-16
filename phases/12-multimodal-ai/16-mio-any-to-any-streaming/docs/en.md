# MIO and Any-to-Any Streaming Multimodal Models

> GPT-4o shipped a product most open models can't replicate: an agent that hears speech, sees video, and talks back in real time. By late 2024, the open ecosystem's answer is MIO (Wang et al., September 2024). MIO tokenizes text, images, speech, and music, trains a causal transformer on interleaved sequences, and generates from any modality to any modality. AnyGPT (Zhan et al., February 2024) is the proof of concept; MIO is the scaled-up version; Unified-IO 2 (Allen AI, December 2023) is the cousin with visual + action grounding. This lesson reads through the any-to-any pattern — four tokenizers, one transformer, streaming-friendly decoding.

**Type:** Learn
**Languages:** Python (standard library, four-modality token allocator + streaming decode loop)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech & Audio)
**Time:** ~120 minutes

## Learning Objectives

- Design a shared vocabulary accommodating text, image, speech, and music tokens without collisions.
- Compare SEED-Tokenizer (image) and SpeechTokenizer residual-VQ (speech) on compression + reconstruction trade-offs.
- Explain the four-stage curriculum that builds any-to-any generation layer by layer.
- State three open any-to-any recipes and their main trade-offs: MIO, AnyGPT, Unified-IO 2.

## The Problem

Unified multimodal models are easy to claim and hard to build at scale. As of 2024, most "any-to-any" systems are pipelines: vision model → text representation → speech model → audio. Each hop loses information, adds latency, and complicates training. GPT-4o's demo video showed a single-model alternative with sub-second response; open systems lagged by months.

The engineering challenges:

- Each modality needs a tokenizer that compresses losslessly enough for reconstruction and produces tokens at a rate the transformer can consume.
- A single vocabulary must allocate space for text (32k+), image (16k+), speech (4k+), music (8k+). At least 40k+ entries minimum.
- Training data must cover every input-output pair (text→image, image→speech, speech→image, etc.), or the model must compose on its own.
- Inference must stream output tokens fast enough to support conversational latency (time-to-first-audio-byte <500ms).

## The Concept

### Four Tokenizers for Four Modalities

MIO's tokenizer stack:

- Text: standard BPE, ~32000 vocabulary.
- Image: SEED-Tokenizer (2023) — quantized VAE with discrete codebook, 4096 entries, 32x32 tokens per image.
- Speech: SpeechTokenizer residual-VQ (2023) — encodes 16kHz waveform into 8 hierarchical codebooks; first level is coarse content, subsequent levels add prosody and speaker identity.
- Music: similar residual-VQ (Meta's MusicGen / Encodec family), 4-8 codebooks.

Each modality outputs integer tokens. These tokens get non-overlapping ID ranges in the shared vocabulary:

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

Total: ~48k vocabulary. The input embedding and output projection span all of it.

### Streaming Decode

Speech generation uses residual-VQ. The transformer predicts base (level-0) speech tokens; a parallel-decoded residual quantizer predicts subsequent levels. Each level-0 token is roughly 50ms of audio at 16kHz.

Streaming mode:

1. User speaks into microphone; real-time audio tokenizer emits batches of speech tokens every ~50ms.
2. MIO consumes tokens as they arrive (prompt prefill + incremental forward).
3. Output tokens stream out as generated; a parallel speech decoder converts them to audio samples at ~50-150ms latency.
4. Time-to-first-audio-byte: ~300-500ms in the MIO paper, approaching GPT-4o's ~250ms.

Mini-Omni (arXiv:2408.16725), GLM-4-Voice (arXiv:2412.02612), and Moshi (arXiv:2410.00037) are complementary streaming speech LLM designs. Moshi notably achieves 160ms round-trip on a single GPU.

### Four-Stage Curriculum

MIO's training curriculum:

1. Stage 1 — Alignment. Large-scale modality-pair corpora: text-image, text-speech, text-music. Each pair uses its own token vocabulary segment. Trains the shared vocabulary.
2. Stage 2 — Interleaving. Multimodal interleaved documents (blogs with images + video, podcasts with transcripts, etc.). Trains cross-modal context.
3. Stage 3 — Speech enhancement. Additional audio data to improve speech quality without losing text capabilities.
4. Stage 4 — SFT. Cross-modal instruction tuning: VQA, captioning, narration, speech-to-speech conversation.

Drop a stage and specific capabilities degrade: skip stage 2 and the model loses cross-modal context; skip stage 3 and speech suffers.

### Visual Chain-of-Thought

MIO introduces visual chain-of-thought: the model emits intermediate image tokens as a reasoning step. For "is the cat climbing the tree?" the model:

1. Emits `<image>` tokens rendering the scene (from the input image or a sketch).
2. Emits text analyzing the sketch.
3. Emits the final answer.

The rendered intermediate image acts as scratch paper. Benchmark gains on spatial reasoning tasks. This idea parallels chain-of-thought for text reasoning.

### Competitors in the Any-to-Any Space

- AnyGPT (arXiv:2402.12226): 4 modalities (text, image, speech, music), similar design.
- Unified-IO 2 (arXiv:2312.17172): adds visual action outputs, depth, normals. More diverse tasks, smaller scale.
- NExT-GPT (arXiv:2309.05519): LLM + modality-specific diffusion decoders. Not the single-model route.
- CoDi (arXiv:2305.11846): composable diffusion; any-to-any via shared latents.

MIO is closest to pure-token any-to-any. AnyGPT is its conceptual ancestor.

### Latency Budget

For a conversational product, every component's latency matters:

- Microphone to audio tokens: ~50ms.
- Prefill (audio tokens + history): ~100ms on an 8B model.
- First output token: ~50ms.
- Parallel residual-VQ + speech decoder: ~100-150ms.

Total time-to-first-audio-byte: minimum ~300ms. GPT-4o claims ~250ms. Moshi claims 160ms. MIO/AnyGPT sits in the 400-600ms range per public benchmarks.

### Why Any-to-Any Is Still Hard

Even in 2026, open any-to-any models lag behind closed-source on two axes:

- Speech quality. Residual-VQ tokenizers are lossy; conversational speech sounds robotic compared to ElevenLabs-tier voices.
- Cross-modal reasoning. Getting the model to "sing a song about what you see" still fails more often than pure-visual tasks.

These are open research problems. Qwen3-Omni (Lesson 12.20) is the 2025 state-of-the-art open attempt.

## Use It

`code/main.py`:

- Defines the four-modality vocabulary allocation and prints it.
- Routes a set of multimodal inputs (text, image, audio clip, music) through a tokenizer router.
- Simulates streaming decode for a text-to-speech response and counts latency.
- Computes expected time-to-first-audio-byte given encoder, prefill, and decoder latencies.

## Ship It

This lesson produces `outputs/skill-any-to-any-pipeline-auditor.md`. Given a conversational product spec (input modalities, output modalities, latency target), it audits MIO-family design choices and computes the latency budget.

## Exercises

1. Your product accepts speech input and returns speech output. What's the end-to-end latency budget target? List the components that consume time.

2. SpeechTokenizer residual-VQ uses 8 codebooks. Argue why parallel decoding of residual levels is necessary (vs sequential) and what latency savings it buys.

3. Your vocabulary has 32k text + 4k image + 4k speech. Add 8k music and ~10 separators. What's the embedding matrix parameter cost at hidden dimension 4096?

4. Visual chain-of-thought emits an intermediate image. What kinds of questions benefit? What kinds are hurt by the extra tokens?

5. Read Moshi (arXiv:2410.00037). Describe its "inner monologue" technique and compare with MIO's visual chain-of-thought.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Any-to-any | "multimodal in/out" | Single model accepting and producing text, image, speech, music in any direction |
| Residual-VQ | "speech tokenizer stack" | Multi-codebook tokenization where each layer adds information; base layer is content, subsequent layers are prosody |
| SEED-Tokenizer | "image encoding" | Discrete image tokenizer with 4096-entry codebook used by MIO |
| Visual chain-of-thought | "visual scratch paper" | Model generates an intermediate image as a reasoning step before the final answer |
| Time-to-first-audio-byte | "TTFAB" | Latency from user speech to first audio output; conversational feel requires <500ms |
| Four-stage curriculum | "training recipe" | Alignment → interleaving → speech enhancement → SFT, in that order |

## Further Reading

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
