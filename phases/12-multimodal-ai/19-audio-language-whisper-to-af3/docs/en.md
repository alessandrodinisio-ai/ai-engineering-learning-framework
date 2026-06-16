# Audio-Language Models: The Arc from Whisper to Audio Flamingo 3

> Whisper (Radford et al., December 2022) closed the book on speech recognition — 680k hours of weakly-supervised multilingual speech, a simple encoder-decoder transformer, and a benchmark that every subsequent ASR release cites. But recognition is not reasoning. Asking "what instruments are in this recording," "what emotion is the speaker expressing," "what happens at minute 3" requires audio understanding, not transcription. Qwen-Audio, SALMONN, LTU, and NVIDIA's Audio Flamingo 3 (AF3, July 2025) built up this stack layer by layer: keep a Whisper-class encoder, attach a Q-former, train on audio-text instruction data, add chain-of-thought reasoning. This lesson walks the arc.

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech & Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 minutes

## Learning Objectives

- Compute a log-Mel spectrogram from a waveform: windowing, FFT, filterbank, log transform.
- Compare encoder options: Whisper encoder, BEATs, AF-Whisper hybrid. When each wins.
- Build an audio Q-former: N learnable queries cross-attend to spectrogram patches.
- Explain cascade (Whisper-then-LLM) vs end-to-end audio LLM training: why end-to-end scales better for reasoning.

## The Problem

Speech recognition is solved by Whisper. OCR for audio became a commodity. But "commodity" stops at transcription. If the model cannot reason about what it hears — temporal ordering, speakers, emotion, music structure, environmental sounds — transcription alone won't power product features.

Three obvious paths:

1. Cascade: Whisper transcribes, LLM reasons over the transcript text. Works for pure-speech scenarios. Fails for music, environmental audio, multi-speaker overlap, emotion.

2. End-to-end audio LLM: an audio encoder feeds audio tokens directly into the LLM, skipping transcription. Preserves acoustic information (emotion, speaker, environment). Requires new training data.

3. Hybrid: audio encoder + a text decoder that can both transcribe and reason. Qwen-Audio and Audio Flamingo took this path.

## The Concept

### Log-Mel Spectrogram: The Input Feature

Every audio encoder starts from the same feature: a log-Mel spectrogram.

1. Resample to 16 kHz.
2. Short-time Fourier transform with 25ms window, 10ms hop.
3. Take magnitude of the FFT output.
4. Apply a Mel filterbank (typically 80 filters, log-spaced over 0–8000 Hz) to warp to perceptual frequencies.
5. Log compression (log(1 + x)) to stretch dynamic range.

Result: a 2D array of shape (T, 80), where T is the number of time frames. For a 30-second clip at 100 Hz frame rate: (3000, 80).

### Whisper's Encoder

Whisper's encoder is a 12-layer ViT-style transformer that processes the log-Mel spectrogram as a sequence of time frames. Output: one hidden-state vector per time frame.

For ASR, Whisper's decoder is a cross-attention transformer that generates text tokens conditioned on encoder output. Standard encoder-decoder.

For ALMs (audio LLMs), you want the encoder output as input to a different LLM. The pattern: Whisper encoder frozen, Q-former trainable, LLM frozen or fine-tuned.

### BEATs and Audio-Specific Encoders

Whisper was trained on speech-dominant data. It's weaker on music and environmental audio.

BEATs (Chen et al., 2022) is a self-supervised transformer trained on AudioSet. Captures music and environmental sounds better than Whisper at the same parameter count.

AF-Whisper (Audio Flamingo 3's hybrid): concatenates Whisper + BEATs features as audio input. Whisper carries linguistic signal, BEATs carries acoustic signal.

### Audio Q-former

Same pattern as BLIP-2's visual Q-former. A fixed number of learnable queries (commonly 32 or 64) cross-attend to the audio encoder's output frames. The queries become the audio tokens the LLM consumes.

Alignment training stage: Q-former alone, contrastive + captioning loss on audio-text pairs (AudioCaps, Clotho). Instruction stage: end-to-end, unfreeze LLM, train on instruction data.

### The Arc — SALMONN, Qwen-Audio, AF3

SALMONN (Tang et al., 2023): Whisper + BEATs + Q-former + LLaMA. First open audio LLM with serious reasoning capability. Benchmarks on MMAU show composite score ~0.55.

Qwen-Audio (Chu et al., 2023): similar architecture, trained on richer datasets, tuned for multi-turn conversation. MMAU ~0.60.

LTU — Listen, Think, Understand (Gong et al., 2023): explicit reasoning data, focused on chain-of-thought over audio clips. Smaller but more focused.

Audio Flamingo 3 (Goel et al., July 2025): current open SOTA. 8B LLM backbone (Qwen2 7B), Whisper-large encoder concatenated with BEATs, 64-query Q-former, trained on 1M+ audio-text instruction pairs. MMAU 0.72, matching proprietary frontier on some subtasks.

AF3 also introduces on-demand chain-of-thought for audio: the model can optionally emit thinking tokens before the final answer ("let me first identify instruments: ..."). When thinking is enabled, accuracy on complex reasoning tasks improves 3–5 points.

### Cascade vs End-to-End

Cascade pipeline:

1. Whisper transcribes audio → text.
2. LLM reasons over text.

Works perfectly for "summarize this podcast episode." Fails for:
- "What's the mood of this song?" — mood is in the sound, not the words.
- "Is that Alice or Bob speaking?" — requires speaker identification.
- "At what second does the explosion happen?" — temporal grounding is lost in text.
- "Is this real audio or generated?" — deepfake detection needs acoustic features.

End-to-end preserves acoustic signal. Qwen-Audio and AF3 natively handle music, environment, and emotion.

### 2026 Production Recipe

For a new audio understanding product:

- Cascade, if: transcription is the goal, no music, no emotion inference.
- AF3 / Qwen-Audio family, if: music, emotion, multi-speaker, or complex audio reasoning.

Cascade is cheaper and simpler. End-to-end is more capable.

### MMAU — Audio Reasoning Benchmark

MMAU (Massive Multimodal Audio Understanding) is the 2024–2025 audio reasoning benchmark:

- 10,000 audio-text QA pairs spanning speech, music, environmental sound.
- Covers classification, temporal reasoning, causal reasoning, open-ended QA.
- Tests what cascade pipelines systematically miss.

Open SOTA (AF3) at 0.72; proprietary frontier ~0.78 (Gemini 2.5 Pro, Claude Opus 4.7). This gap is smaller than VideoMME's open-vs-closed gap, indicating audio LLMs are maturing.

## Use It

`code/main.py`:

- Implements log-Mel spectrogram computation with stdlib: windowing, naive DFT, Mel filterbank.
- Audio Q-former skeleton: given encoder output frames, computes Q, K, V, attention, and emits N tokens.
- Cascade vs end-to-end comparison on a toy task.

## Ship It

This lesson produces `outputs/skill-audio-llm-pipeline-picker.md`. Given an audio task (transcription, music tagging, emotion inference, multi-speaker diarization, environment classification), it picks cascade, end-to-end AF3, or hybrid.

## Exercises

1. Calculate the log-Mel spectrogram dimensions for a 30-second, 16kHz clip with 25ms window, 10ms hop, 80 Mel bins. How does this change at 48kHz?

2. Why does Whisper underperform on music? What audio features does BEATs capture that Whisper does not?

3. 64-query vs 32-query audio Q-former: at what task complexity does 64 pay off? For what scenarios does 32 save compute?

4. Read AF3 Section 4 on on-demand thinking. Propose three audio tasks where chain-of-thought helps the most.

5. Implement a minimal speaker diarization pipeline using AF3's output. How do you demarcate speaker switches?

## Key Terms

| Term | How people say it | What it actually means |
|------|-----------------|------------------------|
| Log-Mel spectrogram | "Mel features" | 2D (time, frequency) array of log-amplitude values after Mel filterbank |
| Audio Q-former | "audio Perceiver" | Cross-attention bottleneck from audio encoder output to fixed-length queries fed to LLM |
| Cascade | "ASR-then-LLM" | Pipeline of Whisper transcription then text LLM reasoning; loses acoustic information |
| End-to-end | "audio LLM" | Audio features via Q-former directly into LLM; preserves acoustic signal |
| BEATs | "AudioSet encoder" | SSL transformer trained on AudioSet; strong on music + environmental sound |
| MMAU | "audio reasoning benchmark" | 10k QA pairs spanning speech, music, environment; 2024 evaluation standard |
| On-demand thinking | "audio CoT" | Model optionally emits reasoning tokens before final answer, improving accuracy 3–5 points |

## Further Reading

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
