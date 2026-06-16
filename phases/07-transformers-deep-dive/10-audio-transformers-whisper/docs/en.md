# Audio Transformers — Whisper Architecture

> Audio is an image of frequency over time. Whisper is a ViT that eats mel spectrograms and speaks back.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 min

## The Problem

Before Whisper (OpenAI, Radford et al. 2022), state-of-the-art automatic speech recognition (ASR) meant wav2vec 2.0 and HuBERT — self-supervised feature extractors plus a fine-tuned head. High quality, expensive data pipelines, brittle to domain shifts. Multilingual ASR required a separate model per language family.

Whisper made three bets:

1. **Train on everything.** 680k hours of weakly-labeled audio scraped from the internet, across 97 languages. No clean academic corpora. No phoneme labels.
2. **One model, many tasks.** A single decoder jointly trained on transcription, translation, voice activity detection, language identification, and timestamps via task tokens.
3. **Standard encoder-decoder transformer.** Encoder eats log-mel spectrograms. Decoder autoregressively produces text tokens. No vocoder, no CTC, no HMM.

The result: Whisper large-v3 is robust to accents, noise, and languages with zero clean labeled data. It's the default speech frontend for every open-source voice assistant and most commercial ones in 2026.

## The Concept

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Step 1 — Resample + window

Audio at 16 kHz. Clip/pad to 30 seconds. Compute log-mel spectrogram: 80 mel bins, 10 ms stride → ~3,000 frames × 80 features. This is the "input image" Whisper sees.

### Step 2 — Convolutional stem

Two Conv1D layers with kernel 3, stride 2 reduce 3,000 frames to 1,500. Halves sequence length without adding many parameters.

### Step 3 — Encoder

A 24-layer (large) transformer encoder over 1,500 time steps. Sinusoidal positional encoding, self-attention, GELU FFN. Outputs 1,500 × 1,280 hidden states.

### Step 4 — Decoder

A 24-layer transformer decoder. It autoregressively produces tokens from a BPE vocabulary — a superset of GPT-2's vocabulary plus a few audio-specific special tokens.

### Step 5 — Task tokens

The decoder prompt starts with control tokens that tell the model what to do:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

Or:

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

The model was trained on this convention. You control the task with the prefix. The 2026 equivalent of instruction tuning, applied to speech.

### Step 6 — Output

Beam search (width 5) with log-prob thresholding. When the `<|notimestamps|>` token is absent, a timestamp is predicted every 0.02 seconds of audio.

### Whisper model sizes

| Model | Params | Layers | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo (2024) cuts the decoder from 32 layers to 4. Decoding is 8× faster with less than 1 point WER regression. This decoding speed unlock made Whisper-turbo the default for real-time voice agents in 2026.

### What Whisper doesn't do

- No speaker diarization (who is speaking). Pair with pyannote for that.
- No native real-time streaming — the 30-second window is fixed. Modern wrappers (`faster-whisper`, `WhisperX`) add streaming via VAD + overlap.
- No long-form context beyond 30 seconds without external chunking. Works well in practice because human speech transcription rarely needs long-range context.

### 2026 landscape

| Task | Model | Notes |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine is 4× faster on edge |
| Multilingual ASR | Whisper-large-v3 | 97 languages |
| Streaming ASR | faster-whisper + VAD | 150 ms latency target achievable |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder pattern, but Whisper-shaped |
| Audio + language | AudioLM, SeamlessM4T | Text tokens + audio tokens in one transformer |

## Build It

See `code/main.py`. We don't train Whisper — we build the log-mel spectrogram pipeline + task token prompt formatter. These are the parts you actually touch in production.

### Step 1: Synthetic audio

Generate a 1-second, 440 Hz sine wave sampled at 16 kHz. 16,000 samples.

### Step 2: Log-mel spectrogram (simplified)

A full mel spectrogram requires FFT. We build a simplified frame + per-frame energy version to show the pipeline without needing `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms, hop = 10 ms. Matches Whisper's windowing. Per-frame energy substitutes for mel bins in this pedagogical version.

### Step 3: Pad to 30 seconds

Whisper always processes 30-second chunks. Pad (or clip) the spectrogram to 3,000 frames.

### Step 4: Build prompt tokens

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

This is the entire task control surface. A 4-token prefix.

## Use It

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Faster, OpenAI-compatible:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**When to pick Whisper in 2026:**

- Multilingual ASR with a single model.
- Robust transcription on noisy, diverse audio.
- Research / prototype ASR — fastest starting point.

**When to pick something else:**

- Ultra-low latency streaming on edge — Moonshine beats Whisper at equal quality.
- Real-time conversational AI needing <200 ms — dedicated streaming ASR.
- Speaker diarization — Whisper doesn't do this; add pyannote.

## Ship It

See `outputs/skill-asr-configurator.md`. This skill picks an ASR model, decoding parameters, and preprocessing pipeline for a new speech application.

## Exercises

1. **Easy.** Run `code/main.py`. Confirm that a 1-second signal at 16 kHz with 10 ms hop produces ~100 frames. 30 seconds: ~3,000 frames.
2. **Medium.** Build a full log-mel spectrogram using `numpy.fft`. Verify that 80 mel bins match `librosa.feature.melspectrogram(n_mels=80)` within numerical tolerance.
3. **Hard.** Implement streaming inference: split audio into 10-second windows with 2-second overlap, run Whisper on each chunk, merge transcriptions. Measure word error rate delta vs. single-pass on a 5-minute podcast sample.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| Mel spectrogram | "audio image" | 2D representation: one axis is frequency bins, the other is time frames; each cell is log-scaled energy. |
| Log-mel | "what Whisper sees" | Mel spectrogram after log transform; approximates human loudness perception. |
| Frame | "one time slice" | A 25 ms window of samples; overlapping with 10 ms stride. |
| Task token | "speech prompt prefix" | Special tokens like `<|transcribe|>` / `<|translate|>` in the decoder prompt. |
| Voice Activity Detection (VAD) | "find the speech" | Gating that removes silence before ASR; drastically cuts cost. |
| CTC | "Connectionist Temporal Classification" | Classic ASR loss for alignment-free training; Whisper doesn't use it. |
| Whisper-turbo | "small decoder, full encoder" | large-v3 encoder + 4-layer decoder; 8× faster decoding. |
| Faster-whisper | "production wrapper" | CTranslate2 reimplementation; int8 quantization; 4× faster than OpenAI reference. |

## Further Reading

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — The Whisper paper.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — Reference code + model weights. Read `whisper/model.py`, ~400 lines covering Conv1D stem + encoder + decoder end-to-end.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — The beam-search + task token logic described in steps 5–6 lives here; 500 lines, fully readable.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — Predecessor; still SOTA features in some scenarios.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Production wrapper, 4× faster than reference.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 edge-friendly ASR, Whisper-shaped but smaller.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — Canonical fine-tuning recipe with mel spectrogram preprocessor and token timestamp handling.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — Full implementation (encoder, decoder, cross-attention, generation), corresponding to the architecture diagram in this lesson.
