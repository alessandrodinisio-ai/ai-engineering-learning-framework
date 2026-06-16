# Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC, and the Semantic-Acoustic Split

> Audio generation in 2026 runs almost entirely on tokens. EnCodec, SNAC, Mimi, and DAC turn continuous waveforms into discrete sequences that transformers can predict. The semantic-acoustic token split — first codebook as semantic, the rest as acoustic — is the most important architectural shift in audio since the Transformer itself.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## The Problem

Language models work on discrete tokens. Audio is continuous. If you want an LLM-style model for speech / music — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — you first need a **neural audio codec**: a learned encoder that discretizes audio into a small-vocabulary token sequence, paired with a decoder that reconstructs the waveform.

Two schools emerged:

1. **Reconstruction-first codecs** — EnCodec, DAC. Optimize perceptual audio quality. Tokens are "acoustic" — they capture everything including speaker identity, timbre, background noise.
2. **Semantic-first codecs** — Mimi (Kyutai), SpeechTokenizer. Force the first codebook to encode linguistic / phonemic content (often by distilling from WavLM). Later codebooks are acoustic detail.

The 2024-2026 insight: **pure reconstruction codecs give you mushy speech when you try to generate from text.** An LM over codec tokens has to learn both linguistic structure and acoustic structure in the same codebook, and that doesn't scale. Splitting them — semantic codebook 0, acoustic codebooks 1-N — is what makes Moshi and Sesame CSM work.

## The Concept

![Four codec landscapes: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### The Core Trick: Residual Vector Quantization (RVQ)

Rather than one giant codebook (millions of codes needed for good quality), all modern audio codecs use **RVQ**: a cascade of small codebooks. The first codebook quantizes the encoder output; the second quantizes the residual; and so on. Each codebook has 1024 codes. 8 codebooks = effective vocabulary of 1024^8 = 10^24.

At inference, the decoder sums the selected codes from all codebooks per frame to reconstruct.

### The Four Codecs That Matter in 2026

**EnCodec (Meta, 2022).** The baseline. Encoder-decoder on waveform with RVQ bottleneck. 24 kHz, up to 32 codebooks, default 4 codebooks @ 1.5 kbps. Uses a `1D conv + transformer + 1D conv` architecture. MusicGen runs on it.

**DAC (Descript, 2023).** RVQ with L2-normalized codebooks, periodic activation functions, improved losses. Highest reconstruction fidelity of any open-source codec — sometimes indistinguishable from original speech at 12 codebooks. 44.1 kHz full-band.

**SNAC (Hubert Siuzdak, 2024).** Multi-scale RVQ — coarse codebooks operate at lower frame rates than fine codebooks. Effectively models audio hierarchically: a coarse "sketch" at ~12 Hz plus detail at 50 Hz. Orpheus-3B uses it because this hierarchical structure maps well to LM-based generation.

**Mimi (Kyutai, 2024).** The 2026 game-changer. 12.5 Hz frame rate (extremely low), 8 codebooks @ 4.4 kbps. Codebook 0 is **distilled from WavLM** — trained to predict WavLM's speech content features. Codebooks 1-7 are acoustic residuals. This split drives Moshi (Lesson 15) and Sesame CSM.

### Frame Rate Matters for Language Modeling

Lower frame rate = shorter sequences = faster LM.

| Codec | Frame Rate | 1 s = N frames | Good for |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | Music, general audio |
| DAC-44.1k | 86 Hz | 86 | Hi-fi music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficiency |
| Mimi | 12.5 Hz | 12.5 | Streaming speech |

At 12.5 Hz, 10 seconds of speech is only 125 codec frames — a transformer can predict them trivially.

### Semantic Tokens vs Acoustic Tokens

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token (codebook 0 in Mimi).** Encodes what was said — phonemes, words, content. Distilled from WavLM via an auxiliary prediction loss.
- **Acoustic tokens (codebooks 1-7).** Encode timbre, speaker identity, prosody, background noise, detail.

An AR LM predicts the semantic token first (conditioned on text), then the acoustic tokens (conditioned on semantic + speaker reference). This factorization is why modern TTS can zero-shot clone voices: the semantic model handles content; the acoustic model handles timbre.

### 2026 Reconstruction Quality (bits per second, lower bitrate = better)

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Traditional codecs like Opus still win on per-bit perceptual quality. Neural codecs win on **discrete tokens** (Opus doesn't produce them) and **generative model quality** (what an LM can do with those tokens).

## Build It

### Step 1: Encode with EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

At 6 kbps, `n_codebooks=8`. Each code is 0-1023 (10 bits).

### Step 2: Decode and Measure Reconstruction

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Step 3: Semantic-Acoustic Split (Mimi-style)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantic codebook 0 is WavLM-aligned. You can train a text-to-semantic transformer — much smaller vocabulary than going directly to audio. Then a separate acoustic-to-waveform decoder conditions on a speaker reference.

### Step 4: Why AR LMs Over Codec Tokens Work

For 10 seconds of speech at Mimi's 12.5 Hz × 8 codebooks:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 tokens is a trivial context for a transformer. A 256M-parameter transformer generates 10 seconds of speech in milliseconds on a modern GPU.

## Use It

Map your problem to a codec:

| Task | Codec |
|------|-------|
| General music generation | EnCodec-24k |
| Highest-fidelity reconstruction | DAC-44.1k |
| AR LM over speech (TTS) | SNAC or Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| Sound effect library with text | EnCodec + T5 conditioning |
| Fine-grained audio editing | DAC + local inpainting |

Rule of thumb: **Building a generative model? Start with Mimi or SNAC. Building a compression pipeline? Use Opus.**

## Pitfalls

- **Too many codebooks.** Adding codebooks improves fidelity linearly, but LM sequence length also grows linearly. Stop at 8-12.
- **Frame rate mismatch.** Training an LM on 12.5 Hz Mimi then fine-tuning on 50 Hz EnCodec silently fails.
- **Assuming all codebooks are equal.** In Mimi, codebook 0 carries content; losing it destroys intelligibility. Losing codebook 7 is barely noticeable.
- **Using only reconstruction quality as a metric.** A codec can reconstruct well yet be useless for LM-based generation if its semantic structure is poor.

## Ship It

Save as `outputs/skill-codec-picker.md`. Select a codec for a given generation or compression task.

## Exercises

1. **Easy.** Run `code/main.py`. It implements a toy scalar + residual quantizer, measuring reconstruction error as you add codebooks.
2. **Medium.** Install `encodec`, compare 1, 4, 8, and 32 codebooks on a held-out speech clip. Plot PESQ or MSE vs bitrate.
3. **Hard.** Load Mimi. Encode an audio clip. Replace codebook 0 with random integers; decode. Then do the same for codebook 7. Compare the two corruptions — codebook 0 corruption should destroy intelligibility; codebook 7 corruption should change almost nothing.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | A cascade of small codebooks; each quantizes the previous one's residual. |
| Frame rate | Codec speed | How many token frames per second. Lower = faster LM. |
| Semantic codebook | Codebook 0 (Mimi) | Distilled from SSL features; encodes content. |
| Acoustic codebooks | Everything else | Timbre, prosody, noise, detail. |
| PESQ / ViSQOL | Perceptual quality | Objective metrics correlated with MOS. |
| EnCodec | Meta codec | RVQ baseline; powers MusicGen. |
| Mimi | Kyutai codec | 12.5 Hz frame rate; semantic-acoustic split; powers Moshi. |

## Further Reading

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — The RVQ baseline.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — Highest-fidelity open-source.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — Multi-scale RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — Semantic-acoustic split, WavLM distillation.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — The two-stage semantic/acoustic paradigm.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — The first streamable RVQ codec.
