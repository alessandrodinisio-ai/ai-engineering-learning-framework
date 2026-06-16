# Text-to-Speech (TTS) — From Tacotron to F5 and Kokoro

> ASR inverts speech into text; TTS inverts text into speech. The 2026 toolstack has three stages: text → tokens, tokens → mel, mel → waveform. Each stage has a default model that fits in a notebook.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## The Problem

You have a string: "Please remind me to water the plants at 6 pm." You need a natural-sounding 3-second audio clip with correct prosody (pauses, stress), the right vowel in "plants," and it must run in under 300 ms on CPU for a real-time voice assistant. You also need to swap voices, handle mixed-language input ("remind me at 6 pm, daijoubu?"), and not butcher proper nouns.

The modern TTS pipeline looks like this:

1. **Text frontend.** Normalizes text (dates, numbers, emails), converts to phonemes or subword tokens, predicts prosody features.
2. **Acoustic model.** Text → mel spectrogram. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Vocoder.** Mel → waveform. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), neural codec vocoders post-2024.

By 2026, the boundary between acoustic model and vocoder is blurring with end-to-end diffusion and flow-matching models. But this three-stage mental model still holds when debugging.

## The Concept

![Side-by-side comparison of Tacotron, FastSpeech, VITS, F5/Kokoro](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: character embeddings → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder outputting mel frames. Slow (autoregressive), drifts on long text. Still cited as a baseline.

**FastSpeech 2 (2020).** Non-autoregressive. A duration predictor outputs how many mel frames each phoneme gets. Single pass, 10× faster than Tacotron. Loses some naturalness (monotonic alignment), but deployed everywhere.

**VITS (2021).** Uses variational inference to jointly train encoder + flow-based duration + HiFi-GAN vocoder end-to-end. High quality, single model. Dominated open-source TTS 2022–2024. Variants: YourTTS (multi-speaker zero-shot), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Diffusion transformer on top of flow matching. Natural prosody, zero-shot voice cloning from a 5-second reference. #1 on open-source TTS leaderboards in 2026. 335M parameters.

**Kokoro (2024).** Small (82M), runs on CPU, best-in-class English TTS for real-time scenarios. Closed vocabulary, English-only, Apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** Commercial state-of-the-art. ElevenLabs v2.5's emotion tags ("[whispered]", "[laughing]") and character voices dominate audiobook production in 2026.

### Vocoder Evolution

| Era | Vocoder | Latency | Quality |
|-----|---------|---------|---------|
| 2016 | WaveNet | Offline only | SOTA at release |
| 2018 | WaveRNN | ~Real-time | Decent |
| 2020 | HiFi-GAN | 100× real-time | Near-human |
| 2022 | BigVGAN | 50× real-time | Cross-speaker/language generalization |
| 2024 | SNAC, DAC (neural codecs) | Integrated with AR models | Discrete tokens, bit-efficient |

By 2026, most "TTS" models are end-to-end text-to-waveform; the mel spectrogram is just an internal representation.

### Evaluation

- **MOS (Mean Opinion Score).** 1–5, crowdsourced. Still the gold standard; painfully slow.
- **CMOS (Comparative MOS).** A-vs-B preference. Tighter confidence intervals per annotation.
- **UTMOS, DNSMOS.** Reference-free neural MOS predictors. Used for leaderboards.
- **CER via ASR (Character Error Rate).** Pass TTS output through Whisper, compute CER against input text. Proxy for intelligibility.
- **SECS (Speaker Embedding Cosine Similarity).** Voice cloning quality.

2026 numbers on LibriTTS test-clean:

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## Build It

### Step 1: Convert Input to Phonemes

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phonemes are the universal bridge. Anything below VITS-level quality — don't feed raw text directly.

### Step 2: Run Kokoro (2026 CPU Default)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Runs offline, single file, 82M parameters.

### Step 3: Voice Cloning with F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Pass a 5-second reference audio + its transcript; F5 clones the prosody and timbre.

### Step 4: Build a HiFi-GAN Vocoder from Scratch

Too large to fit in a single tutorial script, but the shape is:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, 256× total, from mel rate to audio rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Training: adversarial (discriminator on short windows) + mel spectrogram reconstruction loss + feature matching loss. Commoditized — use the `hifi-gan` repo or nvidia-NeMo pretrained checkpoints.

### Step 5: Full Pipeline (Pseudocode)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Use It

2026 toolstack:

| Scenario | Pick |
|-----------|------|
| Real-time English voice assistant | Kokoro (CPU) or XTTS v2 (GPU) |
| Voice cloning from 5s reference | F5-TTS |
| Commercial character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 or XTTS v2 + fine-tune |
| Low-resource language | Train VITS on 5–20 hours of target language data |
| Expressiveness / emotion tags | ElevenLabs v2.5 or StyleTTS 2 fine-tune |

Open-source leaders as of 2026: **F5-TTS for quality, Kokoro for efficiency**. Don't touch Tacotron unless you're a historian.

## Pitfalls

- **No text normalizer.** "Dr. Smith" — pronounced "Doctor" or "Drive"? "2026" — "twenty twenty six" or "two zero two six"? Normalize *before* phonemization.
- **OOV proper nouns.** "Ghumare" → "ghyu-mair"? Have a fallback grapheme-to-phoneme model for unknown tokens.
- **Clipping.** Vocoder output rarely clips, but mel scaling mismatch at inference can push past ±1.0. Always `np.clip(wav, -1, 1)`.
- **Sample rate mismatch.** Kokoro outputs 24 kHz; your downstream pipeline expects 16 kHz → resample, or you get aliasing.

## Ship It

Save as `outputs/skill-tts-designer.md`. Design a TTS pipeline for a given voice, latency, and language target.

## Exercises

1. **Easy.** Run `code/main.py`. It builds a phoneme dictionary from a toy vocabulary, estimates per-phoneme durations, and prints a fake "mel" schedule.
2. **Medium.** Install Kokoro, synthesize the same sentence with voices `af_bella` and `am_adam`. Compare audio duration and subjective quality.
3. **Hard.** Record a 5-second reference of your own voice. Clone it with F5-TTS. Report the SECS between reference and cloned output.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Phoneme | Sound unit | An abstract sound category; English has 39 (ARPABet). |
| Duration predictor | How long each phoneme lasts | Output of non-autoregressive models; integer frame count per phoneme. |
| Vocoder | Mel → waveform | Neural network mapping mel spectrograms to raw samples. |
| HiFi-GAN | The standard vocoder | GAN-based; dominated 2020–2024. |
| MOS | Subjective quality | Mean opinion score, 1–5, rated by human evaluators. |
| SECS | Voice cloning metric | Cosine similarity between target and output speaker embeddings. |
| F5-TTS | 2024 open-source SOTA | Flow-matching diffusion; zero-shot cloning. |
| Kokoro | CPU English leader | 82M parameter model, Apache 2.0. |

## Further Reading

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — the seq2seq baseline.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end, flow-based.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — current open-source SOTA.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — the vocoder still in production in 2026.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 CPU-friendly English TTS.
