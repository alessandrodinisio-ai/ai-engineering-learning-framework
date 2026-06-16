# Audio Generation

> Audio is a 1D signal at 16–48 kHz. A five-second clip is 80,000 to 240,000 samples. No transformer can attend to a sequence that long directly. In 2026 every production audio model uses the same solution: a neural codec (Encodec, SoundStream, DAC) compresses audio into discrete tokens at 50–75 Hz, then a transformer or diffusion model generates those tokens.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## The Problem

Three audio generation tasks:

1. **Text-to-speech.** Given text, produce speech. Clean speech is narrowband with strong phonetic structure—transformer-over-tokens solves it well. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Music generation.** Given a prompt (text, melody, chord progression, genre), produce music. Much wider distribution. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **Sound effects / sound design.** Given a prompt, produce ambient sound or foley. AudioGen, AudioLDM 2, Stable Audio Open.

All three run on the same foundation: neural audio codec + token autoregressive or diffusion generator.

## The Concept

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### Neural Audio Codecs

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). A convolutional encoder compresses the waveform into per-timestep vectors; residual vector quantization (RVQ) converts each vector into a sequence of K codebook indices. The decoder reverses it. 24 kHz audio at 2 kbps uses 8 RVQ codebooks at 75 Hz = 600 tokens per second.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Two Generation Paradigms on Top

**Token autoregressive.** Flatten RVQ tokens into a single sequence, run a decoder-only transformer. MusicGen uses "delay parallel" to emit K codebook streams in parallel with per-stream offsets. VALL-E generates speech tokens from a text prompt + 3-second voice sample.

**Latent diffusion.** Pack codec tokens into continuous latents, or model them with categorical diffusion. Stable Audio 2.5 uses flow matching on continuous audio latents. AudioLDM 2 uses text-to-mel-to-audio diffusion.

2024–2026 trend: flow matching wins on music (faster inference, cleaner samples), while token autoregressive still dominates speech because it is naturally causal and streams well.

## Production Landscape

| System | Task | Backbone | Latency |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms to first token |
| OpenAI GPT-4o audio | Full-duplex voice | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | ~10s for 1-min clip |
| Suno v4 | Full songs | Undisclosed; likely token-AR | ~30s per song |
| Udio v1.5 | Full songs | Undisclosed | ~30s per song |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s for 5s clip |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

## Build It

`code/main.py` simulates the core idea: train a mini next-token transformer on synthetic "audio token" sequences from two different "styles" (style A alternates low-high tokens, style B ramps monotonically). Sample conditioned on style.

### Step 1: Synthetic Audio Tokens

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Step 2: Train a Mini Token Predictor

A style-conditioned bigram-style predictor. The point is the pattern: codec tokens → cross-entropy training → autoregressive sampling.

### Step 3: Conditional Sampling

Given a style token and a start token, sample the next token from the predicted distribution. Continue for 20–40 tokens.

## Pitfalls

- **Codec quality caps output quality.** If the codec cannot faithfully represent a sound, no amount of generator quality can save it. DAC is current open-source best.
- **RVQ error accumulates.** Each RVQ layer models the residual of the previous one. Errors in layer 1 propagate. Sampling with temperature 0 at higher layers helps.
- **Musical structure.** 30 seconds of tokens at 75 Hz is 2,000+ tokens. Hard for transformers. MusicGen uses sliding window + prompt continuation; Stable Audio uses shorter clips + crossfade.
- **Boundary artifacts.** Crossfading between generated clips requires careful overlap-add.
- **Appetite for clean data.** Music generators need tens of thousands of hours of licensed music. Suno/Udio's RIAA lawsuits (2024) surfaced this.
- **Voice cloning ethics.** A 3-second sample plus a text prompt is enough for VALL-E / XTTS / ElevenLabs to clone a voice. Every production model needs abuse detection + opt-out lists.

## Use It

| Task | 2026 Stack |
|------|------------|
| Commercial TTS | ElevenLabs, OpenAI TTS, or Azure Neural |
| Voice cloning (verified consent) | XTTS v2 (open-source) or ElevenLabs Pro |
| Background music, fast | Stable Audio 2.5 API, Suno, or Udio |
| Music with lyrics | Suno v4 or Udio v1.5 |
| Sound effects / foley | AudioCraft 2, ElevenLabs SFX, or Stable Audio Open |
| Real-time voice agent | GPT-4o realtime or Gemini Live |
| Open-weight music research | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Dubbing / translation | HeyGen, ElevenLabs Dubbing |

## Ship It

Save as `outputs/skill-audio-brief.md`. The skill accepts an audio requirement (task, duration, style, voice, licensing), outputs: model + hosting, prompt format (genre tags, style descriptors, structure markers), codec + generator + vocoder chain, seed workflow, and evaluation plan (MOS / CLAP score / CER for TTS / user A/B).

## Exercises

1. **Easy.** Run `code/main.py` and explicitly set the style. Verify the generated sequence matches that style's pattern.
2. **Medium.** Add delay-parallel decoding: simulate two token streams that must stay offset by 1 step. Train a joint predictor.
3. **Hard.** Run MusicGen-small locally with HuggingFace transformers. Generate 10-second clips with three different prompts; A/B for style adherence.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Codec | "neural compression" | Audio encoder/decoder; typical output is tokens at 50–75 Hz. |
| RVQ | "residual VQ" | Cascade of K quantizers; each models the residual of the previous one. |
| Token | "a codec symbol" | Discrete index into a codebook; typically 1024 or 2048. |
| Delay parallel | "offset codebooks" | Emit K token streams at staggered offsets to shorten sequence length. |
| Flow matching | "2024's audio winner" | Straighter-path alternative to diffusion; faster sampling. |
| Voice prompt | "3-second sample" | Speaker embedding or token prefix that steers cloned voice. |
| Mel spectrogram | "that visual" | Log-magnitude perceptual spectrogram; used by many TTS systems. |
| Vocoder | "mel to waveform" | Neural component that converts mel spectrograms back to audio. |

## Production Notes: Audio Is a Streaming Problem

Audio is the only output modality where users expect delivery *as it generates* rather than all at once. In production terms, this means TPOT (time per output token) is critical because the user's listening rate is the target throughput—not their reading speed. For 16kHz audio tokenized at ~75 tokens/second (Encodec), the server must produce ≥75 tokens/second per user to keep playback smooth.

Two architectural consequences:

- **Flow-matching audio models cannot easily stream.** Stable Audio 2.5 and AudioCraft 2 render a fixed-length clip in one pass. To stream, you chunk clips and overlap boundaries—think sliding-window diffusion—adding 100–300ms latency overhead versus codec AR models.

If the product is "real-time voice chat" or "live music continuation," pick the codec AR path. If it is "submit and render a 30-second clip," flow matching wins on quality and total latency.

## Further Reading

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — the codec standard.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — first widely used neural audio codec.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — flow matching for text-to-music in 2025.
