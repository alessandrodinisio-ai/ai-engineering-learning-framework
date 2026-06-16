# Music Generation — MusicGen, Stable Audio, Suno, and the Copyright Earthquake

> Music generation in 2026: Suno v5 and Udio v4 dominate the commercial space; MusicGen, Stable Audio Open, and ACE-Step lead open-source. The technical problems are largely solved. The legal problems (Warner Music $500M settlement, Universal Music settlement) reshaped the entire field in 2025–2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## The Problem

Text → a 30-second to 4-minute piece of music with lyrics, vocals, and structure. Three sub-problems:

1. **Instrumental generation.** Text like "lo-fi hip-hop drums with warm keys, 128 BPM" → audio. MusicGen, Stable Audio, AudioLDM.
2. **Song generation (with vocals + lyrics).** "A country song about a rainy night in Texas" → complete song. Suno, Udio, YuE, ACE-Step.
3. **Conditional / controllable generation.** Continue an existing audio clip, regenerate a transition, change genre, separate stems, or inpaint. Udio's inpainting + stem separation is the 2026 feature to benchmark against.

## The Concept

![Music generation: token-LM vs diffusion, 2026 model map](../assets/music-generation.svg)

### Token LMs over Neural Codec Tokens

Meta's **MusicGen** (2023, MIT) and many derivatives: autoregressive prediction of EnCodec tokens (32 kHz, 4 codebooks), conditioned on text/melody embeddings, decoded by EnCodec. 300M – 3.3B parameters. Strong baseline; struggles beyond 30 seconds.

**ACE-Step** (open-source, 4B XL released April 2026) extends this to full-song lyrics-conditioned generation. The closest open-source thing to Suno.

### Diffusion over Mel or Latents

**Stable Audio (2023)** and **Stable Audio Open (2024)**: latent diffusion on compressed audio. Excels at loops, sound design, ambient textures. Not great at structured full songs.

**AudioLDM / AudioLDM2**: text-to-audio via T2I-style latent diffusion, generalizes to music, sound effects, speech.

### Hybrid (Production) — Suno, Udio, Lyria

Closed weights. Likely AR codec LM + diffusion-based vocoder with specialized vocal / drum / melody heads. Suno v5 (2026) is the quality leader at ELO 1293. Udio v4 adds inpainting + stem separation (bass, drums, vocals downloaded separately).

### Evaluation

- **FAD (Fréchet Audio Distance).** Using VGGish or PANNs features, measures distance between generated and real audio distributions at the embedding level. Lower is better. MusicGen small has FAD 4.5 on MusicCaps; SOTA is ~3.0.
- **Musicality (subjective).** Human preference. Suno v5 ELO 1293 leads.
- **Text-audio alignment.** CLAP score between prompt and output.
- **Musical artifacts.** Off-beat transitions, vocal phrase drift, loss of structure beyond 30 seconds.

## 2026 Model Map

| Model | Parameters | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | No | MIT |
| Stable Audio Open | 1.2B | 47 s | No | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | Yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | Yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | Yes, ELO 1293 | Commercial |
| Udio v4 (closed) | ? | 4 min | Yes + stems | Commercial |
| Google Lyria 3 (closed) | ? | Real-time | Yes | Commercial |
| MiniMax Music 2.5 | ? | 4 min | Yes | Commercial API |

## Legal Landscape (2025–2026)

- **Warner Music v. Suno settlement.** $500M. WMG now has oversight over AI voice likenesses, music copyright, and user-generated tracks on Suno. Similar Universal Music settlement on Udio.
- **EU AI Act** + **California SB 942**: AI-generated music must be disclosed.
- **Riffusion / MusicGen** under MIT have no compliance baggage, but also no commercial vocals.

Patterns safe to ship:

1. Generate instrumentals only (MusicGen, Stable Audio Open, MIT/CC0 outputs).
2. Use commercial APIs with per-generation licensing (Suno, Udio, ElevenLabs Music).
3. Train on owned or licensed catalogs (where most enterprises end up).
4. Watermark generated outputs + metadata tagging.

## Build It

### Step 1: Generate with MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Three sizes: `small` (300M, fast), `medium` (1.5B), `large` (3.3B). `small` is enough to judge "does this idea work."

### Step 2: Melody Conditioning

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody takes a chromagram, preserving the tune while changing timbre. Good for "turn this melody into a string quartet."

### Step 3: FAD Evaluation

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Computes VGGish embedding distance. Suitable for genre-level regression tests; no substitute for human listeners.

### Step 4: Plug into LLM-Music Workflow

Combine ideas from lessons 7–8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Use It

| Goal | Toolstack |
|------|-----------|
| Instrumental sound design | Stable Audio Open |
| Game / adaptive music | Google Lyria RealTime (closed) |
| Full song with vocals (commercial) | Suno v5 or Udio v4 with explicit licensing |
| Full song with vocals (open-source) | ACE-Step XL or YuE |
| Short ad jingles | MusicGen with melody conditioning from a hum reference |
| Music video background | MusicGen + Stable Video Diffusion |

## Pitfalls Still Shipping in 2026

- **Copyright-laundering prompts.** "A song in Taylor Swift's style" — commercial Suno/Udio now filter these, open-source models don't. Add your own filter list.
- **Repetition / drift beyond 30s.** AR models loop. Crossfade multiple generations, or use ACE-Step for structural coherence.
- **Tempo drift.** Models wander off BPM. Use BPM tags in prompts and post-filter with librosa's `beat_track`.
- **Vocal intelligibility.** Suno excels; open-source models often mumble lyrics. Use commercial APIs or fine-tune if lyrics matter.
- **Mono output.** Open-source models generate mono or fake stereo. Upmix with proper stereo reconstruction (ezst, Cartesia's stereo diffusion).

## Ship It

Save as `outputs/skill-music-designer.md`. Pick the model, licensing strategy, length/structure approach, and disclosure metadata for a music generation deployment.

## Exercises

1. **Easy.** Run `code/main.py`. It produces a "generative" chord progression + drum pattern in ASCII notation — a cartoon of music generation. Render with any MIDI renderer to hear it.
2. **Medium.** Install `audiocraft`, generate 10-second clips with MusicGen-small across 4 genre prompts, measure FAD against a set of genre references.
3. **Hard.** Using ACE-Step (or MusicGen-melody), generate three variations of the same melody with different timbre prompts. Compute CLAP similarity to the prompt to verify alignment.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| FAD | Audio FID | Fréchet distance between real and generated embedding distributions. |
| Chromagram | Melody as pitch | Per-frame 12-dimensional vector; input for melody conditioning. |
| Stems | Individual instrument tracks | Separated bass / drums / vocals / melody WAVs. |
| Inpainting | Regenerate a section | Mask a time window; the model regenerates only that section. |
| CLAP | Text-audio CLIP | Contrastive audio-text embeddings; evaluates text-audio alignment. |
| EnCodec | Music codec | Meta's neural codec used by MusicGen; 32 kHz, 4 codebooks. |

## Further Reading

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — open-source autoregressive baseline.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — the default for sound design.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — open-source 4B full-song generator, April 2026.
- [Suno v5 platform docs](https://suno.com) — commercial quality leader.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — latent diffusion for music + sound effects.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — the November 2025 precedent.
