# Audio Evaluation — WER, MOS, UTMOS, MMAU, FAD, and the Open Leaderboards

> What you can't measure, you can't ship. This lesson names the 2026 metrics for every audio task: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, ASR round-trip WER), audio-language (MMAU, LongAudioBench), music (FAD, CLAP), speaker (EER). Plus the leaderboards you benchmark against.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## The Problem

Every audio task has multiple metrics, each measuring a different dimension. Use the wrong metric and you ship a model that looks great on the dashboard and falls apart in production. The 2026 checklist:

| Task | Primary Metric | Secondary Metrics |
|------|---------|-----------|
| ASR | WER | CER · RTFx · First-token latency |
| TTS | MOS / UTMOS | SECS · ASR round-trip WER · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · FAR / FRR at operating point |
| Speaker diarization | DER | JER · Speaker confusion |
| Audio classification | top-1 · mAP | Macro F1 · Per-class recall |
| Music generation | FAD | CLAP · Listening panel MOS |
| Audio-language models | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | Latency P50/P95 | WER · MOS |

## The Concept

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR Metrics

**WER (Word Error Rate).** `(S + D + I) / N`. Lowercase, strip punctuation, normalize numbers before scoring. Use `jiwer` or OpenAI's `whisper_normalizer`. < 5% = human-level on read speech.

**CER (Character Error Rate).** Same formula, character-level. Used for tonal languages with ambiguous word segmentation (Mandarin, Cantonese).

**RTFx (Inverse Real-Time Factor).** Audio seconds processed per wall-clock second. Higher is better. Parakeet-TDT achieves 3380×. Whisper-large-v3 is ~30×.

**First-token latency.** Wall-clock time from audio input to first transcription token. Critical for streaming. Deepgram Nova-3: ~150 ms.

### TTS Metrics

**MOS (Mean Opinion Score).** 1-5 human rating. Gold standard but slow. Collect 20+ listeners per sample, 100+ samples per model.

**UTMOS (2022-2026).** Learned MOS predictor. Correlates ~0.9 with human MOS on standard benchmarks. F5-TTS: UTMOS 3.95; ground truth: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** For voice cloning. ECAPA embedding cosine between reference and cloned output. > 0.75 = recognizable clone.

**ASR round-trip WER.** Run Whisper on TTS output, compute WER against input text. Catches intelligibility degradation. 2026 SOTA: < 2% CER.

**TTFA (Time to First Audio).** Wall-clock latency. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.

### Voice Cloning Specifics

**SECS + MOS + CER** triplet. High SECS but low MOS means timbre matches but sounds unnatural; the reverse means natural sound but wrong speaker.

### Speaker Verification

**EER (Equal Error Rate).** Threshold where false acceptance = false rejection. ECAPA on VoxCeleb1-O: 0.87%.

**minDCF (Minimum Detection Cost).** Weighted cost at a chosen operating point (commonly FAR=0.01). More production-relevant than EER.

### Speaker Diarization

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. Missed speech + false alarm speech + speaker confusion, each as a proportion. AMI meetings: DER ~10-20% is realistic. pyannote 3.1 + Precision-2 commercial: <10% DER with good recording.

**JER (Jaccard Error Rate).** DER alternative, more robust to short-segment bias.

### Audio Classification

Multi-label: **mAP (mean Average Precision)** across all classes. AudioSet: BEATs-iter3 is 0.548 mAP.

Multi-class exclusive: **top-1, top-5 accuracy**. Speech Commands v2: 99.0% top-1 (Audio-MAE).

Imbalanced: **macro F1** + **per-class recall**. Report by class — aggregate accuracy hides which classes fail.

### Music Generation

**FAD (Fréchet Audio Distance).** Distance between real and generated audio VGGish embedding distributions. MusicGen-small on MusicCaps: 4.5. MusicLM: 4.0. Lower is better.

**CLAP score.** Text-audio alignment score using CLAP embeddings. > 0.3 = reasonable alignment.

**Listening panel MOS.** For consumer-grade music, still the final verdict. Suno v5 at ELO 1293 on TTS Arena (from pairwise human preference).

### Audio-Language Benchmarks

**MMAU (Massive Multi-Audio Understanding).** 10k audio QA pairs.

**MMAU-Pro.** 1800 hard questions across four categories: speech / sound / music / multi-audio. Random chance on 4-way multiple choice = 25%. Gemini 2.5 Pro overall ~60%; all models ~22% on multi-audio.

**LongAudioBench.** Minutes-long clips with semantic queries. Audio Flamingo Next beats Gemini 2.5 Pro.

**AudioCaps / Clotho.** Captioning benchmarks. SPICE, CIDEr, FENSE metrics.

### Streaming Speech-to-Speech

**Latency P50 / P95 / P99.** Wall-clock time from user end-of-speech to first audible response. Moshi: 200 ms; GPT-4o Realtime: 300 ms.

**WER / MOS** on output.

**Barge-in responsiveness.** Time from user interruption to assistant silence. Target < 150 ms.

### 2026 Leaderboards

| Leaderboard | Track | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long audio | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, pairwise ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

## Build It

### Step 1: WER with Normalization

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Step 2: TTS Round-Trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Step 3: SECS for Voice Cloning

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Step 4: FAD for Music Generation

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Step 5: EER for Speaker Verification (Same Code as Lesson 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Use It

Every deployment gets a fixed evaluation test bed, run on every model update. Three iron rules:

1. **Normalize before scoring.** Lowercase, strip punctuation, expand numbers. Write down the normalization rules.
2. **Report distributions, not means.** Latency as P50/P95/P99. Classification as per-class recall. MMAU as per-category.
3. **Run a canonical public benchmark.** Even if your production data differs, reporting on Open ASR / TTS Arena / MMAU lets reviewers compare apples-to-apples.

## Pitfalls

- **UTMOS extrapolation.** Trained on VCTK-style clean speech; scores poorly on noisy / cloned / emotional audio.
- **MOS panel bias.** 20 Amazon Mechanical Turk workers ≠ 20 target users. When stakes are high, pay for a domain panel.
- **FAD depends on reference set.** Use the same reference distribution when comparing across models.
- **Aggregate WER.** Overall 5% WER can mask 30% WER on accented speech. Report by demographic slice.
- **Public benchmark saturation.** Most frontier models are near ceiling on standard benchmarks. Build an internal held-out set that reflects your traffic.

## Ship It

Save as `outputs/skill-audio-evaluator.md`. Select metrics, benchmarks, and reporting format for any audio model release.

## Exercises

1. **Easy.** Run `code/main.py`. Computes WER / CER / EER / SECS / FAD-like / MMAU-like on toy inputs.
2. **Medium.** Build a TTS round-trip WER test bed. Pass your Kokoro or F5-TTS output through Whisper. Compute WER on 50 prompts. Flag prompts with WER > 10%.
3. **Hard.** Score your Lesson 10 LALM on MMAU-Pro speech + multi-audio subsets (50 items each). Report per-category accuracy and compare against published numbers.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| WER | ASR score | Word-level `(S+D+I)/N` after normalization. |
| CER | Character-level WER | Used for tonal languages or character-level systems. |
| MOS | Human opinion | 1-5 rating; 20+ listeners × 100 samples. |
| UTMOS | ML MOS predictor | Learned model; ~0.9 correlation with human MOS. |
| SECS | Voice clone similarity | ECAPA cosine between reference and clone. |
| EER | Speaker verification score | Threshold where FAR = FRR. |
| DER | Diarization score | (FA + Miss + Confusion) / total. |
| FAD | Music generation quality | Fréchet distance on VGGish embeddings. |
| RTFx | Throughput | Audio seconds processed per wall-clock second. |

## Further Reading

- [jiwer](https://github.com/jitsi/jiwer) — WER/CER library with normalization tools.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — Learned MOS predictor.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — Music generation standard.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026 live rankings.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — Human-voted TTS leaderboard.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM reasoning leaderboard.
- [HEAR benchmark](https://hearbenchmark.com/) — Audio SSL benchmark.
