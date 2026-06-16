# Voice Anti-Spoofing & Audio Watermarking — ASVspoof 5, AudioSeal, WaveVerify

> Voice cloning runs faster than defense. Production voice systems in 2026 need two things: a detector (AASIST, RawNet2) to classify real vs synthetic speech, and a watermark (AudioSeal) that survives compression and editing. Ship both or don't ship voice cloning.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## The Problem

Three related defenses:

1. **Anti-spoofing / deepfake detection.** Given audio, is it synthetic or genuine? The ASVspoof benchmark (ASVspoof 2019 → 2021 → 5) is the gold standard.
2. **Audio watermarking.** Embed an imperceptible signal in generated audio that a detector can later extract. AudioSeal (Meta) and WavMark are the open-source options.
3. **Certifiable provenance.** Cryptographically sign audio files + metadata. C2PA / Content Authenticity Initiative.

Detection addresses uncooperative adversaries. Watermarking addresses compliance — AI-generated audio should be identifiable as such. Both are required in 2026.

## The Concept

![Anti-spoofing vs watermarking vs provenance — three layers of defense](../assets/spoofing-watermark.svg)

### ASVspoof 5 — The 2024-2025 Benchmark

Major changes from prior editions:

- **Crowdsourced data** (not studio-clean recordings) — real-world conditions.
- **~2000 speakers** (previously ~100).
- **32 attack algorithms.** TTS + voice conversion + adversarial perturbations.
- **Two tracks.** Countermeasures (CM) detecting independently; spoofing-aware ASV (SASV) for biometric systems.

SOTA on ASVspoof 5: ~7.23% EER. On the older ASVspoof 2019 LA: 0.42% EER. Real-world deployment: expect 5-10% EER on in-the-wild audio.

### AASIST and RawNet2 — The Detector Family

**AASIST** (2021, updated through 2026). Graph attention on spectral features. Current SOTA on the ASVspoof 5 CM task.

**RawNet2.** Convolutional front-end on raw waveform + TDNN backbone. Simpler baseline; still competitive after fine-tuning.

**NeXt-TDNN + SSL features.** 2025 variant: ECAPA-style + WavLM features + focal loss. Achieves 0.42% EER on ASVspoof 2019 LA.

### AudioSeal — The 2024 Watermarking Default

Meta's **AudioSeal** (January 2024, v0.2 December 2024). Key design:

- **Localization.** Detects the watermark frame-by-frame at 16 kHz sample resolution (1/16000 s).
- **Generator + detector jointly trained.** Generator learns to embed an inaudible signal; detector learns to find it through various augmentations.
- **Robust.** Survives MP3 / AAC compression, EQ, speed changes ±10%, noise mixing at +10 dB SNR.
- **Fast.** Detector runs at 485× real-time; 1000× faster than WavMark.
- **Capacity.** 16-bit payload (can encode model ID, generation timestamp, user ID), embeddable in each utterance.

### WavMark

Pre-AudioSeal open-source baseline. Invertible neural network, 32 bits/second. Problems:

- Synchronization by brute-force search, slow.
- Can be removed by Gaussian noise or MP3 compression.
- Not suitable for real-time.

### WaveVerify (July 2025)

Targets AudioSeal's weaknesses — particularly temporal manipulation (reversal, speed changes). Uses a FiLM-based generator + mixture-of-experts detector. Matches AudioSeal on standard attacks; handles temporal edits.

### The Gap Adversaries Exploit

From AudioMarkBench: "Under pitch-shifting, all watermarks' bit recovery accuracy drops below 0.6, meaning near-complete removal." **Pitch-shifting is the universal attack.** No watermark in 2026 is fully robust to aggressive pitch manipulation. This is why you need detection (AASIST) on top of watermarking.

### C2PA / Content Authenticity Initiative

Not ML — it's a manifest format. Audio files carry cryptographically signed metadata about the creation tool, author, and date. Audiobox / Seamless use it. Good for provenance; powerless once a bad actor re-encodes and strips the metadata.

## Build It

### Step 1: A Simple Spectral Feature Detector (Toy)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Synthetic speech often has abnormally flat high-frequency energy. Production detectors use AASIST, not this. But the intuition holds.

### Step 2: AudioSeal Embedding + Detection

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Step 3: Evaluation — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Step 4: Production Integration

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Every generation ships with: (1) watermark, (2) signed manifest, (3) audit log under retention policy.

## Use It

| Use Case | Defense |
|----------|---------|
| Shipping TTS / voice cloning | Embed AudioSeal on every output (non-negotiable) |
| Biometric voice unlock | AASIST + ECAPA ensemble; liveness challenge |
| Call center fraud detection | Run AASIST on a 20% sample of incoming calls |
| Podcast authenticity | C2PA sign on upload, AudioSeal on AI-generated segments |
| Research / training detectors | ASVspoof 5 train/dev/eval partitions |

## Pitfalls

- **Watermarking without running the detector.** Pointless. Put the detector in your CI.
- **Uncalibrated detection.** AASIST trained on ASVspoof LA overfits; real-world accuracy drops. Calibrate on your domain.
- **The pitch-shift gap.** Aggressive pitch-shifting removes most watermarks. Have detection as a fallback.
- **Stripping metadata and re-hosting.** C2PA is trivially bypassed by re-encoding. Always pair cryptographic + perceptual (watermark) defenses.
- **Treating liveness as detection.** Asking users to read a random phrase stops replay attacks, not real-time cloning.

## Ship It

Save as `outputs/skill-spoof-defender.md`. Select the detection model, watermark, provenance manifest, and runbook for a voice generation deployment.

## Exercises

1. **Easy.** Run `code/main.py`. Toy detector + toy watermark embedding/detection on synthetic audio.
2. **Medium.** Install `audioseal`, embed a 16-bit payload in a TTS output, then decode. Corrupt the audio with noise and measure bit recovery accuracy.
3. **Hard.** Fine-tune a RawNet2 or AASIST on ASVspoof 2019 LA. Measure EER. Test on a held-out set of F5-TTS generations — observe how out-of-distribution detection degrades.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| ASVspoof | The benchmark | Biennial challenge; 2024 = ASVspoof 5. |
| CM (Countermeasure) | Detector | Classifier: genuine vs synthetic / converted speech. |
| SASV | Speaker verification + CM | Integrated biometric + spoof detection. |
| AudioSeal | Meta watermark | Localized, 16-bit payload, 485× faster than WavMark. |
| Bit recovery accuracy | Watermark survival | Fraction of payload bits recovered after attack. |
| C2PA | Provenance manifest | Cryptographic metadata about creation / authorship. |
| AASIST | Detector family | Graph-attention-based anti-spoofing SOTA. |

## Further Reading

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — The current benchmark.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — Watermarking default.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — MoE detector for temporal attacks.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA detection backbone.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — Robustness evaluation.
- [C2PA specification](https://c2pa.org/specifications/specifications/) — Provenance manifest format.
