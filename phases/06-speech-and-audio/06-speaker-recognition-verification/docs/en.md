# Speaker Recognition & Verification

> ASR asks "what did they say," speaker recognition asks "who said it." The math looks the same — embeddings plus cosine — but every production decision hinges on a single EER number.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## The Problem

A user speaks a passphrase. You want to know: is this the person they claim to be (*verification*, 1:1), or the first person in your enrolled gallery (*identification*, 1:N)? Or neither — is this an unknown speaker (*open-set*)?

Pre-2018: GMM-UBM + i-vectors. Decent EER but brittle to channel shift (phone vs laptop) and emotion. 2018–2022: x-vectors (TDNN backbone trained with angular margin). Post-2022: ECAPA-TDNN and WavLM-large embeddings. By 2026, the field is dominated by three models and one metric.

That metric is **EER** — Equal Error Rate. Set your decision threshold where false acceptance rate = false rejection rate; the crossover point is EER. Every paper, every leaderboard, every procurement review uses it.

## The Concept

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Pipeline.** Enrollment: record 5–30 seconds of the target speaker's voice; compute a fixed-dimension embedding (ECAPA-TDNN is 192-d, WavLM-large is 256-d). Verification: take the test utterance's embedding; compute cosine similarity; compare to a threshold.

**ECAPA-TDNN (2020, still dominant through 2026).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. 1D conv blocks with squeeze-excitation, multi-head attentive pooling, followed by a linear layer to 192 dimensions. Trained on VoxCeleb 1+2 (2,700 speakers, 1.1M utterances) with additive angular margin loss (AAM-softmax).

**WavLM-SV (post-2022).** A pre-trained WavLM-large SSL backbone fine-tuned with AAM loss. Higher quality but slower — 300+ MB vs 15 MB.

**x-vector (baseline).** TDNN + statistics pooling. Classic; still useful on CPU / edge.

**AAM-softmax.** Standard softmax with an angular margin `m` added to the correct class: `cos(θ + m)`. Forces angular separation between classes. Typical `m=0.2`, scale `s=30`.

### Scoring

- **Cosine** between enrollment and test embeddings. Threshold-based decision.
- **PLDA (Probabilistic LDA).** Projects embeddings into a latent space where "same speaker vs different speaker" has a closed-form likelihood ratio. Stacked on top of cosine drops EER by another 10–20%. Standard before 2020; now only used in closed-set scenarios.
- **Score normalization.** `S-norm` or `AS-norm`: normalize each score against the mean and standard deviation of a cohort of impostors. Essential for cross-domain evaluation.

### Numbers you should know (2026)

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× real-time |
| ECAPA-TDNN | 0.87% | 15 M | 200× real-time |
| WavLM-SV large | 0.42% | 316 M | 20× real-time |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× real-time |
| ReDimNet (2024) | 0.39% | 24 M | 100× real-time |

### Speaker Diarization

"Who spoke when" in multi-speaker audio. Pipeline: VAD → segmentation → embed each segment → clustering (hierarchical or spectral) → smooth boundaries. Modern toolkit: `pyannote.audio` 3.1, which packages speaker segmentation + embedding + clustering behind a single call. 2026 SOTA DER on AMI is ~15% (down from 23% in 2022).

## Build It

### Step 1: Toy embedding from MFCC statistics

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Nowhere near SOTA — pedagogical only. `code/main.py` uses it as a proof-of-concept on synthetic speaker data.

### Step 2: Cosine similarity + threshold

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Step 3: Compute EER from similarity pairs

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Returns (eer, threshold_at_eer). Report both.

### Step 4: Production with SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# Enrollment: average embeddings over 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# Verification
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # Typical threshold for ECAPA; tune on your own data
```

### Step 5: Speaker diarization with pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Use It

The 2026 toolkit:

| Scenario | Choose |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + cosine threshold |
| Open-set verification, cloud | WavLM-SV + AS-norm |
| Speaker diarization (meetings, podcasts) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing (replay / deepfake detection) | AASIST or RawNet2 |
| Tiny embedded (KWS + enrollment) | Titanet-Small (NeMo) |

## Pitfalls

- **Channel mismatch.** A model trained on VoxCeleb (web video) ≠ telephone audio. Always evaluate on the target channel.
- **Short utterances.** EER degrades sharply when test audio is under 3 seconds.
- **Noisy enrollment.** A single noisy enrollment poisons the anchor. Average over ≥3 clean samples.
- **Fixed threshold across conditions.** Always tune the threshold on a held-out dev set from the target domain.
- **Cosine on unnormalized embeddings.** L2-normalize first; otherwise magnitude dominates.

## Ship It

Save as `outputs/skill-speaker-verifier.md`. Select the model, enrollment protocol, threshold tuning strategy, and anti-spoofing protection.

## Exercises

1. **Easy.** Run `code/main.py`. It constructs synthetic "speakers" (different timbral profiles), enrolls them, and computes EER on a 100-pair trial list.
2. **Medium.** Use SpeechBrain ECAPA on 30 VoxCeleb1 utterances (5 speakers × 6 each). Compute EER with cosine vs PLDA.
3. **Hard.** Build a full enrollment → diarization → verification pipeline using `pyannote.audio`. Evaluate DER on the AMI dev set.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| EER | The headline metric | The threshold where false acceptance = false rejection. |
| Verification | 1:1 | "Is this Alice?" |
| Identification | 1:N | "Who is speaking?" |
| Open-set | Unknown possible | Test set may contain unenrolled speakers. |
| Enrollment | Registration | Computing a speaker's reference embedding. |
| AAM-softmax | The loss | Softmax with additive angular margin; forces cluster separation. |
| PLDA | Classic scoring | Probabilistic LDA; likelihood-ratio scoring on top of embeddings. |
| DER | Diarization metric | Diarization Error Rate — missed + false alarm + confusion. |

## Further Reading

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — The classic deep embedding paper.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — The dominant architecture from 2020–2026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — The SSL backbone for SV and diarization.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — Production-grade diarization + embedding toolkit.
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — Current EER rankings across models.
