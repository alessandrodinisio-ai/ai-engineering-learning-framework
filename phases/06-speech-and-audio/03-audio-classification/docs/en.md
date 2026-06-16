# Audio Classification — From k-NN on MFCC to AST and BEATs

> From "dog bark or siren" to "what language is this," it's all audio classification. The features are mel. The architecture changes every decade. Evaluation is always AUC, F1, and per-class recall.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNN), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## The Problem

You have a 10-second audio clip and want to know "what is this": urban sounds (siren, drill, dog bark), speech commands (yes/no/stop), language identification (en/es/ar), speaker emotion (angry/neutral), or ambient sound (indoor/outdoor, crowd noise). These are all *audio classification*, and by 2026 the baseline architecture is mature: log-mel → CNN or Transformer → softmax.

The core difficulty isn't the network — it's the data. Audio datasets have brutal class imbalance, strong domain shift (clean vs noisy), and label noise (who defines the boundary between "urban noise" and "restaurant noise"?). This problem is 80% data curation, augmentation, and evaluation, not swapping CNN for Transformer.

## The Concept

![Audio classification ladder: k-NN on MFCC to AST to BEATs](../assets/audio-classification.svg)

**k-NN on MFCC (1990s baseline).** Flatten each clip's MFCCs, compute cosine similarity against a labeled bank, return majority vote of top K. Surprisingly strong on clean small datasets (Speech Commands, ESC-50). Runs without a GPU.

**2D CNN on log-mel (2015-2019).** Treat the `(T, n_mels)` log-mel as an image, apply a ResNet-18 or VGG-style network. Global average pooling over the time axis, softmax over classes. Still the baseline in most 2026 kaggle competitions.

**Audio Spectrogram Transformer, AST (2021-2024).** Slice log-mel into patches (e.g., 16×16), add positional embeddings, feed to a ViT. Supervised SOTA on AudioSet at the time (mAP 0.485).

**BEATs and WavLM-base (2024-2026).** Self-supervised pre-training on millions of hours. Fine-tune on your task with 1-10% of the supervised data you'd otherwise need. In 2026 this is the default starting point for non-speech audio. BEATs-iter3 beats AST by 1-2 mAP on AudioSet with 1/4 the compute.

**Whisper encoder as frozen backbone (2024).** Take Whisper's encoder, drop the decoder, attach a linear classifier. Near-SOTA on language identification and simple event classification with zero audio augmentation. A "free lunch" baseline.

### Class imbalance is the real challenge

ESC-50: 50 classes, 40 clips each — balanced, easy. UrbanSound8K: 10 classes, 10:1 imbalance. AudioSet: 632 classes, 100,000:1 long tail. Techniques that work:

- Balanced sampling during training (not during evaluation).
- Mixup: linearly interpolate two audio clips (and their labels) as augmentation.
- SpecAugment: randomly mask time and frequency bands. Simple; critical.

### Evaluation

- Multi-class exclusive (Speech Commands): top-1 accuracy, top-5 accuracy.
- Multi-class multi-label (AudioSet, UrbanSound-like): mean Average Precision (mAP).
- Severe imbalance: per-class recall + macro F1.

2026 numbers you should know:

| Benchmark | Baseline | 2026 SOTA | Source |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

## Build It

### Step 1: Featurization

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Step 2: Fixed-length summary

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Simple but powerful: mean + variance over time gives a 26-dimensional fixed-length embedding for 13-coefficient MFCCs. Runs instantly. Until 2017 it could beat the then-SOTA neural baseline on ESC-50.

### Step 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Step 4: Upgrade to CNN on log-mel

With PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M parameters. Trains on ESC-50 in ~10 minutes on a single RTX 4090 to 80%+ accuracy.

### Step 5: The 2026 default — fine-tune BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

For BEATs, use `microsoft/BEATs-base` via the `beats` library; the transformers API shape is the same.

## Use It

The 2026 toolkit:

| Scenario | Start here |
|-----------|-----------|
| Very small dataset (<1000 clips) | k-NN on MFCC means (your baseline) + audio augmentation |
| Medium dataset (1K–100K) | Fine-tune BEATs or AST |
| Large dataset (>100K) | Train from scratch, or fine-tune Whisper encoder |
| Real-time, edge | 40-MFCC CNN, quantized to int8 (KWS-style) |
| Multi-label (AudioSet) | BEATs-iter3 + BCE loss + mixup + SpecAugment |
| Language identification | MMS-LID, SpeechBrain VoxLingua107 baseline |

Decision rule: **Start from a frozen backbone, not a fresh model.** Fine-tuning a BEATs head gets you 95% of SOTA in hours, not weeks.

## Ship It

Save as `outputs/skill-classifier-designer.md`. For a given audio classification task, select the architecture, augmentation strategy, class balancing approach, and evaluation metrics.

## Exercises

1. **Easy.** Run `code/main.py`. It trains a k-NN MFCC baseline on a 4-class synthetic dataset (pure tones at different pitches). Report the confusion matrix.
2. **Medium.** Replace `summarize` with [mean, variance, skewness, kurtosis]. On the same synthetic dataset, does 4th-moment pooling beat mean+variance?
3. **Hard.** Train a 2D CNN on ESC-50 fold 1 using `torchaudio`. Report 5-fold cross-validation accuracy. Add SpecAugment (time masking = 20, frequency masking = 10) and report the difference.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|-----------------|-----------------------|
| AudioSet | The ImageNet of audio | Google's 2M clip, 632-class weakly-labeled YouTube dataset. |
| ESC-50 | Small classification benchmark | 50 classes × 40 clips of environmental sounds. |
| AST | Audio Spectrogram Transformer | ViT on log-mel patches; 2021 SOTA. |
| BEATs | Self-supervised audio | Microsoft's model; iter3 leads AudioSet in 2026. |
| Mixup | Pairwise augmentation | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Masking-based augmentation | Zero out random time and frequency bands of the spectrogram. |
| mAP | The multi-label metric | Mean Average Precision across classes and thresholds. |

## Further Reading

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — The benchmark architecture for 2021–2024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — The default choice post-2024.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — The dominant audio augmentation method.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — The enduring 50-class benchmark.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632-class YouTube ontology; still the gold standard.
