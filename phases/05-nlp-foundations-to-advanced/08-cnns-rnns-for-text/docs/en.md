# CNNs and RNNs for Text

> Convolutions learn n-grams, recurrence remembers. Both were replaced by attention, but both still matter on constrained hardware.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (Intro to PyTorch), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 min

## The Problem

TF-IDF and Word2Vec produce flat vectors that ignore word order. A classifier built on them can't distinguish `dog bites man` from `man bites dog`. And word order is sometimes the signal.

Before transformers, two architecture families filled this gap.

**TextCNN (Convolutional networks for text).** Apply 1D convolutions over a sequence of word embeddings. A filter of width 3 is a learnable trigram detector: it slides over three words and outputs a score. Stack different widths (2, 3, 4, 5) to detect multi-scale patterns. Max-pool to get a fixed-length representation. Flat, parallel, fast.

**RNNs (RNN, LSTM, GRU).** Process one token at a time, maintaining a hidden state that carries information forward. Sequential, memory-equipped, flexible on input length. Dominated sequence modeling from 2014 to 2017, then attention arrived.

This lesson builds both and highlights the failure that gave rise to attention.

## The Concept

**TextCNN** (Kim, 2014). Tokens are embedded. A 1D convolution of width `k` slides a filter over consecutive `k`-gram embeddings, producing a feature map. Global max-pooling over that map picks the strongest activation. Concatenate max-pooled outputs from several filter widths and feed to a classification head.

Why it works. A single filter is a learnable n-gram. Max-pooling is position-invariant, so "not good" activates the same feature whether it's at the start or middle of a review. Three filter widths × 100 filters each gives you 300 learned n-gram detectors. Training is parallel with no sequential dependency.

**RNN.** At each timestep `t`, hidden state `h_t = f(W * x_t + U * h_{t-1} + b)`. `W`, `U`, `b` are shared across time. The hidden state at time `T` summarizes the entire prefix. For classification, pool over `h_1 ... h_T` (max, mean, or take the last).

Vanilla RNNs suffer from vanishing gradients. **LSTM** adds gates that decide what to forget, store, and output, stabilizing gradients across long sequences. **GRU** simplifies LSTM to two gates; fewer parameters with comparable performance.

**Bidirectional RNNs** run one RNN forward and another backward, concatenating hidden states. Each token's representation sees both left and right context. Essential for tagging tasks.

## Build It

### Step 1: TextCNN in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` reshapes `[batch, seq_len, embed_dim]` to `[batch, embed_dim, seq_len]` because `nn.Conv1d` treats the middle dimension as channels. Pooled output is fixed-length regardless of input length.

### Step 2: LSTM classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool over the sequence rather than taking the last state. For classification, max-pooling usually beats taking the final hidden state because information at the end of long sequences tends to dominate the last state.

### Step 3: Vanishing gradient demo (intuition)

A vanilla RNN without gating can't learn long-range dependencies. Consider a toy task: predict whether token `A` appeared anywhere in the sequence. If `A` is at position 1 and the sequence is 100 tokens long, the gradient of the loss must flow back through 99 multiplications by the recurrent weight. If the weight is less than 1, the gradient vanishes. If greater than 1, it explodes.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# With weight = 0.9 across 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# Gradient from step 100 to step 1 is effectively zero.
```

LSTM fixes this with a **cell state** that passes through the network with only additive interactions (the forget gate multiplicatively scales it, but gradients still flow along this "highway"). GRU does something similar with fewer parameters. Both let you train stably on sequences of 100+ steps.

### Step 4: Why this still isn't enough

Even with LSTMs, three problems remain.

1. **Sequential bottleneck.** Training an RNN on a 1000-token sequence requires 1000 serial forward/backward steps. No parallelism across time.
2. **Fixed-length context vector in encoder-decoder setups.** The decoder only sees the encoder's final hidden state, which compresses the entire input. Long inputs lose detail. Lesson 09 addresses this directly.
3. **Accuracy ceiling on long-range dependencies.** LSTMs beat vanilla RNNs, but propagating specific information across 200+ steps is still unreliable.

Attention solves all three. The transformer drops recurrence entirely. Lesson 10 is that turning point.

## Use It

PyTorch's `nn.LSTM`, `nn.GRU`, and `nn.Conv1d` are all production-ready. Training code is standard boilerplate.

Hugging Face provides pretrained embeddings you can wire as an input layer:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Checklist for when the constraint fits.

- **Edge/on-device inference.** A TextCNN with GloVe embeddings is 10–100× smaller than a transformer. If your deploy target is a phone, use this.
- **Streaming/online classification.** RNNs process one token at a time; transformers need the full sequence. For text arriving in real time, LSTMs still win.
- **Small-model baselines.** Iterate quickly on a new task. Training a TextCNN takes 5 minutes on CPU.
- **Sequence labeling with limited data.** For 1k–10k labeled sentences, BiLSTM-CRF (lesson 06) is still a production-grade NER architecture.

Everything else goes to transformers.

## Ship It

Save as `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Exercises

1. **Easy.** Train a TextCNN on a 3-class toy dataset (invent the data yourself). Verify that filter widths (2, 3, 4) beat a single width (3) on average F1.
2. **Medium.** Implement max, mean, and last-state pooling for the LSTM classifier. Compare on a small dataset; document which pooling wins and hypothesize why.
3. **Hard.** Build a BiLSTM-CRF NER tagger (combining lesson 06 and this one). Train on CoNLL-2003. Compare against the lesson 06 pure CRF baseline and a BERT fine-tune. Report training time, memory, and F1.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| TextCNN | CNN for text | 1D convolutions over word embeddings + global max-pooling. Kim (2014). |
| RNN | Recurrent network | Updates a hidden state at each timestep: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | Gated RNN | Adds input/forget/output gates + a cell state. Stabilizes training on long sequences. |
| GRU | Simpler LSTM | Two gates instead of three. Comparable accuracy, fewer parameters. |
| Bidirectional | Both directions | Forward + backward RNN concatenated. Each token sees both sides of context. |
| Vanishing gradient | Training signal dies | Repeated multiplication by weights < 1 in vanilla RNNs makes early-step gradients effectively zero. |

## Further Reading

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — The TextCNN paper. Eight pages, readable.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — The LSTM paper. Surprisingly clear.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — The diagrams that made LSTMs accessible to everyone.
