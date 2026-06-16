# Sequence-to-Sequence Models

> Two RNNs pretending to be translators. The bottleneck they hit is the reason attention exists.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (Intro to PyTorch)
**Time:** ~75 min

## The Problem

Classification maps a variable-length sequence to a single label. Translation maps a variable-length sequence to another variable-length sequence. Input and output live in different vocabularies, possibly different languages, with no guarantee of equal length.

The seq2seq architecture (Sutskever, Vinyals, Le, 2014) cracked this with a deliberately simple recipe. Two RNNs. One reads the source sentence and produces a fixed-length context vector. The other reads that vector and generates the target sentence token by token. Same code you wrote in lesson 08, just wired differently.

Worth studying for two reasons. First, the context vector bottleneck is the most pedagogically valuable failure in NLP. It explains everything attention and transformers excel at. Second, the training recipe (teacher forcing, scheduled sampling, beam search at inference) still applies to every modern generative system, including LLMs.

## The Concept

**Encoder.** An RNN that reads the source sentence. Its final hidden state is the **context vector** — a fixed-length summary of the entire input. Supposed to lose nothing except the source sentence itself.

**Decoder.** Another RNN initialized from the context vector. At each step it takes the previous generated token as input and produces a distribution over the target vocabulary. Sample or argmax to pick the next token. Feed it back. Repeat until an `<EOS>` token is produced or max length is reached.

**Training:** Cross-entropy loss at each decoder step, summed over the sequence. Standard backpropagation through time on both networks.

**Teacher forcing.** During training, the decoder's input at step `t` is the *ground-truth* token at position `t-1`, not the decoder's own previous prediction. This stabilizes training; without it, early errors cascade and the model never learns. At inference you can only use the model's own predictions, so there's always a distribution gap between training and inference. This gap is called **exposure bias**.

**The bottleneck.** Everything the encoder learns about the source sentence must squeeze into that one context vector. Long sentences lose detail. Rare words get smeared. Reordering (chat noir vs. black cat) can only be memorized, not computed.

Attention (lesson 10) fixes this by letting the decoder see *every* encoder hidden state, not just the last one. That's the entire pitch in one sentence.

## Build It

### Step 1: An encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` has shape `[batch, seq_len, hidden_dim]` — one hidden state per input position. `hidden` has shape `[1, batch, hidden_dim]` — the final step. Lesson 08 said "pool over outputs for classification." Here we keep the final hidden state as the context vector and ignore the per-step outputs.

### Step 2: A decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

The decoder is called one step at a time. Input: a batch of single tokens and the current hidden state. Output: vocabulary logits for the next token and the updated hidden state.

### Step 3: Training loop with teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Two knobs worth naming. `ignore_index=0` skips loss on padding tokens. `teacher_forcing_ratio` is the probability of using the ground-truth token vs. the model's prediction at each step. Start at 1.0 (full teacher forcing) and anneal to ~0.5 during training to close the exposure bias gap.

### Step 4: Inference loop (greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding picks the highest-probability token at each step. It drifts: once you commit to a token, you can't take it back. **Beam search** keeps the top-`k` partial sequences alive and picks the highest-scoring complete sequence at the end. Beam width 3–5 is standard.

### Step 5: Demonstrating the bottleneck

Train the model on a toy copy task: source `[a, b, c, d, e]`, target `[a, b, c, d, e]`. Increase sequence length and observe accuracy.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

A single GRU hidden state can't losslessly remember a 40-token input. The information is there at every encoder step, but the decoder only sees the final state. Attention fixes this directly.

## Use It

PyTorch has seq2seq templates built on `nn.Transformer` and `nn.LSTM`. Hugging Face's `transformers` library provides complete encoder-decoder models trained on billions of tokens (BART, T5, mBART, NLLB).

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Modern encoder-decoders replace RNNs with transformers. The high-level shape (encoder, decoder, token-by-token generation) is identical to the 2014 seq2seq paper. The internal mechanism of each block changed.

### When to reach for RNN-based seq2seq

For new projects, almost never. Specific exceptions:

- Streaming translation where you consume input one token at a time with bounded memory.
- On-device text generation where transformer memory costs are unaffordable.
- Teaching. Understanding the encoder-decoder bottleneck is the fastest path to understanding why transformers won.

### Exposure bias and its mitigations

- **Scheduled sampling.** Anneal the teacher forcing ratio during training so the model learns to recover from its own mistakes.
- **Minimum risk training.** Train on sentence-level BLEU scores instead of token-level cross-entropy. Closer to what you actually want.
- **RL fine-tuning.** Reward a sequence generator with a metric. Used in modern LLM RLHF.

All three still apply to transformer-based generation.

## Ship It

Save as `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Exercises

1. **Easy.** Implement the toy copy task. Train a GRU seq2seq on input-output pairs where target equals source. Measure accuracy at lengths 5, 10, and 20. Reproduce the bottleneck.
2. **Medium.** Add beam search decoding with beam width 3. Measure BLEU on a small parallel corpus and compare against greedy. Document where beam search wins (usually end-of-sequence tokens) and where it makes no difference.
3. **Hard.** Fine-tune `facebook/bart-base` on a 10k-pair paraphrase dataset. Compare the fine-tuned model's beam-4 output against the base model on held-out inputs. Report BLEU and pick 10 qualitative examples.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Encoder | Input RNN | Reads the source sentence. Produces per-step hidden states and a final context vector. |
| Decoder | Output RNN | Initialized from the context vector. Generates one target token at a time. |
| Context vector | The summary | Encoder's final hidden state. Fixed-length. The bottleneck attention solves. |
| Teacher forcing | Use ground-truth tokens | Feed the real previous token during training. Stabilizes learning. |
| Exposure bias | Train/test gap | A model trained on ground-truth tokens never practices recovering from its own mistakes. |
| Beam search | Better decoding | Keep top-k partial sequences alive at each step instead of greedily committing. |

## Further Reading

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — The original seq2seq paper. Four pages.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Introduced the GRU and the encoder-decoder framework.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — The attention paper. Read it right after this lesson.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — Buildable seq2seq + attention code.
