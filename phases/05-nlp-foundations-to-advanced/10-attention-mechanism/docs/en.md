# The Attention Mechanism — The Breakthrough

> The decoder stopped squinting at a compressed summary and started looking at the entire source sentence. Everything after this is attention plus engineering.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45 min

## The Problem

Lesson 09 ended on a quantified failure. A GRU encoder-decoder trained on a toy copy task hit 89% accuracy at length 5 and dropped to near-random at length 80. The cause is structural, not a training bug: every bit of information the encoder gathers must squeeze into a fixed-length hidden state, and the decoder never sees anything else.

Bahdanau, Cho, and Bengio published a three-line fix in 2014. Instead of giving the decoder only the final encoder state, keep every encoder state around. At each decoder step, compute a weighted average of encoder states where the weights say "how much should the decoder look at encoder position `i` right now?" That weighted average is the context, and it changes at every decoder step.

That's the entire idea. The transformer scaled it. Self-attention applied it within a single sequence. Multi-head attention ran it in parallel. But the 2014 version already broke the bottleneck, and once you have it, moving to transformers is engineering, not conceptual.

## The Concept

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

At each decoder step `t`:

1. Use the previous decoder hidden state `s_{t-1}` as the **query**.
2. Score it against every encoder hidden state `h_1, ..., h_T`. One scalar per encoder position.
3. Softmax the scores to get attention weights `α_{t,1}, ..., α_{t,T}` summing to 1.
4. Context vector `c_t = Σ α_{t,i} * h_i`. Weighted average of encoder states.
5. The decoder takes `c_t` plus an output token and produces the next token.

The weighted average is the key. When the decoder needs to translate "Je" into "I", it assigns high weight to the encoder state over "Je" and low weight elsewhere. When it needs "not", it gives "pas" high weight. The context vector reshapes at every step.

## Shapes (the thing that bites everyone)

This is where every attention implementation goes wrong the first time. Read slowly.

| Thing | Shape | Notes |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | If BiLSTM, `d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | A single vector |
| Attention score `e_{t,i}` | scalar | One per encoder position |
| Attention weight `α_{t,i}` | scalar | After softmax over all `i` |
| Context vector `c_t` | `(d_h,)` | Same shape as one encoder state |

**Bahdanau (additive) scoring.** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` shape `(d_s,)`, `h_i` shape `(d_h,)`.
- `W_a` shape `(d_attn, d_s)`. `U_a` shape `(d_attn, d_h)`.
- Their sum inside tanh has shape `(d_attn,)`.
- `v_α` shape `(d_attn,)`. Dot product with `v_α` collapses to a scalar. **That's what `v_α` does.** It's not magic — it's the projection that turns an attention-dimension vector into a scalar score.

**Luong (multiplicative) scoring.** Three variants:

- `dot`: `e_{t,i} = s_t^T * h_i`. Requires `d_s == d_h`. Hard constraint. Skip if your encoder is bidirectional.
- `general`: `e_{t,i} = s_t^T * W * h_i`, `W` shape `(d_s, d_h)`. Removes the equal-dimension constraint.
- `concat`: Essentially the Bahdanau form. Rarely used because the first two are cheaper.

**One Bahdanau/Luong gotcha worth naming.** Bahdanau uses `s_{t-1}` (the decoder state *before* generating the current word). Luong uses `s_t` (the state *after*). Mixing them produces subtly wrong gradients that are extremely hard to debug. Pick one paper and stick to its convention.

## Build It

### Step 1: Additive (Bahdanau) attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Cross-check your shapes against the table above. `encoder_states` is `(T_enc, d_h)`. `projected_enc` is `(T_enc, d_attn)`. `projected_dec` is `(d_attn,)` and broadcasts. `combined` is `(T_enc, d_attn)`. `scores` is `(T_enc,)`. `weights` is `(T_enc,)`. `context` is `(d_h,)`. Done.

### Step 2: Luong dot and general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

Three lines each. That's why the Luong paper holds up. Same accuracy on most tasks, much less code.

### Step 3: A worked numerical example

Given three encoder states (roughly corresponding to "cat", "sat", "mat") and a decoder state most aligned with the first, the attention distribution concentrates on position 0. If the decoder state shifts to align with the last, attention moves to position 2. The context vector follows.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

First row wins. Shift the decoder state closer to the third encoder state and watch the weights transfer. That's it. Attention is explicit alignment.

### Step 4: Why this is the bridge to transformers

Translate the above into Q/K/V terms:

- **Query** = decoder state `s_{t-1}`
- **Key** = encoder states (what we score against)
- **Value** = encoder states (what we weighted-sum over)

In classical attention, key and value are the same thing. Self-attention separates them: you can have a sequence query itself with different learned projections for K and V. Multi-head attention runs it in parallel with different learned projections. The transformer stacks this many layers deep and drops the RNN.

The math is the same. The shapes are the same. The pedagogical leap from Bahdanau attention to scaled dot-product attention is mostly notational.

## Use It

PyTorch and TensorFlow provide attention directly.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

That's a transformer attention layer. The query batch is 5 positions, key/value batch is 10 positions, each 128-dimensional, 8 heads. `output` is the new context-enriched query. `weights` is the 5×10 alignment matrix you can visualize.

### When classical attention still matters

- Teaching. The single-head, single-layer, RNN-based version makes every concept visible.
- On-device sequence tasks where a transformer won't fit.
- Any 2014–2017 paper. You'll misread it without knowing the Bahdanau convention.
- Fine-grained alignment analysis in machine translation. Raw attention weights remain an interpretability tool even on transformer models, and reading them requires understanding what they are.

### The trap of using attention weights as explanations

Attention weights look interpretable. They're weights across positions that sum to one; you can plot them; high means "looked at this." Reviewers love them.

They're less interpretable than they look. Jain and Wallace (2019) showed that for some tasks, attention distributions can be permuted or replaced by arbitrary alternatives without changing model predictions. Never present attention weights as evidence of reasoning without an ablation or counterfactual check.

## Ship It

Save as `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Exercises

1. **Easy.** Implement a `softmax` mask so padding tokens in the encoder get zero attention weight. Test on a batch of variable-length sequences.
2. **Medium.** Add multi-head attention to the Luong `general` form. Split `d_h` into `n_heads` groups, run attention per head, then concatenate. Verify the single-head case matches your earlier implementation.
3. **Hard.** Train a GRU encoder-decoder with Bahdanau attention on the lesson 09 toy copy task. Plot accuracy vs. sequence length. Compare against the no-attention baseline. You should see the gap widen with length, confirming attention lifts the bottleneck.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Attention | Looking at things | Weighted average of a value sequence, weights determined by query-key similarity. |
| Query, Key, Value | QKV | Three projections: Q asks, K is what gets matched, V is what gets returned. |
| Additive attention | Bahdanau | Feed-forward scoring: `v^T tanh(W q + U k)`. |
| Multiplicative attention | Luong dot / general | Score is `q^T k` or `q^T W k`. Cheaper, same accuracy on most tasks. |
| Alignment matrix | The pretty picture | Attention weights as a `(T_dec, T_enc)` grid. Read it to see what the model focuses on. |

## Further Reading

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — The paper.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — Three scoring variants and their comparison.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — The interpretability caveat.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — Runnable walkthrough with PyTorch.
