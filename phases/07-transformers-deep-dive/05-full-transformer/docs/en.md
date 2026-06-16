# The Full Transformer — Encoder + Decoder

> Attention is the star. Everything else — residuals, normalization, feed-forward, cross-attention — is scaffolding that lets you stack it deep.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 min

## The Problem

A single attention layer is a feature extractor, not a model. One matmul per layer isn't enough capacity to feed language. You need depth — and without proper plumbing, depth collapses.

The 2017 Vaswani paper packaged six design decisions that turn an attention layer into a stackable block. Every transformer since — encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) — inherits the same skeleton. By 2026 the blocks are polished (RMSNorm, SwiGLU, pre-norm, RoPE), but the skeleton is identical.

This lesson covers the skeleton. Later lessons specialize it — 06 for encoder, 07 for decoder, 08 for encoder-decoder.

## The Concept

![Encoder and decoder block internal wiring](../assets/full-transformer.svg)

### Six components

1. **Embedding + position signal.** token → vector. Position injected via RoPE (modern) or sinusoidal (classic).
2. **Self-attention.** Every position attends to every other position. Masked in the decoder.
3. **Feed-forward network (FFN).** Per-position two-layer MLP: `W_2 · activation(W_1 · x)`. Default expansion ratio 4×.
4. **Residual connections.** `x + sublayer(x)`. Without them, gradients vanish past ~6 layers.
5. **Layer normalization.** `LayerNorm` or `RMSNorm` (modern). Stabilizes the residual stream.
6. **Cross-attention (decoder only).** Queries come from the decoder; keys and values from the encoder output.

Watch a vector flow through one block: attention mixes across positions, residuals carry it forward, FFN transforms, normalization keeps the residual stream stable.

```figure
transformer-block
```

### Encoder block (used by BERT, T5 encoder)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

The encoder is bidirectional. No mask. Every position sees every other position.

### Decoder block (used by GPT, T5 decoder)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

The decoder has three sublayers per block. The middle one — cross-attention — is the only place information flows from encoder to decoder. In decoder-only architectures (GPT), cross-attention is dropped; you get masked self-attention + FFN only.

### Pre-norm vs post-norm

Original paper: `x + sublayer(LN(x))` vs `LN(x + sublayer(x))`. Post-norm fell out of favor around 2019 — hard to train deep without careful warmup. Pre-norm (`LN` *before* the sublayer) is the 2026 default: Llama, Qwen, GPT-3+, Mistral all use it.

### The 2026 modernized block

Vaswani 2017 shipped LayerNorm + ReLU. The modern stack swaps both. A production block actually looks like:

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices; total params align) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm drops LayerNorm's mean centering (one fewer subtraction), saves compute, and is empirically at least as stable. SwiGLU (`Swish(W1 x) ⊙ W3 x`) consistently beats ReLU/GELU FFN by ~0.5 ppl points in Llama, PaLM, and Qwen papers.

### Parameter count

For a block with `d_model = d` and FFN expansion `r`:

- MHA: `4 · d²` (Q, K, V, O projections)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Normalization: negligible

At `d = 4096, r = 2.6, layers = 32` (roughly Llama 3 8B): total `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B` (plus embeddings and output head). Matches published figures.

## Build It

### Step 1: Building blocks

Using the mini `Matrix` class from lesson 03 (copied here for standalone use):

- `layer_norm(x, eps=1e-5)` — subtract mean, divide by std.
- `rms_norm(x, eps=1e-6)` — divide by RMS. No mean subtraction.
- `gelu(x)` and `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` and `decoder_block(x, enc_out, params)`.

Full wiring in `code/main.py`.

### Step 2: Wire a 2-layer encoder and 2-layer decoder

Stack them. Pass encoder output into each decoder's cross-attention. Add a final LN before the output projection.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Step 3: Run forward on a toy example

Feed a 6-token source and a 5-token target. Verify output shape is `(5, vocab)`. No training — this lesson cares about architecture, not loss.

### Step 4: Swap in RMSNorm + SwiGLU

Replace LayerNorm and ReLU-FFN with RMSNorm and SwiGLU. Confirm shapes still align. This is the 2026 modernization — a single function swap.

## Use It

PyTorch/TF reference implementations: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. But most production code in 2026 writes its own block because:

- Flash Attention is called inside attention, not through `nn.MultiheadAttention`.
- GQA / MLA are not in the standard library reference.
- RoPE, RMSNorm, SwiGLU are not PyTorch defaults.

HF `transformers` has clean reference blocks worth reading: `modeling_llama.py` is the canonical 2026 decoder-only block. ~500 lines, worth a full read.

**Encoder vs decoder vs encoder-decoder — when to pick which:**

| Need | Choice | Examples |
|------|--------|----------|
| Classification, embeddings, text QA | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Decoder-only won language because it scales most cleanly while handling both understanding and generation. Encoder-decoder still wins when the input has a clear "source sequence" identity (translation, speech recognition, structured tasks).

## Ship It

See `outputs/skill-transformer-block-reviewer.md`. This skill reviews a new transformer block implementation against 2026 defaults, flagging missing components (pre-norm, RoPE, RMSNorm, GQA, FFN expansion ratio).

## Exercises

1. **Easy.** Count your encoder_block's parameters at `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Verify by implementing the block and using `sum(p.numel() for p in block.parameters())`.
2. **Medium.** Switch from post-norm to pre-norm. Initialize both, stack 12 layers on random input, and measure activation norms. Post-norm activations should explode; pre-norm should stay bounded.
3. **Hard.** Implement a 4-layer encoder-decoder on a toy reverse-copy task (copy `x` in reverse). Train for 100 steps. Report loss. Swap in RMSNorm + SwiGLU + RoPE — does loss improve?

## Key Terms

| Term | How people talk about it | What it actually means |
|------|--------------------------|------------------------|
| Block | "a transformer layer" | A stack of normalization + attention + normalization + FFN, wrapped with residual connections. |
| Residual | "skip connection" | `x + f(x)` output; lets gradients flow through deep stacks. |
| Pre-norm | "normalize before, not after" | Modern practice: `x + sublayer(LN(x))`. Trains deeper without warmup gymnastics. |
| RMSNorm | "LayerNorm without the mean" | Divides by RMS; one fewer op, same empirical stability. |
| SwiGLU | "the FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Beats ReLU/GELU on LM ppl. |
| Cross-attention | "how the decoder reads the encoder" | MHA where Q comes from decoder, K/V from encoder output. |
| FFN expansion | "how wide the middle MLP is" | Hidden-size-to-d_model ratio, typically 4 (LayerNorm) or 2.6 (SwiGLU). |
| No bias | "drop the +b term" | Modern stacks omit bias in linear layers; slight ppl improvement, smaller model. |

## Further Reading

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original block spec.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — why pre-norm beats post-norm at depth.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — the SwiGLU paper.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — the canonical 2026 decoder-only block.
