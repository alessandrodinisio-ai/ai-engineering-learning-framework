# GPT — Causal Language Modeling

> BERT looks both ways. GPT only looks back. That triangular mask is the most impactful single line of code in modern AI.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 min

## The Problem

A language model answers one question: given the first `t-1` tokens, what is the probability distribution over token `t`? Train on this signal — next-token prediction — and you get a model that can generate arbitrary text one token at a time.

To train it end-to-end in parallel over the full sequence, you need each position's prediction to depend only on earlier positions. Otherwise the model cheats by looking at the answer.

The causal mask does exactly this. It is an upper-triangular matrix of `-inf` values added to attention scores before softmax. After softmax, those positions become 0. Each position can only attend to itself and earlier positions. And because you apply it once over the full sequence, you get N parallel next-token predictions in one forward pass.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — all decoder-only causal transformers with the same core loop. Just bigger, better data, better RLHF.

## The Concept

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### The mask

Given a sequence of length `N`, build an `N × N` matrix:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Add `M` to raw attention scores before softmax. `exp(-inf) = 0`, so masked positions contribute zero weight. Each row of the attention matrix is a probability distribution over earlier positions only.

Implementation cost: one `torch.tril()` call. Compute time: nanoseconds. Impact on the entire field: everything.

### Parallel training, serial inference

Training: forward the full `(N, d_model)` sequence once, compute N cross-entropy losses (one per position), sum, backprop. Parallel along the sequence. This is why GPT training scales — you process a million tokens per batch in a single GPU pass.

Inference: you generate one token at a time. Feed `[t1, t2, t3]`, get `t4`. Feed `[t1, t2, t3, t4]`, get `t5`. Feed `[t1, t2, t3, t4, t5]`, get `t6`. KV cache (lesson 12) saves hidden states of `t1…tn` so you don't recompute each step. But serial depth at inference = output length. This is the autoregressive tax and why decoding is every LLM's latency bottleneck.

### The loss — shift by one

Given tokens `[t1, t2, t3, t4]`:

- Input: `[t1, t2, t3]`
- Target: `[t2, t3, t4]`

At each position `i`, compute `-log P(target_i | inputs[:i+1])`. Sum. This is the full-sequence cross-entropy.

Every transformer LM you've heard of trains on this loss. Pretraining, fine-tuning, SFT — same loss, different data.

### Decoding strategies

After training, sampling choices matter more than people think.

| Method | What it does | When to use |
|--------|--------------|-------------|
| Greedy | Take argmax each step | Deterministic tasks, code completion |
| Temperature | Divide logits by T before sampling | Creative tasks; higher T = more diversity |
| Top-k | Sample only from top-k tokens | Cuts low-probability long tail |
| Top-p (nucleus) | Sample from smallest set whose cumulative probability ≥ p | Post-2020 default; adapts to distribution shape |
| Min-p | Keep tokens with `p > min_p * max_p` | 2024+; better at rejecting long tail than top-p |
| Speculative decoding | Draft model proposes N tokens, big model verifies | 2–3× latency reduction at same quality |

In 2026, min-p + temperature 0.7 is a reasonable default for open-weight models. Speculative decoding is table stakes for any production inference stack.

### Why the "GPT recipe" works

1. **Decoder-only.** No encoder overhead. One attention + FFN per layer.
2. **Scaling.** 124M → 1.5B → 175B → trillions. Chinchilla scaling laws (lesson 13) tell you how to spend compute.
3. **In-context learning.** Emerges around 6B–13B. The model follows few-shot examples without fine-tuning.
4. **RLHF.** Post-training on human preferences turns raw pretrained text into a chat assistant.
5. **Pre-norm + RoPE + SwiGLU.** Stable training at scale.

The core architecture hasn't changed much since GPT-2. All the interesting stuff happens in data, scale, and post-training.

## Build It

### Step 1: Causal mask

See `code/main.py`. One line:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Add it to attention scores before softmax. That's the entire mechanism.

### Step 2: A 2-layer GPT-like model

Stack two decoder blocks (masked self-attention + FFN, no cross-attention). Add a token embedding, a positional encoding, and an unembedding (tied with the token embedding matrix — standard trick since GPT-2).

### Step 3: End-to-end next-token prediction

On a 20-token toy vocab, produce logits at each position. Compute cross-entropy loss against shifted-by-one targets. No gradients — this is a forward-pass sanity check.

### Step 4: Sampling

Implement greedy, temperature, top-k, top-p, min-p. Run each on a fixed prompt and compare outputs. A sampling function is 10 lines.

## Use It

PyTorch, the 2026 way:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Under the hood, `generate()` runs a forward pass, takes logits at the last position, samples the next token, appends, repeats. Every production LLM inference stack (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) implements the same loop with heavy optimization — batched prefill, continuous batching, paged KV cache, speculative decoding.

**GPT vs BERT, one line each:** GPT predicts `P(x_t | x_{<t})`. BERT predicts `P(x_masked | x_unmasked)`. The loss determines whether the model can generate.

## Ship It

See `outputs/skill-sampling-tuner.md`. This skill picks sampling parameters for a new generation task and flags when deterministic decoding is needed.

## Exercises

1. **Easy.** Run `code/main.py`, verify the causal attention matrix is lower-triangular after softmax. Spot-check: row 3 should have weights only in columns 0–3.
2. **Medium.** Implement beam search with width 4. Compare beam-4 vs greedy perplexity on 10 short prompts. Does beam always win? (Hint: usually wins for translation, not for open-ended chat.)
3. **Hard.** Implement speculative decoding: use a tiny 2-layer model as draft and a 6-layer model as verifier. Measure wall-clock speedup on 100 completions of length 64. Confirm output matches greedy from the verifier.

## Key Terms

| Term | How people talk about it | What it actually means |
|------|--------------------------|------------------------|
| Causal mask | "the triangle" | Upper-triangular `-inf` matrix added to attention scores so position `i` only sees positions `≤ i`. |
| Next-token prediction | "the loss" | Cross-entropy of model distribution against the true next token at each position. |
| Autoregressive | "one at a time" | Feed output back as input; parallel at training, serial at generation. |
| Logits | "scores before softmax" | Raw output of the LM head before softmax; sampling happens here. |
| Temperature | "creativity knob" | Divide logits by T; T→0 = greedy, T→∞ = uniform. |
| Top-p | "nucleus sampling" | Truncate distribution to smallest set summing ≥p; sample from the rest. |
| Min-p | "better than top-p" | Keep tokens with `p ≥ min_p × max_p`; adaptive cutoff by distribution sharpness. |
| Speculative decoding | "draft + verify" | Cheap model proposes N tokens; big model verifies in parallel. |
| Teacher forcing | "training trick" | Feed ground-truth previous token during training, not model's own prediction. Standard for every seq2seq LM. |

## Further Reading

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 and in-context learning.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — the speculative decoding paper.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — the canonical causal LM reference.
