# BERT — Masked Language Modeling

> GPT predicts the next word. BERT predicts the missing word. One sentence apart — and it powered everything embedding-related for the next five years.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representations)
**Time:** ~45 min

## The Problem

In 2018, every NLP task — sentiment, NER, QA, entailment — trained its own model from scratch on its own labeled data. No pretrained "understands English" checkpoint existed for fine-tuning. ELMo (2018) proved you could pretrain contextual embeddings with a bidirectional LSTM; it helped but didn't generalize.

BERT (Devlin et al. 2018) asked: what if we take a transformer encoder, train it on every sentence on the internet, and force it to predict missing words from both sides of context? Then you fine-tune a head on downstream tasks. Parameter efficiency was a revelation.

The result: within 18 months, BERT and variants (RoBERTa, ALBERT, ELECTRA) swept every NLP leaderboard that existed. By 2020, every search engine, content moderation pipeline, and semantic search system on Earth had a BERT inside.

In 2026, encoder-only models remain the right tool for classification, retrieval, and structured extraction — they run 5–10× faster per token than decoders, and their embeddings are the backbone of every modern retrieval stack. ModernBERT (Dec 2024) pushes the architecture to 8K context with Flash Attention + RoPE + GeGLU.

## The Concept

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### Training signal

Take a sentence: `the quick brown fox jumps over the lazy dog`.

Randomly mask 15% of tokens:

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

Train the model to predict original tokens at masked positions. Because the encoder is bidirectional, predicting the `[MASK]` at position 1 can use `brown fox jumps` at position 2 and beyond. This is exactly what GPT cannot do.

### BERT's masking rules

Of the 15% of tokens selected for prediction:

- 80% are replaced with `[MASK]`.
- 10% are replaced with a random token.
- 10% are left unchanged.

Why not always `[MASK]`? Because `[MASK]` never appears at inference. Training the model to expect `[MASK]` at 100% of predicted positions creates distribution shift between pretraining and fine-tuning. The 10% random + 10% unchanged keeps the model honest.

### Next Sentence Prediction (NSP) — and why it was killed

Original BERT also trained NSP: given two sentences A and B, predict whether B follows A. RoBERTa (2019) ablated it, proving NSP hurts rather than helps. Modern encoders skip it.

### What changed by 2026: ModernBERT

The 2024 ModernBERT paper rebuilds the block with 2026 primitives:

| Component | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Position | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

Unlike the 2018 stack, it is Flash-Attention native. At sequence length 8K, inference is 2–3× faster than DeBERTa-v3 with higher GLUE scores.

### Use cases that still pick encoder in 2026

| Task | Why encoder beats decoder |
|------|---------------------------|
| Retrieval / semantic search embeddings | Bidirectional context = better per-token embedding quality |
| Classification (sentiment, intent, toxicity) | Single forward pass; no generation overhead |
| NER / token tagging | Per-position output, natively bidirectional |
| Zero-shot entailment (NLI) | Classification head on top of encoder |
| Rerankers for RAG | Cross-encoder scoring, 10× faster than LLM rerankers |

## Build It

### Step 1: Masking logic

See `code/main.py`. The function `create_mlm_batch` takes a list of token IDs, vocab size, and mask probability. Returns input IDs (with masking applied) and labels (only at masked positions; -100 elsewhere — PyTorch's ignore index convention).

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### Step 2: Run MLM prediction on a tiny corpus

Train a 2-layer encoder + MLM head on a 20-word vocab, 200-sentence corpus. No gradients — we do a forward-pass sanity check. Full training requires PyTorch.

### Step 3: Compare masking types

Show how the three-way rule keeps the model usable without `[MASK]`. Predict on an unmasked sentence and a masked sentence. Both should produce reasonable token distributions because the model saw both modes during training.

### Step 4: Fine-tuning head

On a toy sentiment dataset, swap the MLM head for a classification head. Only the head trains; encoder is frozen. This is the paradigm every BERT application follows.

## Use It

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding models are fine-tuned BERTs.** `sentence-transformers` models like `all-MiniLM-L6-v2` are BERTs trained with contrastive loss. Same encoder, different loss.

**Cross-encoder rerankers are also fine-tuned BERTs.** Pairwise classification on `[CLS] query [SEP] doc [SEP]`. Bidirectional attention between query and doc is exactly why cross-encoders are higher quality than bi-encoders.

**When NOT to pick BERT in 2026.** Any generative task. Encoders have no reasonable way to produce tokens autoregressively. Also: any scenario under 1B params where a small decoder can match quality with more flexibility (Phi-3-Mini, Qwen2-1.5B).

## Ship It

See `outputs/skill-bert-finetuner.md`. This skill plans a BERT fine-tuning for a new classification or extraction task (backbone selection, head spec, data, evaluation, stopping criteria).

## Exercises

1. **Easy.** Run `code/main.py`, print the masking distribution over 10,000 tokens. Confirm ~15% are selected, and of those ~80% become `[MASK]`.
2. **Medium.** Implement whole-word masking: if a word is split into multiple subwords, either mask all subwords together or none. Test whether this improves MLM accuracy on a 500-sentence corpus.
3. **Hard.** Train a tiny (2-layer, d=64) BERT on 10,000 sentences from a public dataset. Fine-tune the `[CLS]` token for SST-2 sentiment. Compare against a decoder-only baseline at equal parameters — who wins?

## Key Terms

| Term | How people talk about it | What it actually means |
|------|--------------------------|------------------------|
| MLM | "masked language modeling" | Training signal: randomly replace 15% of tokens with `[MASK]`, predict originals. |
| Bidirectional | "sees both sides" | Encoder attention has no causal mask — every position sees every other position. |
| `[CLS]` | "the pooling token" | Special token prepended to every sequence; its final embedding serves as sentence-level representation. |
| `[SEP]` | "segment separator" | Separates paired sequences (e.g. query/doc, sentence A/B). |
| NSP | "next sentence prediction" | BERT's second pretraining task; RoBERTa proved it useless and killed it post-2019. |
| Fine-tuning | "adapt to task" | Encoder largely frozen; train a small head on top for downstream task. |
| Cross-encoder | "a reranker" | BERT that takes both query and doc as input and outputs a relevance score. |
| ModernBERT | "the 2024 refresh" | Encoder rebuilt with RoPE, RMSNorm, GeGLU, alternating local/global attention, 8K context. |

## Further Reading

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — the original paper.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — how to train BERT right; killed NSP.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — replaced token detection beats MLM at equal compute.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — the ModernBERT paper.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — the canonical encoder reference.
