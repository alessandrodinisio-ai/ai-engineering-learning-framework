# GloVe, FastText, and Subword Embeddings

> Word2Vec trains one embedding per word. GloVe factorizes the co-occurrence matrix directly. FastText embeds word parts. BPE bridges the gap to transformers.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 min

## The Problem

Word2Vec left two open questions.

First, a parallel research line had been factorizing co-occurrence matrices directly (LSA, HAL) rather than running online skip-gram updates. Is Word2Vec's iterative approach fundamentally better, or is the difference an artifact of how each method handles counts? **GloVe** answers this: matrix factorization with a carefully chosen loss matches or beats Word2Vec at lower training cost.

Second, neither method can say anything about a word it's never seen. `Zoomer-approved`, `dogecoin`, any proper noun coined last week, every inflected form of a rare root. **FastText** fixes this by embedding character n-grams: a word is the sum of its parts (including morphemes), so even OOV words get a reasonable vector.

Third, once transformers arrived the problem shifted again. Word-level vocabularies cap at roughly one million entries; real language is more open than that. **Byte Pair Encoding (BPE)** and its cousins solve this: learn a vocabulary of high-frequency subword units that covers everything. Every modern tokenizer for every modern LLM is a subword tokenizer.

This lesson walks through all three, then clarifies when to reach for which.

## The Concept

**GloVe (Global Vectors).** Build a word–word co-occurrence matrix `X` where `X[i][j]` is the frequency of word `j` appearing in word `i`'s context. Train vectors so that `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Weight the loss so high-frequency pairs don't dominate. Done.

**FastText.** A word is the sum of its character n-grams plus the word itself. `where` decomposes into `<wh, whe, her, ere, re>, <where>`. The word vector is the sum of these part vectors. Trained like Word2Vec. Benefit: an unseen word (`whereupon`) can be assembled from known n-grams.

**BPE (Byte Pair Encoding).** Start with a vocabulary of individual bytes (or characters). Count every adjacent token pair in the corpus. Merge the most frequent pair into a new token. Repeat `k` times. The result is a vocabulary of `k + 256` tokens where high-frequency sequences (`ing`, `tion`, `the`) are single tokens and rare words are split into familiar parts. Every sentence can be tokenized.

## Build It

### GloVe: Factorize the co-occurrence matrix

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Two moving parts worth naming. The weighting function `f(x) = (x/x_max)^alpha` down-weights very high-frequency pairs (like `(the, and)`) so they don't dominate the loss. The final embedding is the sum of the `W` (center) and `W_tilde` (context) tables. Summing both is a published trick that usually outperforms using either alone.

### FastText: Subword-aware embeddings

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Each word is represented by its set of n-grams (typically 3 to 6 characters). The word embedding is the sum of its n-gram embeddings. During skip-gram training, this replaces the single vector Word2Vec uses.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

For an unseen word, as long as some of its n-grams are known, you still get a vector. `whereupon` shares `<wh`, `her`, `ere`, `<where` with `where`, so the two land close together.

### BPE: A learned subword vocabulary

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

The first merge joins the most common adjacent pair. After enough merges, high-frequency substrings (`low`, `est`, `tion`) become single tokens and rare words split cleanly.

Real GPT / BERT / T5 tokenizers learn 30k–100k merges. The result: any text becomes a bounded-length sequence of known IDs, with no OOV ever.

## Use It

In practice you rarely train these yourself — you load pretrained checkpoints.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

BPE-style subword tokenization in the transformer era:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

The `Ġ` prefix marks word boundaries (GPT-2 convention). Every modern tokenizer is either a BPE variant, WordPiece (BERT), or SentencePiece (T5, LLaMA).

### Which to choose

| Scenario | Choice |
|-----------|------|
| Pretrained general word vectors, no need to handle OOV | GloVe 300d |
| Pretrained general word vectors, must handle typos / neologisms / morphologically rich languages | FastText |
| Anything going into a transformer (training or inference) | The model's own tokenizer. Never swap it. |
| Training your own language model from scratch | Train a BPE or SentencePiece tokenizer on your corpus first |
| Production text classification with linear models | Still TF-IDF. See lesson 02. |

## Ship It

Save as `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Exercises

1. **Easy.** Run `char_ngrams("playing")` and `char_ngrams("played")`. Compute the Jaccard overlap of the two n-gram sets. You should see substantial shared parts (`pla`, `lay`, `play`) — this is why FastText transfers well across morphological variants.
2. **Medium.** Extend `learn_bpe` to track vocabulary growth. Plot "tokens per corpus character" as a function of merge count. You should see rapid compression initially, then asymptotic approach to ~2–3 characters per token.
3. **Hard.** Train a 1k-merge BPE on the complete works of Shakespeare. Compare tokenization results for common words vs rare proper nouns. Measure the average tokens per word before and after. Document what surprises you.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Co-occurrence matrix | Word–word frequency table | `X[i][j]` = frequency of word `j` appearing in word `i`'s context window. |
| Subword | A piece of a word | A character n-gram (FastText) or learned token (BPE/WordPiece/SentencePiece). |
| BPE | Byte Pair Encoding | Iteratively merge the highest-frequency adjacent pair until the vocabulary reaches target size. |
| OOV | Out of vocabulary | A word the model has never seen. Word2Vec/GloVe fail; FastText and BPE handle it. |
| Byte-level BPE | BPE on raw bytes | GPT-2's approach. Vocabulary starts from 256 bytes, so OOV is impossible. |

## Further Reading

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — The GloVe paper, seven pages, still the best loss derivation.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — The paper that brought BPE into modern NLP.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — How BPE, WordPiece, and SentencePiece differ in practice.
