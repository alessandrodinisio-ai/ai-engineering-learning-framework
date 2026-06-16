# Word Embeddings — Word2Vec from Scratch

> A word is defined by the company it keeps. Feed that idea into a shallow network, and geometric structure emerges on its own.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 min

## The Problem

TF-IDF knows `dog` and `puppy` are different words but doesn't know they mean nearly the same thing. A classifier trained on `dog` can't generalize to a review about `puppy`. You can hack around it with synonym lists, but that breaks on rare words, domain jargon, and any language you didn't anticipate.

What you want is a representation where `dog` and `puppy` land close together in space; `king - man + woman` lands near `queen`; and a model trained on `dog` transfers some signal to `puppy` for free.

Word2Vec gives us this space. A two-layer neural network, trained on trillion-token scale, published in 2013. The architecture is almost embarrassingly simple, yet the results reshaped NLP for a full decade.

## The Concept

**The distributional hypothesis** (Firth, 1957): "You shall know a word by the company it keeps." If two words appear in similar contexts, they likely have similar meanings.

Word2Vec has two flavors, both consuming this idea.

- **Skip-gram.** Given a center word, predict surrounding words. With window size 2: `cat -> (the, sat, on)`.
- **CBOW (Continuous Bag of Words).** Given surrounding words, predict the center word. `(the, sat, on) -> cat`.

Skip-gram trains slower but handles rare words better. It became the default.

The network has one hidden layer, no nonlinearity. Input is a one-hot vector over the vocabulary, output is a softmax over the vocabulary. After training, discard the output layer. The hidden-layer weights are the embeddings.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

The trick: computing softmax over 100k words is prohibitively expensive. Word2Vec uses **negative sampling** to turn it into a binary classification task — predict "does this context word appear near this center word, yes or no." Each training pair samples a small number of negative (non-co-occurring) words instead of normalizing over the full vocabulary.

## Build It

### Step 1: Generate training pairs from a corpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Every (center, context) pair within the window is a positive training sample.

### Step 2: Embedding tables

Two matrices. `W` is the center-word embedding table (the one you keep). `W'` is the context-word table (usually discarded, sometimes averaged with `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Small random initialization. Vocabulary 10k, dimension 100 is near-realistic; for teaching, 50 words × 16 dimensions is enough to see geometric structure.

### Step 3: Negative sampling objective

For each positive pair `(center, context)`, sample `k` random words from the vocabulary as negatives. Train the model so the dot product `W[center] · W'[context]` is high for positives and low for negatives.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

The magic formula: logistic loss on the positive pair (push sigmoid toward 1) plus logistic loss on negative pairs (push sigmoid toward 0). Gradients flow into both tables. Full derivation is in the original paper; to internalize it, derive it on paper from scratch.

### Step 4: Train on a toy corpus

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

After enough epochs on a large corpus, words sharing context develop similar center embeddings. On a toy corpus you can faintly see this. On billions of tokens, the effect is striking.

### Step 5: The analogy trick

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

On pretrained 300d Google News vectors:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Not because the model understands royalty, but because the vector `(king - man)` captures something like "regal-ness," and adding it to `woman` lands near the female-royal region.

## Use It

Writing Word2Vec from scratch is for learning. Production NLP uses `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

In real work you almost never train Word2Vec yourself — you download pretrained vectors.

- **GloVe** — Stanford's co-occurrence matrix factorization approach. Available in 50d, 100d, 200d, 300d checkpoints. Good general coverage. Lesson 04 covers GloVe in depth.
- **fastText** — Facebook's extension of Word2Vec that also embeds character n-grams. Handles OOV words by composing subwords. Covered in lesson 04.
- **Pretrained Word2Vec on Google News** — 300d, 3M-word vocabulary, released 2013. Still downloaded daily.

### Where Word2Vec still wins in 2026

- Lightweight domain-specific retrieval. Train for an hour on medical abstracts on a laptop, and you get specialized vectors that general models miss.
- Analogy-style feature engineering. `gender_vector = mean(man - woman pairs)`. Subtract it from other words to get a gender-neutral axis. Still used in fairness research.
- Interpretability. 100d is small enough to visualize with PCA or t-SNE — you can actually see clusters forming.
- Anywhere inference must run on a GPU-free device. Word2Vec lookup is just fetching a row.

### Where Word2Vec fails

The polysemy wall. `bank` has one vector shared by `river bank` and `financial bank`; `table` (spreadsheet vs furniture) shares one too. A downstream classifier can't disambiguate sense from that vector.

Contextual embeddings (ELMo, BERT, and every transformer since) solve this: they produce a different vector for each occurrence of a word based on surrounding context. That's the leap from Word2Vec to BERT — static to contextual. Phase 7 covers the transformer half.

Another failure is the OOV problem. If `Zoomer-approved` wasn't in training data, Word2Vec has never seen it and has no fallback. fastText patches this with subword composition (lesson 04).

## Ship It

Save as `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Exercises

1. **Easy.** Run the training loop on a tiny corpus (20 sentences about cats and dogs). After 200 epochs, verify that `nearest(vocab, W, W[vocab["cat"]])` returns `dog` in the top 3. If not, increase epochs or vocabulary.
2. **Medium.** Add high-frequency subsampling. Words with frequency above `10^-5` are dropped from training pairs with probability proportional to their frequency. Measure its effect on rare-word similarity.
3. **Hard.** Train a model on the 20 Newsgroups corpus. Compute two bias axes: `he - she` and `doctor - nurse`. Project occupation words onto these axes. Report which occupations have the largest bias gap. This is exactly the kind of probe fairness researchers use.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Word embedding | Words as vectors | Dense, low-dimensional (typically 100–300) representations learned from context. |
| Skip-gram | The Word2Vec trick | Predict context words from a center word. Slower than CBOW but better for rare words. |
| Negative sampling | Training shortcut | Binary classification against `k` random words replaces full-vocabulary softmax. |
| Static embedding | One vector per word | Same vector regardless of context. Fails on polysemy. |
| Contextual embedding | Context-dependent vector | A different vector for each occurrence based on surrounding words. What transformers produce. |
| OOV | Out of vocabulary | A word never seen in training. Word2Vec can't produce a vector for it. |

## Further Reading

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — The negative sampling paper. Short and readable.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — The clearest gradient derivation, if you find the original paper's math too dense.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — Production training configurations that actually work.
