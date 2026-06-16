# Bag of Words, TF-IDF, and Text Representation

> Count first, think later. On well-defined tasks, TF-IDF still beats embeddings in 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 min

## The Problem

Models want numbers; you have strings.

Every NLP pipeline must answer the same question: how do you turn a variable-length stream of tokens into a fixed-length vector a classifier can consume? The earliest production answer in this field was the dumbest one that works — count words, make a vector.

That vector powered more production NLP than any embedding model. Spam filtering, topic classification, log anomaly detection, search ranking (pre-BM25), the first wave of sentiment analysis, the first decade of academic NLP benchmarks. In 2026, practitioners still reach for it first on narrow classification tasks. It's fast, interpretable, and on tasks where word presence is what matters, it often ties a 400M-parameter embedding model.

This lesson builds bag-of-words from scratch, then TF-IDF. Then shows scikit-learn doing the same thing in three lines. Finally, it points out the failure scenario that will force you toward embeddings.

## The Concept

**Bag of Words (BoW)** discards order. For each document, count how many times each vocabulary word appears. Vector length equals vocabulary size; position `i` holds the count of word `i`.

**TF-IDF** re-weights BoW. A word appearing in every document carries no information — scale its weight down. A word rare across the corpus but frequent in one document is a signal — scale its weight up.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

`TF` is term frequency within the document, `df` is document frequency (how many documents contain the word), `N` is the total document count. The `log` compresses the weight of ubiquitous words into a bounded range.

Key property: both produce sparse vectors with interpretable axes. You can inspect a trained classifier's weights and read which words push documents toward which class. You can't do that with a 768-dimensional BERT embedding.

## Build It

### Step 1: Build the vocabulary

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Input: a list of tokenized documents (any word-level tokenizer works; this lesson's `code/main.py` uses a simplified lowercase variant). Output: a `{word: index}` dictionary. Stable insertion order means word index 0 is the first word seen in the first document. Conventions vary — scikit-learn sorts alphabetically.

### Step 2: Bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Rows are documents, columns are vocabulary indices. Entry `[i][j]` is "how many times word `j` appears in document `i`". Document 1 has `cat` count 2 because it appears twice. Document 0 has `ran` count 0 because it doesn't appear.

### Step 3: Term frequency and document frequency

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Two smoothing tricks worth naming. `(n+1)/(d+1)` avoids `log(x/0)`. The trailing `+1` ensures a word appearing in every document still gets an IDF of 1 (not 0), matching scikit-learn's default. Other implementations use raw `log(N/df)`. Both work; the smoothed version is friendlier.

### Step 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Three documents, five vocabulary words (`the`, `cat`, `sat`, `dog`, `ran`). `the` appears in all three, so its IDF is low. `dog` appears in only one, so its IDF is high. The vectors are sparse (most entries are small), and discriminative words stand out.

### Step 5: L2-normalize rows

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Without normalization, longer documents get larger vectors and dominate similarity scores. L2 normalization places each document on the unit hypersphere. Cosine similarity between rows then becomes a simple dot product.

## Use It

scikit-learn ships the production version.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` handles tokenization, vocabulary building, and BoW in a single call. `TfidfVectorizer` adds IDF weighting and L2 normalization on top. Both return sparse matrices. At 100k documents the dense version won't fit in memory — keep it sparse until a classifier requires dense input.

Knobs that change everything:

| Parameter | Effect |
|-----|--------|
| `ngram_range=(1, 2)` | Include bigrams. Usually improves classification. |
| `min_df=2` | Drop words appearing in fewer than 2 documents. Trims vocabulary on noisy data. |
| `max_df=0.95` | Drop words appearing in over 95% of documents. Approximates stopword removal without a hardcoded list. |
| `stop_words="english"` | scikit-learn's built-in English stopword list. Task-dependent — sentiment analysis **should not** remove negation words. |
| `sublinear_tf=True` | Replace raw `tf` with `1 + log(tf)`. Helps when a word repeats many times in one document. |

### Where TF-IDF still wins (as of 2026)

- Spam detection, topic tagging, log anomaly flagging. Word presence is what matters; semantic nuance doesn't.
- Low-data regimes (a few hundred labeled samples). TF-IDF plus logistic regression has no pretraining cost.
- Anywhere latency matters. TF-IDF plus a linear model returns results in microseconds. Passing a document through a transformer for embedding takes 10–100ms.
- Systems that must explain their predictions. Look at the classifier's coefficients — the top positive-weight words are the explanation.

### Where TF-IDF fails

Semantic blindness. Consider these two documents:

- "The movie was not good at all."
- "The movie was excellent."

One is negative, one is positive. Their TF-IDF overlap is exactly `{the, movie, was}`. A BoW classifier can only memorize that `not` near `good` flips the label. With enough data it learns this, but it's never as graceful as a model that understands syntax.

Another failure: out-of-vocabulary words at inference time. A BoW model trained on IMDb reviews has no idea what to do with a token like `Zoomer-approved` that never appeared in training. Subword embeddings (lesson 04) handle this; TF-IDF cannot.

### Hybrid approach: TF-IDF–weighted embeddings

The pragmatic default for mid-data classification in 2026: use TF-IDF weights as attention over word embeddings.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

You get semantic capability from embeddings and emphasis on rare words from TF-IDF. The classifier trains on the pooled vector. On sentiment, topic, and intent classification tasks below ~50k labeled samples, it outperforms either alone.

## Ship It

Save as `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Exercises

1. **Easy.** Implement `cosine_similarity(doc_vec_a, doc_vec_b)` on L2-normalized TF-IDF output. Verify that identical documents score 1.0 and documents with disjoint vocabularies score 0.0.
2. **Medium.** Add n-gram support to `bag_of_words`. A parameter `n` produces counts over `n`-grams. Test that `n=2` on `["the", "cat", "sat"]` produces bigram counts for `["the cat", "cat sat"]`.
3. **Hard.** Build the TF-IDF–weighted embedding hybrid above using GloVe 100d vectors (download once and cache). Compare its classification accuracy against pure TF-IDF and pure mean-pooled embeddings on the 20 Newsgroups dataset. Report which scenarios each wins.

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| BoW | Term-frequency vector | Count of each vocabulary word in a document. Discards order. |
| TF | Term frequency | Count of a word in a document, optionally normalized by document length. |
| DF | Document frequency | Number of documents containing the word at least once. |
| IDF | Inverse document frequency | Smoothed `log(N / df)`. Down-weights ubiquitous words. |
| Sparse vector | Mostly zeros | Vocabularies are typically 10k–100k words; for any given document, most don't appear. |
| Cosine similarity | Angle between vectors | Dot product of L2-normalized vectors. 1 means identical, 0 means orthogonal. |

## Further Reading

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — The authoritative API reference with explanations for every knob.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — The paper that made TF-IDF the default for a decade.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — A 2026 perspective: when and why the old method wins.
