# Text Generation Before the Transformer — N-gram Language Models

> If a word is surprising, the model is bad. Perplexity turns "surprise" into a number. Smoothing keeps it finite.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (text processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## The Problem

Before transformers, before RNNs, before word embeddings, language models predicted the next word by counting how often it followed the previous `n-1` words. Count "the cat" → "sat" 47 times, "the cat" → "jumped" 12 times, "the cat" → "refrigerator" 0 times. Normalize to get a probability distribution.

That is the n-gram language model. From 1980 to 2015 it ran inside every speech recognizer, every spell checker, every phrase-based machine translation system. It still runs today when you need cheap on-device language modeling.

The interesting question: what to do about unseen n-grams. A raw-count model assigns zero probability to anything it hasn't seen, which is catastrophic because sentences are long and nearly every long sentence contains at least one unseen sequence. Fifty years of smoothing research fixed this. Kneser-Ney smoothing is its result, and modern deep learning inherited its empiricist tradition.

## The Concept

![N-gram model: counting, smoothing, generation](../assets/ngram.svg)

**N-gram probability:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Fix `n` (typically 3 for trigrams, 4 for 4-grams). Compute from counts:

```text
P(w | context) = count(context, w) / count(context)
```

**The zero-count problem.** Any n-gram not seen in training gets probability zero. A 2007 study on the Brown corpus found that even for a 4-gram model, 30% of 4-grams in a held-out set never appeared in training. Without smoothing, you cannot evaluate on any real text.

**Smoothing methods, ordered by sophistication:**

1. **Laplace (add-one).** Add 1 to every count. Simple, terrible for rare events.
2. **Good-Turing.** Based on frequency of frequencies, redistributes probability mass from high-frequency events to unseen events.
3. **Interpolation.** Combines n-gram, (n-1)-gram, etc. estimates with tunable weights.
4. **Backoff.** If the n-gram count is zero, fall back to the (n-1)-gram. Katz backoff normalizes this.
5. **Absolute discounting.** Subtract a fixed discount `D` from all counts, redistribute to unseen events.
6. **Kneser-Ney.** Absolute discounting plus a clever choice for the lower-order model: use *continuation probability* (how many contexts a word appears in) instead of raw frequency.

The Kneser-Ney insight runs deep. "San Francisco" is a common bigram. The unigram "Francisco" mostly appears after "San". Naive absolute discounting gives "Francisco" a high unigram probability (because the count is high). Kneser-Ney notices that "Francisco" appears in only one context and lowers its continuation probability accordingly. Result: a novel bigram ending in "Francisco" gets an appropriately low probability.

**Evaluation: perplexity.** The exponent of average per-word negative log-likelihood on a held-out test set. Lower is better. Perplexity 100 means the model is as confused as choosing uniformly among 100 words.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## Build It

### Step 1: Trigram counting

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Input is a list of tokenized sentences. Output is n-gram counts and context counts. `<s>` and `</s>` are sentence boundaries.

### Step 2: Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Add 1 to every count. Smooths, but assigns too much mass to unseen events and hurts observed rare events.

### Step 3: Kneser-Ney (bigram, interpolated)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Three moving parts. `continuation_prob` captures "how many distinct contexts does this word appear in?" (Kneser-Ney's innovation). `lambda_prev` is the mass freed by discounting, used to weight the backoff. The final probability is the discounted main term plus the weighted continuation term.

### Step 4: Text generation via sampling

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Sampling proportional to probability. Each seed gives different output. For beam-search-like output, take the argmax at each step (greedy) and add a small randomness knob (temperature).

### Step 5: Perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Lower is better. On the Brown corpus, a well-tuned 4-gram KN model hits perplexity around 140. A transformer LM on the same test set scores 15–30. The gap is roughly 10×. That gap is why the field moved forward.

## Use It

- **Classic NLP teaching.** The clearest exposure you can get to smoothing, MLE, and perplexity.
- **KenLM.** Production-grade n-gram library. Used as a rescorer in speech and MT systems where low latency matters.
- **On-device autocomplete.** Trigram models inside keyboards. Still running today.
- **Baseline.** Always compute an n-gram LM perplexity before declaring your neural LM good. If your transformer doesn't beat KN by a wide margin, something is wrong.

## Ship It

Save as `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Exercises

1. **Easy.** Train a trigram LM on a 1000-sentence Shakespeare corpus. Generate 20 sentences. They will be locally plausible, globally incoherent. This is the classic demonstration.
2. **Medium.** Implement perplexity for your KN model on a held-out Shakespeare split. Compare against Laplace. You should see KN reduce perplexity by 30–50%.
3. **Hard.** Build a trigram spell corrector: given a misspelled word and its context, generate candidate corrections ranked by context probability under the LM. Evaluate on the Birkbeck spelling corpus (public).

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|---------------------|
| N-gram | Word sequence | A sequence of `n` consecutive tokens. |
| Smoothing | Avoiding zeros | Redistributing probability mass so unseen events get non-zero probability. |
| Perplexity | LM quality metric | `exp(-average log probability)` on held-out data. Lower is better. |
| Backoff | Fall back to shorter context | Use the bigram when the trigram count is zero. Katz backoff formalizes this. |
| Kneser-Ney | Best n-gram smoothing | Absolute discounting + continuation probability for the lower-order model. |
| Continuation probability | KN-specific | `P(w)` weighted by the number of contexts `w` appears in, not raw count. |

## Further Reading

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — The classic treatment of n-gram LMs and smoothing.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — The paper that established Kneser-Ney as the best n-gram smoother.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — The original KN paper.
- [KenLM](https://kheafield.com/code/kenlm/) — Fast production-grade n-gram LM, still used in latency-sensitive applications as of 2026.
