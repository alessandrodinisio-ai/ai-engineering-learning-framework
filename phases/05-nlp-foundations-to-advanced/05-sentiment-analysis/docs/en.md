# Sentiment Analysis

> The most classic NLP task. Most of what you need to know about classical text classification surfaces here.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 min

## The Problem

"The food was not great." Positive or negative?

Sentiment analysis sounds trivial: a reviewer said they liked or disliked something — label it. It became the most classic NLP task because every simple-looking example hides a hard one behind it. Negation flips meaning. Sarcasm inverts it. "Not bad at all" is positive despite containing two negative-toned words. Emojis carry more signal than surrounding text. Domain vocabulary matters (`tight` in a music review vs `tight` in a fashion review).

Sentiment analysis is a living lab for classical NLP. If you understand why each naive baseline has one specific failure mode, you understand why each more complex model was invented. This lesson builds a Naive Bayes baseline from scratch, adds logistic regression, then highlights the traps that turn production sentiment analysis into a compliance-level problem.

## The Concept

Classical sentiment analysis is a two-step recipe.

1. **Representation.** Turn text into a feature vector. BoW, TF-IDF, or n-grams.
2. **Classification.** Fit a linear model on labeled samples (Naive Bayes, logistic regression, SVM).

Naive Bayes is the dumbest model that works. It assumes every feature is independent given the label. Estimate `P(word | positive)` and `P(word | negative)` from counts. At inference, multiply probabilities. The "naive" independence assumption is laughably wrong, yet results are surprisingly strong. The reason: with sparse text features and moderate data, the classifier cares which side each word leans toward, not how much.

Logistic regression fixes the independence assumption. It learns a weight for each feature, including negative weights. `not good` as a bigram feature gets a negative weight. Naive Bayes can't do this for bigrams it hasn't explicitly counted.

## Build It

### Step 1: A real mini dataset

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Intentionally small. Real work uses tens of thousands of samples (IMDb, SST-2, Yelp polarity). The math is identical.

### Step 2: Multinomial Naive Bayes from scratch

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing (alpha=1.0) is Laplace smoothing. Without it, a word unseen in one class gets zero probability and log explodes. In practice `alpha=0.01` is common. `alpha=1.0` is the teaching default.

### Step 3: Logistic regression from scratch

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization matters here. Text features are sparse; without L2 the model memorizes training samples. Start at `0.01` and tune.

### Step 4: Handling negation (the failure mode)

Consider "not good" and "not bad". A BoW classifier sees `{not, good}` and `{not, bad}` and learns from whichever appeared more in training. A bigram classifier sees `not_good` and `not_bad` as distinct features. This usually suffices.

A cruder fix that works without bigrams: **negation scope**. Prefix tokens after a negation word with `NOT_` until the next punctuation.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Now `good` and `NOT_good` are different features and the classifier can assign opposite weights. Three lines of preprocessing for a measurable accuracy boost on sentiment benchmarks.

### Step 5: The metrics that actually matter

If classes are imbalanced, accuracy alone misleads. Real sentiment corpora are often 70–80% positive or 70–80% negative; a constant-majority classifier hits 80% accuracy and is worthless. Report all of the following:

- **Per-class precision and recall.** One pair per class. Macro-average them for a single number that respects class balance.
- **Macro-F1 (primary metric for imbalanced data).** Unweighted mean of per-class F1. Use it instead of accuracy when classes are imbalanced.
- **Weighted-F1 (alternative).** Same as macro but weighted by class frequency. Report alongside macro-F1 when the imbalance itself has business meaning.
- **Confusion matrix.** Raw counts. Look at it before trusting any scalar metric — it reveals which class pairs the model confuses.
- **Per-class error samples.** Pull 5 misclassifications per class and read them. Nothing substitutes for reading real errors.

For severely imbalanced data (> 95–5 ratio), report **AUROC** and **AUPRC** instead of accuracy. AUPRC is more sensitive to the minority class, which is often the one you care about (spam, fraud, rare sentiment).

**Common bug to avoid.** Reporting micro-F1 instead of macro-F1 on imbalanced data gives a deceptively high number because it's dominated by the majority class. Macro-F1 forces you to see minority-class performance.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Use It

scikit-learn does it correctly in six lines.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Note three things. `stop_words=None` preserves negation words. `ngram_range=(1, 2)` adds bigrams so `not_good` becomes a feature. `sublinear_tf=True` dampens repeated words. On SST-2, these three switches are the difference between a 75% accuracy baseline and an 85% accuracy baseline.

### When to reach for a transformer

- Sarcasm detection. Classical models fail here, period.
- Long reviews where sentiment shifts mid-document.
- Aspect-based sentiment. "Camera was great but battery was terrible." You need to attribute sentiment to aspects. Only transformers or structured-output models handle this.
- Non-English, low-resource languages. Multilingual BERT gives you a zero-shot baseline for free.

If you need any of the above, jump to phase 7 (transformers deep dive). Otherwise, Naive Bayes or logistic regression on TF-IDF with bigrams and negation handling is your 2026 production baseline.

### Reproducibility trap (again)

Retraining sentiment models is routine; re-evaluating is not. Accuracy numbers in papers use specific splits, specific preprocessing, specific tokenizers. If you compare your new model to a baseline without using the exact same pipeline, the delta is misleading. Always regenerate baselines on your own pipeline rather than citing paper numbers.

## Ship It

Save as `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Exercises

1. **Easy.** Add `apply_negation` as a preprocessing step to a scikit-learn pipeline and measure F1 change on a small sentiment dataset.
2. **Medium.** Implement class-weighted logistic regression (pass `class_weight="balanced"` to scikit-learn, or derive the gradient yourself). Test its effect on a synthetic 90–10 class imbalance.
3. **Hard.** Train a second classifier on the residuals of a sentiment model to build a sarcasm detector. Document your experimental setup. Warn the reader when accuracy falls below chance level (binary sarcasm chance is ~50%; most first attempts land there).

## Key Terms

| Term | What people say | What it actually is |
|------|-----------------|-----------------------|
| Polarity | Positive or negative | A binary label; sometimes extended to neutral or fine-grained (five stars). |
| Aspect-based sentiment | Per-aspect polarity | Attribute sentiment to specific entities or attributes mentioned in the text. |
| Negation scope | Flip nearby tokens | Prefix tokens after "not" with `NOT_` until punctuation. |
| Laplace smoothing | Add-one to counts | Prevents zero-probability features in Naive Bayes. |
| L2 regularization | Shrink weights | Add `lambda * sum(w^2)` to the loss. Essential for sparse text features. |

## Further Reading

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — The foundational survey. Long, but the first four sections cover all classical content.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — The paper proving bigrams + Naive Bayes are hard to beat on short text.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — Reference for `CountVectorizer`, `TfidfVectorizer`, and every knob you'll tune.
