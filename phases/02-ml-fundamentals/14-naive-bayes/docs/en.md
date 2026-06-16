# Naive Bayes

> The "naive" assumption is wrong, but it works anyway. That's the beauty of it.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 Lessons 01-07 (Classification, Bayes' Theorem)
**Time:** ~75 minutes

## Learning Objectives

- Implement Multinomial Naive Bayes from scratch with Laplace smoothing for text classification
- Explain why the naive independence assumption is mathematically wrong yet produces correct class rankings in practice
- Compare the Multinomial, Bernoulli, and Gaussian Naive Bayes variants and choose the right one for a given feature type
- Benchmark Naive Bayes against logistic regression on high-dimensional sparse data and explain the bias-variance tradeoff at play

## The Problem

You need to classify text. Sort emails into spam or not-spam. Categorize customer reviews as positive or negative. Route support tickets to the right team. You have thousands of features (one per word) and limited training data.

Most classifiers choke here. Logistic regression needs enough samples to reliably estimate thousands of weights. Decision trees split on one word at a time and overfit catastrophically. KNN in 10,000 dimensions is meaningless because every point is equidistant from every other point.

Naive Bayes handles this. It makes a mathematically wrong assumption (given the class, every feature is independent of every other feature) and still outperforms "smarter" models on text classification, especially with small training sets. It trains in a single pass over the data. It scales to millions of features. It produces probability estimates (though often poorly calibrated due to the independence assumption).

Understanding why a wrong assumption leads to good predictions teaches you something fundamental about machine learning: the best model isn't the most correct one — it's the one with the best bias-variance tradeoff for your data.

## The Concept

### Bayes' Theorem (Quick Recap)

Bayes' theorem flips conditional probabilities:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

We want `P(class | features)` — the probability a document belongs to a class given its words. We can compute it from:
- `P(features | class)` — the likelihood of seeing these words in documents of this class
- `P(class)` — the prior probability of the class (how common is spam overall?)
- `P(features)` — the evidence, same for all classes, so we can ignore it when comparing

The class with the highest `P(class | features)` wins.

### The Naive Independence Assumption

Computing `P(features | class)` exactly requires estimating the joint probability of all features together. With a vocabulary of 10,000 words, you'd need to estimate a distribution over 2^10,000 possible combinations. Impossible.

The naive assumption: given the class, every feature is conditionally independent.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Instead of estimating one impossible joint distribution, you estimate n simple per-feature distributions. Each one just needs a count.

This assumption is obviously wrong. The words "machine" and "learning" are not independent in any document. But the classifier doesn't need correct probability estimates — it needs correct rankings — which class has the highest probability. The independence assumption introduces systematic errors, but these errors affect all classes similarly, so the ranking is preserved.

### Why It Still Works

Three reasons:

1. **Ranking over calibration.** Classification only requires the top-ranked class to be correct. Even if P(spam) = 0.99999 when the true probability is 0.7, the classifier still correctly picks spam. We don't need correct probabilities — we need the correct winner.

2. **High bias, low variance.** The independence assumption is a strong prior. It heavily constrains the model, preventing overfitting. With limited training data, a slightly wrong but stable model beats a theoretically correct but wildly unstable one. This is the bias-variance tradeoff in action.

3. **Feature redundancy cancels out.** Correlated features provide redundant evidence. The classifier double-counts this evidence, but it double-counts it for the correct class. If "machine" and "learning" always co-occur, both provide evidence for the "tech" class. NB counts them twice, but it counts them twice for the right class.

A fourth practical reason: Naive Bayes is extremely fast. Training is a single pass counting frequencies. Prediction is a matrix multiplication. You can train on a million documents in seconds. This speed means you can iterate faster, try more feature sets, and run more experiments than slower models allow.

### Step-by-Step Math

Let's walk through a concrete example. Suppose two classes: spam and not-spam. Our vocabulary has three words: "free", "money", "meeting".

Training data:
- Spam emails mention "free" 80 times, "money" 60 times, "meeting" 10 times (150 total words)
- Not-spam emails mention "free" 5 times, "money" 10 times, "meeting" 100 times (115 total words)
- 40% of emails are spam, 60% are not-spam

With Laplace smoothing (alpha=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

New email contains: "free" (2 times), "money" (1 time), "meeting" (0 times).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam wins by a large margin. "free" appearing twice is strong evidence for spam. Notice that "meeting" not appearing contributes zero to both log sums (0 * log(P)) — in Multinomial NB, absent words have no effect. Bernoulli NB is the variant that explicitly models word absence.

### Three Variants

Naive Bayes comes in three flavors. Each models `P(feature | class)` differently.

#### Multinomial Naive Bayes

Models each feature as a count. Best for text data where features are word frequencies or TF-IDF values.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` is Laplace smoothing (explained below). This variant is the workhorse for text classification.

#### Gaussian Naive Bayes

Models each feature as a normal distribution. Best for continuous features.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

Each class has its own mean and variance for each feature. Works well when features are truly bell-shaped within each class.

#### Bernoulli Naive Bayes

Models each feature as binary (present or absent). Best for short text or binary feature vectors.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

Unlike Multinomial, Bernoulli explicitly penalizes word absence. If "free" usually appears in spam but doesn't appear in this email, Bernoulli treats that as evidence against spam.

### When to Use Which Variant

| Variant | Feature Type | Best For | Examples |
|---------|-------------|----------|---------|
| Multinomial | Counts or frequencies | Text classification, bag of words | Email spam, topic classification |
| Gaussian | Continuous values | Tabular data with near-normal features | Iris classification, sensor data |
| Bernoulli | Binary (0/1) | Short text, binary feature vectors | SMS spam, presence/absence features |

### Laplace Smoothing

What happens when a word appears in test data but never appeared in a class's training data?

Without smoothing: `P(word | class) = 0/N = 0`. One zero kills the entire product, making `P(class | features) = 0` regardless of all other evidence. A single unseen word destroys the prediction no matter how much other evidence supports it.

Laplace smoothing adds a small count `alpha` (usually 1) to every feature count:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

With alpha=1, every word has at least a tiny probability. "Discombobulate" appearing in a test email no longer kills the spam probability. This smoothing has a Bayesian interpretation: it's equivalent to placing a uniform Dirichlet prior on the word distribution.

Higher alpha means more smoothing (flatter distribution). Lower alpha means trusting the data more. Alpha is a hyperparameter you tune.

Effect of alpha:

| Alpha | Effect | When to Use |
|-------|--------|-------------|
| 0.001 | Almost no smoothing, trust the data | Very large training set, no unseen features expected |
| 0.1 | Light smoothing | Large training set |
| 1.0 | Standard Laplace smoothing | Default starting point |
| 10.0 | Heavy smoothing, flattens distribution | Very small training set, many unseen features expected |

### Log-Space Computation

Multiplying hundreds of probabilities (each less than 1) causes floating-point underflow. The product becomes zero in floating-point even though the true value is a tiny positive number.

Solution: work in log-space. Instead of multiplying probabilities, add their logarithms:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

This turns prediction into a dot product:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Matrix multiplication. This is why Naive Bayes prediction is so fast — it's the same operation as a single-layer linear model.

### Naive Bayes vs Logistic Regression

Both are linear classifiers for text. The difference is what they model.

| Dimension | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| Type | Generative (models P(X\|Y)) | Discriminative (models P(Y\|X)) |
| Training | Count frequencies | Optimize loss function |
| Small data | Better (strong prior helps) | Worse (not enough to estimate weights) |
| Large data | Worse (wrong assumption hurts) | Better (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, very fast | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

Rule of thumb: start with Naive Bayes. If you have enough data and NB plateaus, switch to logistic regression.

### Classification Pipeline

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

In practice, we work in log-space to avoid floating-point underflow. Instead of multiplying many small probabilities, we add their logs:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## Build It

The code in `code/naive_bayes.py` implements MultinomialNB and GaussianNB from scratch.

### MultinomialNB

From-scratch implementation:

1. **fit(X, y)**: For each class, count the frequency of each feature. Add Laplace smoothing. Compute log probabilities. Store class priors (log of class frequencies).

2. **predict_log_proba(X)**: For each sample, compute log P(class) + sum of log P(feature_i | class) for all classes. This is a matrix multiplication: X @ log_probs.T + log_priors.

3. **predict(X)**: Return the class with the highest log probability.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

Key insight: after fitting, prediction is a matrix multiplication plus a bias. That's why Naive Bayes is so fast.

### GaussianNB

For continuous features, we estimate the mean and variance per class per feature:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

Prediction uses the Gaussian PDF for each feature, multiplied across features (summed in log-space).

### Demo: Text Classification

The code generates synthetic bag-of-words data simulating two classes (tech articles vs sports articles). Each class has a different word frequency distribution. MultinomialNB classifies them using word counts.

The synthetic data works like this: we create 200 "words" (feature columns). Words 0-39 are high-frequency in tech, low in sports. Words 80-119 are high-frequency in sports, low in tech. Words 40-79 are medium-frequency in both. This creates a realistic scenario where some words are strong class indicators and others are noise.

### Demo: Continuous Features

The code generates iris-like data (3 classes, 4 features, Gaussian clusters). GaussianNB classifies using per-class means and variances. Each class has a different center (mean vector) and spread (variance), mimicking real data where different categories have systematically different measurements.

The code also demonstrates:
- **Smoothing comparison:** Training MultinomialNB with different alpha values, showing how smoothing strength affects accuracy.
- **Training size experiment:** How NB accuracy improves as training data grows from 20 to 1600 samples. NB achieves decent accuracy even with very few samples — that's its main advantage.
- **Confusion matrix:** Per-class precision, recall, and F1 scores, showing where NB makes errors.

### Prediction Speed

Naive Bayes prediction is a matrix multiplication. For n samples, d features, k classes:
- MultinomialNB: one matrix multiply (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k Gaussian PDF computations, each across d features = O(n * d * k)

Both are linear in every dimension. Compared to KNN (which needs distance to all training points) or SVM with RBF kernel (which needs kernel evaluations against all support vectors), NB is orders of magnitude faster at prediction time.

## Use It

With sklearn, both variants are one-liners:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Text classification with sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

The code in `naive_bayes.py` compares the from-scratch implementation against sklearn on the same data to verify correctness.

### TF-IDF with Naive Bayes

Raw word counts give equal weight to every word occurrence. But common words like "the" and "is" appear frequently in every class — they carry no information. TF-IDF (Term Frequency - Inverse Document Frequency) downweights common words and upweights rare, distinctive ones.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF values are non-negative, so they work with MultinomialNB. The TF-IDF + MultinomialNB combination is one of the strongest baselines for text classification. On datasets with fewer than 10,000 training samples, it often beats more complex models.

### BernoulliNB for Short Text

For short text (tweets, SMS, chat messages), BernoulliNB can outperform MultinomialNB. Short texts have low word counts, so the frequency information MultinomialNB relies on is noisy. BernoulliNB only cares about presence or absence, which is more reliable for short texts.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

The `binary=True` flag in CountVectorizer converts all counts to 0/1. Without it, BernoulliNB still runs but it sees counts it wasn't designed to handle.

### Calibrating NB Probabilities

NB probabilities are poorly calibrated. When NB says P(spam) = 0.95, the true probability might be 0.7. If you need reliable probability estimates (e.g., for thresholding or combining with other models), use sklearn's CalibratedClassifierCV:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

This fits a logistic regression on top of NB's raw scores using cross-validation. The resulting probabilities are closer to true class frequencies.

### Pitfalls

1. **Negative feature values.** MultinomialNB requires non-negative features. If you have negative values (e.g., TF-IDF with certain settings, or standardized features), use GaussianNB instead, or shift features to be positive.

2. **Zero-variance features.** GaussianNB divides by variance. If a feature has zero variance for a class (all values are identical), the probability computation breaks. The code adds a small smoothing term (1e-9) to all variances to prevent this.

3. **Class imbalance.** If 99% of emails are not-spam, the prior P(not-spam) = 0.99 is so strong it overwhelms the likelihood evidence. You can manually set class priors, or use the class_prior parameter in sklearn.

4. **Feature scaling.** MultinomialNB doesn't need scaling (it works on counts). GaussianNB doesn't need scaling either (it estimates per-feature statistics). This is an advantage over logistic regression and SVM, which are sensitive to feature scales.

## Ship It

This lesson produces:
- `outputs/skill-naive-bayes-chooser.md` -- A decision skill for picking the right NB variant
- `code/naive_bayes.py` -- From-scratch MultinomialNB and GaussianNB with sklearn comparison

### When Naive Bayes Fails

NB fails when the independence assumption causes the ranking to be wrong (not just the probabilities). This happens when:

1. **Strong feature interactions.** If the class depends on the combination of two features rather than either alone (XOR-like patterns), NB will completely miss it. Each feature alone provides no evidence, and NB can't combine them nonlinearly.

2. **Highly correlated features with opposing evidence.** If feature A says "spam" and feature B says "not-spam", but A and B are perfectly correlated (they always agree in reality), NB sees conflicting evidence where none exists.

3. **Very large training sets.** With enough data, discriminative models like logistic regression learn the true decision boundary and outperform NB. The independence assumption that helped with small data now holds the model back.

In practice, these failure modes are rare for text classification. Text features are numerous and individually weak, and the independence assumption's errors tend to cancel out. For tabular data with few strongly correlated features, prefer logistic regression or tree-based models.

## Exercises

1. **Smoothing experiment.** Train MultinomialNB on text data with alpha values of 0.01, 0.1, 1.0, 10.0, and 100.0. Plot accuracy vs alpha. Where does performance peak? Why is very high alpha harmful?

2. **Feature independence test.** Take a real text dataset. Pick two obviously correlated words ("machine" and "learning"). Compute P(word1 | class) * P(word2 | class) and compare to P(word1 AND word2 | class). How wrong is the independence assumption? Does it affect classification accuracy?

3. **Bernoulli implementation.** Extend the code with a BernoulliNB class. Convert bag-of-words to binary (presence/absence) and compare accuracy against MultinomialNB on text data. When does Bernoulli win?

4. **NB vs logistic regression.** Train both on text data. Start with 100 training samples and increase to 10,000. Plot both accuracies vs training set size. At what point does logistic regression overtake Naive Bayes?

5. **Spam filter.** Build a complete spam classifier: tokenize raw email text, build vocabulary, create bag-of-words features, train MultinomialNB, evaluate with precision and recall (not just accuracy — why?).

## Key Terms

| Term | What People Say | What It Actually Is |
|------|----------------|----------------------|
| Naive Bayes | "Simple probabilistic classifier" | A classifier that applies Bayes' theorem with the assumption that features are conditionally independent given the class |
| Conditional independence | "Features don't affect each other" | P(A, B \| C) = P(A \| C) * P(B \| C) — once you know C, knowing B tells you nothing new about A |
| Laplace smoothing | "Add-one smoothing" | Adding a small count to every feature to prevent zero probabilities from dominating predictions |
| Prior | "What you believe before seeing data" | P(class) — the probability of each class before observing any features |
| Likelihood | "How well the data fits" | P(features \| class) — the probability of observing these features given the class |
| Posterior | "What you believe after seeing data" | P(class \| features) — the updated probability of a class after observing features |
| Generative model | "Models how data is generated" | A model that learns P(X \| Y) and P(Y), then uses Bayes' theorem to get P(Y \| X) |
| Discriminative model | "Models the decision boundary" | A model that directly learns P(Y \| X) without modeling how X is generated |
| Log-probabilities | "Avoiding underflow" | Using log P instead of P to prevent the product of many small numbers from becoming zero in floating-point |

## Further Reading

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) -- All three variants with mathematical details
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- Classic comparison of Multinomial vs Bernoulli for text
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- Improvements to NB for text
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- Shows NB converges faster than LR with less data
