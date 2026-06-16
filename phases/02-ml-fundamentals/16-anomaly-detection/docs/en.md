# Anomaly Detection

> Normal is easy to define. An anomaly is anything that doesn't fit in.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 Lessons 01-09
**Time:** ~75 minutes

## Learning Objectives

- Implement Z-score, IQR, and Isolation Forest anomaly detection methods from scratch
- Distinguish point anomalies, contextual anomalies, and collective anomalies, choosing the right detection method for each
- Explain why anomaly detection is framed as modeling normal data rather than classifying anomalies
- Compare unsupervised anomaly detection to supervised classification, evaluating tradeoffs between novel anomaly coverage and precision

## The Problem

A credit card is swiped at 2 PM in New York, then again at 2:05 PM in Tokyo. A factory sensor reads 150 degrees when the normal range is 80-120. A server sends 50,000 requests per second when the daily average is 200.

These are anomalies. Finding them matters. Fraud costs billions. Equipment failures cause downtime. Network intrusions cause data breaches.

The hard part: you rarely have labeled examples of anomalies. Fraud is only 0.1% of transactions. Equipment failures happen a few times a year. You can't train a standard classifier because there's almost nothing in the "anomaly" class to learn from. Even if you have some labels, the anomalies you've seen aren't all the types you'll encounter. Tomorrow's fraud looks different from today's.

Anomaly detection flips the problem. Instead of learning what's anomalous, learn what's normal. Anything that deviates from normal is suspicious. This requires no labels, adapts to novel anomaly types, and scales to massive datasets.

## The Concept

### Types of Anomalies

Not all anomalies are alike:

- **Point anomalies.** A single data point that is unusual by itself, regardless of context. A temperature reading of 500 degrees. A $50,000 transaction on an account that usually spends $50.
- **Contextual anomalies.** A data point that is unusual given its context. A temperature of 90 degrees is normal in summer, anomalous in winter. Same value, different context.
- **Collective anomalies.** A sequence of data points that is unusual as a group, even though individual points may be normal. Five failed logins is normal. Fifty consecutive failed logins is a brute-force attack.

Most methods detect point anomalies. Contextual anomalies require time or location features. Collective anomalies require sequence-aware methods.

```mermaid
flowchart TD
    A[Anomaly Types] --> B[Point Anomaly]
    A --> C[Contextual Anomaly]
    A --> D[Collective Anomaly]

    B --> B1["Single unusual value<br/>Temperature: 500F"]
    C --> C1["Unusual in context<br/>90F in January"]
    D --> D1["Unusual sequence<br/>50 failed logins"]

    style B fill:#fdd,stroke:#333
    style C fill:#ffd,stroke:#333
    style D fill:#fdf,stroke:#333
```

### The Unsupervised Framing

In standard classification, you have labels for both classes. In anomaly detection, you're typically in one of three situations:

1. **Fully unsupervised.** No labels at all. You fit a detector on all data, hoping anomalies are rare enough not to contaminate the "normal" model.
2. **Semi-supervised.** You have a clean dataset of only normal data. You fit on this clean set and score everything else. This is the strongest setup when possible.
3. **Weakly supervised.** You have a few labeled anomalies. Use them for evaluation, not training. Train unsupervised, then measure precision/recall on the labeled subset.

Key insight: anomaly detection is fundamentally different from classification. You model the distribution of normal data, not a decision boundary between two classes.

### Supervised vs Unsupervised: Tradeoffs

If you do have labeled anomalies, should you use them for training (supervised classification) or only for evaluation (unsupervised detection)?

**Supervised (treat as classification):**
- Catches the exact anomaly types you've seen before
- Higher precision on known anomaly types
- Completely misses novel anomaly types
- Requires retraining when new anomaly types appear
- Needs enough anomaly samples (often too few)

**Unsupervised (model normal, flag deviations):**
- Catches anything that deviates from normal, including novel types
- Doesn't require labeled anomalies
- Higher false positive rate (not everything unusual is bad)
- More robust to distribution shift

In practice, the best systems combine both: unsupervised detection for broad coverage, supervised models for known high-priority anomaly types, and human review for ambiguous cases.

### Z-Score Method

The simplest approach. Compute the mean and standard deviation for each feature. Flag any point more than k standard deviations from the mean.

```text
z_score = (x - mean) / std
anomaly if |z_score| > threshold
```

Default threshold is 3.0 (for a Gaussian distribution, 99.7% of normal data falls within 3 standard deviations).

**Strengths:** Simple. Fast. Interpretable ("this value is 4.5 standard deviations from normal").

**Weaknesses:** Assumes normally distributed data. Sensitive to outliers in training data (outliers pull the mean and inflate standard deviation, making them harder to detect). Fails on multimodal distributions.

**When it works well:** Single-feature monitoring on roughly bell-shaped data. Server response times, manufacturing tolerances, sensor readings with a stable baseline.

**When it fails:** Multi-cluster data (two office locations with different baseline temperatures), skewed data ($1000 is rare but not anomalous in transaction amounts), data with outliers in the training set.

### IQR Method

More robust than Z-score. Uses the interquartile range instead of mean and standard deviation.

```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
lower_bound = Q1 - factor * IQR
upper_bound = Q3 + factor * IQR
anomaly if x < lower_bound or x > upper_bound
```

Default factor is 1.5.

**Strengths:** Robust to outliers (percentiles aren't affected by extreme values). Works on skewed distributions. No normality assumption.

**Weaknesses:** Univariate only (applies to each feature independently). Can't detect anomalies that are only unusual when features are considered together (a point that's normal in each feature alone but anomalous in the joint space).

**Practical note:** The 1.5 factor in IQR corresponds to the whiskers in a box plot. Points beyond the whiskers are potential outliers. Use 3.0 instead of 1.5 for a more conservative detector (fewer flags, fewer false positives). The correct factor depends on your tolerance for false alarms.

### Isolation Forest

Key insight: anomalies are few and different. In random partitions of the data, anomalies are easier to isolate — they take fewer random splits to separate from the rest.

```mermaid
flowchart TD
    A[All Data Points] --> B{Random Feature + Random Split}
    B --> C[Left Partition]
    B --> D[Right Partition]
    C --> E{Random Feature + Random Split}
    E --> F[Normal Point - Deep in Tree]
    E --> G[Needs More Splits...]
    D --> H["Anomaly - Isolated Quickly (Short Path)"]

    style H fill:#fdd,stroke:#333
    style F fill:#dfd,stroke:#333
```

**How it works:**
1. Build many random trees (a forest of isolation trees)
2. At each node, pick a random feature and a random split value between that feature's min and max
3. Keep splitting until every point is isolated (alone in a leaf)
4. Anomalies have shorter average path lengths across all trees

**Why it works:** Normal points live in dense regions. Isolating one point from its neighbors requires many random splits. Anomalies live in sparse regions. One or two random splits are enough to isolate them.

The anomaly score is based on the average path length across all trees, normalized by the expected path length of a random binary search tree:

```
score(x) = 2^(-average_path_length(x) / c(n))
```

Where `c(n)` is the expected path length for n samples. Scores close to 1 mean anomalous. Close to 0.5 means normal. Close to 0 means very normal (deep in a dense cluster).

**Strengths:** No distributional assumptions. Works in high dimensions. Scales well (sub-linear in sample size because each tree uses a subsample). Handles mixed feature types.

**Weaknesses:** Struggles with anomalies in dense regions (masking effect). Random splits are less effective when many features are irrelevant.

**Key hyperparameters:**
- `n_estimators`: Number of trees. 100 is usually enough. More trees give more stable scores but slower computation.
- `max_samples`: Samples per tree. Original paper defaults to 256. Smaller values make individual trees less accurate but increase diversity. Subsampling is precisely why Isolation Forest is fast — each tree sees only a small fraction of the data.
- `contamination`: Expected proportion of anomalies. Only used for threshold setting, doesn't affect the scores themselves.

### Local Outlier Factor (LOF)

LOF compares the local density around a point to the density around its neighbors. A point in a sparse region surrounded by dense regions is anomalous.

**How it works:**
1. For each point, find its k nearest neighbors
2. Compute the local reachability density (how dense the neighborhood is)
3. Compare each point's density to its neighbors' densities
4. If a point is much less dense than its neighbors, it's an outlier

**LOF scores:**
- LOF close to 1.0 means similar density to neighbors (normal)
- LOF greater than 1.0 means lower density than neighbors (possibly anomalous)
- LOF much greater than 1.0 (e.g., 2.0+) means significantly lower density (likely anomalous)

The "local" part is crucial. Imagine a dataset with two clusters: a dense cluster of 1000 points and a sparse cluster of 50 points. A point at the edge of the sparse cluster isn't globally unusual — it has 50 neighbors. But if its immediate neighbors are denser than it is, it's locally unusual. LOF captures this nuance that global methods miss.

**Strengths:** Detects local anomalies (points unusual in their neighborhood, even if not globally unusual). Works on clusters with different densities.

**Weaknesses:** Slow on large datasets (naive implementation is O(n^2)). Sensitive to the choice of k. Doesn't work well in very high dimensions (curse of dimensionality affects distance computation).

### Comparison

| Method | Assumptions | Speed | Handles High Dimensions | Detects Local Anomalies |
|--------|------------|-------|-------------------|------------------------|
| Z-score | Normal distribution | Very fast | Yes (per-feature) | No |
| IQR | None (per-feature) | Very fast | Yes (per-feature) | No |
| Isolation Forest | None | Fast | Yes | Partially |
| LOF | Distances are meaningful | Slow | Poor | Yes |

### Evaluation Challenges

Evaluating anomaly detectors is harder than evaluating classifiers:

- **Extreme class imbalance.** At 0.1% anomalies, predicting "normal" for everything achieves 99.9% accuracy. Accuracy is useless.
- **AUROC can mislead.** Under heavy imbalance, AUROC can look good even when the model misses most anomalies at practical thresholds.
- **Better metrics:** Precision@k (of the top k flagged items, how many are true anomalies), AUPRC (area under the precision-recall curve), recall at a fixed false positive rate.

```mermaid
flowchart LR
    A[Raw Data] --> B[Train on Normal Data Only]
    B --> C[Score All Test Data]
    C --> D[Rank by Anomaly Score]
    D --> E[Evaluate Top K Flagged Items]
    E --> F[Precision at K / AUPRC]

    style A fill:#f9f,stroke:#333
    style F fill:#9f9,stroke:#333
```

### Anomaly Detection Pipeline

In practice, anomaly detection follows this workflow:

1. **Collect baseline data.** Ideally, a period you know is free of (or has very few) anomalies.
2. **Feature engineering.** Raw features plus derived features (rolling statistics, time features, ratios).
3. **Train detector.** Fit on baseline data. The model learns what "normal" looks like.
4. **Score new data.** Each new observation gets an anomaly score.
5. **Threshold selection.** Pick a score cutoff. This is a business decision: higher thresholds mean fewer false alarms but more missed anomalies.
6. **Alert and investigate.** Flagged points go to human review or automated response.
7. **Collect feedback.** Record whether flagged items were true anomalies or false positives. Use this to evaluate the detector and adjust thresholds over time.

This pipeline is never "done." Data distributions drift, new anomaly types emerge, and thresholds need adjustment. Treat anomaly detection as a living system, not a one-time model.

## Build It

The code in `code/anomaly_detection.py` implements Z-score, IQR, and Isolation Forest from scratch.

### Z-Score Detector

```python
def zscore_detect(X, threshold=3.0):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    z = np.abs((X - mean) / std)
    return z.max(axis=1) > threshold
```

Simple and vectorized. Any feature exceeding the threshold flags a point.

### IQR Detector

```python
def iqr_detect(X, factor=1.5):
    q1 = np.percentile(X, 25, axis=0)
    q3 = np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1.0
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    outside = (X < lower) | (X > upper)
    return outside.any(axis=1)
```

### Isolation Forest from Scratch

The from-scratch implementation builds isolation trees that randomly partition the feature space:

```python
class IsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, depth=0):
        n, p = X.shape
        if depth >= self.max_depth or n <= 1:
            self.is_leaf = True
            self.size = n
            return self
        self.is_leaf = False
        self.feature = np.random.randint(p)
        x_min = X[:, self.feature].min()
        x_max = X[:, self.feature].max()
        if x_min == x_max:
            self.is_leaf = True
            self.size = n
            return self
        self.threshold = np.random.uniform(x_min, x_max)
        left_mask = X[:, self.feature] < self.threshold
        self.left = IsolationTree(self.max_depth).fit(X[left_mask], depth + 1)
        self.right = IsolationTree(self.max_depth).fit(X[~left_mask], depth + 1)
        return self
```

The path length to isolate a point determines its anomaly score. Shorter paths mean more anomalous.

The `IsolationForest` class wraps multiple trees:

```python
class IsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, seed=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples

    def fit(self, X):
        sample_size = min(self.max_samples, X.shape[0])
        max_depth = int(np.ceil(np.log2(sample_size)))
        for _ in range(self.n_estimators):
            idx = rng.choice(X.shape[0], size=sample_size, replace=False)
            tree = IsolationTree(max_depth=max_depth)
            tree.fit(X[idx])
            self.trees.append(tree)

    def anomaly_score(self, X):
        avg_path = average path length across all trees
        scores = 2.0 ** (-avg_path / c(max_samples))
        return scores
```

The normalization factor `c(n)` is the expected path length of an unsuccessful search in a binary search tree with n elements. It equals `2 * H(n-1) - 2*(n-1)/n`, where `H` is the harmonic number. This normalization ensures scores are comparable across datasets of different sizes.

### Demo Scenarios

The code generates several test scenarios:

1. **Single cluster with outliers.** A 2D Gaussian cluster with injected anomalies far from center. All methods should work here.
2. **Multimodal data.** Three clusters of different sizes and densities. Points between clusters are anomalies. Z-score struggles because per-feature ranges are wide.
3. **High-dimensional data.** 50 features, but anomalies differ in only 5 of them. Tests whether methods can find anomalies in a subset of features.

Each demo compares all methods using precision, recall, F1, and Precision@k.

## Use It

With sklearn (library implementation, not from scratch):

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_train)
predictions = iso.predict(X_test)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof.fit(X_train)
predictions = lof.predict(X_test)
```

Note that `contamination` sets the expected proportion of anomalies. Setting it correctly is critical — too low misses anomalies, too high creates false positives.

The code in `anomaly_detection.py` compares the from-scratch implementation against sklearn on the same data.

### sklearn's contamination Parameter

The `contamination` parameter in sklearn determines the threshold that converts continuous anomaly scores into binary predictions. It doesn't change the underlying scores.

```python
iso_5 = IsolationForest(contamination=0.05)
iso_10 = IsolationForest(contamination=0.10)
```

Both produce the same anomaly scores. But `iso_5` flags the top 5% and `iso_10` flags the top 10%. If you don't know the true anomaly rate (you usually don't), set contamination to "auto" and work with raw scores. Set your own threshold based on the cost tradeoff between false positives and false negatives.

### One-Class SVM

Another unsupervised anomaly detector worth knowing. One-Class SVM fits a boundary around normal data in a high-dimensional feature space (using the kernel trick).

```python
from sklearn.svm import OneClassSVM

oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
oc_svm.fit(X_train)
predictions = oc_svm.predict(X_test)
```

The `nu` parameter approximately bounds the fraction of anomalies. One-Class SVM works well on small-to-medium datasets but doesn't scale to very large data (kernel matrix grows quadratically).

### Autoencoder Approach (Preview)

Autoencoders are neural networks that learn to compress and reconstruct data. Train on normal data. At test time, anomalies have high reconstruction error because the network only learned to reconstruct normal patterns.

This is covered in Phase 3 (Deep Learning), but the principle is the same: model what's normal, flag what deviates.

### Ensemble Anomaly Detection

Just as ensemble methods improve classification (Lesson 11), combining multiple anomaly detectors can improve detection. The simplest approach:

1. Run multiple detectors (Z-score, IQR, Isolation Forest, LOF)
2. Normalize each detector's scores to [0, 1]
3. Average the normalized scores
4. Flag points whose average score exceeds a threshold

This reduces false positives because different methods have different failure modes. A point flagged by all four methods is almost certainly anomalous. A point flagged by only one might be a quirk of that method.

More sophisticated ensembles weight each detector by its estimated reliability (measured on a validation set with known anomalies, if available).

### Production Considerations

1. **Threshold drift.** As data distributions shift, fixed thresholds become stale. Monitor the distribution of anomaly scores and adjust periodically.
2. **Alert fatigue.** Too many false positives and operators stop paying attention. Start with a high threshold (fewer, more reliable alerts) and lower it as trust builds.
3. **Ensemble approach.** Combine multiple detectors in production. Flag a point only when multiple methods agree it's anomalous. This significantly reduces false positives.
4. **Feature engineering.** Raw features are rarely enough. Add rolling statistics, ratios, time-since-last-event, and domain-specific features. A good feature set matters more than the choice of detector.
5. **Feedback loop.** When operators investigate flagged items and confirm or dismiss them, feed this back into the system. Accumulate labeled data over time to evaluate and improve the detector.

## Ship It

This lesson produces:
- `outputs/skill-anomaly-detector.md` -- A decision skill for choosing the right detector
- `code/anomaly_detection.py` -- From-scratch Z-score, IQR, and Isolation Forest with sklearn comparison

### Choosing a Threshold

Anomaly scores are continuous. You need a threshold to make binary decisions. This is a business decision, not a technical one.

Consider two scenarios:
- **Fraud detection.** Missing fraud is expensive (chargebacks, customer trust). A false positive costs an analyst 5 minutes of investigation. Set the threshold low to catch more fraud, accepting more false positives.
- **Equipment maintenance.** A false positive means unnecessary downtime costing $50,000. A missed failure means a $500,000 repair. Set the threshold to balance these costs.

In both cases, the optimal threshold depends on the cost ratio between false positives and false negatives. Plot precision and recall at different thresholds, overlay the cost function, and pick the point with minimum cost.

### Scaling to Production

Real-time anomaly detection in production:

1. **Batch train, online score.** Train the model periodically (daily, weekly) on recent normal data. Score each new observation as it arrives.
2. **Feature computation must match.** If you train with 30-day rolling statistics, you need 30 days of history to compute features for a new observation. Cache the required history.
3. **Score distribution monitoring.** Track the distribution of anomaly scores over time. If the median score drifts upward, either the data is changing or the model is stale.
4. **Explainability.** When flagging an anomaly, explain why. Z-score: "Feature X is 4.2 standard deviations above normal." Isolation Forest: "This point was isolated in an average of 3.1 splits (normal points take 8.5)."

## Exercises

1. **Threshold tuning.** Run the Z-score detector with thresholds from 1.0 to 5.0 in steps of 0.5. Plot precision and recall at each threshold. Where is the sweet spot for your data?

2. **Multivariate anomalies.** Create 2D data where each feature alone looks normal, but the combination is anomalous (e.g., points far from the main cluster's diagonal). Show that per-feature Z-score misses these but Isolation Forest catches them.

3. **LOF from scratch.** Implement Local Outlier Factor using k-nearest neighbors. Compare against sklearn's LocalOutlierFactor on the same data. Use k=10 and k=50 — how does the choice of k affect results?

4. **Streaming anomaly detection.** Modify the Z-score detector to work in a streaming setting: update the running mean and variance as new points arrive (Welford's online algorithm). Compare to batch Z-score on the same data.

5. **Real-world evaluation.** Take a dataset with known anomalies (e.g., Kaggle's credit card fraud). Evaluate all four methods using precision@100, precision@500, and AUPRC. Which method works best? Why?

## Key Terms

| Term | What People Say | What It Actually Is |
|------|----------------|----------------------|
| Anomaly | "Outlier, unusual point" | A data point that deviates significantly from the expected pattern of normal data |
| Point anomaly | "Single weird value" | An observation that is unusual by itself, regardless of context |
| Contextual anomaly | "Normal value, wrong context" | An observation that is unusual given its context (time, location, etc.) but may be normal in another context |
| Isolation Forest | "Find outliers with random splits" | An ensemble of random trees that isolates anomalies with fewer splits than normal points |
| Local Outlier Factor | "Compare density to neighbors" | A method that flags points whose local density is much lower than their neighbors' density |
| Z-score | "How many standard deviations from the mean" | (x - mean) / std, measuring how far a point is from center in units of standard deviation |
| IQR | "Interquartile range" | Q3 - Q1, measuring the spread of the middle 50% of data, used for robust outlier detection |
| Contamination | "Expected proportion of anomalies" | A hyperparameter that tells the detector what fraction of the data to flag as anomalous |
| Precision@k | "How many of the top k flags are real" | Precision computed only on the k most suspicious points, useful for imbalanced anomaly detection |
| AUPRC | "Area under precision-recall curve" | A metric summarizing precision-recall performance across all thresholds, better than AUROC for imbalanced data |

## Further Reading

- [Liu et al., Isolation Forest (2008)](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) -- The original Isolation Forest paper
- [Breunig et al., LOF: Identifying Density-Based Local Outliers (2000)](https://dl.acm.org/doi/10.1145/342009.335388) -- The original LOF paper
- [scikit-learn Outlier Detection docs](https://scikit-learn.org/stable/modules/outlier_detection.html) -- Overview of all sklearn anomaly detectors
- [Chandola et al., Anomaly Detection: A Survey (2009)](https://dl.acm.org/doi/10.1145/1541880.1541882) -- Comprehensive survey of anomaly detection methods
- [Goldstein and Uchida, A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms (2016)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152173) -- Empirical comparison of 10 methods on real datasets
