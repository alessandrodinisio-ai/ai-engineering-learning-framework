# ML Pipelines

> The model isn't the product — the pipeline is. The pipeline is everything from raw data to production predictions, and every step must be reproducible.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 Lesson 12 (Hyperparameter Tuning)
**Time:** ~120 minutes

## Learning Objectives

- Build an ML pipeline from scratch that chains imputation, scaling, encoding, and model training into a single reproducible object
- Identify data leakage scenarios and explain how pipelines prevent it by fitting transformers only on training data
- Construct a ColumnTransformer that applies different preprocessing to numerical and categorical features
- Implement pipeline serialization and demonstrate that the same fitted pipeline produces identical results in training and production

## The Problem

You have a notebook that loads data, imputes missing values with the median, scales features, trains a model, and prints accuracy. It works. You ship it.

A month later, someone retrains the model and gets different results. The median was computed on the entire dataset including test data (data leakage). Scaling parameters weren't saved, so inference uses different statistics. Feature engineering code was copy-pasted between training and serving, and the two copies diverged. A categorical column has new values in production that the encoder never saw.

These aren't hypothetical. They're the most common reasons ML systems fail in production. Pipelines solve all of them by packaging every transformation step into a single, ordered, reproducible object.

## The Concept

### What a Pipeline Is

A pipeline is a sequence of ordered data transformations followed by a model. Each step takes the output of the previous step as input. The entire pipeline is fitted once on training data. At inference time, the same fitted pipeline transforms new data and produces predictions.

```mermaid
flowchart LR
    A[Raw Data] --> B[Impute Missing Values]
    B --> C[Scale Numerical Features]
    C --> D[Encode Categoricals]
    D --> E[Train Model]
    E --> F[Predictions]
```

The pipeline guarantees:
- Transformations are fitted only on training data (no leakage)
- The same transformations are applied at inference
- The entire object can be serialized and deployed as a single artifact
- Cross-validation applies the pipeline to each fold, preventing subtle leakage

### Data Leakage: The Silent Killer

Data leakage happens when information from the test set or future data contaminates training. Pipelines prevent the most common forms.

**With leakage (wrong):**
```python
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = X_scaled[:800], X_scaled[800:]
y_train, y_test = y[:800], y[800:]
```

The scaler saw the test data. The mean and standard deviation include test samples. This inflates accuracy estimates.

**Correct:**
```python
X_train, X_test = X[:800], X[800:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

With a pipeline, you don't have to worry about this at all. The pipeline handles it automatically.

### sklearn Pipeline

sklearn's `Pipeline` chains transformers and an estimator. It exposes `.fit()`, `.predict()`, and `.score()`, applying all steps in order.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])

pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

When you call `pipe.fit(X_train, y_train)`:
1. The scaler calls `fit_transform` on X_train
2. The model calls `fit` on the scaled X_train

When you call `pipe.predict(X_test)`:
1. The scaler calls `transform` on X_test (not fit_transform)
2. The model calls `predict` on the scaled X_test

The scaler never sees the test data during fitting. That's the key.

### ColumnTransformer: Different Pipelines for Different Columns

Real datasets have numerical and categorical columns that need different preprocessing. `ColumnTransformer` handles this.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["age", "income", "score"]),
    ("cat", categorical_pipe, ["city", "gender", "plan"]),
])

full_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", GradientBoostingClassifier()),
])
```

The `handle_unknown="ignore"` in OneHotEncoder is critical for production. When a new category appears (a city the model never saw), it produces a zero vector instead of crashing.

### Experiment Tracking

Pipelines make training reproducible, but you also need to track what happened across experiments: which hyperparameters were used, which dataset version, what the metrics were, which code was run.

**MLflow** is the most common open-source solution:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)

    pipe.fit(X_train, y_train)
    accuracy = pipe.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipe, "model")
```

Every run is logged with parameters, metrics, artifacts, and the full model. You can compare runs, reproduce any experiment, and deploy any model version.

**Weights & Biases (wandb)** offers the same functionality plus a hosted dashboard:

```python
import wandb

wandb.init(project="my-pipeline")
wandb.config.update({"max_depth": 5, "n_estimators": 100})

pipe.fit(X_train, y_train)
accuracy = pipe.score(X_test, y_test)

wandb.log({"accuracy": accuracy})
```

### Model Versioning

After tracking experiments, you need to manage model versions. Which model is in production? Which is in staging? Which was last week's?

MLflow's Model Registry provides:
- **Version tracking:** Each saved model gets a version number
- **Stage transitions:** "Staging", "Production", "Archived"
- **Approval workflows:** Models must be explicitly promoted to production
- **Rollback:** Instantly switch back to a previous version

### Data Versioning with DVC

Code is versioned with git. Data should be versioned too, but git can't handle large files. DVC (Data Version Control) solves this.

```
dvc init
dvc add data/training.csv
git add data/training.csv.dvc data/.gitignore
git commit -m "Track training data"
dvc push
```

DVC stores the actual data in remote storage (S3, GCS, Azure) and keeps a small `.dvc` file in git that records the hash. When you checkout a git commit, `dvc checkout` restores the exact data that was used at that time.

This means every git commit pins both code and data. Full reproducibility.

### Reproducible Experiments

A reproducible experiment requires four things:

1. **Fixed random seeds:** Set seeds for numpy, random, and frameworks (torch, sklearn)
2. **Pinned dependencies:** requirements.txt or poetry.lock with exact versions
3. **Versioned data:** DVC or similar
4. **Configuration files:** All hyperparameters in config, not hardcoded

```python
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
```

### From Notebook to Production Pipeline

```mermaid
flowchart TD
    A[Jupyter Notebook] --> B[Extract Functions]
    B --> C[Build Pipeline Object]
    C --> D[Add Config for Hyperparameters]
    D --> E[Add Experiment Tracking]
    E --> F[Add Data Validation]
    F --> G[Add Tests]
    G --> H[Package for Deployment]

    style A fill:#fdd,stroke:#333
    style H fill:#dfd,stroke:#333
```

Typical progression path:

1. **Notebook exploration:** Rapid experiments, visualizations, feature ideas
2. **Extract functions:** Move preprocessing, feature engineering, evaluation into modules
3. **Build Pipeline:** Chain transformations into a sklearn Pipeline or custom class
4. **Configuration management:** Move all hyperparameters into YAML/JSON config
5. **Experiment tracking:** Add MLflow or wandb logging
6. **Data validation:** Check schema, distributions, and missing-value patterns before training
7. **Testing:** Unit tests for transformers, integration tests for the full pipeline
8. **Deployment:** Serialize the pipeline, wrap in an API (FastAPI, Flask), containerize

### Common Pipeline Mistakes

| Mistake | Why It's Bad | Fix |
|---------|-------------|-----|
| Fitting on all data before splitting | Data leakage | Use Pipeline with cross_val_score |
| Feature engineering outside the pipeline | Different transformations at training and serving | Put all transformations inside the Pipeline |
| Not handling unknown categories | Crashes on new values in production | OneHotEncoder(handle_unknown="ignore") |
| Hardcoded column names | Breaks when schema changes | Use column name lists from config |
| No data validation | Silently wrong predictions on bad data | Add schema checks before prediction |
| Training/serving skew | Model sees different features in production | Share one Pipeline object for training and serving |

## Build It

The code in `code/pipeline.py` builds a complete ML pipeline from scratch:

### Step 1: Custom Transformer

```python
class CustomTransformer:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.means) / self.stds

    def fit_transform(self, X):
        return self.fit(X).transform(X)
```

### Step 2: Pipeline from Scratch

```python
class PipelineFromScratch:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.fit_transform(X_current)
        name, model = self.steps[-1]
        model.fit(X_current, y)
        return self

    def predict(self, X):
        X_current = X.copy()
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        name, model = self.steps[-1]
        return model.predict(X_current)
```

### Step 3: Cross-Validation with a Pipeline

The code demonstrates how cross-validation with a pipeline prevents data leakage: the scaler is fitted independently on each fold's training data.

### Step 4: Full Production Pipeline with sklearn

A complete pipeline with `ColumnTransformer`, multiple preprocessing paths, and a model, trained with proper cross-validation and experiment logging.

## Ship It

This lesson produces:
- `outputs/prompt-ml-pipeline.md` -- A skill for building and debugging ML pipelines
- `code/pipeline.py` -- Complete pipeline from scratch to sklearn

## Exercises

1. Build a pipeline that handles a dataset with 3 numerical and 2 categorical columns. Use `ColumnTransformer` to apply median imputation + scaling on numerical columns, and mode imputation + one-hot encoding on categorical columns. Train with 5-fold cross-validation.

2. Intentionally introduce data leakage: fit the scaler on the entire dataset before splitting. Compare the cross-validation score (with leakage) to the pipeline cross-validation score (clean). How big is the difference?

3. Serialize your pipeline with `joblib.dump`. Load it in another script and run predictions. Verify the predictions are exactly identical.

4. Add a custom transformer to the pipeline that creates polynomial features (degree 2) for the two most important numerical columns. Where should it go in the pipeline?

5. Set up MLflow tracking for the pipeline. Run 5 experiments with different hyperparameters. Use the MLflow UI (`mlflow ui`) to compare runs and pick the best model.

## Key Terms

| Term | What People Say | What It Actually Is |
|------|----------------|----------------------|
| Pipeline | "Chain of transformations + model" | A sequence of fitted transformers plus a model, applied as one unit to prevent leakage |
| Data leakage | "Test info leaking into training" | Using information from outside the training set to build the model, inflating performance estimates |
| ColumnTransformer | "Different preprocessing per column" | Applies different pipelines to different column subsets and concatenates results |
| Experiment tracking | "Log your runs" | Recording parameters, metrics, artifacts, and code version for each training run |
| MLflow | "Track and deploy models" | Open-source platform for experiment tracking, model registry, and deployment |
| DVC | "Git for data" | Version control system for large data files, storing hashes in git and data in remote storage |
| Model registry | "Model version catalog" | A system that tracks model versions with stage labels (staging, production, archived) |
| Training/serving skew | "It was fine in the notebook" | Differences in data processing between training and inference that cause silent errors |
| Reproducibility | "Same code, same results" | The ability to get identical results from the same code, data, and configuration |

## Further Reading

- [scikit-learn Pipeline docs](https://scikit-learn.org/stable/modules/compose.html) -- Official pipeline reference
- [MLflow documentation](https://mlflow.org/docs/latest/index.html) -- Experiment tracking and model registry
- [DVC documentation](https://dvc.org/doc) -- Data versioning
- [Sculley et al., Hidden Technical Debt in Machine Learning Systems (2015)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) -- Seminal paper on ML system complexity
- [Google ML Best Practices: Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) -- Practical production ML advice
