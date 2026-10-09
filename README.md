# Task 5 — Decision Trees & Random Forests

**Elevate Lab — Machine Learning Internship**

A supervised learning project that builds, evaluates, and interprets tree-based
classification models on the **Heart Disease dataset** to predict the presence of
heart disease. The project compares a single **Decision Tree** against a
**Random Forest** ensemble, analyzes overfitting through depth tuning, and
interprets model behavior using feature importances and cross-validation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Workflow](#workflow)
5. [Prerequisites](#prerequisites)
6. [Implementation](#implementation)
7. [Results](#results)
8. [Interview Questions & Answers](#interview-questions--answers)
9. [How to Run](#how-to-run)
10. [Author](#author)

---

## Project Overview

The objective of this task is to gain hands-on experience with tree-based models
for classification. Specifically, the project:

- Trains and visualizes a Decision Tree Classifier.
- Diagnoses overfitting and controls it by tuning tree depth.
- Trains a Random Forest Classifier and compares its accuracy to the single tree.
- Interprets feature importances to understand which clinical features matter most.
- Evaluates both models using 5-fold cross-validation.

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | Heart Disease Dataset |
| **File** | `heart.csv` |
| **Source** | [Kaggle — Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| **Rows** | 1025 |
| **Features** | 13 |
| **Target** | `target` — 1 = disease, 0 = no disease |

### Features

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible) |
| `target` | 1 = disease, 0 = no disease |

---

## Project Structure

```
task5-decision-trees/
├── heart.csv                     # Dataset
├── task5.ipynb                   # Executed notebook
├── tree.png                      # Decision tree visualization
├── feature_importance.png        # Feature importance bar chart
├── depth_vs_accuracy.png         # Overfitting analysis plot
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Workflow

1. **Data Inspection** — loaded `heart.csv`, checked shape, data types, missing values, and class balance.
2. **Exploratory Data Analysis** — summary statistics and correlation heatmap.
3. **Preprocessing** — separated features (`X`) and target (`y`); performed an 80/20 stratified train/test split.
4. **Decision Tree Training** — trained an unpruned `DecisionTreeClassifier` and visualized the resulting tree.
5. **Overfitting Analysis** — swept `max_depth` from 2 to `None` and compared training vs. test accuracy.
6. **Random Forest Training** — trained a `RandomForestClassifier` with 100 estimators.
7. **Feature Importance** — extracted and plotted impurity-based importances from the forest.
8. **Cross-Validation** — performed 5-fold CV on both models.
9. **Evaluation** — compared accuracy, confusion matrices, and CV stability.

---

## Prerequisites

- Python 3.8 or higher
- Required libraries:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
graphviz
```

Install with:

```bash
pip install -r requirements.txt
```

> **Graphviz note:** To render `tree.png`, Graphviz must be installed on your system.
> - Ubuntu: `sudo apt-get install graphviz`
> - macOS: `brew install graphviz`
> - Windows: [download installer](https://graphviz.org/download/)

---

## Implementation

### 1. Load and split the data

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 2. Train a Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("Train accuracy:", dt.score(X_train, y_train))
print("Test  accuracy:", dt.score(X_test, y_test))
```

### 3. Control overfitting via `max_depth`

```python
for depth in [2, 3, 4, 5, 6, 8, 10, None]:
    m = DecisionTreeClassifier(max_depth=depth, random_state=42)
    m.fit(X_train, y_train)
    print(depth, m.score(X_train, y_train), m.score(X_test, y_test))
```

### 4. Train a Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest test accuracy:", rf.score(X_test, y_test))
```

### 5. Feature importance

```python
import pandas as pd
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp)
```

### 6. Cross-validation

```python
from sklearn.model_selection import cross_val_score

dt_cv = cross_val_score(DecisionTreeClassifier(max_depth=5, random_state=42), X, y, cv=5)
rf_cv = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=5)

print("Decision Tree CV:", dt_cv.mean(), "±", dt_cv.std())
print("Random Forest CV:", rf_cv.mean(), "±", rf_cv.std())
```

---

## Results

> Replace the bracketed values below with your actual outputs.

### Decision Tree (unpruned)

| Metric | Value |
|--------|-------|
| Training accuracy | [1.00] |
| Test accuracy | [0.98] |

The unpruned tree perfectly fits the training data — a clear sign of overfitting.

### Effect of `max_depth`

| `max_depth` | Train Accuracy | Test Accuracy |
|-------------|----------------|---------------|
| 2 | [0.79] | [0.76] |
| 3 | [0.83] | [0.81] |
| 4 | [0.87] | [0.84] |
| 5 | [0.90] | [0.86] |
| 6 | [0.93] | [0.88] |
| 8 | [0.97] | [0.90] |
| 10 | [0.99] | [0.91] |
| None | [1.00] | [0.92] |

The gap between training and test accuracy narrows as depth decreases. A depth
of **[5–6]** offers the best bias–variance trade-off.

### Model Comparison

| Model | Test Accuracy | 5-Fold CV Mean | CV Std |
|-------|---------------|----------------|--------|
| Decision Tree (`max_depth=5`) | [0.86] | [0.85] | [0.03] |
| Random Forest (100 trees) | [0.99] | [0.99] | [0.01] |

The Random Forest achieves higher accuracy with lower variance across folds.

### Top 5 Feature Importances (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | [cp] | [0.13] |
| 2 | [thal] | [0.12] |
| 3 | [ca] | [0.11] |
| 4 | [thalach] | [0.10] |
| 5 | [oldpeak] | [0.09] |

### Key Findings

- The unpruned Decision Tree overfits (training accuracy 1.00 vs. lower test accuracy).
- Limiting `max_depth` to **[5–6]** controls overfitting and improves generalization.
- The Random Forest outperforms the single tree in both accuracy and CV stability.
- **[cp, thal, ca, thalach, oldpeak]** are the strongest predictors of heart disease.

### Visualizations

- `tree.png` — full Decision Tree structure
- `feature_importance.png` — feature importance bar chart
- `depth_vs_accuracy.png` — train vs. test accuracy across depths

---

## Interview Questions & Answers

### 1. How does a decision tree work?

A decision tree recursively splits the dataset on the feature and threshold that
best separates the classes. Each internal node is a test, each branch is an
outcome, and each leaf is a prediction. The algorithm chooses splits that
maximize the reduction in impurity — measured by Gini impurity or entropy — and
stops when a stopping criterion is met (maximum depth, minimum samples per leaf,
or full purity). The result is a set of interpretable if/else rules.

### 2. What is entropy and information gain?

**Entropy** measures the impurity of a node:

```
H(S) = -Σ pᵢ · log₂(pᵢ)
```

A pure node has entropy 0; a 50/50 binary split has entropy 1.

**Information gain** is the reduction in entropy achieved by a split:

```
IG(S, A) = H(S) - Σ (|Sᵥ| / |S|) · H(Sᵥ)
```

The tree picks the split that maximizes information gain — i.e., the split that
produces the purest children.

### 3. How is random forest better than a single tree?

A single tree is high variance — small changes in data produce very different
trees, and an unpruned tree overfits easily. A Random Forest trains many trees
on bootstrap samples, considers a random subset of features at each split, and
aggregates their votes. This decorrelates the trees so their errors cancel,
yielding lower variance, higher accuracy, and more stable performance.

### 4. What is overfitting and how do you prevent it?

Overfitting occurs when a model learns the training data too well — including
its noise — and performs poorly on unseen data. Symptoms: training accuracy much
higher than test accuracy.

Prevention techniques:
- Limit complexity (`max_depth`, `min_samples_split`, `min_samples_leaf`).
- Prune with cost-complexity pruning (`ccp_alpha`).
- Use cross-validation to detect it.
- Collect more data.
- Use ensembles such as Random Forest or Bagging.
- Apply regularization where applicable.

### 5. What is bagging?

**Bagging** (Bootstrap Aggregating) is an ensemble technique that draws `N`
bootstrap samples (with replacement), trains a base model on each, and
aggregates predictions by majority vote (classification) or averaging
(regression). It reduces variance without increasing bias and works especially
well with high-variance models like decision trees. Random Forest is bagging
applied to trees with additional feature-level randomness.

### 6. How do you visualize a decision tree?

Three common ways:

1. `sklearn.tree.plot_tree` — quick matplotlib rendering.
2. `sklearn.tree.export_graphviz` + Graphviz — high-quality PNG/PDF output.
3. `dtreeviz` — modern, more readable visuals.

Each node shows the split feature and threshold, impurity, sample count, and
class distribution. A plain-text alternative is `sklearn.tree.export_text`.

### 7. How do you interpret feature importance?

Tree-based models compute impurity-based importance: the total decrease in
impurity contributed by a feature, weighted by samples reaching each node and
averaged over all trees. Higher values mean the feature was more useful for
splitting.

Alternatives: **permutation importance** (drop in accuracy when a feature is
shuffled) and **SHAP values** (per-prediction contributions).

Caveat: impurity importance is biased toward high-cardinality features and is
relative — it tells you what the model used, not what causes the outcome.

### 8. What are the pros and cons of random forests?

**Pros**
- High accuracy and low variance.
- Robust to noise and outliers.
- Handles mixed data types without scaling.
- Provides feature importance.
- Parallelizable.

**Cons**
- Less interpretable than a single tree.
- Slower to train and predict as the number of trees grows.
- Large memory footprint.
- Can still overfit on small noisy datasets.
- Biased toward categorical features with many levels.

