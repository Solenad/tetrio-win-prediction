---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: "1.3"
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# k-Nearest Neighbors (kNN) Training and Hyperparameter Tuning

This notebook demonstrates the implementation of a k-Nearest Neighbors classifier to predict Tetr.io replay outcomes. We explore the impact of the distance-based "neighborhood" on accuracy and perform cross-validation to identify the optimal _k_ value.

---

## Initialization and Data Prep

Before training a distance-based model like kNN, data consistency and scaling are paramount. We begin by importing the necessary analytical and visualization suites.

### Imports and Initializations

We utilize `sklearn` for the core model logic and evaluation metrics. `seaborn` and `matplotlib` are used to visualize the "elbow" of the hyperparameter tuning curve and the final confusion matrix.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load processed replay data
df = pd.read_csv("../data/data_processed.csv")
df.info()
df.head()
```

### Target Separation and Normalization

kNN is highly sensitive to the scale of features because it calculates the
Euclidean distance between data points. If one feature (like `rating`) has a
much larger range than another (like `pps`), it will dominate the distance calculation.

We apply Standardization (Z-score normalization) to ensure each TETR.IO metric
contributes equally to the distance metric.

```python
# Target separation
X = df.drop(columns=["won"])  # Features
y = df["won"]  # Target variable

# Normalize features to have mean=0 and std=1
# Vital for distance-based algorithms
X = (X - X.mean()) / X.std()
```

### Data Split

We split the dataset into training and testing sets to evaluate how well the
model generalizes to unseen replays. A 85%/15% split is utilized here, where
85% of the training data is validated automatically by a single fold created
by `cross_val_score`, and tested on the remaining 15%.

```python
# Split data into training and test sets
# Stratification ensures win/loss proportions are maintained
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print(f"Training data: {X_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Testing data: {X_test.shape}")
print(f"Testing labels: {X_test.shape}")
```

## Model Training

We begin by training a "pilot" model with a default value of _k=5_. This
allows us to establish a baseline performance before beginning the search for
the optimal neighborhood size. The model training is almost instantaneous due to
the nature of kNN's training pipeline.

```python
# Train initial kNN model
k_initial = 5
model = KNeighborsClassifier(n_neighbors=k_initial)
model.fit(X_train, y_train)

```

### Initial Model Performance

We can then test the current performance of the model with the set hyperparameter
of k=5.

```python
# Initial Predictions
y_pred = model.predict(X_test)
print(f"Baseline Accuracy (k=5): {accuracy_score(y_test, y_pred):.2f}")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
```

## Hyperparameter Tuning

Next is to explore the ideal hyperparameter that yields the highest accuracy
or least loss. A small _k_ such as 5 can lead to overfitting, while a high _k_
may lead to underfitting. Thus, there must be a right hyperparameter for this model.

### Identifying the best _k_

We set a reasonable range for the hyperparameter tuning, which is from 1-31. The
typical range is up to 21, but to consider the dataset and to visualize the curve
better, we add an extra 10.

```python
# Set a hyperparameter range from 1-31
k_values = range(1, 31)
cv_scores = []

# Iterate through the range to test for accuracy
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores.append(scores.mean())
```

```python
# Cross validation accuracy
plt.plot(k_values, cross_val_scores, marker="o")
plt.title("Cross-Validation Accuracy for Different k")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy")
plt.grid(True)
plt.show()
```

```python
# Best k selection
best_k = k_values[np.argmax(cross_val_scores)]
print(f"Best k: {
      best_k} with Cross-Validation Accuracy: {max(cross_val_scores):.2f}")
```

### Retraining the model

Now that the best _k_ is identified, we can retrain the model using that _k_ to
clearly visualize its performance.

```python
# Model retraining using best k
final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)

# Final model evaluation
final_y_pred = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_y_pred)

print(f"Final Model Accuracy (k={best_k}): {final_accuracy:.2f}")
print("\nFinal Classification Report:\n")
print(classification_report(y_test, final_y_pred))

# Confusion matrix for final model
final_conf_matrix = confusion_matrix(y_test, final_y_pred)
sns.heatmap(final_conf_matrix, annot=True, cmap="Blues", fmt="d")
plt.title(f"Confusion Matrix (k={best_k})")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
```
