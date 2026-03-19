# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Initialization
from sklearn.metrics import accuracy_score
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from logreg import fit_minibatch_sgd
import sys

sys.path.append("C:/Users/Rohann/Documents/Github/stintsy-mco/src")


df = pd.read_csv("../data/data_processed.csv")

df.info()
df.head()

# %%
# Target separation
X = df.drop(columns=["won"])  # Features
y = df["won"]  # Target variable

# Normalize features
X = (X - X.mean()) / X.std()

# %%
# Split data into training, validation, and test
# 0.1765 ≈ 0.15 / 0.85 to get 15% of original data for validation
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, stratify=y_temp, random_state=42
)

# Check sizes of splits
print(f"Training data: {X_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Validation data: {X_val.shape}")
print(f"Validation labels: {y_val.shape}")
print(f"Testing data: {X_test.shape}")
print(f"Testing labels: {y_test.shape}")

# %%
# Logistic Regression
# Instantiate SDGClassifier and fit model
epochs = 200
batch_size = 32
random_state = 1
logreg_model = SGDClassifier(
    loss="log_loss",
    eta0=0.001,
    max_iter=epochs,
    learning_rate="constant",
    random_state=random_state,
    verbose=0,
)
logreg_model = fit_minibatch_sgd(
    logreg_model,
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    random_state=1,
)

# %%
# Test model on training set
y_pred_train = logreg_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {train_accuracy:.4f}")
print("\nClassification Report (Train Set):\n")
print(classification_report(y_train, y_pred_train))

# %%
# Test model on validation set
y_pred_val = logreg_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")
print("\nClassification Report (Validation Set):\n")
print(classification_report(y_val, y_pred_val))

# %%
# Perform hyperparameter tuning


# %%
# Test model on test data
y_pred_test = logreg_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("\nClassification Report (Test Set):\n")
print(classification_report(y_test, y_pred_test))

# %%
# Test data confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (Test Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# %%
# Cross-validation
# Evaluate the model using 10-fold cross-validation
# 10-fold was used as a common default, 5-fold may cause higher variance
cv_scores = cross_val_score(logreg_model, X, y, cv=10, scoring="accuracy")

print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")

# %%
# Cross-validation visualization
plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker="o")
plt.title("Cross-Validation Accuracy per Fold")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# %%
# Predictions on test set
# SGDClassifier supports the `predict_proba` method for estimating probabilities
y_proba = logreg_model.predict_proba(X_test)

# Display head predictions with probabilities
print("Predicted Probabilities (First 5 Test Examples):\n")
for i, (prob_0, prob_1) in enumerate(y_proba[:5]):
    print(f"Example {i+1}: P(Loss)={prob_0:.4f}, P(Win)={prob_1:.4f}")

# %%
# Feature impact
# SGDClassifier stores weights in coef_
feature_weights = logreg_model.coef_[0]

# Map weights to feature names for better interpretability
feature_importances = pd.Series(data=feature_weights, index=X.columns).sort_values(
    ascending=False
)

# Display top and bottom weights (most and least impactful features)
print("Top 5 Positive Features (Increase P(Win)):\n")
print(feature_importances.head())

print("\nTop 5 Negative Features (Increase P(Loss)):\n")
print(feature_importances.tail())

# %%
# Feature impact visualization
plt.figure(figsize=(10, 6))
feature_importances.sort_values().plot(kind="barh", color="steelblue")
plt.title("Feature Importances (Weights from Logistic Regression)")
plt.xlabel("Weight Value (Impact on P(Win))")
plt.ylabel("Features")
plt.show()
