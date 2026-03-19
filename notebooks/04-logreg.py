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

from scipy.stats import loguniform
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import comet_ml.integration.sklearn
import comet_ml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.logreg import MinibatchSGDWrapper
load_dotenv()

api_key = os.getenv("COMET_API_KEY")
comet_ml.login(api_key=api_key)
exp = comet_ml.start(project_name="logreg-hyperparam-tuning")
exp.set_name("LogReg Hyperparam Tuning")
exp.add_tag("logreg")

df = pd.read_csv("../data/data_processed.csv")

df.info()
df.head()

# %%
# Target separation
X = df.drop(columns=["won"])  # Features
y = df["won"]  # Target variable

# %%
# Split data into training and validation
# 85% / 15% split
# RandomizedSearchCV will automatically do a 15% split for validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Check sizes of splits
print(f"Training data: {X_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Testing data: {X_test.shape}")
print(f"Testing labels: {y_test.shape}")

# %%
# Run pilot model to test optimal epochs
pilot_model = MinibatchSGDWrapper(eta0=0.001, epochs=150, batch_size=64)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Run the training
print("Training pilot model...")
pilot_model.fit(X_train_scaled, y_train)
print("Done!")

# %%
# Check learning curve
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(pilot_model.loss_history_) + 1),
    pilot_model.loss_history_,
    color="firebrick",
    linewidth=2,
)

plt.title("Learning Curve: Loss vs. Epochs", fontsize=14)
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Log Loss (Training)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()

print(f"Final Loss at Epoch 300: {pilot_model.loss_history_[-1]:.4f}")
print(f"Loss at Epoch 130: {pilot_model.loss_history_[129]:.4f}")

# %%
# Perform hyperparameter tuning
# Instantiate pipeline with SGDWrapper
pipeline = Pipeline([("scaler", StandardScaler()), ("logreg", MinibatchSGDWrapper())])

# Define hyperparameter ranges
param_dist = {
    "logreg__eta0": loguniform(1e-4, 1e-1),
    # 
    "logreg__batch_size": [32, 64, 128],
    # Set best epochs
    "logreg__epochs": [50],
}

# RandomizedSearchCV is more robust than manual iterations as the former runs jobs in parallel
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

# %%
# Tune hyperparameter via random_search
random_search.fit(X_train, y_train)

# %%
# Log best model
best_model = random_search.best_estimator_

print(f"Best Params: {random_search.best_params_}")
print(f"Best CV Accuracy: {random_search.best_score_:.4f}")


# %%
# Load results into a df
results_df = pd.DataFrame(random_search.cv_results_)
results_df["param_logreg__eta0"] = pd.to_numeric(results_df["param_logreg__eta0"])
results_df["param_logreg__epochs"] = pd.to_numeric(results_df["param_logreg__epochs"])
results_df["param_logreg__batch_size"] = pd.to_numeric(
    results_df["param_logreg__batch_size"]
)

# %%
# Impact of adjusting the learning rate on accruacy
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=results_df,
    x="param_logreg__eta0",
    y="mean_test_score",
    hue="param_logreg__batch_size",
    palette="viridis",
    size=200,
    legend=False,
)
sns.regplot(
    data=results_df,
    x="param_logreg__eta0",
    y="mean_test_score",
    scatter=False,
    order=1,  # Uses polynomial regression instead of LOWESS
    color="red",
    line_kws={"linestyle": "--", "linewidth": 2},
)

plt.xscale("log")
plt.title("Impact of Learning Rate on Accuracy")
plt.xlabel("Initial Learning Rate (eta0)")
plt.ylabel("Mean Test Accuracy")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# %%
# Impact of batch size on accuracy
plt.figure(figsize=(8, 6))

sns.lineplot(
    data=results_df,
    x="param_logreg__batch_size",
    y="mean_test_score",
    marker="o",
    color="royalblue",
    errorbar="sd",  # Shows the standard deviation as a shaded band
)

# Forces the x-axis to only show your exact batch sizes
plt.xticks([32, 64, 128])
plt.title("Impact of Batch Size on Accuracy")
plt.xlabel("Batch Size")
plt.ylabel("Mean Test Accuracy")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# %%
# Test best model on test data
y_pred_test = best_model.predict(X_test)
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
# Predictions on test set
# SGDClassifier supports the `predict_proba` method for estimating probabilities
y_proba = best_model.predict_proba(X_test)

# Display head predictions with probabilities
print("Predicted Probabilities (First 5 Test Examples):\n")
for i, (prob_0, prob_1) in enumerate(y_proba[:5]):
    print(f"Example {i+1}: P(Loss)={prob_0:.4f}, P(Win)={prob_1:.4f}")

# %%
# Feature impact
inner_model = best_model.named_steps["logreg"].model
feature_weights = inner_model.coef_[0]

# Map weights to feature names
feature_importances = pd.Series(data=feature_weights, index=X.columns).sort_values(
    ascending=False
)

print("Top 5 Positive Features (Increase P(Win)):\n")
print(feature_importances.head())

# %%
# Feature impact visualization
plt.figure(figsize=(10, 6))
feature_importances.sort_values().plot(kind="barh", color="steelblue")
plt.title("Feature Importances (Weights from Logistic Regression)")
plt.xlabel("Weight Value (Impact on P(Win))")
plt.ylabel("Features")
plt.show()
