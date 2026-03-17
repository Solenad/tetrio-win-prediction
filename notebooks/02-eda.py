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
# Initializations
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./data/data_processed.csv")
df.head()

# %%
# Data Overview and Quality
df.info()
df.describe(include="all")
print("Missing values per column:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# %%
# Distribution
sns.countplot(x="won_first", data=df)
plt.title("Win/Loss Distribution")
plt.xlabel("Won (1=Win, 0=Loss)")
plt.ylabel("Count")
plt.show()

df["won_first"].value_counts(normalize=True)

# %%
# Distribution of player ratings
plt.figure(figsize=(8, 4))
sns.histplot(df["rating_first"], kde=True, bins=30)
plt.title("Distribution of Player Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# %%
# Distribution of game lengths in seconds
plt.figure(figsize=(8, 4))
sns.histplot(df["duration_sec"], kde=True, bins=30)
plt.title("Distribution of Game Lengths (Seconds)")
plt.xlabel("Game Duration (Seconds)")
plt.ylabel("Frequency")
plt.show()

# %%
# Distribution of pieces per second (pps) and attack per second
plt.figure(figsize=(8, 4))
sns.histplot(df["pps"], kde=True, color="blue", bins=30, label="Pieces per Second")
plt.title("Distribution of Pieces per Second")
plt.xlabel("PPS")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(
    df["attack_per_sec"], kde=True, color="orange", bins=30, label="Attack per Second"
)
plt.title("Distribution of Attack per Second")
plt.xlabel("Attack/sec")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %%
# Average attack per minute: Compare between winning and losing games
plt.figure(figsize=(8, 4))
sns.boxplot(
    x="won_first",
    y="attack_per_sec",
    data=df,
    hue="won_first",
    palette="Set2",
    dodge=False,
)
plt.title("Attack per Second Comparison (Win vs Loss)")
plt.xlabel("Won (1=Win, 0=Loss)")
plt.ylabel("Attack per Second")
plt.show()
