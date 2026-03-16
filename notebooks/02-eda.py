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

# %%
# Data Overview and Quality
df.info()
df.describe(include="all")
print("Missing values per column:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
# %%

# %%
# Distribution
sns.countplot(x="won_first", data=df)
plt.title("Win/Loss Distribution")
plt.xlabel("Won (1=Win, 0=Loss)")
plt.ylabel("Count")
plt.show()

df["won_first"].value_counts(normalize=True)
# %%
print(test)
