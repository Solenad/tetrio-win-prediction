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
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/data_processed.csv")
df.head()

# %%
# Data Overview and Quality
df.info()
df.describe(include="all")
print("Missing values per column:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# %% [markdown]
# ### What is the Distribution of Winning and Losing Games?

# %%
sns.countplot(x="won", data=df)
plt.title("Win/Loss Distribution")
plt.xlabel("Won (1=Win, 0=Loss)")
plt.ylabel("Count")
plt.show()

df["won"].value_counts(normalize=True)

# %% [markdown]
# ### What is the Distribution of Player Ratings?

# %%
plt.figure(figsize=(8, 4))
sns.histplot(df["rating"], kde=True, bins=30)
plt.title("Distribution of Player Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# ### What is the Distrbution of Game Lengths in Seconds?

# %%
plt.figure(figsize=(8, 4))
sns.histplot(df["duration_sec"], kde=True, bins=30)
plt.title("Distribution of Game Lengths (Seconds)")
plt.xlabel("Game Duration (Seconds)")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# ### What is the distribution of Pieces Per Second (PPS) and Attack Per Minute (APM)?

# %%
plt.figure(figsize=(8, 4))
sns.histplot(df["pps"], kde=True, color="blue",
             bins=30, label="Pieces per Second")
plt.title("Distribution of Pieces per Second")
plt.xlabel("PPS")
plt.ylabel("Frequency")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(
    df["apm"], kde=True, color="orange", bins=30, label="Attack per minute"
)
plt.title("Distribution of Attack per Minute")
plt.xlabel("APM")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %% [markdown]
# ### How does average Attack Per Minute (APM) differ between winning games and losing games?

# %%
plt.figure(figsize=(8, 4))
sns.boxplot(
    x="won",
    y="apm",
    data=df,
    hue="won",
    palette="Set2",
    dodge=False,
)
plt.title("Attack per Minute Comparison (Win vs Loss)")
plt.xlabel("Won (1=Win, 0=Loss)")
plt.ylabel("Attack per Minute")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# %% [markdown]
# ### Do winning games exhibit a higher average Attack Efficiency (garbage sent per piece) than losing games?

# %%
plt.figure(figsize=(8, 6))

sns.boxplot(
    x="won",
    y="attack_per_piece",
    hue="won",
    data=df,
    palette=["#FF9999", "#99CCFF"],
    legend=False
)

plt.title("Attack Efficiency: Winning vs. Losing Games", fontsize=14, pad=15)
plt.xlabel("Game Outcome (0 = Loss, 1 = Win)", fontsize=12)
plt.ylabel("Attack Efficiency (Garbage Sent per Piece)", fontsize=12)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

print(df.groupby("won")["attack_per_piece"].describe())

# %% [markdown]
# ### How does the T-Spin Rate differ between games that were won versus games that were lost?

# %%
df['won_category'] = df['won'].map({0: 'Loss', 1: 'Win'})

plt.figure(figsize=(8, 6))

sns.boxplot(
    x="won_category",
    y="tspin_rate",
    data=df,
    palette="Set2",
    hue="won_category",
    legend=False
)

plt.title("T-Spin Rate Distribution: Wins vs Losses", fontsize=14)
plt.xlabel("Match Outcome", fontsize=12)
plt.ylabel("T-Spin Rate (T-Spins per Piece Placed)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

df.drop(columns=['won_category'], inplace=True)

# %% [markdown]
# ### How does the average pressure faced (Incoming Garbage queue) compare between winning and losing games?

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(
    data=df,
    x="won",
    y="incoming_garbage_mean",
    hue="won",
    ax=axes[0],
    palette="Set1",
    legend=False
)
axes[0].set_title("Average Pressure Faced vs Match Outcome")
axes[0].set_xlabel("Match Outcome (0 = Loss, 1 = Win)")
axes[0].set_ylabel("Mean Incoming Garbage (Rows)")

sns.boxplot(
    data=df,
    x="won",
    y="incoming_garbage_max",
    hue="won",
    ax=axes[1],
    palette="Set2",
    legend=False
)
axes[1].set_title("Peak Pressure Faced vs Match Outcome")
axes[1].set_xlabel("Match Outcome (0 = Loss, 1 = Win)")
axes[1].set_ylabel("Max Incoming Garbage (Rows)")

plt.tight_layout()

plt.show()

# %% [markdown]
# ### Is there a linear relationship between a player's raw speed (PPS) and their damage output (APM)?

# %%

corr, p_value = pearsonr(df['pps'], df['apm'])
print(f"Pearson Correlation Coefficient (r): {corr:.3f}")
print(f"P-value: {p_value:.3e}")

plt.figure(figsize=(10, 6))

sns.regplot(
    data=df,
    x='pps',
    y='apm',
    scatter_kws={'alpha': 0.3, 'color': 'dodgerblue', 's': 20},
    line_kws={'color': 'red', 'linewidth': 2}
)

plt.title(f"Linear Relationship Between Speed (PPS) and Damage (APM)\nPearson r = {
          corr:.3f}", fontsize=14)
plt.xlabel("Pieces Per Second (PPS)", fontsize=12)
plt.ylabel("Attack Per Minute (APM)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Does relying heavily on maximum combo chains correlate with higher Back-to-Back (B2B) chains?

# %%
correlation = df["combo_max"].corr(df["btb_max"])
print(f"Pearson Correlation between Max Combo and Max B2B: {correlation:.4f}")

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="combo_max",
    y="btb_max",
    hue="won",
    alpha=0.5,
    palette="coolwarm"
)

sns.regplot(
    data=df,
    x="combo_max",
    y="btb_max",
    scatter=False,
    color='black',
    line_kws={"linestyle": "--"}
)

plt.title(f"Maximum Combo vs. Maximum Back-to-Back (B2B) Chains\nCorrelation: {
          correlation:.2f}", fontsize=14)
plt.xlabel("Maximum Combo Chain", fontsize=12)
plt.ylabel("Maximum Back-to-Back (B2B) Chain", fontsize=12)
plt.legend(title="Won", labels=["Loss (0)", "Win (1)"])
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

# %% [markdown]
# ### When plotting Speed (PPS) against Attack (APM), do winning games cluster in a specific region compared to losing games?

# %%
df['Outcome'] = df['won'].map({1: 'Win', 0: 'Loss'})

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='pps',
    y='apm',
    hue='Outcome',
    palette={'Win': '#1f77b4', 'Loss': '#d62728'},
    alpha=0.3,
    s=15
)

plt.title('Pieces Per Second (PPS) vs Attack Per Minute (APM) by Game Outcome',
          fontsize=14, fontweight='bold')
plt.xlabel('Pieces Per Second (PPS)', fontsize=12)
plt.ylabel('Attack Per Minute (APM)', fontsize=12)

plt.legend(title='Match Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

df.drop(columns=['Outcome'], inplace=True)
