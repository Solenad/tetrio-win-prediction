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
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
# Initializations and Imports
import pandas as pd

df = pd.read_csv("./data/data.csv")
df.head()

# %%
# Removing Unnecessary Variables
unneeded_cols = ["playfield", "x", "y", "r", "next", "hold", "placed"]
present_unneeded = [col for col in unneeded_cols if col in df.columns]
df = df.drop(columns=present_unneeded)

df.head()

# %%
# Checking for Multiple Representations
df["won"].value_counts()
df["t_spin"].value_counts()

# %%
# Checking For Missing Data
df.isnull().any()

# %%
# Checking for Negative Values
numeric_cols = df.select_dtypes(include=["number"])
negative_counts = (numeric_cols < 0).sum()
print(negative_counts)

# %%
# Checking for Incorrect Datatypes
df.dtypes

# %%
# Data Preprocessing: Convert column types
int_cols = [
    "won",
    "game_id",
    "subframe",
    "cleared",
    "garbage_cleared",
    "attack",
    "btb",
    "combo",
    "immediate_garbage",
    "incoming_garbage",
]
float_cols = ["rating", "glicko", "glicko_rd"]
for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast="integer", errors="coerce")
for col in float_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")

# state that this is for optimization purposes
df.dtypes

# %%
# Checking for Default Values and Inconsistent Formatting
for col in df:
    print(f"{col}: {df[col].unique()}")
    print("")

# %%
# Checking for Duplicate Data
df.duplicated().any()

df = df.drop_duplicates(subset=["game_id", "subframe"])
df.shape

# %%
# Checking for Inconsistent Games
if {"game_id", "won", "rating", "glicko", "glicko_rd"}.issubset(df.columns):
    gchecks = df.groupby("game_id")[
        ["won", "rating", "glicko", "glicko_rd"]].nunique()
    inconsistent_games = gchecks[(gchecks > 1).any(axis=1)].index.tolist()
    print(f"Inconsistent games found: {len(inconsistent_games)}")
    df = df[~df["game_id"].isin(inconsistent_games)]

# %%
# Checking for Outliers
num_cols = [
    "cleared",
    "garbage_cleared",
    "attack",
    "btb",
    "combo",
    "immediate_garbage",
    "incoming_garbage",
]
for col in num_cols:
    if col in df.columns:
        high = df[col].quantile(0.999)
        low = df[col].quantile(0.001)
        print(f"{col}: min={df[col].min()}, max={
              df[col].max()}, 0.1%={low}, 99.9%={high}")

num_cols = [
    "cleared",
    "garbage_cleared",
    "attack",
    "btb",
    "combo",
    "immediate_garbage",
    "incoming_garbage",
]

for col in num_cols:
    if col in df.columns:
        top_values = df[col].value_counts().nlargest(5).to_dict()

        max_val = df[col].max()

        print(f"--- {col} ---")
        print(f"Max Value: {max_val}")
        print(f"Top 5 most common values: {top_values}\n")

# %%
# Feature Engineering
print("CLEANING SUMMARY:")
print(f"Rows after cleaning: {len(df)}")
print(f"Unique games: {df['game_id'].nunique()
      if 'game_id' in df.columns else 'N/A'}")

print("Num placement rows:", len(df))
print("Num unique game_id:", df["game_id"].nunique())
print("game_id dtype:", df["game_id"].dtype)

# %%
# Aggregation
agg_dict = {
    "subframe": ["max", "count"],
    "cleared": "sum",
    "garbage_cleared": "sum",
    "attack": "sum",
    "t_spin": lambda x: x.isin(["M", "F"]).sum(),
    "btb": ["mean", "max"],
    "combo": ["mean", "max"],
    "immediate_garbage": ["mean", "max"],
    "incoming_garbage": ["mean", "max"],
    "won": "first",
    "rating": "first",
    "glicko": "first",
    "glicko_rd": "first",
}

game_level = df.groupby("game_id").agg(agg_dict)

game_level.columns = ["_".join([c for c in col if c])
                      for col in game_level.columns]
game_level = game_level.reset_index()

rename_map = {
    "t_spin_<lambda>": "t_spin_count",
    "won_first": "won",
    "rating_first": "rating",
    "glicko_first": "glicko",
    "glicko_rd_first": "glicko_rd",
}
game_level = game_level.rename(columns=rename_map)

game_level["duration_sec"] = game_level["subframe_max"] / 600
game_level["pps"] = game_level["subframe_count"] / game_level["duration_sec"]
game_level["attack_per_piece"] = game_level["attack_sum"] / \
    game_level["subframe_count"]
game_level["apm"] = (game_level["attack_sum"] /
                     game_level["duration_sec"]) * 60
game_level["tspin_rate"] = game_level["t_spin_count"] / \
    game_level["subframe_count"]

print("Num rows in game_level:", len(game_level))

for col in game_level:
    print(f"{col}: {game_level[col].unique()}")
    print("")


# %%
# Further Preprocessing
game_level2 = game_level[game_level["duration_sec"] >= 10.0].copy()
game_level2.shape

# %%
# Finalizing
cols_to_drop = [
    "game_id",
    "subframe_max",
    "subframe_count",
    "cleared_sum",
    "garbage_cleared_sum",
    "attack_sum",
    "t_spin_count",
]

final_df = game_level.drop(columns=cols_to_drop)
final_df.head()

# Export to csv
final_df.to_csv("./data/data_processed.csv", index=False)
# %%
