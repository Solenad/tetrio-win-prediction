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
# Initializations and Imports
import pandas as pd
import numpy as np

df = pd.read_csv("./data/data.csv")
# %%

# %%
# Check
df.head()
# %%

# %%
# Data Preprocessing
unneeded_cols = ["playfield", "x", "y", "r", "next", "hold"]
present_unneeded = [col for col in unneeded_cols if col in df.columns]
df = df.drop(columns=present_unneeded)
# %%

# %%
# Data Preprocessing: Clean and filter 't_spin' column
# Standardize 't_spin' values to uppercase strings, strip whitespace,
# and filter to keep only valid entries: 'N', 'M', or 'F'.
if "t_spin" in df.columns:
    df["t_spin"] = df["t_spin"].astype(str).str.upper().str.strip()
    df = df[df["t_spin"].isin({"N", "M", "F"})]
# %%

# %%
# Data Preprocessing: Replace negative values with NaN
# For selected numeric columns, replace negative values with NaN,
# except for 't_spin'.
for col in [
    "cleared",
    "garbage_cleared",
    "attack",
    "btb",
    "combo",
    "immediate_garbage",
    "incoming_garbage",
    "rating",
    "glicko",
    "glicko_rd",
    "subframe",
]:
    if col in df.columns and col not in ["t_spin"]:
        df.loc[df[col] < 0, col] = np.nan
# %%

# %%
# Data Preprocessing: Drop rows with missing critical values
# Remove rows with NaN in any of the critical columns required for modeling.
critical_cols = [
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
    "rating",
    "glicko",
    "glicko_rd",
]
present_critical = [c for c in critical_cols if c in df.columns]
df = df.dropna(subset=present_critical)
# %%

# %%
# Data Preprocessing: Convert column types
# Convert integer columns to smallest possible integer type,
# and float columns to smallest possible float type.
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
# %%

# %%
# Data Preprocessing: Remove duplicate rows
# Remove exact duplicates, then duplicates based on 'game_id' and 'subframe'.
before_dupes = len(df)
df = df.drop_duplicates()
after_exact = len(df)
if "game_id" in df.columns and "subframe" in df.columns:
    df = df.drop_duplicates(subset=["game_id", "subframe"])
after_pair = len(df)
# %%

# %%
# Data Preprocessing: Filter valid values for 'won' and 't_spin'
# Keep only rows where 'won' is 0 or 1, and 't_spin' is one of 'N', 'M', 'F'.
if "won" in df.columns:
    df = df[df["won"].isin([0, 1])]
if "t_spin" in df.columns:
    df = df[df["t_spin"].isin(["N", "M", "F"])]
# %%

# %%
# Data Preprocessing: Outlier statistics
# Print min, max, and extreme quantiles for key numeric columns.
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
# %%

# %%
# Data Preprocessing: Remove inconsistent games
# Drop games where 'won', 'rating', 'glicko', or 'glicko_rd' are not consistent within a game.
if {"game_id", "won", "rating", "glicko", "glicko_rd"}.issubset(df.columns):
    gchecks = df.groupby("game_id")[["won", "rating", "glicko", "glicko_rd"]].nunique()
    inconsistent_games = gchecks[(gchecks > 1).any(axis=1)].index.tolist()
    print(f"Inconsistent games found: {len(inconsistent_games)}")
    df = df[~df["game_id"].isin(inconsistent_games)]
# %%

# %%
# Print summary statistics after cleaning.
print("CLEANING SUMMARY:")
print(f"Rows after cleaning: {len(df)}")
print(f"Unique games: {df['game_id'].nunique()
      if 'game_id' in df.columns else 'N/A'}")

print("Num placement rows:", len(df))
print("Num unique game_id:", df["game_id"].nunique())
print("game_id dtype:", df["game_id"].dtype)
# %%

# %%
# Aggregation
# groupby game_id
agg_dict = {
    "subframe": ["max", "count"],
    "cleared": "sum",
    "garbage_cleared": "sum",
    "attack": "sum",
    "t_spin": lambda x: (x == "M").sum() + (x == "F").sum(),
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

game_level.columns = ["_".join([c for c in col if c]) for col in game_level.columns]
game_level = game_level.reset_index()

game_level["duration_sec"] = game_level["subframe_max"] / 600
game_level["pps"] = game_level["subframe_count"] / game_level["duration_sec"]
game_level["attack_per_piece"] = game_level["attack_sum"] / game_level["subframe_count"]
game_level["attack_per_sec"] = game_level["attack_sum"] / game_level["duration_sec"]
game_level["tspin_rate"] = game_level["t_spin_<lambda>"] / game_level["subframe_count"]

game_level.head()
print("Num rows in game_level:", len(game_level))
# %%

# %%
# Export
game_level.to_csv("./data/data_processed.csv", index=False)
# %%
