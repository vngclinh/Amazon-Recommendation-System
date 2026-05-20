"""
03_has_sampler.py
Heterogeneity-Aware Sampler: apply different sampling strategies per sparsity type.
CONTRIBUTION: HAS sampling pipeline

Reads  : has_recsys/data/train_with_category.parquet  (has pseudo_category column)
Reads  : has_recsys/results/category_labels.csv
Writes : has_recsys/data/train_has.parquet
         has_recsys/results/has_sampling_log.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path("has_recsys/data")
RESULTS_DIR = Path("has_recsys/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COL_USER   = "reviewerID"
COL_ITEM   = "asin"
COL_RATING = "overall"

# ── Load ────────────────────────────────────────────────────────
# Load train_with_category.parquet (saved by 02_sparsity_diagnosis.py).
# This file contains the pseudo_category column needed for per-type sampling.
train = pd.read_parquet(DATA_DIR / "train_with_category.parquet")
if "pseudo_category" not in train.columns:
    raise ValueError(
        "pseudo_category column missing. "
        "Run 02_sparsity_diagnosis.py first — it saves train_with_category.parquet."
    )

cat_labels = pd.read_csv(RESULTS_DIR / "category_labels.csv")
type_map   = dict(zip(cat_labels["category"], cat_labels["type_flag"]))

print("=== Before HAS ===")
print(f"Total rows: {len(train):,}")
print(f"\nReviews per pseudo-category:")
print(train["pseudo_category"].value_counts().to_string())

# ── HAS Strategies ──────────────────────────────────────────────

def apply_type_A(df_cat: pd.DataFrame, n_cat: int, all_n: int) -> pd.DataFrame:
    """
    Type A: Category-sparse → oversample the category.
    Oversampling weight is inversely proportional to log(n_reviews).
    """
    weight      = 1.0 / np.log1p(n_cat)
    target_ratio = weight / np.log1p(all_n)
    target_n    = max(n_cat, int(n_cat * (1 + target_ratio * 5)))
    if target_n > n_cat:
        extra = df_cat.sample(n=target_n - n_cat, replace=True, random_state=42)
        return pd.concat([df_cat, extra], ignore_index=True)
    return df_cat


def apply_type_B(df_cat: pd.DataFrame) -> pd.DataFrame:
    """
    Type B: Item long-tail → popularity-inverse sampling.
    Each interaction is weighted by 1/sqrt(item_review_count).
    Sample with replacement keeping the same number of rows.

    Bug 5 fix: use .merge() instead of .join() to attach item_n.
    pandas .join(on=col) can behave unexpectedly with Series; .merge() is explicit.
    """
    item_counts = (
        df_cat[COL_ITEM]
        .value_counts()
        .rename("item_n")
        .reset_index()
    )
    item_counts.columns = [COL_ITEM, "item_n"]

    df_cat = df_cat.merge(item_counts, on=COL_ITEM, how="left")
    df_cat["sample_weight"] = 1.0 / np.sqrt(df_cat["item_n"].clip(lower=1))
    df_cat["sample_weight"] /= df_cat["sample_weight"].sum()

    sampled = df_cat.sample(
        n=len(df_cat), replace=True,
        weights="sample_weight", random_state=42,
    )
    return sampled.drop(columns=["item_n", "sample_weight"])


def apply_type_C(df_cat: pd.DataFrame, K: int = 3, threshold: int = 5) -> pd.DataFrame:
    """
    Type C: User cold-start → duplicate cold-user rows K times.
    Cold user = fewer than `threshold` reviews in this pseudo-category.
    """
    user_counts = df_cat[COL_USER].value_counts()
    cold_users  = user_counts[user_counts < threshold].index
    cold_rows   = df_cat[df_cat[COL_USER].isin(cold_users)]
    if len(cold_rows) == 0:
        return df_cat
    duplicated = pd.concat([cold_rows] * (K - 1), ignore_index=True)
    return pd.concat([df_cat, duplicated], ignore_index=True)


# ── Apply HAS per category ──────────────────────────────────────
all_n = len(train)
sampled_parts = []
log_rows = []

for cat, flag in type_map.items():
    df_cat   = train[train["pseudo_category"] == cat].copy()
    n_before = len(df_cat)

    if "A" in flag:
        df_cat = apply_type_A(df_cat, n_before, all_n)
    if "B" in flag:
        df_cat = apply_type_B(df_cat)
    if "C" in flag:
        df_cat = apply_type_C(df_cat)

    n_after = len(df_cat)
    sampled_parts.append(df_cat)
    log_rows.append({
        "category": cat,
        "type_flag": flag,
        "n_before": n_before,
        "n_after" : n_after,
        "ratio"   : round(n_after / max(n_before, 1), 3),
    })
    print(f"  [{flag:6s}] {cat:22s}: {n_before:>7,} → {n_after:>7,}")

train_has = pd.concat(sampled_parts, ignore_index=True)

# Shuffle to prevent the model from learning row order
train_has = train_has.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n=== After HAS ===")
print(f"Total rows: {len(train_has):,}  (×{len(train_has)/len(train):.2f} original)")

# ── Save ────────────────────────────────────────────────────────
train_has.to_parquet(DATA_DIR / "train_has.parquet", compression="zstd", index=False)

log_df = pd.DataFrame(log_rows)
log_df.to_csv(RESULTS_DIR / "has_sampling_log.csv", index=False)

print("\n✓ train_has.parquet saved to", DATA_DIR)
print("✓ has_sampling_log.csv saved to", RESULTS_DIR)
print("Stage 3 complete")

input("\nPress Enter to continue to 04_train_eval.py, or Ctrl+C to stop...")
