"""
02_sparsity_diagnosis.py
Compute 3 sparsity metrics per pseudo-category (brand-based), assign Type A/B/C labels.
CONTRIBUTION: Heterogeneity-Aware Sparsity Diagnosis

Saves train_with_category.parquet so 03_has_sampler.py can access pseudo_category.
"""
import re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from pathlib import Path

DATA_DIR    = Path("has_recsys/data")
RESULTS_DIR = Path("has_recsys/results")
FIG_DIR     = Path("has_recsys/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

COL_USER   = "reviewerID"
COL_ITEM   = "asin"
COL_RATING = "overall"

# HAS hyperparameters — justify in report
THRESHOLD_A = 0.005   # density < this  → Type A (category-sparse)
THRESHOLD_B = 0.60    # Gini > this     → Type B (item long-tail)
THRESHOLD_C = 0.50    # cold_ratio > this → Type C (user cold-start)
TOP_N_BRANDS = 20     # number of top brands kept as individual pseudo-categories

# ── Load train ──────────────────────────────────────────────────
train = pd.read_parquet(DATA_DIR / "train.parquet")
print(f"Train loaded: {len(train):,} rows")

# ── Load brand info from raw parquet files ──────────────────────
# train.parquet only has [reviewerID, asin, overall, review_date].
# Brand lives in the original raw files; we re-read only [asin, brand].
raw_dir = DATA_DIR / "raw" / "electronics"
parquet_files = sorted(raw_dir.rglob("*.parquet"))
print(f"Reading brand info from {len(parquet_files)} raw files...")

brand_chunks = []
for path in parquet_files:
    if not re.search(r'overall=\d', str(path)):
        continue
    try:
        chunk = pq.ParquetFile(path).read(columns=[COL_ITEM, "brand"]).to_pandas()
        brand_chunks.append(chunk)
    except Exception as e:
        print(f"  WARN: could not read brand from {path.name}: {e}")

brand_df = pd.concat(brand_chunks, ignore_index=True)
brand_df = brand_df.drop_duplicates(COL_ITEM)[[COL_ITEM, "brand"]]
print(f"Unique items with brand info: {len(brand_df):,}")

# ── Build pseudo_category from top brands ───────────────────────
# Join brand onto train, keep top-N brands as individual categories
train_tmp = train.merge(brand_df, on=COL_ITEM, how="left")
top_brands = (
    train_tmp["brand"]
    .value_counts()
    .head(TOP_N_BRANDS)
    .index
)
train_tmp["pseudo_category"] = train_tmp["brand"].where(
    train_tmp["brand"].isin(top_brands), other="other_brands"
)
train_tmp["pseudo_category"] = train_tmp["pseudo_category"].fillna("unknown")

categories = train_tmp["pseudo_category"].unique()
print(f"\nPseudo-categories: {len(categories)}")
print(f"Distribution (top 10):\n{train_tmp['pseudo_category'].value_counts().head(10)}")

# ── Compute 3 sparsity metrics per category ─────────────────────

def compute_gini(counts: np.ndarray) -> float:
    """Gini index of review-count distribution per item."""
    x = np.sort(counts)
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)

rows = []
for cat in categories:
    sub      = train_tmp[train_tmp["pseudo_category"] == cat]
    n_reviews = len(sub)
    n_users   = sub[COL_USER].nunique()
    n_items   = sub[COL_ITEM].nunique()

    density    = n_reviews / max(n_users * n_items, 1)
    item_counts = sub.groupby(COL_ITEM)[COL_RATING].count().values
    gini        = compute_gini(item_counts)
    user_counts = sub.groupby(COL_USER)[COL_RATING].count()
    cold_ratio  = float((user_counts < 5).mean())

    flags = []
    if density    < THRESHOLD_A: flags.append("A")
    if gini       > THRESHOLD_B: flags.append("B")
    if cold_ratio > THRESHOLD_C: flags.append("C")
    type_flag = "".join(flags) if flags else "none"

    rows.append({
        "category"  : cat,
        "n_reviews" : n_reviews,
        "n_users"   : n_users,
        "n_items"   : n_items,
        "density"   : density,
        "gini"      : gini,
        "cold_ratio": cold_ratio,
        "type_flag" : type_flag,
    })

cat_labels = pd.DataFrame(rows).sort_values("n_reviews", ascending=False)
cat_labels.to_csv(RESULTS_DIR / "category_labels.csv", index=False)

print("\n=== Category Sparsity Diagnosis ===")
print(cat_labels[["category","n_reviews","density","gini","cold_ratio","type_flag"]].to_string())
print(f"\nType distribution:\n{cat_labels['type_flag'].value_counts()}")

# ── Visualize 3 metric distributions ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Sparsity Heterogeneity across Pseudo-Categories (brand-based)", fontsize=13)

axes[0].hist(cat_labels["density"], bins=15, color="#4C72B0", edgecolor="white")
axes[0].axvline(THRESHOLD_A, color="red", linestyle="--", label=f"threshold={THRESHOLD_A}")
axes[0].set_title("Metric A: Interaction Density")
axes[0].set_xlabel("density"); axes[0].legend()

axes[1].hist(cat_labels["gini"], bins=15, color="#55A868", edgecolor="white")
axes[1].axvline(THRESHOLD_B, color="red", linestyle="--", label=f"threshold={THRESHOLD_B}")
axes[1].set_title("Metric B: Gini Index (item imbalance)")
axes[1].set_xlabel("Gini"); axes[1].legend()

axes[2].hist(cat_labels["cold_ratio"], bins=15, color="#C44E52", edgecolor="white")
axes[2].axvline(THRESHOLD_C, color="red", linestyle="--", label=f"threshold={THRESHOLD_C}")
axes[2].set_title("Metric C: Cold User Ratio")
axes[2].set_xlabel("ratio"); axes[2].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_sparsity_distributions.png", dpi=150)
plt.show()

# ── Save train_with_category (Bug 4 fix) ────────────────────────
# 03_has_sampler.py loads this file to get the pseudo_category column.
train_with_cat = train_tmp.drop(columns=["brand"])   # drop raw brand, keep pseudo_category
train_with_cat.to_parquet(
    DATA_DIR / "train_with_category.parquet",
    compression="zstd",
    index=False,
)

print("\n✓ category_labels.csv saved to", RESULTS_DIR)
print("✓ fig1_sparsity_distributions.png saved to", FIG_DIR)
print("✓ train_with_category.parquet saved to", DATA_DIR)
print("  Columns:", train_with_cat.columns.tolist())
print(f"  Shape: {train_with_cat.shape}")
print("Stage 2 complete")

input("\nPress Enter to continue to 03_has_sampler.py, or Ctrl+C to stop...")
