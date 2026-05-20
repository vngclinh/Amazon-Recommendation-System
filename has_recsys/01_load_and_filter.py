"""
01_load_and_filter.py
Load Amazon Electronics from HuggingFace -> 5-core filter -> temporal split -> save artifacts
"""
import re, json, gc
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from huggingface_hub import snapshot_download

# ── Config ─────────────────────────────────────────────────────
REPO_ID     = "datdong2004/amazonNew-cleaned"
CATEGORY    = "electronics"
HF_TOKEN    = None          # fill in if repo is private
K_CORE      = 5
TRAIN_RATIO = 0.8
DATA_DIR    = Path("has_recsys/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

COL_USER    = "reviewerID"
COL_ITEM    = "asin"
COL_RATING  = "overall"
COL_DATE    = "review_date"
NEEDED_COLS = [COL_USER, COL_ITEM, COL_DATE]

# ── Step 1: Download ────────────────────────────────────────────
local_dir = DATA_DIR / "raw" / CATEGORY
local_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {CATEGORY} from HuggingFace...")
snapshot_download(
    repo_id        = REPO_ID,
    repo_type      = "dataset",
    local_dir      = str(local_dir),
    token          = HF_TOKEN,
    allow_patterns = [f"*{CATEGORY.capitalize()}*", f"*{CATEGORY}*"],
    ignore_patterns= ["*.json", "*.md", "*.txt", "*.crc"],
)

# ── Step 2: Load parquet files, parse overall from folder name ──
parquet_files = sorted(local_dir.rglob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

dfs = []
for path in parquet_files:
    match = re.search(r'overall=(\d)', str(path))
    if not match:
        print(f"  SKIP (no overall): {path.name}")
        continue
    overall_val = int(match.group(1))
    df_chunk = pq.ParquetFile(path).read(columns=NEEDED_COLS).to_pandas()
    df_chunk[COL_RATING] = overall_val
    dfs.append(df_chunk)
    print(f"  overall={overall_val}: {len(df_chunk):,} rows from {path.name}")

df = pd.concat(dfs, ignore_index=True)
df[COL_DATE]   = pd.to_datetime(df[COL_DATE])
df[COL_RATING] = df[COL_RATING].astype(int)

print(f"\nRaw data: {len(df):,} reviews, "
      f"{df[COL_USER].nunique():,} users, "
      f"{df[COL_ITEM].nunique():,} items")
print(f"Rating distribution:\n{df[COL_RATING].value_counts().sort_index()}")

# ── Step 3: K-core filtering ────────────────────────────────────
print(f"\nRunning {K_CORE}-core filtering...")
df_core = df.copy()
del df; gc.collect()

for iteration in range(1, 50):
    valid_users = df_core[COL_USER].value_counts()
    valid_items = df_core[COL_ITEM].value_counts()
    valid_users = valid_users[valid_users >= K_CORE].index
    valid_items = valid_items[valid_items >= K_CORE].index

    df_filtered = df_core[
        df_core[COL_USER].isin(valid_users) &
        df_core[COL_ITEM].isin(valid_items)
    ]
    removed = len(df_core) - len(df_filtered)
    print(f"  Iter {iteration}: removed {removed:,} → {len(df_filtered):,} remaining")
    if removed == 0:
        break
    df_core = df_filtered

n_users  = df_core[COL_USER].nunique()
n_items  = df_core[COL_ITEM].nunique()
sparsity = 1 - len(df_core) / (n_users * n_items)
print(f"\nAfter {K_CORE}-core: {len(df_core):,} reviews, "
      f"{n_users:,} users, {n_items:,} items, sparsity={sparsity:.4%}")

# ── Step 4: Temporal split ──────────────────────────────────────
df_core    = df_core.sort_values(COL_DATE).reset_index(drop=True)
split_idx  = int(len(df_core) * TRAIN_RATIO)
split_date = df_core.iloc[split_idx][COL_DATE]

train = df_core.iloc[:split_idx].copy()
test  = df_core.iloc[split_idx:].copy()

train_users = set(train[COL_USER])
train_items = set(train[COL_ITEM])
test["is_warm"]      = test[COL_USER].isin(train_users) & test[COL_ITEM].isin(train_items)
test["is_cold_user"] = ~test[COL_USER].isin(train_users)
test["is_cold_item"] = ~test[COL_ITEM].isin(train_items)

print(f"\nSplit date      : {split_date}")
print(f"Train           : {len(train):,}")
print(f"Test            : {len(test):,}")
print(f"Warm test pairs : {test['is_warm'].mean():.1%}  ({test['is_warm'].sum():,})")
print(f"Cold-user pairs : {test['is_cold_user'].mean():.1%}")
print(f"Cold-item pairs : {test['is_cold_item'].mean():.1%}")

# ── Step 5: Save ────────────────────────────────────────────────
train.to_parquet(DATA_DIR / "train.parquet", compression="zstd", index=False)
test.to_parquet(DATA_DIR  / "test.parquet",  compression="zstd", index=False)
df_core[[COL_USER, COL_ITEM, COL_RATING, COL_DATE]].to_parquet(
    DATA_DIR / "df_core.parquet", compression="zstd", index=False)

meta = {
    "category"   : CATEGORY,
    "k_core"     : K_CORE,
    "n_train"    : len(train),
    "n_test"     : len(test),
    "n_users"    : int(n_users),
    "n_items"    : int(n_items),
    "sparsity"   : float(sparsity),
    "split_date" : str(split_date),
    "train_ratio": TRAIN_RATIO,
    "warm_pairs" : int(test["is_warm"].sum()),
    "warm_ratio" : float(test["is_warm"].mean()),
}
with open(DATA_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✓ train.parquet, test.parquet, df_core.parquet, meta.json saved")
print(f"Stage 1 complete — {CATEGORY.upper()}")

input("\nPress Enter to continue to 02_sparsity_diagnosis.py, or Ctrl+C to stop...")
