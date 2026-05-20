"""
04_train_eval.py
Train 3 models and compare:
  - Baseline 1 : Simple average  (user_mean + item_mean) / 2
  - Baseline 2 : SVD on raw train data
  - HAS model  : SVD on HAS-sampled train data

Bug 1 fix: replaced scipy.svds + .toarray() with sklearn TruncatedSVD,
           which operates directly on sparse CSR matrices (no densification).
           The original .toarray() on a 609K×145K matrix would allocate ~700GB RAM.

Bug 2 fix: replaced row-by-row Python loop in predict_svd with vectorized
           numpy indexing + einsum, reducing prediction cost from O(n) Python
           iterations to a single batch matrix operation.
"""
import json, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

DATA_DIR    = Path("has_recsys/data")
RESULTS_DIR = Path("has_recsys/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

COL_USER   = "reviewerID"
COL_ITEM   = "asin"
COL_RATING = "overall"
K_FACTORS  = 50

# ── Load ────────────────────────────────────────────────────────
train     = pd.read_parquet(DATA_DIR / "train_with_category.parquet")
train_has = pd.read_parquet(DATA_DIR / "train_has.parquet")
test      = pd.read_parquet(DATA_DIR / "test.parquet")

print(f"Train     : {len(train):,} rows")
print(f"Train HAS : {len(train_has):,} rows")
print(f"Test      : {len(test):,} rows")
print(f"Warm test : {test['is_warm'].sum():,} rows ({test['is_warm'].mean():.1%})")

# ── Helpers ─────────────────────────────────────────────────────

def build_index(df: pd.DataFrame):
    users     = df[COL_USER].unique()
    items     = df[COL_ITEM].unique()
    user2idx  = {u: i for i, u in enumerate(users)}
    item2idx  = {a: i for i, a in enumerate(items)}
    return user2idx, item2idx


def build_matrix(df: pd.DataFrame, user2idx: dict, item2idx: dict) -> sp.csr_matrix:
    rows = df[COL_USER].map(user2idx).values
    cols = df[COL_ITEM].map(item2idx).values
    vals = df[COL_RATING].astype(np.float32).values
    mask = (rows >= 0) & (cols >= 0)
    return sp.csr_matrix(
        (vals[mask], (rows[mask], cols[mask])),
        shape=(len(user2idx), len(item2idx)),
    )


def train_svd(df: pd.DataFrame, k: int = K_FACTORS) -> dict:
    """
    Train SVD without ever calling .toarray().

    Steps:
      1. Build user/item encoders.
      2. Safety check: sample to 500K rows if matrix has >1e10 cells.
      3. Build CSR sparse matrix.
      4. Compute per-user means via groupby (pure pandas, no matrix operations).
      5. Center non-zero entries in-place using mat.data and row index expansion
         from mat.indptr — works entirely in the sparse data array.
      6. Run TruncatedSVD (randomized algorithm, sparse-native, O(nnz*k) memory).
      7. Return model dict for use in predict_svd.
    """
    user2idx, item2idx = build_index(df)
    n_users, n_items   = len(user2idx), len(item2idx)

    # Step 2: safety check — too large for even the sparse representation
    if n_users * n_items > 1e10:
        print(f"  Matrix has {n_users}×{n_items} cells. Sampling to 500K rows.")
        df = df.sample(n=min(500_000, len(df)), random_state=42)
        user2idx, item2idx = build_index(df)
        n_users, n_items   = len(user2idx), len(item2idx)

    mat = build_matrix(df, user2idx, item2idx)
    print(f"  Sparse matrix: {mat.shape}, nnz={mat.nnz:,}")

    # Step 4: per-user mean from pandas groupby (not from the sparse matrix)
    global_mean      = float(df[COL_RATING].mean())
    user_mean_series = df.groupby(COL_USER)[COL_RATING].mean()
    users_ordered    = list(user2idx.keys())
    user_means       = np.array([
        user_mean_series.get(u, global_mean) for u in users_ordered
    ], dtype=np.float64)

    # Step 5: center non-zero entries without .toarray()
    # In CSR format, mat.indptr[i]:mat.indptr[i+1] are the nnz indices for row i.
    # np.repeat maps each nnz element back to its row index in O(nnz) memory.
    mat_c = mat.astype(np.float64).copy()
    nz_row = np.repeat(np.arange(n_users), np.diff(mat_c.indptr))
    mat_c.data -= user_means[nz_row]

    # Step 6: TruncatedSVD — handles sparse CSR natively via randomized SVD
    n_components = min(k, min(n_users, n_items) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U   = svd.fit_transform(mat_c)   # shape (n_users, k); equals U_raw @ diag(sigma)
    Vt  = svd.components_            # shape (k, n_items)

    print(f"  TruncatedSVD done. U={U.shape}, Vt={Vt.shape}")
    print(f"  Explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.4f}")

    return {
        "U"          : U,
        "Vt"         : Vt,
        "user2idx"   : user2idx,
        "item2idx"   : item2idx,
        "user_means" : user_means,
        "global_mean": global_mean,
    }


def predict_svd(test_df: pd.DataFrame, model: dict) -> np.ndarray:
    """
    Vectorized SVD prediction — no Python-level row iteration.

    Strategy:
      - Map test user/item IDs to integer indices; unknown IDs → -1 sentinel.
      - For rows where both are known: dot product + user_mean (batch einsum).
      - For known-user / unknown-item: use user_mean.
      - For fully unknown: use global_mean.
      - Clip predictions to [1.0, 5.0].

    The einsum "nk,kn->n" contracts over the k-factor dimension:
      result[n] = sum_k( U[u_n, k] * Vt[k, i_n] )
    This is equivalent to (U[u_valid] * Vt[:, i_valid].T).sum(axis=1)
    but avoids the intermediate (n, k) transpose allocation.
    """
    U, Vt       = model["U"], model["Vt"]
    user2idx    = model["user2idx"]
    item2idx    = model["item2idx"]
    user_means  = model["user_means"]
    global_mean = model["global_mean"]

    u_idx = test_df[COL_USER].map(user2idx).fillna(-1).astype(int).values
    i_idx = test_df[COL_ITEM].map(item2idx).fillna(-1).astype(int).values

    preds = np.full(len(test_df), global_mean, dtype=np.float64)

    known_both = (u_idx >= 0) & (i_idx >= 0)
    if known_both.any():
        u_k = u_idx[known_both]
        i_k = i_idx[known_both]
        # U[u_k]: (n_warm, k)   Vt[:, i_k]: (k, n_warm)
        dot = np.einsum("nk,kn->n", U[u_k], Vt[:, i_k])
        preds[known_both] = user_means[u_k] + dot

    known_u_only = (u_idx >= 0) & (i_idx < 0)
    if known_u_only.any():
        preds[known_u_only] = user_means[u_idx[known_u_only]]

    return np.clip(preds, 1.0, 5.0)


def evaluate(preds: np.ndarray, actuals: np.ndarray, label: str) -> dict:
    rmse = float(mean_squared_error(actuals, preds) ** 0.5)
    mae  = float(mean_absolute_error(actuals, preds))
    print(f"  {label:30s}  RMSE={rmse:.4f}  MAE={mae:.4f}  n={len(actuals):,}")
    return {"label": label, "rmse": rmse, "mae": mae, "n": len(actuals)}


# ── Baseline 1: Simple Average ───────────────────────────────────
print("\n=== BASELINE 1: Simple Average ===")
global_mean = float(train[COL_RATING].mean())
user_means  = train.groupby(COL_USER)[COL_RATING].mean().rename("user_mean")
item_means  = train.groupby(COL_ITEM)[COL_RATING].mean().rename("item_mean")

test = test.join(user_means, on=COL_USER).join(item_means, on=COL_ITEM)
test["pred_b1"] = (
    test["user_mean"].fillna(global_mean) + test["item_mean"].fillna(global_mean)
) / 2.0
test["pred_b1"] = test["pred_b1"].clip(1.0, 5.0)

test_warm = test[test["is_warm"]].copy()
score_b1_warm = evaluate(
    test_warm["pred_b1"].values, test_warm[COL_RATING].values, "Baseline-avg  (warm)"
)
score_b1_all  = evaluate(
    test["pred_b1"].values, test[COL_RATING].values, "Baseline-avg  (all)"
)

# ── Baseline 2: SVD on raw train ─────────────────────────────────
print("\n=== BASELINE 2: SVD on raw train ===")
t0 = time.time()
model_b2 = train_svd(train)
print(f"  Trained in {time.time() - t0:.1f}s")

preds_b2    = predict_svd(test_warm, model_b2)
score_b2    = evaluate(preds_b2, test_warm[COL_RATING].values, "SVD-raw       (warm)")

# ── HAS Model: SVD on HAS-sampled train ─────────────────────────
print("\n=== HAS MODEL: SVD on HAS-sampled train ===")
t0 = time.time()
model_has = train_svd(train_has)
print(f"  Trained in {time.time() - t0:.1f}s")

preds_has = predict_svd(test_warm, model_has)
score_has = evaluate(preds_has, test_warm[COL_RATING].values, "HAS+SVD       (warm)")

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY — Warm test set")
print("="*60)
print(f"{'Model':30s}  {'RMSE':>8s}  {'MAE':>8s}")
print("-"*50)
print(f"{'Baseline (avg)':30s}  {score_b1_warm['rmse']:>8.4f}  {score_b1_warm['mae']:>8.4f}")
print(f"{'Baseline (SVD raw)':30s}  {score_b2['rmse']:>8.4f}  {score_b2['mae']:>8.4f}")
print(f"{'HAS + SVD':30s}  {score_has['rmse']:>8.4f}  {score_has['mae']:>8.4f}")
print()
delta = score_b2["rmse"] - score_has["rmse"]
sign  = "↓ IMPROVEMENT" if delta > 0 else "↑ REGRESSION"
print(f"HAS vs SVD-raw RMSE delta: {delta:+.4f}  {sign}")

# ── Save scores & predictions ────────────────────────────────────
scores = {
    "baseline_avg": score_b1_warm,
    "baseline_svd": score_b2,
    "has_svd"     : score_has,
}
with open(RESULTS_DIR / "model_scores.json", "w") as f:
    json.dump(scores, f, indent=2)

test_warm = test_warm.copy()
test_warm["pred_b2"]  = preds_b2
test_warm["pred_has"] = preds_has
test_warm.to_parquet(DATA_DIR / "test_with_preds.parquet", compression="zstd", index=False)

print("\n✓ model_scores.json saved to", RESULTS_DIR)
print("✓ test_with_preds.parquet saved to", DATA_DIR)
print("Stage 4 complete")

input("\nPress Enter to continue to 05_analysis.py, or Ctrl+C to stop...")
