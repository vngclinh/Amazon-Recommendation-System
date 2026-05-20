"""
05_analysis.py
Generate figures for the HAS RecSys report:
  1. RMSE per rating star — 3 models
  2. Gini vs RMSE improvement per pseudo-category (scatter)
  3. Average RMSE improvement per sparsity type (bar chart)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from pathlib import Path

DATA_DIR    = Path("has_recsys/data")
RESULTS_DIR = Path("has_recsys/results")
FIG_DIR     = Path("has_recsys/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

COL_RATING = "overall"

# ── Load ────────────────────────────────────────────────────────
test_preds = pd.read_parquet(DATA_DIR / "test_with_preds.parquet")
cat_labels = pd.read_csv(RESULTS_DIR / "category_labels.csv")

with open(RESULTS_DIR / "model_scores.json") as f:
    model_scores = json.load(f)

print(f"Test with preds : {len(test_preds):,} rows")
print(f"Columns         : {test_preds.columns.tolist()}")
print(f"\nModel scores (warm):")
for name, s in model_scores.items():
    print(f"  {name:20s}  RMSE={s['rmse']:.4f}  MAE={s['mae']:.4f}")

# ── Figure 1: RMSE per star — 3 models ──────────────────────────
stars  = sorted(test_preds[COL_RATING].unique())
x      = np.arange(len(stars))
width  = 0.25

models = [
    ("pred_b1",  "Baseline (avg)",  "#4C72B0"),
    ("pred_b2",  "Baseline (SVD)",  "#55A868"),
    ("pred_has", "HAS + SVD",       "#C44E52"),
]

fig, ax = plt.subplots(figsize=(9, 5))
for i, (col, label, color) in enumerate(models):
    rmses = []
    for s in stars:
        mask = test_preds[COL_RATING] == s
        rmse = mean_squared_error(
            test_preds.loc[mask, COL_RATING],
            test_preds.loc[mask, col],
        ) ** 0.5
        rmses.append(rmse)
    ax.bar(x + i * width, rmses, width, label=label, color=color, alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels([f"★{int(s)}" for s in stars])
ax.set_xlabel("Actual Rating")
ax.set_ylabel("RMSE")
ax.set_title("RMSE per Rating Star — 3 Models")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_rmse_per_star.png", dpi=150)
plt.show()
print("✓ fig2_rmse_per_star.png")

# ── Figure 2: Gini vs RMSE improvement ──────────────────────────
if "pseudo_category" in test_preds.columns:
    def safe_rmse(g, col):
        if len(g) < 5:
            return np.nan
        return mean_squared_error(g[COL_RATING], g[col]) ** 0.5

    cat_perf = (
        test_preds
        .groupby("pseudo_category")
        .apply(lambda g: pd.Series({
            "rmse_b2" : safe_rmse(g, "pred_b2"),
            "rmse_has": safe_rmse(g, "pred_has"),
            "n"       : len(g),
        }))
        .reset_index()
    )
    cat_perf = cat_perf.merge(
        cat_labels[["category", "gini", "type_flag"]],
        left_on="pseudo_category", right_on="category", how="left",
    )
    cat_perf["improvement"] = cat_perf["rmse_b2"] - cat_perf["rmse_has"]
    cat_perf = cat_perf.dropna(subset=["rmse_b2", "rmse_has", "gini"])

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        cat_perf["gini"],
        cat_perf["improvement"],
        s=cat_perf["n"] / cat_perf["n"].max() * 400,
        c=cat_perf["improvement"], cmap="RdYlGn",
        alpha=0.85, edgecolors="black", linewidths=0.5,
    )
    for _, row in cat_perf.iterrows():
        ax.annotate(
            str(row["pseudo_category"])[:12],
            (row["gini"], row["improvement"]),
            fontsize=7, alpha=0.7,
            xytext=(4, 4), textcoords="offset points",
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Gini Index (item imbalance)")
    ax.set_ylabel("RMSE Improvement (SVD_raw − HAS_SVD)")
    ax.set_title("Higher Gini → HAS Helps More\n(bubble size = category size)")
    plt.colorbar(scatter, label="RMSE improvement")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_gini_vs_improvement.png", dpi=150)
    plt.show()
    print("✓ fig3_gini_vs_improvement.png")
else:
    print("SKIP fig3: pseudo_category column not found in test_with_preds.parquet")
    cat_perf = None

# ── Figure 3: Improvement per sparsity type ──────────────────────
if cat_perf is not None and "type_flag" in cat_perf.columns:
    type_perf = (
        cat_perf
        .dropna(subset=["improvement"])
        .groupby("type_flag")["improvement"]
        .mean()
        .reset_index()
    )

    colors = {
        "none": "#AAAAAA", "A": "#4C72B0", "B": "#55A868", "C": "#C44E52",
        "AB"  : "#8172B2", "BC": "#CCB974", "ABC": "#64B5CD", "AC": "#E8762D",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        type_perf["type_flag"],
        type_perf["improvement"],
        color=[colors.get(t, "#888888") for t in type_perf["type_flag"]],
        edgecolor="white", alpha=0.9,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Sparsity Type (HAS label)")
    ax.set_ylabel("Avg RMSE Improvement")
    ax.set_title("HAS Improvement by Sparsity Type\n(positive = HAS better than SVD raw)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_improvement_per_type.png", dpi=150)
    plt.show()
    print("✓ fig4_improvement_per_type.png")
else:
    print("SKIP fig4: type_flag not available")

print(f"\n✓ All figures saved to {FIG_DIR}")
print("Analysis complete — all stages done!")
