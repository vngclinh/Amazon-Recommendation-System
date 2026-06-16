# -*- coding: utf-8 -*-
"""
Render a pipeline diagram for the Amazon RecSys (Phase 1 - Amazon Reviews 2018).
5-stage pipeline: Baseline -> Feature Engineering -> 4 auxiliary models -> Ensemble -> Ranking Eval.
Output: src/visualize_data/amazon_recsys_pipeline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(17, 10.5), dpi=170)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---- palette ----
C_DATA   = "#37474F"   # data source / preprocessing
C_CLEAN  = "#546E7A"
STAGE_COL = ["#1565C0", "#00838F", "#2E7D32", "#EF6C00", "#6A1B9A"]
STAGE_BG  = ["#E3F2FD", "#E0F2F1", "#E8F5E9", "#FFF3E0", "#F3E5F5"]
C_CALL   = "#B71C1C"
C_RESULT = "#1B5E20"


def box(x, y, w, h, fc, ec, title, body, tcol="white", bcol="#263238",
        tsize=11, bsize=8.2, round_pad=0.02, lw=1.6):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.15,rounding_size={round_pad*100}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
                       mutation_aspect=0.6)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h - 0.18 * h, title, ha="center", va="top",
            fontsize=tsize, fontweight="bold", color=tcol, zorder=4)
    if body:
        ax.text(x + w / 2, y + h - 0.42 * h, body, ha="center", va="top",
                fontsize=bsize, color=bcol, zorder=4, linespacing=1.45)


def arrow(x1, y1, x2, y2, col="#455A64", lw=2.2, style="-|>", mut=18):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=mut, lw=lw, color=col, zorder=2,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


# ===== Title =====
ax.text(50, 97.5, "Amazon RecSys — 5-Stage Pipeline (Phase 1: Amazon Reviews 2018)",
        ha="center", va="top", fontsize=17, fontweight="bold", color="#212121")
ax.text(50, 93.7,
        "Electronics & Home·Kitchen  ·  ~14M reviews  →  5-core: 5.5M reviews / 609K users / 145K items  ·  sparsity 99.99%",
        ha="center", va="top", fontsize=9.5, color="#546E7A", style="italic")

# ===== Data & preprocessing band (top) =====
box(3, 80, 27, 9.5, "#ECEFF1", C_DATA,
    "1) Data source",
    "Amazon Reviews 2018\nHuggingFace: datdong2004/\namazonNew-cleaned",
    tcol=C_DATA, tsize=11.5, bsize=8.5)

box(36.5, 80, 27, 9.5, "#ECEFF1", C_CLEAN,
    "2) Cleaning — 7 steps",
    "strip HTML/emoji · unicode norm\ntimestamp · drop rating=0 / empty\ndedup SimHash · keep 10–2000 tokens",
    tcol=C_CLEAN, tsize=11.5, bsize=8.0)

box(70, 80, 27, 9.5, "#ECEFF1", C_CLEAN,
    "3) 5-core + Temporal split",
    "keep users/items with ≥5 reviews\nsplit train/test by time\n→ zstd-compressed Parquet",
    tcol=C_CLEAN, tsize=11.5, bsize=8.0)

arrow(30, 84.7, 36.5, 84.7)
arrow(63.5, 84.7, 70, 84.7)
# arrow down into pipeline
arrow(83.5, 80, 83.5, 73.5, col=C_DATA, lw=2.6)
ax.text(85, 76.6, "anti-leakage: fit/transform on train only · test scored once",
        ha="left", va="center", fontsize=8, color="#78909C", style="italic")

# ===== 5-stage pipeline (horizontal row) =====
stage_y = 50
stage_h = 19
stage_w = 17.0
gap = 2.6
x0 = 3.0

stages = [
    ("Stage 0", "Baseline",
     "canonical data\n+ Bayesian shrinkage\n(global → item/user mean)\n+ cold-start profiling\n(new/cold/medium/warm)"),
    ("Stage 2", "Feature Engineering",
     "TF-IDF text → SVD-128\nitem_profiles · user_profiles\nM_norm (normalization)\nC_implicit (confidence)"),
    ("Stage 3", "4 auxiliary models",
     "A · Item-Item CF (cosine)\nB · MF-SGD (Numba, 10–50×)\nC · Content-based (SVD-128)\nD · ALS implicit (c=1+α·x)"),
    ("Stage 4", "Ensemble & Meta",
     "16 candidates → selected by\nOOF validation\n★ ridge_positive\nB 57.5% · D 25%\nBase 8.8% · C 8.8%"),
    ("Stage 5", "Ranking Eval",
     "RMSE · MAE\nPrecision/Recall/NDCG@K\nCoverage\nfull-catalog ranking\n(143K items)"),
]

centers = []
for i, (st, title, body) in enumerate(stages):
    x = x0 + i * (stage_w + gap)
    cx = x + stage_w / 2
    centers.append((x, cx))
    box(x, stage_y, stage_w, stage_h, STAGE_BG[i], STAGE_COL[i], "", "",
        round_pad=0.02, lw=2.0)
    ax.text(cx, stage_y + stage_h - 1.1, st, ha="center", va="top",
            fontsize=11.5, fontweight="bold", color=STAGE_COL[i])
    ax.text(cx, stage_y + stage_h - 4.4, title, ha="center", va="top",
            fontsize=9.8, fontweight="bold", color="#263238")
    ax.text(cx, stage_y + stage_h - 7.3, body, ha="center", va="top",
            fontsize=8.0, color="#37474F", linespacing=1.5)
    if i > 0:
        px = x0 + (i - 1) * (stage_w + gap) + stage_w
        arrow(px, stage_y + stage_h / 2, x, stage_y + stage_h / 2,
              col="#455A64", lw=2.4)

# ===== Positive result (RMSE) =====
box(3, 30, 45, 12, "#E8F5E9", C_RESULT,
    "✓ Result — Electronics (warm test)",
    "Ensemble ridge_positive:  RMSE 1.1994  (−1.15% vs baseline)\n"
    "Best single model B (MF-SGD):  RMSE 1.2127\n"
    "Home & Kitchen:  RMSE 1.1358 — generalizes better than HGBR",
    tcol=C_RESULT, tsize=11, bsize=8.6, bcol="#1B5E20")

# ===== Turning point — Stage 5 failure =====
box(52, 30, 45, 12, "#FFEBEE", C_CALL,
    "⚠ Turning point — full-catalog ranking fails",
    "Recommending from all 143K items: Recall@10 ≈ 0.0001\n"
    "Coverage only 1.56% — always recommends a few popular items\n"
    "Root cause: 71% of items have ≤20 reviews → embeddings are noise",
    tcol=C_CALL, tsize=11, bsize=8.6, bcol="#B71C1C")

# arrows from Stage 4 / Stage 5 down to the two result boxes
arrow(centers[3][1], stage_y, 25, 42.3, col="#455A64", lw=2.0)
arrow(centers[4][1], stage_y, 74, 42.3, col=C_CALL, lw=2.0, style="-|>")

# ===== Lesson (bottom) =====
box(3, 17.5, 94, 9.5, "#FFF8E1", "#F9A825",
    "Lesson → motivation to switch to Goodreads (Phase 2/3)",
    "A 'good' RMSE of 1.20 is MEANINGLESS for real recommendation: within-test NDCG ≈ 0.98 but full-catalog ≈ 0.\n"
    "Amazon's extreme sparsity breaks ranking → choosing a dataset that fits the problem matters as much as choosing the algorithm.",
    tcol="#F57F17", tsize=11, bsize=8.8, bcol="#5D4037")

# ===== Footnote =====
ax.text(3, 4.2,
        "Source: src/mining_data/ (stage-0, stage-2-electronics, stage-3-electronics, stage-4-elec, dedup_simhash)  ·  "
        "EDA: src/visualize_data/EDA.ipynb",
        ha="left", va="center", fontsize=7.8, color="#90A4AE")

plt.tight_layout()
for out in ("amazon_recsys_pipeline.png", "amazon_recsys_pipeline.svg"):
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print("Saved:", out)
