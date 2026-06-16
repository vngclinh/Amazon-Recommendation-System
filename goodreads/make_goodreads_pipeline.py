# -*- coding: utf-8 -*-
"""
Render a pipeline diagram for the Goodreads RecSys (Phase 2 & Phase 3).
Phase 2: edge-weighting ablation (F1-F5) + LDA taste + ALS blend  (sampled@500).
Phase 3: behavior chains + chainRec + shared full-ranking evaluator (S0->S5).
Output: goodreads/goodreads_recsys_pipeline.{png,svg}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(18, 13.5), dpi=165)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")

# ---- palette ----
C_DATA   = "#37474F"
C_CLEAN  = "#546E7A"
P2_COL   = ["#00695C", "#00897B", "#26A69A", "#4DB6AC"]   # Phase 2 (teal)
P2_BG    = ["#E0F2F1"] * 4
P3_COL   = ["#4527A0", "#5E35B1", "#7E57C2", "#9575CD"]   # Phase 3 (purple)
P3_BG    = ["#EDE7F6"] * 4
C_FIND   = "#BF360C"
C_RESULT = "#1B5E20"


def box(x, y, w, h, fc, ec, title, subtitle, body,
        tcol="#263238", bcol="#37474F", tsize=10.5, sbsize=9.0,
        bsize=7.8, lw=1.9):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.15,rounding_size=1.6",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
                       mutation_aspect=0.7)
    ax.add_patch(p)
    cur = y + h - 1.0
    ax.text(x + w / 2, cur, title, ha="center", va="top",
            fontsize=tsize, fontweight="bold", color=ec, zorder=4)
    cur -= 2.7
    if subtitle:
        ax.text(x + w / 2, cur, subtitle, ha="center", va="top",
                fontsize=sbsize, fontweight="bold", color=tcol, zorder=4)
        cur -= 2.5
    if body:
        ax.text(x + w / 2, cur, body, ha="center", va="top",
                fontsize=bsize, color=bcol, zorder=4, linespacing=1.5)


def arrow(x1, y1, x2, y2, col="#455A64", lw=2.3, style="-|>", mut=18):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=mut, lw=lw, color=col, zorder=2,
                 shrinkA=2, shrinkB=2))


def lane(x, y, w, h, fc, ec, label, lcol):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                 linewidth=1.4, zorder=1, alpha=0.55, linestyle="--"))
    ax.text(x + 0.9, y + h - 1.0, label, ha="left", va="top",
            fontsize=12.5, fontweight="bold", color=lcol, zorder=4)


# ===== Title =====
ax.text(50, 98.3, "Goodreads RecSys — Pipeline (Phase 2 & Phase 3)",
        ha="center", va="top", fontsize=18, fontweight="bold", color="#212121")
ax.text(50, 95.0,
        "Goodreads UCSD Book Graph  ·  15.7M reviews  ·  10 genres  ·  temporal split: Train ≤2015 · Val 2016 · Test 2017",
        ha="center", va="top", fontsize=9.8, color="#546E7A", style="italic")

# ===== Data & preprocessing band =====
box(3, 84.5, 28, 8.0, "#ECEFF1", C_DATA,
    "1) Raw data", "6 JSON.gz (~16 GB)",
    "reviews · interactions ·\nbook metadata · genres",
    tcol=C_DATA, tsize=11, sbsize=8.6, bsize=8.0)

box(36, 84.5, 28, 8.0, "#ECEFF1", C_CLEAN,
    "2) Preprocess", "Parquet by genre (10)",
    "dedup · text clean · schema fix\nedge_weight · temporal split",
    tcol=C_CLEAN, tsize=11, sbsize=8.6, bsize=8.0)

box(69, 84.5, 28, 8.0, "#ECEFF1", C_CLEAN,
    "3) HuggingFace", "vngclinh/goodreads-*",
    "reviews · concats · preprocessed\n(+ lda / phase3 / graph artifacts)",
    tcol=C_CLEAN, tsize=11, sbsize=8.6, bsize=8.0)

arrow(31, 88.5, 36, 88.5)
arrow(64, 88.5, 69, 88.5)
arrow(50, 84.5, 50, 78.7, col=C_DATA, lw=2.6)    # down to Phase 2
arrow(94, 84.5, 94, 53.2, col=C_DATA, lw=2.4, style="-|>")  # down to Phase 3 (right margin)
ax.text(95, 68.0, "raw\ninteractions", ha="left", va="center",
        fontsize=7.5, color=C_DATA, style="italic")

# ===== Phase 2 lane =====
p2y, p2h = 57.5, 21.0
lane(2, p2y, 96, p2h, "#E0F2F1", "#00897B",
     "Phase 2 · ALS + edge-weighting + LDA taste   (sampled@500)", "#00695C")

bx, bw, bh, gy = 6.5, 19.0, 12.5, 59.5
gaps = 3.5
p2_boxes = [
    (P2_COL[0], P2_BG[0], "Edge-weighting", "ablation F1–F5",
     "F1 votes · F2 rating · F3\nrating×log(len+1) · F4 · F5 binary\n★ F3 winner (+3.6% R@10)"),
    (P2_COL[1], P2_BG[1], "LDA taste profile", "40 topics + 10 genres",
     "50-dim vector per user\nweighted by (rating−3.5)\n× exp(−λ·days_ago)"),
    (P2_COL[2], P2_BG[2], "ALS ⊕ Taste blend", "final = 0.7·ALS + 0.3·taste",
     "ALS dominates,\ntaste regularizes\n(+3.8% over ALS-only)"),
    (P2_COL[3], P2_BG[3], "Eval — sampled@500", "1 pos + 500 neg",
     "Recall@10 = 0.6122\nNDCG@10 = 0.5369\nRecall@20 = 0.6859"),
]
p2cx = []
for i, (ec, fc, t, s, b) in enumerate(p2_boxes):
    x = bx + i * (bw + gaps)
    p2cx.append(x + bw / 2)
    box(x, gy, bw, bh, fc, ec, t, s, b)
    if i > 0:
        arrow(bx + (i - 1) * (bw + gaps) + bw, gy + bh / 2, x, gy + bh / 2,
              col="#00695C", lw=2.4)

# ===== Phase 3 lane =====
p3y, p3h = 32.0, 21.0
lane(2, p3y, 96, p3h, "#EDE7F6", "#5E35B1",
     "Phase 3 · chainRec + shared full-ranking evaluator   (S0→S5)", "#4527A0")

gy3 = 34.0
p3_boxes = [
    (P3_COL[0], P3_BG[0], "Behavior chains", "shelve→read→rate→recommend",
     "48.3K users × 1.57M items\n13.7M interactions × 4 stages\nfrom goodreads_interactions"),
    (P3_COL[1], P3_BG[1], "chainRec (PyTorch)", "monotonic scoring",
     "samplers: stagewise / uniform\nstagewise > uniform\n(+2.9% R@10 @ sampled)"),
    (P3_COL[2], P3_BG[2], "Full-ranking rank_eval", "model-agnostic",
     "mask seen · score all 1.57M\nAUC + Recall@K + NDCG@K\nsame test / pool / mask"),
    (P3_COL[3], P3_BG[3], "S0→S5 experiments", "head-to-head + hybrid",
     "ALS vs chainRec · F3 transfer\nhybrid ALS⊕chainRec\nweb demo (results_demo.html)"),
]
p3cx = []
for i, (ec, fc, t, s, b) in enumerate(p3_boxes):
    x = bx + i * (bw + gaps)
    p3cx.append(x + bw / 2)
    box(x, gy3, bw, bh, fc, ec, t, s, b)
    if i > 0:
        arrow(bx + (i - 1) * (bw + gaps) + bw, gy3 + bh / 2, x, gy3 + bh / 2,
              col="#4527A0", lw=2.4)

# link Phase 2 -> Phase 3 (F3 / ALS feed forward)
arrow(p2cx[0], gy, p2cx[0], p3y + p3h, col="#7E57C2", lw=2.0, style="-|>")
ax.text(p2cx[0] + 1.2, (gy + p3y + p3h) / 2, "re-test F3 under\nfull-ranking",
        ha="left", va="center", fontsize=7.5, color="#7E57C2", style="italic")

# ===== Key findings band =====
fy = 16.5
box(3, fy, 46.5, 13.0, "#FBE9E7", C_FIND,
    "Key findings (full-ranking)", "",
    "• ALS (AUC 0.9646) ≥ chainRec (0.9525); hybrid ≈ ALS\n"
    "• sampled@500 inflates gains: F3 +3.6% → −0.8% AUC\n"
    "• votes → popularity bias across protocols (−17.6% / −5.8%)\n"
    "• review-length covers only 5.5% of chain edges → no transfer",
    tcol=C_FIND, bcol="#5D4037", tsize=11, bsize=8.4)

box(51.5, fy, 45.5, 13.0, "#E8F5E9", C_RESULT,
    "Net contribution", "",
    "Genuine positives — F3 +3.6% R@10 & stagewise +2.9% (sampled).\n"
    "Methodological — a shared full-ranking evaluator shows when a\n"
    "sampled metric can be trusted, and quantifies a transfer limit.\n"
    "Confirms the paper: Goodreads is unfavorable to chainRec.",
    tcol=C_RESULT, bcol="#1B5E20", tsize=11, bsize=8.4)

# ===== Footnote =====
ax.text(3, 11.0,
        "Source: goodreads/main pipeline/  ·  Phase 2: stage 4 ct1–ct5 (F1–F5), branch A LDA, stage 3 5ct  ·  "
        "Phase 3: phase3_s0…s4 + chainrec-goodreads + phase3_s5_precompute_demo",
        ha="left", va="center", fontsize=7.6, color="#90A4AE")

plt.tight_layout()
for out in ("goodreads_recsys_pipeline.png", "goodreads_recsys_pipeline.svg"):
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    print("Saved:", out)
