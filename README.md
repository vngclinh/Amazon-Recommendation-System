# Large-Scale Recommender System — Amazon → Goodreads → chainRec

> A CS246-style project: building a complete recommendation pipeline on big data, applying
> **Collaborative Filtering, Matrix Factorization, Content-based, and Ensemble** techniques. The project
> documents the full three-phase journey — including failures and lessons learned — not just the final results.

**Detailed report:** [`docs/BAO_CAO_CHI_TIET.md`](docs/BAO_CAO_CHI_TIET.md) (VI, updated/expanded) · older snapshot: [`docs/Nhóm 5.pdf`](docs/Nhóm%205.pdf)
**Results & numbers:** [`docs/RESULTS.md`](docs/RESULTS.md) (VI) · [`docs/RESULTS.en.md`](docs/RESULTS.en.md) (EN) · **Demo:** [`goodreads/results_demo.html`](goodreads/results_demo.html)

> ### Key contributions
> 1. **Positive** — review-length (F3 = `rating × log(length+1)`) is the best engagement weight for ALS: **+3.6% Recall@10** @ sampled@500; and `stagewise > uniform` reproduces chainRec (**+2.9% R@10**).
> 2. **Methodological** — a shared **full-ranking evaluator** shows **sampled@500 inflates** engagement-weighting gains (F3 **+3.6% → −0.8%** under full-ranking), and quantifies a **transfer limit** (review-length covers only **5.5%** of interaction edges → cannot weight chainRec).
> 3. **Robust** — **votes cause popularity bias** across both protocols (−17.6% sampled / −5.8% AUC full-rank); **plain ALS ≥ chainRec** and **hybrid ≈ ALS** on Goodreads — confirming the paper's "Goodreads is unfavorable to chainRec".

---

## 1. Overview — Three Phases

| Phase | Dataset | Content | Status | Code directory |
|-------|---------|---------|--------|----------------|
| **Phase 1** | Amazon Reviews 2018 (Electronics, Home & Kitchen) | 5-stage pipeline: baseline → feature engineering → 4 auxiliary models → ensemble → ranking eval. Discovered structural limitations of Amazon. | ✅ Done | [`src/`](src) |
| **Phase 2** | Goodreads UCSD Book Graph (≈15.7M reviews) | 4-phase pipeline: EDA → ablation of 5 edge-weighting formulas → LDA taste profile → ALS + taste blend. | ✅ Done | [`goodreads/`](goodreads) |
| **Phase 3** | Goodreads (4-stage behavior chain) | Extension via **chainRec** (Wan & McAuley, RecSys'18): monotonic scoring over `shelve → read → rate → recommend`, **trained + evaluated under full-ranking** (S0→S4): head-to-head vs ALS, F3 engagement-weighting, hybrid. See [`docs/RESULTS.md`](docs/RESULTS.md). | ✅ Trained & evaluated | [`goodreads/main pipeline/`](goodreads/main%20pipeline) (`phase3_s0…s4`) |
| **(Extra)** | Amazon Electronics | **HAS** — Heterogeneity-Aware Sparsity diagnosis + sampling. Independent experiment, *not part of the report*. | ✅ Done (negative result) | [`has_recsys/`](has_recsys) |

> **Recurring lesson:** *choosing a dataset that fits the problem matters just as much as choosing the right algorithm.*

---

## 2. Directory Structure

```
Amazon-Recommendation-System/
├── README.md                  # This file
├── docs/
│   └── Nhóm 5.pdf             # Full report (3 phases)
├── configs/
│   └── config.yaml           # Spark / preprocessing / ALS config (Amazon US EDA stage)
│
├── src/                       # ░ PHASE 1 — Amazon Reviews ░
│   ├── visualize_data/
│   │   └── EDA.ipynb          # Spark EDA on Amazon US Customer Reviews (imbalance, sparsity, outliers)
│   ├── utils/
│   │   ├── convert_to_parquet.ipynb   # TSV (37 files) → partitioned Parquet, balanced sampling
│   │   └── spark_utils.py             # Shared SparkSession + canonical schema
│   └── mining_data/           # Main 5-stage pipeline (Amazon Reviews 2018)
│       ├── dedup_simhash.ipynb        # SimHash dedup (numpy-vectorized)
│       ├── stage-0.ipynb              # Canonical data + Bayesian-shrinkage baseline + Stage 2 features
│       ├── stage-2-electronics.ipynb  # Feature engineering: TF-IDF → SVD-128, M_norm, C_implicit
│       ├── stage-3-electronics.ipynb  # 4 auxiliary models: A (Item-CF), B (MF-SGD), C (Content), D (ALS)
│       └── stage-4-elec.ipynb         # Ensemble & meta-learning (16 candidates → ridge_positive)
│
├── goodreads/                 # ░ PHASE 2 & 3 — Goodreads ░
│   ├── preprocess/
│   │   ├── upload_to_hf.py             # JSON dedup → Parquet chunks → HuggingFace
│   │   ├── build_unified_dataset.py    # Join reviews + book metadata + genres → HF
│   │   ├── goodreads-reviews-eda.ipynb # DuckDB EDA on 15.7M reviews (32 parquet)
│   │   └── preprocess-goodreads.ipynb  # Schema fixes, text cleaning, temporal split, edge_weight
│   ├── pre_process.py         # Local 2-pass preprocessing (global stats → per-genre)
│   ├── debug.py / check_files.py       # Utilities to check HF parquet files
│   ├── goodreads_eda_interactive.ipynb # Interactive EDA (DuckDB + Plotly), exports eda_outputs/*.html
│   ├── main pipeline/
│   │   ├── branch A LDA.ipynb          # LDA 40 topics on review text (gensim)
│   │   ├── branch B user-book-graph.ipynb  # User-genre graph affinity (5 variants)
│   │   ├── stage 3 5ct.ipynb           # Build user taste profile (40 topic + 10 genre = 50-dim)
│   │   ├── genre-taste.ipynb           # Genre fingerprint + topic discrimination + t-SNE
│   │   ├── stage 4 ct1.ipynb           # ALS + Taste blend — F1 (vote_weighted)
│   │   ├── stage 4 ct2.ipynb           #                     F2 (rating_only, baseline)
│   │   ├── stage 4 ct3.ipynb           #                     F3 (rating_length) ★ winner
│   │   ├── stage 4 ct4.ipynb           #                     F4 (combined votes+length)
│   │   ├── stage 4 ct5.ipynb           #                     F5 (binary)
│   │   └── chainrec-goodreads.ipynb    # PHASE 3 — chainRec (PyTorch): data loader + model + eval
│   ├── charts/                # Pipeline + EDA charts (PNG/JPG)
│   ├── recsys_demo.html       # Web demo (renders 5000-book subset)
│   ├── demo_data.json         # Data for the demo
│   └── requirements.txt
│
├── has_recsys/                # ░ EXTRA EXPERIMENT — not in the report ░
│   ├── 01_load_and_filter.py      # Amazon Electronics → 5-core → temporal split
│   ├── 02_sparsity_diagnosis.py   # 3 sparsity metrics (density/Gini/cold-ratio) → label Type A/B/C
│   ├── 03_has_sampler.py          # Sample by sparsity type (oversample / popularity-inverse / cold-dup)
│   ├── 04_train_eval.py           # Compare Baseline-avg / SVD-raw / HAS+SVD
│   ├── 05_analysis.py             # Generate analysis figures
│   ├── figures/  results/         # Outputs (PNG, CSV, model_scores.json)
│   └── requirements.txt
│
├── .env.example               # Template — only HF_TOKEN is needed
└── configs/config.yaml
```

> **Environment note:** Most notebooks are designed to run on **Kaggle** (using `kaggle_secrets`,
> `/kaggle/working` paths, GPU T4). The `.py` scripts in `has_recsys/` and `goodreads/preprocess/` run
> **locally** (some preprocessing scripts use hard-coded `D:\goodreads\...` paths — adjust for your machine).

---

## 3. Phase 1 — Amazon Reviews 2018

### 3.1 Data

| Category | Raw reviews | After 5-core | Users | Items | Sparsity |
|----------|------------|--------------|-------|-------|----------|
| Electronics | ≈14M | 5,517,640 | 609K | 145K | 99.99% |
| Home & Kitchen | 13,824,810 | 5,622,812 | 647K | 170K | 99.9949% |

7 cleaning steps: strip HTML/emoji → unicode normalization → timestamp parsing → drop rating=0 / empty text
→ dedup `(user, item, timestamp)` → filter reviews <10 or >2000 tokens. Output: zstd-compressed Parquet,
hosted on HuggingFace at `datdong2004/amazonNew-cleaned`.

### 3.2 The 5-stage pipeline

| Stage | Role | Main output |
|-------|------|-------------|
| 0 | Canonical data + baseline | `train/test.parquet`, **Bayesian shrinkage** baseline |
| 1 | Cold-start profiling | user groups: new / cold / medium / warm |
| 2 | Feature engineering | `item_profiles` (SVD-128), `user_profiles`, `M_norm`, `C_implicit` |
| 3 | 4 auxiliary models | predictions A / B / C / D |
| 4 | Ensemble & meta-learning | 16 candidates → winner **ridge_positive** |
| 5 | Extended ranking metrics | RMSE, MAE, Precision/NDCG/Recall@K, Coverage |

**Anti-leakage principles:** temporal split (not random), fit/transform touch train only, tuning uses
validation OOF only, **test is evaluated exactly once**.

**The 4 Stage-3 models:**
- **A — Item-Item CF**: cosine similarity + shrinkage, falls back to baseline when neighbors are scarce.
- **B — MF-SGD**: UV factorization on the residual `(R − baseline)`, sped up **10–50× with Numba JIT**.
- **C — Content-based**: `baseline + scale × cosine(user_profile, item_profile)` from SVD-128.
- **D — ALS Implicit**: learns from a binary matrix, confidence `c = 1 + α·interaction`.

### 3.3 Results

**Electronics** (test_warm):

| Model | RMSE | MAE |
|-------|------|-----|
| Baseline | 1.21344 | 0.92511 |
| Model B (MF-SGD) — best single | 1.21267 | 0.86822 |
| **Ensemble `ridge_positive`** | **1.19943** (−1.15%) | **0.89474** (−3.28%) |

Ensemble weights: B 57.5% · D 25.0% · Baseline 8.8% · C 8.8%.

**Home & Kitchen**: Ridge-positive ensemble test RMSE **1.1358** / MAE 0.8380 — generalizes better than
HGBR (OOF 1.0879 but test 1.1389) → *lesson: complex models overfit validation*.

### 3.4 The critical Stage-5 failure (project turning point)

When asked to recommend from **all 143K items** (full-catalog ranking):

| K | Precision | Recall | NDCG | Hit |
|---|-----------|--------|------|-----|
| 10 | 0.0000 | 0.0001 | 0.0001 | 0.0004 |
| 20 | 0.0001 | 0.0004 | 0.0001 | 0.0010 |

- **Coverage only 1.56%** (2,239 / 143,281 items) — the model always recommends a few thousand popular items.
- Meanwhile *within-test ranking* was nearly perfect (NDCG@10 = 0.9799).
- → **RMSE 1.20 looks good but is meaningless for real recommendation.** Root cause: uneven sparsity
  distribution (71% of items have ≤20 reviews → embeddings are just noise). This is why we switched datasets.

---

## 4. Phase 2 — Goodreads UCSD Book Graph

### 4.1 Data & processing

6 JSON.gz files (~16GB) → Parquet partitioned by `primary_genre` (10 genres) → HuggingFace.
Temporal split by year: **Train ≤2015 · Val 2016 · Test 2017**. Keep users with ≥5 train reviews.

**HuggingFace repos:**
- `vngclinh/goodreads-reviews` — raw reviews (32 parquet, 15.7M rows)
- `vngclinh/goodreads-concats` — concatenated reviews + interactions + book metadata
- `vngclinh/goodreads-preprocessed` — preprocessed (partitioned by genre) + artifacts (`lda/`, `phase3/`, `graph/`)

Technical challenges handled: RAM overflow (incremental Counter + `iter_batches`), 2-pass global stats,
mixed-format `date_added` parsing, HuggingFace path encoding (`HfFileSystem`).

### 4.2 Main contribution — Ablation of 5 edge-weighting formulas

| # | Notebook | Formula | Recall@10 | NDCG@10 | vs F2 |
|---|----------|---------|-----------|---------|-------|
| **F3** ★ | `stage 4 ct3` | `rating × log(length+1) / 5` | **0.6122** | **0.5369** | **+3.6%** |
| F2 | `stage 4 ct2` | `rating / 5` (baseline) | 0.5757 | 0.5107 | — |
| F5 | `stage 4 ct5` | `1 if rating≥4 else 0` | 0.5613 | 0.4995 | −2.5% |
| F1 | `stage 4 ct1` | `rating × log(votes+1) / 5` | 0.4744 | 0.4217 | −17.6% |
| F4 | `stage 4 ct4` | `rating × log(votes+1) × log(length+1)` | 0.4237 | 0.3752 | −26.4% |

**Two insights:**
- **A — Review length is the best engagement signal** (F3 wins): a user who writes a long review read carefully and felt something.
- **B — n_votes causes popularity bias** (F1/F4 lose heavily): popular books naturally get more votes → inflated, drowning out the long tail.

### 4.3 LDA Taste + ALS blend

- **LDA** with 40 topics on review text (vote-weighted sampling, 200K/genre). User taste = 50-dim
  (40 topic + 10 genre affinity), weighted by `(rating − 3.5) × exp(−λ·days_ago)`.
- **Blend**: `final_score = α·ALS + (1−α)·taste_similarity`. Tuned to **α = 0.7** (ALS 70% + Taste 30%),
  consistent across all 5 variants → ALS dominates, taste acts as regularization (+3.8% over ALS-only).

**Final results:** Recall@10 = **0.6122**, Recall@20 = 0.6859, NDCG@10 = **0.5369**, NDCG@20 = 0.5597.

> ⚠️ **Honest limitation:** uses a **sampled@500** protocol (1 positive + 500 random negatives) → metrics
> are inflated relative to full ranking. Absolute values are not directly comparable to the original paper;
> **the delta between variants (F3 vs F2 = +3.6%) is the real contribution.**

---

## 5. Phase 3 — chainRec + full-ranking evaluation

Goal: compare directly against the paper that created the dataset (Wan & McAuley, RecSys'18), exploit
Goodreads' **monotonic behavior chain**, and — crucially — re-test Phase 2's findings under a **rigorous
full-ranking protocol**. Data loader: 48,307 users × 1,567,258 items × 13.7M interactions × 4 stages
(`shelve 50.3% → read 3.3% → rate 13.9% → recommend 32.5%`).

Full numbers + reproduction notebooks: [`docs/RESULTS.md`](docs/RESULTS.md) (VI) / [`docs/RESULTS.en.md`](docs/RESULTS.en.md) (EN).

### 5.1 What we built (S0→S4)

A shared **full-ranking `rank_eval`** (mask seen items, score all 1.57M items; AUC + Recall@K + NDCG@K),
model-agnostic so ALS, chainRec, hybrid and itemPop are scored on the **same test / pool / mask**.

### 5.2 Key results (full-ranking, *recommend* stage)

| Model | AUC | Recall@10 |
|-------|-----|-----------|
| chainRec (stagewise) | 0.9525 | 0.0500 |
| **ALS (vanilla)** | **0.9646** | **0.1588** |
| Hybrid ALS⊕chainRec (best α) | 0.9647 | 0.1606 |
| itemPop (floor) | 0.2918 | 0.0032 |

- **Plain ALS ≥ chainRec**; **hybrid ≈ ALS** (gain within noise) → reproduces the paper's own statement
  that Goodreads is the least favorable dataset for chainRec.
- **`stagewise > uniform`** sampler: clear at sampled@500 (+2.9% R@10), shrinks into noise at full-ranking.

### 5.3 Engagement-weighting (F3) — does it transfer?

- On chainRec interactions, F3 covers only **5.5%** of training edges (review-length lives in the *reviews*
  source, the chain lives in *interactions*) → F3 is nearly a no-op, **does not transfer** (AUC −0.2%).
- Decisive test (ALS on *reviews*, 100% F3 coverage, full-ranking): **F3 −0.8% AUC** — the Phase 2
  **+3.6% was largely a sampled@500 artifact**. **votes −5.8% AUC** → popularity bias confirmed across protocols.

> **Net Phase-3 contribution (methodological):** a shared full-ranking evaluator reveals that sampled@500
> inflates engagement-weighting gains, quantifies a transfer limit (review ⟷ chain data mismatch), and
> re-confirms the votes popularity-bias — alongside the genuine sampled@500 wins (F3 +3.6%, stagewise +2.9%).

---

## 6. ⚠️ Report ↔ Code Consistency Check

This is the direct comparison between `docs/Nhóm 5.pdf` and the source code:

| Section in the report | Corresponding code | Match? |
|-----------------------|--------------------|--------|
| Phase 1 — 5-stage pipeline, Models A/B/C/D, ridge_positive ensemble, Stage 5 ranking | `src/mining_data/` (stage 0/2/3/4 + dedup) | ✅ Match |
| Phase 1 — EDA, data cleaning | `src/visualize_data/EDA.ipynb`, `src/utils/convert_to_parquet.ipynb` | ✅ Match¹ |
| Phase 2 — preprocessing, 5 edge-weight variants F1–F5, LDA taste, ALS+taste α=0.7 | `goodreads/` (`stage 4 ct1`→F1 … `ct5`→F5) | ✅ Match |
| Phase 3 — chainRec data loader + PyTorch model, "in progress" | `goodreads/main pipeline/chainrec-goodreads.ipynb` | ⚠️ Code now **ahead of the PDF** |
| Demo rendering a 5000-book subset | `goodreads/recsys_demo.html` + `demo_data.json` | ✅ Match |
| **HAS — Heterogeneity-Aware Sparsity** | `has_recsys/` (5 scripts) | ❌ **NOT in the report** |

> **Note on Phase 3:** the PDF report describes Phase 3 as *"in progress"*. The code has since advanced
> well beyond it — full-ranking evaluation, ALS↔chainRec head-to-head, F3 engagement-weighting, and a
> hybrid (notebooks `phase3_s0…s4`). These new results are documented in [`docs/RESULTS.md`](docs/RESULTS.md)
> and should be folded into the next revision of the report.

**Key finding:** the `has_recsys/` directory (0 mentions in the report) is an **independent experiment**
on Amazon Electronics — it diagnoses sparsity per pseudo-category (by brand) and then samples by type
A/B/C. Its own experimental result is **negative**:

| Model (warm test) | RMSE | MAE |
|-------------------|------|-----|
| Baseline-avg | 1.2182 | 0.8921 |
| SVD-raw | 1.4056 | 0.9908 |
| **HAS + SVD** | 1.4075 | 1.0031 |

→ HAS+SVD is *slightly worse* than SVD-raw, and both lose to baseline-avg. This is an approach that was
tried and did **not** improve results; it is **not** the Phase 1 pipeline described in the report (the
report's pipeline uses the ridge_positive ensemble and achieves RMSE 1.1994). To keep the report in sync
with the code, consider either (a) adding an appendix describing HAS as a negative result, or
(b) splitting `has_recsys/` out of the main repo.

¹ *Minor note:* `src/visualize_data/EDA.ipynb` and `convert_to_parquet.ipynb` operate on the
**Amazon US Customer Reviews** dataset (`cynthiarempel/amazon-us-customer-reviews-dataset`, TSV schema
`customer_id/product_id/star_rating`), whereas the main pipeline in `src/mining_data/` uses **Amazon
Reviews 2018** (`datdong2004/amazonNew-cleaned`, schema `reviewerID/asin/overall`). These are two
different Amazon datasets used at two different steps (exploratory EDA vs the main pipeline) — not a
contradiction, but worth knowing.

---

## 7. Setup & Running

### 7.1 Common requirements

```bash
# Create a virtualenv (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# HuggingFace token to read/write datasets
cp .env.example .env            # then fill in HF_TOKEN=hf_xxx
```

### 7.2 Run the HAS experiment (local, sequential)

```bash
pip install -r has_recsys/requirements.txt
python has_recsys/01_load_and_filter.py     # download + 5-core + split
python has_recsys/02_sparsity_diagnosis.py  # label Type A/B/C
python has_recsys/03_has_sampler.py         # sample by type
python has_recsys/04_train_eval.py          # baseline vs SVD vs HAS
python has_recsys/05_analysis.py            # figures
```
> Each script ends with `input()` waiting for Enter before the next step — run it in an interactive terminal.

### 7.3 Goodreads preprocessing (local)

```bash
pip install -r goodreads/requirements.txt
# Adjust the D:\goodreads\... paths in upload_to_hf.py / pre_process.py for your machine
python goodreads/preprocess/upload_to_hf.py
python goodreads/preprocess/build_unified_dataset.py
```

### 7.4 Main pipeline (Kaggle)

The notebooks in `src/mining_data/` and `goodreads/main pipeline/` are designed for **Kaggle**:
enable *Internet* + *GPU T4*, store `HF_TOKEN` in Kaggle Secrets, and run them in stage order. Intermediate
artifacts are pushed to HuggingFace for cross-session reuse.

### 7.5 View the demo

Open `goodreads/recsys_demo.html` in a browser (it reads `demo_data.json` in the same directory).

---

## 8. Current Limitations

- Cold-start is not evaluated separately (both Amazon and Goodreads filter cold users out).
- Phase 2 reports sampled@500; Phase 3 adds a **full-ranking** evaluator and shows sampled@500 inflates
  engagement-weighting gains (F3 +3.6% sampled → −0.8% full-rank). See [`docs/RESULTS.md`](docs/RESULTS.md).
- Phase 3 full-rank AUC (~0.95) is ~0.03 below the chainRec paper (~0.98) — preprocessing / 48K-user
  sample / positive-definition differences, not yet fully traced.
- Phase 3 `stagewise` multi-seed (S4) not finished for all 3 seeds; evaluated on the *recommend* stage only.
- The demo currently renders only a 5000-book subset, not the full 1.35M catalog.
- `has_recsys/` is not yet integrated into the report's narrative (see Section 6).

---

## 9. References

- Wan & McAuley, *"Item Recommendation on Monotonic Behavior Chains"* (chainRec), RecSys 2018 — Goodreads UCSD dataset.
- Netflix Prize (2006–2009) — lessons on RMSE vs ranking metrics, and the marginal returns of ensembling.

---

## 10. Team

Group 5 — Large-Scale Recommender System project. Detailed report: [`docs/Nhóm 5.pdf`](docs/Nhóm%205.pdf).
