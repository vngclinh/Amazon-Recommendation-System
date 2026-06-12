# Phase 3 Results — chainRec, full-ranking & engagement-weighting (English)

> This document collects all Phase 3 experimental results (S0→S4) for direct use in the report.
> Vietnamese version: [`RESULTS.md`](RESULTS.md).

## 0. TL;DR

**POSITIVE results:**
1. **F3 = `rating × log(length+1)` is the best engagement weight** — +3.6% Recall@10 over the `rating/5` baseline, beats votes (Phase 2, sampled@500 protocol).
2. **`stagewise` sampler > `uniform`** for chainRec — reproduces the paper: +2.9% Recall@10 @ sampled@500.

**Methodological contribution (what sets this apart — most coursework stops at sampled@500):**
3. Built a **shared full-ranking evaluator** (mask seen items, score all 1.57M items) across every model → shows **sampled@500 inflates**: the F3 benefit **disappears** under full-ranking; the sampler advantage shrinks into noise.
4. **votes cause popularity bias** — robust across protocols (−17.6% R@10 sampled @ Phase 2; −5.8% AUC full-ranking @ Phase 3).
5. **Transfer limit (quantified)**: review-length covers only **5.5%** of interaction-graph edges → it cannot meaningfully weight chainRec (the chain structure and review signal live in weakly-overlapping data sources).
6. **Plain ALS ≥ chainRec** on Goodreads full-ranking; **hybrid ALS⊕chainRec ≈ ALS** — confirming the paper's own statement that Goodreads is the least favorable dataset for chainRec.

---

## 1. Two evaluation protocols (the crux)

| | sampled@500 | full-ranking |
|---|---|---|
| Candidate pool | 1 positive + 500 random negatives | full catalog (1.05–1.57M items), mask seen items |
| Primary metric | Recall@10 / NDCG@10 | **AUC** (global) + Recall@10 / NDCG@10 (top-K) |
| Character | easy, large numbers, **inflated** | strict, realistic for recommendation |
| Used in | Phase 2 (ALS), partly the chainRec paper | Phase 3 (S1a–S4, shared `rank_eval`) |

> **AUC ~0.95 with Recall@10 ~0.05 is consistent**, not contradictory: AUC = P(positive ranked above a random negative), insensitive to cutoff; with 1.57M items, AUC 0.95 still leaves tens of thousands of items above the positive → it rarely lands in the top-10.

---

## 2. Step-by-step results

### S0 — Reproduce vanilla chainRec (sampled@500, *recommend* stage)

| Sampler | AUC | Recall@10 | NDCG@10 |
|---|---|---|---|
| uniform | 0.9437 | 0.7530 | 0.5921 |
| **stagewise** | **0.9479** | **0.7746** | **0.6193** |

→ Reproduces the paper: `stagewise > uniform` (+2.9% R@10). Sampled@500 numbers are high but inflated.

### S1a — Same checkpoints, full-ranking (*recommend* stage)

| Sampler | AUC (full-rank) | Recall@10 | NDCG@10 |
|---|---|---|---|
| uniform | 0.9500 | 0.0506 | 0.0288 |
| stagewise | 0.9525 | 0.0500 | 0.0301 |

→ Recall@10 drops from ~0.77 to ~0.05 when moving sampled@500 → full-ranking — expected, not a bug.
The paper (Table 3, Goodreads/recommend) reports **AUC** 0.978–0.982; ours ~0.95 (≈0.03 lower, due to different preprocessing / 48K-user sample / positive definition).

### S2 — Head-to-head ALS vs chainRec (full-ranking, same test/pool/mask)

| Model | AUC | Recall@10 | NDCG@10 | Recall@20 |
|---|---|---|---|---|
| chainRec (uniform) | 0.9500 | 0.0506 | 0.0288 | 0.0774 |
| chainRec (stagewise) | 0.9525 | 0.0500 | 0.0301 | 0.0674 |
| **ALS (vanilla)** | **0.9646** | **0.1588** | **0.0988** | **0.2144** |
| itemPop (floor) | 0.2918 | 0.0032 | 0.0016 | — |

→ **Plain ALS beats chainRec** (AUC +0.012; Recall@10 ~3×). ALS is trained *in chainRec's own index space*, so the comparison is apples-to-apples (only the model architecture differs). itemPop ≈ 0.29 < 0.5 because the held-out positives are niche books with all popular ones masked.

### S3 — Engagement-weighted (F3) on chainRec interactions (2×2)

| | vanilla (AUC) | +F3 (AUC) | Δ |
|---|---|---|---|
| chainRec | 0.9525 | 0.9502 | −0.2% (R@10 −2.0%) |
| ALS | 0.9646 | 0.9654 | +0.1% (R@10 −7.7%) |

**F3 coverage on interaction edges: only 5.5%** (recommend-edges 11.8%); weight std σ = 0.087 → F3 is almost a *no-op* during training. → F3 **does not transfer** to chainRec — not because F3 is useless, but because **the chain structure (interactions) and review-length (reviews) live in weakly-overlapping sources**.

### S3b — Decisive test: ALS on *reviews* (100% F3 coverage), full-ranking

| Confidence | AUC | Δ vs vanilla | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| vanilla | **0.8419** | — | 0.0062 | 0.0027 |
| +F3 (length) | 0.8350 | **−0.8%** | 0.0062 | 0.0030 |
| +votes | 0.7931 | **−5.8%** | 0.0064 | 0.0030 |
| itemPop (floor) | 0.4636 | — | 0.0010 | 0.0003 |

→ **F3 does NOT survive full-ranking** even at 100% coverage ⇒ **Phase 2's +3.6% was largely a sampled@500 artifact.** **votes clearly hurt (−5.8%)** ⇒ popularity bias confirmed across protocols.

### S4 — Hybrid ALS⊕chainRec (z-score rank fusion) + multi-seed

α-sweep (α = ALS weight), full-ranking:

| α | AUC | Recall@10 | NDCG@10 |
|---|---|---|---|
| 0.00 (chainRec) | 0.9525 | 0.0500 | 0.0301 |
| 0.40 | 0.9599 | **0.1606** | 0.0983 |
| 0.60 | 0.9594 | 0.1602 | **0.0994** |
| 1.00 (ALS) | **0.9647** | 0.1588 | 0.0986 |

→ Hybrid **≈ ALS** (top-K gap within ~1 SE noise), **far above chainRec**. Fusion gives no significant gain over the stronger model (ALS) → chainRec contributes little on Goodreads.

Multi-seed (full-ranking AUC, 3 seeds): **uniform = 0.9446 ± 0.0068** (`[0.9503, 0.9484, 0.9350]`).
→ chainRec full-rank AUC is fairly seed-sensitive (σ ~0.007); the stagewise–uniform gap at full-ranking lies within noise → the sampler advantage is a **sampled@500 phenomenon**, not clear at full-ranking.

---

## 3. Contributions (report framing)

**Title:** *Engagement-weighting and rigorous evaluation for a Goodreads recommender — review-length improves ranking at sampled@500 but reveals its limits under full-ranking.*

1. **(Positive)** review-length (F3) is the best engagement weight @ sampled@500 (+3.6%); **(Positive)** stagewise > uniform (+2.9%).
2. **(Method)** Shared full-ranking `rank_eval` across ALS/chainRec/itemPop → exposes that **sampled@500 inflates**: F3 −0.8%, sampler advantage ≈ noise.
3. **(Confirm)** votes cause popularity bias — robust across protocols.
4. **(Limit)** review-length does not transfer to chainRec (5.5% coverage, data-source mismatch).
5. **(Baseline)** plain ALS ≥ chainRec; hybrid ≈ ALS — Goodreads is unfavorable to chainRec.

> This is a **methodological** contribution: it has genuine wins AND a cautionary finding about evaluation protocols — more valuable than a fragile +1% improvement.

---

## 4. Limitations & remaining work

- The `stagewise` multi-seed in S4 hasn't completed 3 seeds (only seed=42 = 0.9503) — to firmly state stagewise≈uniform at full-rank, finish the run.
- Our full-rank AUC (~0.95) is ~0.03 below the paper (~0.98) — trace preprocessing / 48K-user sample / positive definition if a direct absolute comparison is needed.
- Evaluation is on the *recommend* stage only; multi-stage (shelve/read/rate) like the paper requires changing `split_train_test` in S0 to hold out every stage.
- `has_recsys/` remains outside the report (negative result) — decide: appendix or split out.

---

## 5. Reproduction (notebooks)

| Step | Notebook | Content |
|---|---|---|
| S0 | `goodreads/main pipeline/phase3_s0_reproduce_chainrec.ipynb` | Reproduce vanilla chainRec (sampled@500) |
| S1a | `phase3_s1a_fullrank_eval.ipynb` | Full-ranking `rank_eval` + checkpoint re-eval |
| S2 | `phase3_s2_headtohead_eval.ipynb` | ALS↔chainRec head-to-head + ID bridge |
| S3 | `phase3_s3_f3_weighted.ipynb` | F3 edge-weighting (2×2 family × weighting) |
| S3b | `phase3_s3b_reviews_fullrank.ipynb` | F3/votes on reviews under full-ranking |
| S4 | `phase3_s4_hybrid_multiseed.ipynb` | Hybrid α-sweep + multi-seed |

Artifacts are pushed to HuggingFace `vngclinh/goodreads-preprocessed` under `chainrec/`, `s2/`, `s3/`, `s3b/`, `s4/`.
