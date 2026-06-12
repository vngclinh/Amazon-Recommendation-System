# Kết quả Phase 3 — chainRec, full-ranking & engagement-weighting (Tiếng Việt)

> Tài liệu này tổng hợp toàn bộ kết quả thực nghiệm Phase 3 (S0→S4) để dán thẳng vào báo cáo.
> Bản tiếng Anh: [`RESULTS.en.md`](RESULTS.en.md).

## 0. TL;DR

**Kết quả DƯƠNG (positive):**
1. **F3 = `rating × log(length+1)` là engagement-weight tốt nhất** — +3.6% Recall@10 so baseline `rating/5`, thắng cả votes (Phase 2, protocol sampled@500).
2. **Sampler `stagewise` > `uniform`** cho chainRec — tái lập đúng paper: +2.9% Recall@10 @ sampled@500.

**Đóng góp phương pháp (điểm khác biệt — đa số đồ án chỉ dừng ở sampled@500):**
3. Dựng **eval full-ranking chung** (mask seen items, chấm toàn bộ 1.57M item) cho mọi mô hình → cho thấy **sampled@500 thổi phồng**: lợi ích F3 **biến mất** dưới full-ranking; ưu thế sampler thu hẹp về mức nhiễu.
4. **votes gây popularity bias** — vững xuyên protocol (−17.6% R@10 sampled @ Phase 2; −5.8% AUC full-ranking @ Phase 3).
5. **Giới hạn transfer (định lượng)**: review-length chỉ phủ **5.5%** edge của đồ thị interactions → không thể weight chainRec một cách có nghĩa (chain-structure và review nằm ở hai nguồn chồng lấn thấp).
6. **ALS thuần ≥ chainRec** trên Goodreads full-ranking; **hybrid ALS⊕chainRec ≈ ALS** — đúng nhận định "Goodreads là dataset bất lợi nhất cho chainRec" trong chính paper.

---

## 1. Hai protocol đánh giá (điểm mấu chốt)

| | sampled@500 | full-ranking |
|---|---|---|
| Pool ứng viên | 1 positive + 500 negative ngẫu nhiên | toàn bộ catalog (1.05–1.57M item), mask item đã tương tác |
| Metric chính | Recall@10 / NDCG@10 | **AUC** (toàn cục) + Recall@10 / NDCG@10 (top-K) |
| Đặc tính | dễ, số lớn, **thổi phồng** | nghiêm, sát thực tế recommendation |
| Dùng ở | Phase 2 (ALS), paper chainRec một phần | Phase 3 (S1a–S4, module `rank_eval` chung) |

> **AUC ~0.95 đi kèm Recall@10 ~0.05 là KHỚP nhau**, không mâu thuẫn: AUC = P(positive xếp trên 1 negative ngẫu nhiên), không nhạy theo cutoff; với 1.57M item, AUC 0.95 vẫn để hàng chục nghìn item xếp trên positive → hiếm khi lọt top-10.

---

## 2. Kết quả theo từng bước

### S0 — Tái lập chainRec vanilla (sampled@500, stage *recommend*)

| Sampler | AUC | Recall@10 | NDCG@10 |
|---|---|---|---|
| uniform | 0.9437 | 0.7530 | 0.5921 |
| **stagewise** | **0.9479** | **0.7746** | **0.6193** |

→ Tái lập paper: `stagewise > uniform` (+2.9% R@10). Số sampled@500 cao nhưng bị thổi phồng.

### S1a — Cùng checkpoint, full-ranking (stage *recommend*)

| Sampler | AUC (full-rank) | Recall@10 | NDCG@10 |
|---|---|---|---|
| uniform | 0.9500 | 0.0506 | 0.0288 |
| stagewise | 0.9525 | 0.0500 | 0.0301 |

→ Recall@10 tụt từ ~0.77 xuống ~0.05 khi chuyển sampled@500 → full-ranking — đúng kỳ vọng, không phải bug.
Paper (Table 3, Goodreads/recommend) báo **AUC** 0.978–0.982; AUC của ta ~0.95 (thấp hơn ~0.03, do khác preprocessing / sample 48K user / định nghĩa positive).

### S2 — Head-to-head ALS vs chainRec (full-ranking, cùng test/pool/mask)

| Model | AUC | Recall@10 | NDCG@10 | Recall@20 |
|---|---|---|---|---|
| chainRec (uniform) | 0.9500 | 0.0506 | 0.0288 | 0.0774 |
| chainRec (stagewise) | 0.9525 | 0.0500 | 0.0301 | 0.0674 |
| **ALS (vanilla)** | **0.9646** | **0.1588** | **0.0988** | **0.2144** |
| itemPop (sàn) | 0.2918 | 0.0032 | 0.0016 | — |

→ **ALS thuần thắng chainRec** (AUC +0.012; Recall@10 gấp ~3×). ALS huấn luyện *trong chính index-space của chainRec* nên so sánh là apples-to-apples (chỉ khác kiến trúc model). itemPop ≈ 0.29 < 0.5 vì positive là sách niche đã bị mask.

### S3 — Engagement-weighted (F3) trên chainRec interactions (2×2)

| | vanilla (AUC) | +F3 (AUC) | Δ |
|---|---|---|---|
| chainRec | 0.9525 | 0.9502 | −0.2% (R@10 −2.0%) |
| ALS | 0.9646 | 0.9654 | +0.1% (R@10 −7.7%) |

**Coverage F3 trên edge interactions: chỉ 5.5%** (rec-edge 11.8%); độ lệch chuẩn trọng số σ = 0.087 → F3 gần như là *no-op* trong huấn luyện. → F3 **không transfer** sang chainRec — không phải vì F3 vô dụng mà vì **chain-structure (interactions) và review-length (reviews) ở hai nguồn chồng lấn thấp**.

### S3b — Phép test quyết định: ALS trên *reviews* (coverage F3 = 100%), full-ranking

| Confidence | AUC | Δ vs vanilla | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| vanilla | **0.8419** | — | 0.0062 | 0.0027 |
| +F3 (length) | 0.8350 | **−0.8%** | 0.0062 | 0.0030 |
| +votes | 0.7931 | **−5.8%** | 0.0064 | 0.0030 |
| itemPop (sàn) | 0.4636 | — | 0.0010 | 0.0003 |

→ **F3 KHÔNG sống sót dưới full-ranking** dù coverage 100% ⇒ **+3.6% của Phase 2 phần lớn là artifact của sampled@500.** **votes làm hại rõ (−5.8%)** ⇒ tái xác nhận popularity bias xuyên protocol.

### S4 — Hybrid ALS⊕chainRec (rank-fusion z-score) + multi-seed

α-sweep (α = trọng số ALS), full-ranking:

| α | AUC | Recall@10 | NDCG@10 |
|---|---|---|---|
| 0.00 (chainRec) | 0.9525 | 0.0500 | 0.0301 |
| 0.40 | 0.9599 | **0.1606** | 0.0983 |
| 0.60 | 0.9594 | 0.1602 | **0.0994** |
| 1.00 (ALS) | **0.9647** | 0.1588 | 0.0986 |

→ Hybrid **≈ ALS** (chênh top-K trong khoảng nhiễu ~1 SE), **bỏ xa chainRec**. Fusion không cải thiện có ý nghĩa so với mô hình mạnh hơn (ALS) → chainRec đóng góp rất ít trên Goodreads.

Multi-seed (full-ranking AUC, 3 seed): **uniform = 0.9446 ± 0.0068** (`[0.9503, 0.9484, 0.9350]`).
→ chainRec full-rank AUC khá nhạy seed (σ ~0.007); gap stagewise–uniform ở full-ranking nằm trong vùng nhiễu → ưu thế sampler là hiện tượng của **sampled@500**, không rõ ở full-ranking.

---

## 3. Đóng góp (khung báo cáo)

**Tiêu đề:** *Engagement-weighting và đánh giá nghiêm ngặt cho recommender Goodreads — review-length cải thiện ranking ở sampled@500 nhưng bộc lộ giới hạn dưới full-ranking.*

1. **(Dương)** review-length (F3) là engagement-weight tốt nhất @ sampled@500 (+3.6%); **(Dương)** stagewise > uniform (+2.9%).
2. **(Phương pháp)** Module `rank_eval` full-ranking chung cho ALS/chainRec/itemPop → vạch rõ **sampled@500 thổi phồng**: F3 −0.8%, ưu thế sampler ≈ nhiễu.
3. **(Khẳng định)** votes gây popularity bias — vững xuyên protocol.
4. **(Giới hạn)** review-length không transfer sang chainRec (coverage 5.5%, lệch nguồn dữ liệu).
5. **(Baseline)** ALS thuần ≥ chainRec; hybrid ≈ ALS — Goodreads bất lợi cho chainRec.

> Đây là đóng góp **methodological**: vừa có win thật, vừa cảnh báo về protocol evaluation — giá trị hơn một cải tiến +1% mong manh.

---

## 4. Hạn chế & việc còn lại

- Multi-seed `stagewise` ở S4 chưa chạy đủ 3 seed (mới seed=42 = 0.9503) — để khẳng định stagewise≈uniform full-rank cần chạy nốt.
- AUC full-rank của ta (~0.95) thấp hơn paper (~0.98) ~0.03 — cần truy nguyên preprocessing / sample 48K user / định nghĩa positive nếu muốn so trực tiếp số tuyệt đối.
- Eval mới ở stage *recommend*; muốn đa tầng (shelve/read/rate) như paper cần sửa `split_train_test` ở S0 để holdout mọi stage.
- `has_recsys/` vẫn ngoài báo cáo (kết quả âm) — quyết định: phụ lục hay tách repo.

---

## 5. Tái lập (notebook)

| Bước | Notebook | Nội dung |
|---|---|---|
| S0 | `goodreads/main pipeline/phase3_s0_reproduce_chainrec.ipynb` | Tái lập chainRec vanilla (sampled@500) |
| S1a | `phase3_s1a_fullrank_eval.ipynb` | `rank_eval` full-ranking + re-eval checkpoint |
| S2 | `phase3_s2_headtohead_eval.ipynb` | ALS↔chainRec head-to-head + cầu nối ID |
| S3 | `phase3_s3_f3_weighted.ipynb` | F3 edge-weighted (2×2 họ model × weighting) |
| S3b | `phase3_s3b_reviews_fullrank.ipynb` | F3/votes trên reviews dưới full-ranking |
| S4 | `phase3_s4_hybrid_multiseed.ipynb` | Hybrid α-sweep + multi-seed |

Artifact đẩy lên HuggingFace `vngclinh/goodreads-preprocessed` các thư mục `chainrec/`, `s2/`, `s3/`, `s3b/`, `s4/`.
