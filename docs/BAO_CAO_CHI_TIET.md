# Báo cáo chi tiết — Hệ gợi ý quy mô lớn: Amazon → Goodreads → chainRec

> Bản báo cáo mở rộng & cập nhật (đặc biệt **Phase 3**). Bản PDF cũ: [`Nhóm 5.pdf`](Nhóm%205.pdf) là snapshot trước khi Phase 3 được huấn luyện & đánh giá đầy đủ.
> Số liệu chi tiết: [`RESULTS.md`](RESULTS.md) (VI) · [`RESULTS.en.md`](RESULTS.en.md) (EN). Demo: [`../goodreads/results_demo.html`](../goodreads/results_demo.html).
> Nhóm 5 — đồ án CS246-style.

---

## Tóm tắt (Abstract)

Đồ án xây dựng hệ gợi ý hoàn chỉnh qua ba phase trên dữ liệu lớn: **Phase 1** (Amazon Reviews 2018) dựng pipeline 5 tầng + ensemble và phát hiện giới hạn cấu trúc của Amazon cho bài toán xếp hạng; **Phase 2** (Goodreads UCSD, 15.7M reviews) so sánh 5 công thức *edge-weighting* cho ALS và tìm ra **review-length là tín hiệu engagement tốt nhất (F3, +3.6% Recall@10)**; **Phase 3** mở rộng bằng **chainRec** (Wan & McAuley, RecSys'18) và — đóng góp then chốt — **dựng một quy trình đánh giá full-ranking nghiêm ngặt** để kiểm chứng lại các kết luận của Phase 2.

Phát hiện trung tâm: **giao thức đánh giá sampled@500 (phổ biến) thổi phồng lợi ích của engagement-weighting** — lợi ích +3.6% của F3 *biến mất (−0.8% AUC)* khi chấm full-ranking; trong khi *tác hại của votes (popularity bias) thì bền vững* qua mọi giao thức. Đồng thời chúng tôi định lượng một **giới hạn khả-transfer**: tín hiệu review-length chỉ phủ 5.5% cạnh của đồ thị hành vi nên không thể "chuyển" sang chainRec. Cuối cùng, **ALS thuần ≥ chainRec** và **hybrid ≈ ALS** trên Goodreads full-ranking — tái xác nhận chính nhận định của paper rằng Goodreads là tập dữ liệu bất lợi nhất cho chainRec.

---

## 1. Giới thiệu & động lực

Bài toán: gợi ý vật phẩm (item recommendation) ở quy mô lớn, dùng Collaborative Filtering, Matrix Factorization, Content-based, Ensemble và mô hình chuỗi hành vi. Mục tiêu xuyên suốt không chỉ là "đạt điểm cao" mà là **hiểu vì sao một phương pháp thắng/thua**, và **đo lường trung thực** — kể cả khi kết quả trái với kỳ vọng.

**Bài học lặp lại của đồ án:** *chọn dữ liệu phù hợp với bài toán quan trọng ngang việc chọn thuật toán*, và *chọn giao thức đánh giá đúng quan trọng ngang việc chọn mô hình*.

---

## 2. Tổng quan ba phase

| Phase | Dữ liệu | Nội dung | Trạng thái |
|---|---|---|---|
| 1 | Amazon Reviews 2018 (Electronics, Home&Kitchen) | Pipeline 5 tầng → ensemble `ridge_positive`; phát hiện giới hạn xếp hạng full-catalog | ✅ |
| 2 | Goodreads UCSD (15.7M reviews) | Ablation 5 công thức edge-weighting cho ALS + LDA taste blend | ✅ |
| 3 | Goodreads (chuỗi 4 tầng) | chainRec + **đánh giá full-ranking** (S0→S5): head-to-head, transfer F3, hybrid, demo | ✅ |

---

## 3. Phase 1 — Amazon Reviews 2018 (tóm tắt)

Pipeline 5 tầng (baseline Bayesian shrinkage → feature engineering SVD-128 → 4 mô hình phụ A/B/C/D → ensemble → ranking eval), chống rò rỉ bằng temporal split, fit/transform chỉ trên train, test chấm đúng một lần.

**Kết quả (Electronics, test warm):** ensemble `ridge_positive` RMSE **1.1994** (−1.15% so baseline). Trọng số B 57.5% · D 25.0% · Baseline 8.8% · C 8.8%.

**Bước ngoặt — thất bại Stage 5:** khi gợi ý từ **toàn bộ 143K item**, mọi metric xếp hạng ≈ 0 (Recall@10 ≈ 0.0001), **coverage chỉ 1.56%** — mô hình luôn gợi ý vài nghìn item phổ biến. RMSE 1.20 "đẹp" nhưng **vô nghĩa cho gợi ý thực tế**. Nguyên nhân: sparsity cực đoan (71% item ≤20 review → embedding chỉ là nhiễu). → Đây là lý do chuyển sang Goodreads.

---

## 4. Phase 2 — Goodreads: ablation edge-weighting + ALS

Tiền xử lý: 6 JSON.gz (~16GB) → Parquet theo genre → HuggingFace; split thời gian **Train ≤2015 · Val 2016 · Test 2017**; giữ user ≥5 review train. Đánh giá: **sampled@500** (1 positive rating≥4 + 500 negative ngẫu nhiên), Recall/NDCG@10.

### 4.1 Đóng góp Phase 2 — Ablation 5 công thức (sampled@500)

| # | Công thức | Recall@10 | NDCG@10 | vs F2 |
|---|---|---|---|---|
| **F3 ★** | `rating × log(length+1) / 5` | **0.6122** | **0.5369** | **+3.6%** |
| F2 | `rating / 5` (baseline) | 0.5757 | 0.5107 | — |
| F5 | `1 if rating≥4 else 0` | 0.5613 | 0.4995 | −2.5% |
| F1 | `rating × log(votes+1) / 5` | 0.4744 | 0.4217 | −17.6% |
| F4 | `rating × log(votes+1) × log(length+1)` | 0.4237 | 0.3752 | −26.4% |

**Hai nhận định:** (A) **review-length là engagement-signal tốt nhất** — viết review dài ⇒ đọc kỹ, có cảm xúc; (B) **votes gây popularity bias** — sách phổ biến nhiều vote ⇒ thổi phồng, lấn át đuôi dài.

LDA taste (40 topic + 10 genre) blend với ALS, `final = 0.7·ALS + 0.3·taste` (+3.8% so ALS-only). Recall@10 cuối = 0.6122.

> ⚠️ Hạn chế đã ý thức từ Phase 2: dùng **sampled@500** ⇒ số bị thổi phồng; đóng góp thật là **delta giữa các biến thể (F3 vs F2 = +3.6%)**. Chính hạn chế này là động lực cho Phase 3.

---

## 5. Phase 3 — chainRec & đánh giá full-ranking (phần trọng tâm)

**Mục tiêu:** (1) so trực tiếp với paper tạo ra dataset; (2) khai thác *chuỗi hành vi đơn điệu* `shelve→read→rate→recommend`; (3) **kiểm chứng lại Phase 2 dưới giao thức nghiêm ngặt**.

Dữ liệu chains: 48,307 user × 1,567,258 item × 13.7M tương tác × 4 tầng (shelve 50.3% → read 3.3% → rate 13.9% → recommend 32.5%), từ `goodreads_interactions.csv` (UCSD).

### 5.1 Hai giao thức đánh giá

| | sampled@500 | full-ranking |
|---|---|---|
| Pool | 1 pos + 500 neg | toàn bộ 1.05–1.57M item, mask item đã xem |
| Metric | R@10 / N@10 | **AUC** + R@10 / N@10 |
| Tính chất | dễ, thổi phồng | nghiêm, sát thực tế |

> **AUC ~0.95 đi kèm R@10 ~0.05 là KHỚP** — AUC = P(positive xếp trên 1 negative ngẫu nhiên), không nhạy cutoff; với 1.57M item, AUC 0.95 vẫn để hàng chục nghìn item trên positive → hiếm lọt top-10.

### 5.2 S0 — Tái lập chainRec vanilla (sampled@500)

| Sampler | AUC | Recall@10 | NDCG@10 |
|---|---|---|---|
| uniform | 0.9437 | 0.7530 | 0.5921 |
| **stagewise** | 0.9479 | **0.7746** | 0.6193 |

→ Tái lập đúng paper: `stagewise > uniform` (**+2.9% R@10**) — hội tụ sạch.

### 5.3 S1a — Cùng checkpoint, full-ranking

| Sampler | AUC | Recall@10 | NDCG@10 |
|---|---|---|---|
| uniform | 0.9500 | 0.0506 | 0.0288 |
| stagewise | 0.9525 | 0.0500 | 0.0301 |

R@10 tụt ~0.77 → ~0.05 khi đổi sampled@500 → full-ranking (đúng kỳ vọng). Paper (Table 3, Goodreads/recommend) báo **AUC 0.978–0.982**; AUC của ta ~0.95 (thấp hơn ~0.03 — khác preprocessing / sample 48K user / định nghĩa positive).

### 5.4 S2 — Head-to-head ALS vs chainRec (full-ranking, cùng test/pool/mask)

ALS được huấn luyện **trong chính index-space của chainRec** (trên recommend-edges) ⇒ so sánh apples-to-apples, chỉ khác *kiến trúc model*.

| Model | AUC | Recall@10 | NDCG@10 |
|---|---|---|---|
| chainRec (stagewise) | 0.9525 | 0.0500 | 0.0301 |
| **ALS (vanilla)** | **0.9646** | **0.1588** | **0.0988** |
| itemPop (sàn) | 0.2918 | 0.0032 | 0.0016 |

→ **ALS thuần thắng chainRec** (AUC +0.012; R@10 gấp ~3×). Củng cố nhận định paper: *Goodreads là dataset bất lợi nhất cho chainRec*. itemPop ≈ 0.29 (<0.5) vì positive là sách niche, sách phổ biến đã bị mask.

### 5.5 S3 + S3b — Tín hiệu engagement có "transfer" không?

**S3 (chainRec interactions, 2×2):** F3 weight `loss_pos`.

| | vanilla AUC | +F3 AUC | Δ |
|---|---|---|---|
| chainRec | 0.9525 | 0.9502 | −0.2% |
| ALS | 0.9646 | 0.9654 | +0.1% |

Coverage F3 trên edge interactions chỉ **5.5%** (rec-edge 11.8%), độ lệch chuẩn trọng số σ = 0.087 → F3 gần như *no-op* khi train ⇒ **không transfer**. Lý do **cấu trúc**: chuỗi hành vi nằm ở *interactions*, review-length nằm ở *reviews* — hai nguồn chồng lấn thấp.

**S3b (phép test quyết định — ALS trên reviews, coverage F3 = 100%, full-ranking):**

| Confidence | AUC | Δ vs vanilla |
|---|---|---|
| vanilla | **0.8419** | — |
| +F3 (length) | 0.8350 | **−0.8%** |
| +votes | 0.7931 | **−5.8%** |
| itemPop | 0.4636 | — |

→ **F3 không sống sót dưới full-ranking** dù coverage 100% ⇒ **+3.6% của Phase 2 phần lớn là artifact của sampled@500.** **votes hại rõ (−5.8%)** ⇒ popularity-bias **bền xuyên giao thức**.

### 5.6 S4 — Hybrid ALS⊕chainRec + multi-seed

Rank-fusion z-score, sweep α (α = trọng số ALS):

| α | 0.0 (chainRec) | 0.40 | 0.60 | 1.0 (ALS) |
|---|---|---|---|---|
| AUC | 0.9525 | 0.9599 | 0.9594 | **0.9647** |
| R@10 | 0.0500 | 0.1606 | 0.1602 | 0.1588 |

→ Hybrid **≈ ALS** (chênh top-K trong nhiễu ~1 SE), **bỏ xa chainRec** → chainRec đóng góp rất ít trên Goodreads.
Multi-seed (full-rank AUC, 3 seed): **uniform = 0.9446 ± 0.0068** → chainRec full-rank khá nhạy seed; ưu thế sampler rõ ở sampled@500 nhưng thu hẹp về nhiễu ở full-ranking.

### 5.7 S5 — Web demo

`results_demo.html` (scrollytelling + dashboard): The Protocol Trap (toggle), head-to-head, ablation, hybrid α-sweep, transfer-limit, và **Live Recommendation Explorer** (top-8 chainRec/ALS/hybrid kèm bìa, từ `phase3_s5_precompute_demo.ipynb`).

---

## 6. ⭐ Đóng góp (Contributions)

**Đóng góp DƯƠNG (positive) — đã chứng minh:**

1. **Review-length là engagement-weight tốt nhất cho MF (F3, +3.6% R@10 @ sampled@500).** Một tín hiệu rẻ, sẵn có, vượt cả votes lẫn baseline rating.
2. **Tái lập chainRec: `stagewise > uniform` (+2.9% R@10).** Xác nhận thiết kế sampler của paper.

**Đóng góp PHƯƠNG PHÁP (điểm khác biệt cốt lõi):**

3. **Một quy trình đánh giá full-ranking thống nhất** (`rank_eval` model-agnostic: chấm toàn bộ 1.57M item, mask seen, AUC+Recall+NDCG) cho phép so **ALS / chainRec / hybrid / itemPop trên cùng test/pool/mask**. Quy trình này phơi bày rằng **sampled@500 thổi phồng lợi ích engagement-weighting**: F3 từ +3.6% (sampled) thành −0.8% (full-ranking). *Đây là cảnh báo phương pháp có giá trị: đa số công trình chỉ báo sampled metrics.*
4. **Định lượng giới hạn khả-transfer của tín hiệu.** Review-length chỉ phủ 5.5% cạnh interactions ⇒ không weight được chainRec — vì chuỗi hành vi và review-length nằm ở hai nguồn dữ liệu chồng lấn thấp. Đây là phát hiện về *điều kiện để một engagement-signal transfer được*, không phải khuyết điểm của F3.
5. **Tái xác nhận votes gây popularity bias xuyên giao thức** (−17.6% R@10 sampled @ Phase 2; −5.8% AUC full-ranking @ Phase 3) — kết luận bền nhất của đồ án.
6. **Bằng chứng độc lập: ALS thuần ≥ chainRec, hybrid ≈ ALS trên Goodreads full-ranking** — khẳng định bằng thực nghiệm chính nhận định "Goodreads bất lợi cho chainRec" của paper.

> **Bản chất đóng góp:** đây là công trình **methodological/empirical** — vừa có kết quả dương thật (F3, stagewise), vừa đưa ra cảnh báo đánh giá và một giới hạn transfer định lượng. Loại đóng góp này bền hơn một cải tiến +1% mong manh, vì nó nói cho cộng đồng biết *khi nào một con số đáng tin*.

---

## 7. Thảo luận & hạn chế

- **Số tuyệt đối AUC (~0.95) thấp hơn paper (~0.98) ~0.03** — chưa truy nguyên hết (preprocessing / sample 48K user / định nghĩa positive). Không ảnh hưởng các so sánh *nội bộ* (cùng pipeline).
- **Eval mới ở tầng *recommend*** — muốn đa tầng (shelve/read/rate) như paper cần sửa `split_train_test` ở S0 để holdout mọi tầng.
- **Multi-seed stagewise (S4)** chưa chạy đủ 3 seed; uniform đã có 3 seed.
- **F3 không transfer** là kết luận trong điều kiện dữ liệu interactions của UCSD; nguồn có review dày hơn có thể khác.
- **`has_recsys/`** (chẩn đoán sparsity, kết quả âm: HAS+SVD 1.4075 > SVD-raw 1.4056) là thí nghiệm độc lập **ngoài** báo cáo gốc — đề xuất đưa thành phụ lục hoặc tách repo.

---

## 8. Kết luận

Qua ba phase, đồ án đi từ "đạt RMSE đẹp" đến hiểu rằng **giao thức đánh giá quyết định kết luận**. Trên Goodreads: review-length là engagement-weight tốt *ở sampled@500* nhưng không bền *ở full-ranking*; votes luôn gây bias; ALS thuần đã đủ mạnh và chainRec không thắng được trên dataset này; và tín hiệu engagement chỉ transfer được khi nó cùng phủ trên cấu trúc dữ liệu mà mô hình học. Đóng góp lớn nhất không phải một thuật toán mới mà là **một khung đánh giá nghiêm ngặt và các kết luận trung thực rút ra từ nó**.

---

## 9. Tái lập & tài nguyên

| Bước | Notebook (`goodreads/main pipeline/`) |
|---|---|
| S0 | `phase3_s0_reproduce_chainrec.ipynb` |
| S1a | `phase3_s1a_fullrank_eval.ipynb` |
| S2 | `phase3_s2_headtohead_eval.ipynb` |
| S3 | `phase3_s3_f3_weighted.ipynb` |
| S3b | `phase3_s3b_reviews_fullrank.ipynb` |
| S4 | `phase3_s4_hybrid_multiseed.ipynb` |
| S5 (demo) | `phase3_s5_precompute_demo.ipynb` + `../results_demo.html` |

Artifact: HuggingFace `vngclinh/goodreads-preprocessed` (`chainrec/ s2/ s3/ s3b/ s4/ s5/`). Môi trường: Kaggle GPU T4 + `HF_TOKEN`.

---

## 10. Tài liệu tham khảo

- M. Wan, J. McAuley. *Item Recommendation on Monotonic Behavior Chains* (chainRec). RecSys 2018.
- Netflix Prize (2006–2009) — bài học RMSE vs ranking metrics, lợi ích cận biên của ensemble.
- Hu, Koren, Volinsky. *Collaborative Filtering for Implicit Feedback Datasets* (ALS confidence). ICDM 2008.
