# Mục lục chi tiết & checklist tài sản — Báo cáo môn học (RecSys: Amazon → Goodreads → chainRec)

> Tài liệu này (a) đối chiếu bản `Untitled document.docx` với code + `RESULTS.md` + `README.md`,
> (b) đề xuất **mục lục chi tiết kiểu báo cáo môn học** đi đúng mạch *dataset → tiền xử lý → EDA →
> pipeline → triển khai → kết quả → nhận xét → insight*, với **Contributions đặt ngay đầu**,
> (c) đánh dấu mọi chỗ cần **ảnh/bảng** và ghi rõ thứ **bạn cần cung cấp thêm**.

---

## A. Đối chiếu nhanh: docx ↔ code/kết quả thực tế

**Kết luận: bản docx đã đúng và nhất quán với code.** Mọi con số kiểm tra được đều khớp:

| Hạng mục | docx | Code / RESULTS.md | Khớp? |
|---|---|---|---|
| Electronics sau 5-core | 5.5M / 609K / 145K / >99.99% | 5,517,640 / 609K / 145K | ✅ |
| Home&Kitchen | 5.6M / 647K / 170K | 5,622,812 / 647K / 170K | ✅ |
| Baseline Electronics | RMSE 1.21344 · MAE 0.92511 | idem | ✅ |
| Model B | RMSE 1.21267 · MAE 0.86822 | idem | ✅ |
| Ensemble ridge_positive | RMSE 1.19943 (−1.15%) | idem | ✅ |
| Coverage full-catalog | 1.56% | 1.56% (2,239/143,281) | ✅ |
| F3 vs F2 (sampled@500) | 0.6122 / 0.5757 (+3.6%) | idem | ✅ |
| chainRec S0 (sampled) | stagewise 0.7746 > uniform 0.7530 | idem (+2.9%) | ✅ |
| ALS vs chainRec (full-rank) | ALS AUC 0.9646 / R@10 0.1588; chainRec ~0.05 | idem | ✅ |
| α taste blend | 0.7 | 0.7 | ✅ |

→ **Không cần sửa số liệu.** Bản docx thiếu một số *phần* và *chiều sâu Phase 3* (xem mục B), chứ không sai.

### Những chỗ docx còn thiếu / nên bổ sung (so với kết quả đã có trong repo)

1. **Chưa có Abstract & chưa có ô "Contributions" ở đầu** — yêu cầu bắt buộc của bạn. (→ thêm mục 0, mục C.0)
2. **Phase 3 bị tóm tắt quá mỏng** so với `RESULTS.md`. Còn thiếu các kết quả *đã chạy thật*:
   - **Giới hạn transfer định lượng**: F3 chỉ phủ **5.5%** cạnh interactions (rec-edge 11.8%), σ trọng số = 0.087 → gần như no-op. docx mới nói "small part".
   - **Phép test quyết định S3b** (ALS trên *reviews*, coverage F3 = **100%**, full-ranking): vanilla AUC 0.8419 → **+F3 −0.8%**, **+votes −5.8%**, itemPop 0.4636. docx chưa có bảng này.
   - **Hybrid α-sweep** (α=0→1): AUC 0.9525→0.9647, R@10 đỉnh 0.1606 @ α=0.40. docx mới nói "tương tự ALS".
   - **Multi-seed**: uniform full-rank AUC = **0.9446 ± 0.0068** → ưu thế sampler nằm trong vùng nhiễu ở full-ranking.
   - **itemPop làm sàn** (AUC 0.2918 full-rank) — cần để đọc đúng AUC.
   - **Quy mô chuỗi hành vi**: 48,307 user × 1,567,258 item × 13.7M tương tác; phân bố `shelve 50.3% · read 3.3% · rate 13.9% · recommend 32.5%`.
3. **Bảng ablation 5.3 nên hiện đủ 5 công thức** F1–F5 (docx mới nêu F2/F3); thêm gain LDA blend **+3.8%** so ALS-only.
4. **Khoảng cách AUC ~0.95 vs paper ~0.98 (~0.03)** nên ghi vào Limitations (do preprocessing / sample 48K user / định nghĩa positive).
5. **`has_recsys/` (kết quả âm: HAS+SVD 1.4075 > SVD-raw 1.4056)** — quyết định: đưa **Phụ lục** hay bỏ. Báo cáo môn học nên để 1 đoạn phụ lục ngắn (kể cả kết quả âm là điểm cộng về tính trung thực).
6. **"Hai giao thức đánh giá" (sampled@500 vs full-ranking)** là đóng góp *phương pháp* mạnh nhất nhưng đang bị rải rác — nên tách thành 1 tiểu mục riêng (5.1) và nhấn trong Contributions.

---

## B. Mục lục chi tiết đề xuất (course report)

> Tiêu đề mục để tiếng Anh cho khớp bản docx; chú thích tiếng Việt. Mạch trình bày mỗi dataset:
> **giới thiệu dataset → tiền xử lý → EDA → pipeline → triển khai → kết quả → nhận xét → insight**.
> Ký hiệu: 🖼 = cần hình, 📊 = cần bảng, ✅ = đã có sẵn trong repo, ⚠️ = **bạn cần tạo/cung cấp**.

### 0. Front matter
- **0.1 Abstract** *(mới)* — 8–12 dòng: bài toán, 2 dataset, 3 phase, phát hiện cốt lõi (sampled@500 thổi phồng; review-length tốt ở sampled nhưng không bền; votes gây bias; ALS ≥ chainRec).
- **0.2 ⭐ Contributions** *(mới — ĐẶT NGAY ĐẦU, yêu cầu của bạn)* — xem nội dung sẵn ở mục C.0 dưới.
- Table of Contents · List of Figures · List of Tables *(đã có trong docx)*.

### Chapter 1 — Overview *(giữ nguyên docx, tốt)*
1.1 Mining Massive Datasets · 1.2 Topic · 1.3 Problem Statement (4 câu hỏi) · 1.4 Objectives & Scope · 1.5 Methodology · 1.6 Report Structure.

### Chapter 2 — Theoretical Background *(giữ nguyên docx)*
2.1 RecSys · 2.2 CF · 2.3 MF · 2.4 Implicit + ALS · 2.5 Content-based · 2.6 Ensemble · 2.7 Metrics (thêm 1 câu định nghĩa **AUC** & vì sao AUC cao đi với Recall@10 thấp) · 2.8 Large-scale processing.

### Chapter 3 — Datasets and Data Processing
- **3.1 Dataset Candidates** — 📊 Table 3.1 (✅ đã có: MovieLens/Yelp/Last.fm/Amazon/Goodreads).
- **3.2 Amazon Reviews 2018** — 📊 Table 3.2 thống kê sau lọc (✅). Thêm: 7 bước làm sạch + SimHash dedup + temporal split.
- **3.3 Goodreads UCSD Book Graph** — mô tả 6 JSON.gz, 10 genre, split Train≤2015/Val2016/Test2017. 📊 *(nên thêm)* bảng quy mô + (Phase 3) phân bố 4 tầng hành vi.
- **3.4 Reason for Changing Amazon → Goodreads** — nối thẳng tới thất bại full-catalog (3.x ↔ 5.3).
- **3.5 Large-Scale Data Processing Pipeline** — 🖼 **Figure 3.1** *(✅ `goodreads/charts/pipeline-data-processing.jpg`)*.
- **3.6 Exploratory Data Analysis** — phần EDA chính:
  - Amazon: rating mất cân bằng (H&K 65.5% 5★), sparsity 99.99%, đuôi dài. 🖼 **Figure 3.2 (Amazon rating dist)** ⚠️ *(chưa export — xem D)*.
  - Goodreads: review-length tăng theo rating; bridge users (≥3 genre); volume theo thời gian; popularity vs quality.
    🖼 **Figure 3.3 review volume over time** ✅ `newplot3.png` · 🖼 **Figure 3.4 review length by rating** ✅ (1 trong `newplot2/4/5`) · 🖼 **Figure 3.5 book/author popularity** ✅ `newplot6.png` · *(overview stats `newplot.png` có thể làm Figure 3.0)*.
- **3.7 Processing Challenges & Solutions** *(giữ nguyên docx — RAM overflow, 2-pass stats, mixed date, Hf path)*.

### Chapter 4 — System Implementation
- **4.1 Overall Development Strategy** — 🖼 **Figure 4.1 overall pipeline** *(✅ có thể dùng 2 hình pipeline mới ghép, hoặc 1 sơ đồ 3-phase)*.
- **4.2 Phase 1: Amazon Pipeline (5 stage)** — 📊 Table 4.1 (✅). 🖼 **Figure 4.2 Amazon five-stage pipeline** ✅ **`src/visualize_data/amazon_recsys_pipeline.png/svg`** *(mình vừa tạo)*. Mô tả anti-leakage, baseline Bayesian, Model A/B/C/D, ensemble 16 ứng viên.
- **4.3 Phase 2: Goodreads Pipeline (4 phase)** — 📊 Table 4.2 (✅). 🖼 **Figure 4.3 Goodreads pipeline** ✅ **`goodreads/goodreads_recsys_pipeline.png/svg`** *(mình vừa tạo; thay cho `pipeline.jpg` cũ đã lỗi thời)*.
- **4.4 Edge Weighting Formulas** — 📊 Table 4.3 (✅ 5 công thức F1–F5).
- **4.5 LDA-Based Taste Profile** — 50-dim (40 topic + 10 genre), trọng số `(rating−3.5)·exp(−λ·days_ago)`.
- **4.6 ALS + Taste Re-ranking** — `final = 0.7·ALS + 0.3·taste`; thêm câu **+3.8% so ALS-only**.
- **4.7 Phase 3: chainRec + Full-Ranking Evaluation** — *(mở rộng)*: chuỗi `shelve→read→rate→recommend`; module `rank_eval` model-agnostic (mask seen, chấm toàn bộ 1.57M item, AUC+R@K+NDCG); liệt kê S0–S5.
- **4.8 Web Demo** — 🖼 **Figure 5.5 web demo** ⚠️ *(cần screenshot `goodreads/results_demo.html`)*.

### Chapter 5 — Results and Evaluation
- **5.1 Evaluation Protocol** — **tách rõ "hai giao thức"**: 📊 *(nên thêm)* bảng sampled@500 vs full-ranking (pool / metric / tính chất). Nhấn: đây là trục phương pháp của báo cáo.
- **5.2 Results on Amazon** — 📊 Table 5.1 Electronics (✅) · 📊 Table 5.2 H&K (✅). 🖼 **Figure 5.1 so sánh model rating-prediction** ⚠️ *(bar chart RMSE/MAE: Baseline vs B vs Ensemble — mình tạo được)*.
- **5.3 Limitations from Amazon (bước ngoặt)** — coverage 1.56%, metric≈0, within-test NDCG≈0.98. 🖼 **Figure 5.2 full-catalog ranking failure** ⚠️ *(mình tạo được)*.
- **5.4 Results on Goodreads** — F3 winner; nhận xét review-length.
- **5.5 Edge Weighting Ablation** — 📊 Table 5.3 **đủ 5 dòng F1–F5** (✅ số có sẵn). 🖼 **Figure 5.3 ablation bar chart** ⚠️ *(mình tạo được)*.
- **5.6 chainRec & Full-Ranking** — 📊 Table 5.4 sampled@500 (✅) · 📊 Table 5.5 **mở rộng**: thêm itemPop, và 📊 *(thêm)* **Table 5.6 S3b** (vanilla/+F3/+votes trên reviews) + **Table 5.7 hybrid α-sweep**. 🖼 **Figure 5.4 sampled@500 vs full-ranking** ⚠️ *(mình tạo được — "protocol trap")*.
- **5.7 Discussion** *(giữ nguyên 6 bài học của docx — rất tốt)*.

### Chapter 6 — Conclusion and Future Work *(giữ nguyên docx)*
6.1 Conclusion · 6.2 Limitations *(thêm: AUC gap ~0.03 vs paper; eval mới ở tầng recommend)* · 6.3 Future Work *(LightGCN, đa tầng, cold-start, tiếng Việt)* · 6.4 Work Allocation 📊 Table 6.1 ⚠️ *(bạn điền tên thành viên)*.

### (Phụ lục) Appendix A — HAS negative result *(tuỳ chọn)*
`has_recsys/` chẩn đoán sparsity theo brand (Type A/B/C) → HAS+SVD 1.4075 > SVD-raw 1.4056. 🖼 ✅ `fig1_sparsity_distributions.png`, `fig2_rmse_per_star.png`. Để 1 trang phụ lục: trung thực về một hướng đã thử và thất bại.

### References *(✅ đã có 10 mục — đủ cho báo cáo môn học)*

---

## C. Nội dung soạn sẵn để dán

### C.0 ⭐ Contributions (đặt ngay đầu báo cáo)

> **Đóng góp của nhóm.** Báo cáo không nhằm đạt một con số cao mà nhằm hiểu *khi nào một kết quả đáng tin*.
> Cụ thể:
>
> **(Kết quả dương — đã chứng minh)**
> 1. **Review-length là tín hiệu engagement tốt nhất cho ALS**: F3 = `rating × log(length+1)/5` đạt
>    **+3.6% Recall@10** so baseline chỉ-rating (sampled@500), vượt cả votes.
> 2. **Tái lập chainRec**: sampler `stagewise > uniform` (**+2.9% Recall@10**) — đúng thiết kế của paper gốc.
>
> **(Đóng góp phương pháp — điểm khác biệt cốt lõi)**
> 3. **Một quy trình đánh giá full-ranking thống nhất** (`rank_eval` model-agnostic: mask seen, chấm toàn bộ
>    1.57M item, AUC+Recall+NDCG) cho phép so ALS / chainRec / hybrid / itemPop trên **cùng test/pool/mask**.
>    Quy trình này phơi bày rằng **sampled@500 thổi phồng** lợi ích engagement-weighting: F3 **+3.6% (sampled)
>    → −0.8% AUC (full-ranking)**.
> 4. **Định lượng giới hạn khả-transfer**: review-length chỉ phủ **5.5%** cạnh đồ thị hành vi → không thể
>    weight chainRec một cách có nghĩa (chuỗi hành vi và review nằm ở hai nguồn dữ liệu chồng lấn thấp).
> 5. **Tái xác nhận votes gây popularity bias xuyên giao thức** (−17.6% R@10 sampled @ Phase 2; −5.8% AUC
>    full-ranking @ Phase 3) — kết luận bền nhất.
> 6. **Bằng chứng độc lập: ALS thuần ≥ chainRec; hybrid ≈ ALS** trên Goodreads full-ranking — khẳng định
>    chính nhận định "Goodreads bất lợi cho chainRec" của paper.
>
> **(Kỹ thuật dữ liệu lớn)** pipeline xử lý ~16GB JSON.gz → Parquet theo genre, 2-pass global stats, SimHash
> dedup, temporal split chống rò rỉ, đẩy artifact lên HuggingFace để tái dùng.

---

## D. Checklist ẢNH & BẢNG — cái nào có, cái nào bạn cần thêm

### ✅ Đã có sẵn trong repo
| Figure/Table | File | Ghi chú |
|---|---|---|
| Fig 3.1 data processing | `goodreads/charts/pipeline-data-processing.jpg` | dùng được |
| Fig 3.3 volume over time | `goodreads/charts/newplot3.png` | dùng được |
| Fig 3.4 review length by rating | `goodreads/charts/newplot2/4/5.png` | **kiểm tra chọn đúng tấm** |
| Fig 3.5 popularity vs quality | `goodreads/charts/newplot6.png` | dùng được |
| (Fig 3.0) dataset overview | `goodreads/charts/newplot.png` | tuỳ chọn |
| Fig 4.2 Amazon pipeline | `src/visualize_data/amazon_recsys_pipeline.png/svg` | **mình vừa tạo** |
| Fig 4.3 Goodreads pipeline | `goodreads/goodreads_recsys_pipeline.png/svg` | **mình vừa tạo** |
| Table 3.1, 3.2, 4.1, 4.2, 4.3, 5.1, 5.2, 5.4 | trong docx | dùng được |
| (Phụ lục) HAS sparsity / rmse | `has_recsys/figures/fig1,fig2.png` | nếu làm appendix |

### ⚠️ CẦN TẠO / BẠN CẦN CUNG CẤP
| # | Cần gì | Vì sao thiếu | Ai làm |
|---|---|---|---|
| Fig 3.2 | **Amazon rating distribution** (mất cân bằng, H&K 65.5% 5★) + sparsity | EDA Amazon (`src/visualize_data/EDA.ipynb`) **chưa export PNG** | **Mình tạo được** nếu bạn xác nhận số phân bố sao/5; hoặc bạn export từ notebook |
| Fig 5.1 | **Bar chart** RMSE/MAE: Baseline vs Model B vs Ensemble (Electronics) | chưa có | **Mình tạo được** (đã có số) |
| Fig 5.2 | **Amazon full-catalog ranking failure** (coverage 1.56%, metric≈0 vs within-test 0.98) | chưa có | **Mình tạo được** |
| Fig 5.3 | **Ablation 5 công thức** F1–F5 (Recall@10 & NDCG@10) | chưa có | **Mình tạo được** (đã có số) |
| Fig 5.4 | **sampled@500 vs full-ranking** ("protocol trap": R@10 0.77→0.05; F3 +3.6%→−0.8%) | chưa có | **Mình tạo được** |
| Fig 5.5 | **Screenshot web demo** `results_demo.html` | cần ảnh chụp màn hình | **Bạn chụp** (mình không mở được browser) |
| Table 5.3 | mở rộng đủ **F1–F5** | docx mới có F2/F3 | mình điền sẵn số bên dưới |
| Table 5.6 | **S3b**: reviews vanilla/+F3/+votes (AUC 0.8419/0.8350/0.7931) | chưa có | mình điền sẵn |
| Table 5.7 | **Hybrid α-sweep** (α=0/0.4/0.6/1.0) | chưa có | mình điền sẵn |
| Table 6.1 | **Work allocation** | cần tên thành viên + phân công | **Bạn điền** |

### Số liệu điền sẵn cho các bảng còn thiếu
**Table 5.3 (ablation đầy đủ, sampled@500):**

| # | Formula | Recall@10 | NDCG@10 | vs F2 |
|---|---|---|---|---|
| F3 ★ | `rating×log(length+1)/5` | 0.6122 | 0.5369 | +3.6% |
| F2 | `rating/5` (baseline) | 0.5757 | 0.5107 | — |
| F5 | `1 if rating≥4 else 0` | 0.5613 | 0.4995 | −2.5% |
| F1 | `rating×log(votes+1)/5` | 0.4744 | 0.4217 | −17.6% |
| F4 | `rating×log(votes+1)×log(length+1)` | 0.4237 | 0.3752 | −26.4% |

**Table 5.6 (S3b — ALS trên reviews, coverage F3=100%, full-ranking):**

| Confidence | AUC | Δ vs vanilla |
|---|---|---|
| vanilla | 0.8419 | — |
| +F3 (length) | 0.8350 | −0.8% |
| +votes | 0.7931 | −5.8% |
| itemPop (sàn) | 0.4636 | — |

**Table 5.7 (Hybrid ALS⊕chainRec, full-ranking):**

| α (trọng số ALS) | AUC | Recall@10 |
|---|---|---|
| 0.00 (chainRec) | 0.9525 | 0.0500 |
| 0.40 | 0.9599 | 0.1606 |
| 0.60 | 0.9594 | 0.1602 |
| 1.00 (ALS) | 0.9647 | 0.1588 |
