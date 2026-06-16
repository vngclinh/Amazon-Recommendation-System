Mở đầu (~20s)

"Sang Goodreads, mục tiêu của tụi em là xây một hệ gợi ý sách và trả lời một câu hỏi: ngoài điểm số sao, còn tín hiệu nào giúp đoán đúng sở thích người đọc hơn không? Toàn
bộ Phase 2 xoay quanh việc thử nghiệm điều đó. Em xin mô tả luồng chạy theo 6 bước."

Bước 1 — Điểm xuất phát: bảng tương tác người đọc–sách (~25s)

"Sau khi làm sạch, tụi em có một bảng khổng lồ: dòng là người đọc, cột là cuốn sách, mỗi ô là một lần tương tác kèm rating. Bảng này rất thưa — mỗi người chỉ đọc vài chục
cuốn trong hàng triệu cuốn. Mô hình gợi ý sẽ học từ bảng này. Nhưng trước khi học, tụi em hỏi: mọi tương tác có đáng tin như nhau không? Câu trả lời là không — và đó là lý
do có bước 2."

Bước 2 — Edge-weight: gán "độ tin cậy" cho mỗi tương tác (~40s)

"Edge-weight nghĩa là: thay vì coi mọi lần đánh giá như nhau, tụi em gán cho mỗi ô một trọng số thể hiện mức độ đáng tin của tín hiệu đó. Câu hỏi là: dùng gì làm trọng số?
Tụi em thử 5 công thức, làm trong notebook branch B và stage 4:

- F2 — chỉ dùng rating (đây là mốc cơ sở).
- F3 — rating nhân với độ dài review. Ý tưởng: ai viết review dài thường đọc kỹ và có cảm xúc thật.
- F1, F4 — đưa thêm lượt vote (số người thấy review hữu ích).
- F5 — đơn giản hoá: thích hay không thích.

Mỗi công thức là một giả thuyết khác nhau về tín hiệu nào phản ánh sở thích thật. Tụi em sẽ để dữ liệu phân xử ở bước cuối."

Bước 3 — LDA: khám phá "gu đọc ẩn" từ nội dung review (~40s)

"Song song, tụi em muốn hiểu nội dung chứ không chỉ con số. Notebook branch A dùng một kỹ thuật tên là LDA — hãy hình dung nó như một cái máy đọc hàng triệu review rồi tự
nhóm các từ hay đi cùng nhau thành 40 chủ đề ẩn: ví dụ một chủ đề toàn từ về 'phép thuật, rồng, phiêu lưu', một chủ đề khác về 'tình cảm, gia đình'. Máy không được dạy
trước tên chủ đề — nó tự phát hiện. Kết quả: mỗi cuốn sách được mô tả bằng tỉ lệ pha trộn các chủ đề. Đây là cách biến văn bản thành con số mà máy hiểu được."

Bước 4 — Hồ sơ khẩu vị 50 chiều cho mỗi người đọc (~30s)

"Notebook stage 3 gộp hai nguồn lại thành hồ sơ khẩu vị cho từng người đọc, gồm 50 con số: 40 chiều là sở thích chủ đề (từ LDA), 10 chiều là sở thích thể loại. Quan trọng:
khi tính hồ sơ này, sách mà người ta chấm cao và đọc gần đây được tính nặng hơn — vì gu đọc thay đổi theo thời gian. Nói nôm na: đây là bản 'chân dung sở thích' của từng
độc giả."

Bước 5 — ALS: cỗ máy gợi ý chính (~40s)

"Trái tim của hệ là ALS — một thuật toán lọc cộng tác. Nguyên lý đơn giản: người có hành vi giống nhau thường thích sách giống nhau. ALS học cho mỗi người đọc và mỗi cuốn
sách một vector ẩn, sao cho nhân hai vector lại thì ra mức độ phù hợp. Điểm mấu chốt là: các trọng số edge-weight ở bước 2 được đưa thẳng vào ALS làm 'độ tin cậy' — tương
tác đáng tin thì ALS học mạnh hơn. Nhờ vậy, đổi công thức trọng số là đổi cách ALS học."

Bước 6 — Trộn ALS với khẩu vị (~25s)

"ALS giỏi bắt tín hiệu cộng tác, nhưng đôi khi bỏ sót gu nội dung. Nên tụi em trộn hai điểm: 70% từ ALS, 30% từ độ giống khẩu vị (hồ sơ 50 chiều ở bước 4). Con số 70/30
không phải đoán mò — tụi em dò trên tập validation và thấy tỉ lệ này tốt nhất. Khẩu vị đóng vai trò như một 'lớp tinh chỉnh' cho ALS."

Bước 7 — Chấm điểm và để dữ liệu phân xử (~40s)

"Cuối cùng là đánh giá. Với mỗi người đọc, tụi em lấy một cuốn họ thực sự thích trộn lẫn với 500 cuốn ngẫu nhiên, rồi xem mô hình có xếp cuốn thật lên top hay không — đo
bằng Recall@10 và NDCG@10. Chạy lần lượt cả 5 công thức trọng số. Kết quả phân xử rất rõ:

- F3 (độ dài review) thắng — tốt hơn chỉ-dùng-rating khoảng 3,6%.
- Các công thức dùng vote thua nặng — vì lượt vote đi theo độ nổi tiếng của sách, không phải sở thích cá nhân, nên nó kéo mô hình về phía sách phổ biến.

Đó là phát hiện chính của Phase 2: độ dài review là tín hiệu engagement tốt nhất, còn lượt vote thì gây thiên lệch phổ biến."

Kết (~15s)

"Tóm lại, Phase 2 là một quy trình khép kín: gán trọng số tin cậy cho tương tác → khám phá gu đọc bằng LDA → ALS sinh gợi ý → trộn thêm khẩu vị → và quan trọng nhất là để
dữ liệu chọn ra tín hiệu tốt nhất qua thí nghiệm 5 công thức. Phát hiện này cũng đặt nền cho Phase 3, nơi tụi em kiểm chứng lại nó dưới một thước đo nghiêm ngặt hơn."
