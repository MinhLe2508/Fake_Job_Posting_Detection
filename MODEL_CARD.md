# Thông tin mô hình đã khóa

## Mục đích và phạm vi

Hỗ trợ người kiểm duyệt ưu tiên xem tin tuyển dụng tiếng Anh đáng ngờ. Đơn vị dự đoán là một tin, lớp dương `fraudulent=1`. Đây là demo nghiên cứu trên Kaggle/EMSCAD, chưa được kiểm chứng cho Việt Nam hoặc dữ liệu tuyển dụng hiện tại. Không dùng làm quyết định tự động khóa tin hoặc kết luận gian lận.

## Dữ liệu và quy trình

- CSV gốc: 17.880 dòng, 18 cột, 866 tin giả (4,84%). Sau loại 315 bản trùng predictor: 17.565 dòng.
- Train / validation / test: 10.836 / 3.340 / 3.389 tin. Nhóm nội dung tạo từ title, description, requirements không giao nhau giữa các tập. Đây không phải phép chia cô lập nhà tuyển dụng.
- 5-fold StratifiedGroupKFold chỉ trên train; mỗi fold đánh giá có 106 tin giả. Tiền xử lý học riêng trong fold.
- Mô hình: LinearSVC C=1, class_weight=None; TF-IDF, độ dài log1p, OHE và cờ thiếu. Chọn cấu hình theo AP CV. Ngưỡng cố định −0,37317585243 được chọn trên validation để tối đa recall với precision ≥0,80.
- `models/lock.json` khóa checksum pipeline và protocol. Điểm SVM là decision score, không phải xác suất.

## Kết quả và điều kiện diễn giải

Test: AP 0,878146; precision 74,53%; recall 83,33%; F1 0,786885; TP/FP/FN/TN = 120/41/24/3.204. Mục tiêu precision 80% chưa đạt; không điều chỉnh mô hình/ngưỡng theo test.

2.548/3.389 tin test có company_profile đã thấy trong train. Với 841 tin không có hoặc chưa thấy profile (114 tin giả), precision là 72,22% và recall 79,82%. Lát cắt này là chẩn đoán hậu kiểm, không phải tập doanh nghiệp mới được thiết kế độc lập; chỉ 187 tin có profile không rỗng chưa thấy, với 11 tin giả. Profile trùng không đồng nghĩa chắc chắn cùng một doanh nghiệp vì thiếu company ID.

Khoảng bootstrap test theo content_group chỉ phản ánh lấy mẫu trên mô hình đã fit; không bao gồm lựa chọn mô hình hoặc toàn bộ phụ thuộc giữa doanh nghiệp. Dữ liệu lịch sử, nhãn kế thừa và văn bản lặp có thể làm kết quả lạc quan.

## Phân tích bổ sung

Naive Bayes dùng TF-IDF; HistGradientBoosting dùng combined + SelectKBest 2.000 đặc trưng. So sánh mở rộng là so sánh pipeline, không cô lập thuật toán. Năm fold có tập fit chồng lấn và được dùng chọn siêu tham số; các p-value, kể cả Nadeau–Bengio/Holm/Friedman, chỉ được giữ để truy nguyên phân tích thăm dò. Không dùng chúng xác nhận ưu thế hay tương đương. Validation đã tham gia chọn ngưỡng; bootstrap trên đó cũng là chẩn đoán.

## Sử dụng và bảo trì

Chạy `demo/app.py` sau cài đặt theo README. Chỉ nạp joblib tin cậy đi kèm. Cần dữ liệu có nhãn mới, chia theo doanh nghiệp/thời gian và đánh giá ngoài nguồn trước pilot. Ngày 18/10 đã bổ sung nested CV 3×3 cho LR/SVM trên train, ablation SVM và sigmoid học từ OOF train, chấm validation. Các phân tích sau khóa chưa xác nhận hiệu quả ngoài nguồn và chưa được đưa vào mô hình triển khai; xem results/robustness và mục 5.10–5.12. Mọi thay đổi cần protocol mới; test hiện tại không được tái sử dụng để lựa chọn.

Nguồn số liệu: `results/test_metrics.json`, `results/company_profile_slices.csv`, `results/extension/`. Tác giả nhóm: Lê Bá Khánh Bình; Nguyễn Minh Đức; Lê Nhật Minh; Chu Đức Thành Nhân. AI Codex hỗ trợ xây dựng và rà soát; hồ sơ đóng góp cá nhân được quản lý riêng bằng nghiệm thu thực tế.

## Chẩn đoán nhóm profile ngày 19/10

SVM cố định được fit trên 3 fold của train, với thành phần liên thông từ content_group hoặc profile chuẩn hóa không rỗng. Cách này chặn giao profile khớp chính xác và giao nhóm nội dung cùng lúc. Kết quả ở `results/generalization_2026-10-19/profile_summary.json`; không thay kết quả test ở trên. Profile thiếu không được gộp thành một công ty. Một doanh nghiệp có thể dùng nhiều profile và nhiều doanh nghiệp có thể dùng một mẫu: chưa chứng minh độc lập theo doanh nghiệp. Cấu hình đã chọn trên train trước đây nên vẫn có thiên lệch lựa chọn; chưa nested toàn bộ năm pipeline.

Benchmark mới lặp 5 lần mỗi pipeline, báo cáo median/Q1/Q3 trong `benchmark_summary.csv`. Các lượt cùng máy/ma trận, không chứng minh tốc độ trên máy khác hay SLA API.


## Thành phần fold và cách dùng số đo

Phân tích ngày 20/10 giữ nguyên mọi dự đoán: fold 3 có tỷ lệ giả 3,96%, ROC-AUC 0,864 và trung vị 237 tin thật đứng trước/bằng điểm một tin giả. Hai nhóm chứa 41 và 11 tin giả chiếm 30,59% tin giả của fold, có thứ hạng trung vị 559 và 1.170. Đây là giải thích mô tả, không cô lập nguyên nhân. Xem PHAN_TICH_FOLD_PROFILE.md.

Test AP 0,878 mô tả pipeline đã khóa trên test nội bộ theo nội dung; profile CV 0,650 mô tả độ nhạy của cấu hình cố định trên ba fold train khác. Cả hai chưa đủ xác nhận hiệu năng trên doanh nghiệp mới hoặc ngoài nguồn. Không sửa ngưỡng hoặc thay số test bằng kết quả hậu kiểm.

