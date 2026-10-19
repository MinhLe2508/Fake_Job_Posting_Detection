# Tổng kết sản phẩm và kế hoạch nghiệm thu cá nhân

## Sản phẩm đã có minh chứng

| Hạng mục hoàn thành ở cấp nhóm | Minh chứng |
|---|---|
| Kiểm toán dữ liệu, chia tập, kiểm tra lớp hiếm | data/*manifest.csv; results/fold_audit.csv; features.py; test_pipeline.py |
| Tuning LR, SVM, RF và ablation | results/cv_results.csv; ablation.csv; experiment.py |
| Phân tích sai số, độ bất định và so sánh bổ sung | results/test_bootstrap.json; company_profile_slices.*; results/extension/ |
| Mô hình, demo, notebook, báo cáo và hướng dẫn tái lập | models/; demo/; Final_CRISP_DM.ipynb; report/; README.md |

AI Codex hỗ trợ viết mã, chạy thí nghiệm, tạo tài liệu và rà soát. Bảng trên xác nhận đầu ra có trong bộ nộp, không quy công việc quá khứ cho từng cá nhân khi chưa có chứng cứ.

## Phân công nghiệm thu sắp tới

Theo kế hoạch của nhóm ngày 14/10/2026, nhóm sẽ tạo GitHub repository sau đợt sửa kỹ thuật này. Các phần dưới đây là việc cần thực hiện, không phải lịch sử đóng góp đã được xác nhận.

| Thành viên | Phần kỹ thuật và tài liệu phụ trách nghiệm thu | Kiểm tra chéo | Minh chứng cần bổ sung |
|---|---|---|---|
| Lê Bá Khánh Bình | data/; features.py; kiểm tra split, fold, trùng profile; mục 2–3 báo cáo | Nguyễn Minh Đức | Lệnh kiểm tra + kết quả + giải thích/sửa thực tế + commit/PR |
| Nguyễn Minh Đức | experiment.py; extension.py; cấu hình và điều kiện đo; mục 4, 5.1, 5.8 | Lê Nhật Minh | Đối chiếu CV + log chạy + nhận xét hội tụ + commit/PR |
| Lê Nhật Minh | analysis.py; company_profile_slices.py; bootstrap, FP/FN; mục 5.2–5.7, 5.9 | Chu Đức Thành Nhân | Đối chiếu metric + giải thích giới hạn + commit/PR |
| Chu Đức Thành Nhân | inference.py; demo/; pyproject.toml; README; scripts/ và đóng gói; mục 6 | Lê Bá Khánh Bình | Cài môi trường mới + kiểm thử demo + log + commit/PR |

Mỗi người cần tự chạy, đọc, giải trình và sửa phần mình nhận; chỉ commit sao chép file có sẵn không chứng minh đã viết hay kiểm chứng file đó. Dùng danh tính thật và thời điểm thực tế; không tạo lịch sử hồi tố. Báo cáo chung do cả bốn người kiểm tra nội dung tương ứng; một người tích hợp để tránh xung đột file Word.

## Nhật ký nghiệm thu

| Ngày | Thành viên | Việc thực tế đã làm | File/commit/PR | Kết quả kiểm tra | Người kiểm tra chéo |
|---|---|---|---|---|---|
| Chưa ghi nhận | — | — | — | — | — |

Khi có nhật ký thực tế, cập nhật mục 7.1 báo cáo và xuất lại PDF, sau đó tạo lại SHA256SUMS và ZIP. Peer assessment được từng người tự điền, nộp riêng; không đưa phiếu đã chấm vào repository/ZIP dùng chung. Phân công dự kiến không thay thế tổng kết đóng góp cá nhân cuối cùng.
