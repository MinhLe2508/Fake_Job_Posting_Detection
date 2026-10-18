# Lê Nhật Minh — Đánh giá, sai số và độ bất định

Trạng thái: **được phân công, chưa xác nhận hoàn thành**.

Nhánh: `work/minh`. Kiểm tra chéo: Chu Đức Thành Nhân.

## Các việc tự thực hiện

1. Tính lại precision/recall/F1 từ TP=120, FP=41, FN=24; đối chiếu test_predictions và test_metrics.
2. Giải thích bootstrap theo content_group, 187 hồ sơ chưa thấy với 11 tin giả và khoảng tin cậy rộng.
3. Đối chiếu LR/SVM ablation, hiệu chỉnh sigmoid và chi phí giả định; giải thích giới hạn suy luận.
4. Kiểm tra các ví dụ FP/FN, MODEL_CARD và tài liệu giải trình; ghi nhận lỗi thực tế nếu có.
5. Gửi nhận xét mục 5.2–5.7, 5.9, 5.11–5.12 cho Nhân tích hợp.

## File nhập vào repo

Copy **nội dung bên trong** ZIP cá nhân vào gốc repo; giữ nguyên đường dẫn. Các file kết quả/log là đầu ra đã có trước khi tạo repo, không phải minh chứng lần chạy của cá nhân.

- `evidence/profile_audit_2026-10-18.log`
- `evidence/profile_counts_verified_2026-10-18.json`
- `HUONG_DAN_GIAI_TRINH.md`
- `MODEL_CARD.md`
- `results/ablation.csv`
- `results/ablation_summary.csv`
- `results/analysis_summary.json`
- `results/company_profile_slices.csv`
- `results/company_profile_slices.json`
- `results/error_cases.json`
- `results/error_slices.csv`
- `results/feature_weights.csv`
- `results/figures/01_cv_ap.png`
- `results/figures/02_ablation.png`
- `results/figures/03_test_curves.png`
- `results/figures/04_confusion.png`
- `results/figures/05_validation_threshold.png`
- `results/figures/06_feature_weights.png`
- `results/robustness/calibration.json`
- `results/robustness/calibration_bins.csv`
- `results/robustness/calibration_train_oof.csv`
- `results/robustness/calibration_validation.csv`
- `results/robustness/cost_sensitivity.csv`
- `results/robustness/svm_ablation.csv`
- `results/robustness/svm_ablation_summary.csv`
- `results/test_bootstrap.json`
- `results/test_metrics.json`
- `results/test_predictions.csv`
- `results/validation_metrics.json`
- `results/validation_predictions.csv`
- `results/validation_threshold_curve.csv`
- `scripts/company_profile_slices.py`
- `src/fakejobs/analysis.py`

## Nhật ký nghiệm thu cần tự điền

| Ngày thực tế | Việc đã tự làm / lệnh chạy | Kết quả và hạn chế | File/log/commit/PR | Người kiểm tra chéo |
|---|---|---|---|---|
| Chưa ghi nhận | | | | |

Lưu log mới tại `contributions/minh/evidence/` khi đã chạy; không chép log lịch sử rồi đổi tên thành log cá nhân. Không ghi đã viết hoặc chạy lại nếu chỉ nhập file.
