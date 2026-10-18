# Nguyễn Minh Đức — Mô hình hóa và nested CV

Trạng thái: **được phân công, chưa xác nhận hoàn thành**.

Nhánh: `work/duc`. Kiểm tra chéo: Lê Nhật Minh.

## Các việc tự thực hiện

1. Đối chiếu cấu hình LR/SVM/RF, quy tắc chọn AP và ngưỡng validation với các file kết quả.
2. Đọc nested CV 3×3: preprocessing fit riêng từng vòng, 144 inner fit và 6 outer fit; ghi nhận cảnh báo hội tụ.
3. Đối chiếu benchmark mới với nguồn lịch sử; giải thích fit/predict, số luồng và phần cứng.
4. Chạy tests/test_extension.py và tests/test_robustness.py sau khi ghép đủ repo.
5. Gửi nhận xét mục 4, 5.1, 5.8, 5.10 cho Nhân; nếu chạy lại thí nghiệm, dùng bản sao và giữ riêng kết quả mới.

## File nhập vào repo

Copy **nội dung bên trong** ZIP cá nhân vào gốc repo; giữ nguyên đường dẫn. Các file kết quả/log là đầu ra đã có trước khi tạo repo, không phải minh chứng lần chạy của cá nhân.

- `evidence/benchmark_2026-10-18.log`
- `evidence/robustness_2026-10-18.log`
- `models/fake_job_detector.joblib`
- `models/lock.json`
- `results/best_per_model.csv`
- `results/cv_results.csv`
- `results/cv_summary.csv`
- `results/environment.json`
- `results/extension/cv_results_extension.csv`
- `results/extension/cv_summary_extension.csv`
- `results/extension/measurement_provenance.json`
- `results/extension/protocol_extension.json`
- `results/extension/run.log`
- `results/extension/stats_cv.json`
- `results/extension/stats_cv_pairwise.csv`
- `results/extension/validation_compare.json`
- `results/protocol.json`
- `results/robustness/benchmark.json`
- `results/robustness/completion_all.json`
- `results/robustness/hardware.json`
- `results/robustness/nested_fold_audit.csv`
- `results/robustness/nested_inner.csv`
- `results/robustness/nested_outer.csv`
- `results/robustness/nested_predictions.csv`
- `results/robustness/nested_summary.csv`
- `results/robustness/protocol.json`
- `results/robustness/runtime_threads.json`
- `results/selection.json`
- `scripts/benchmark_extension.py`
- `src/fakejobs/experiment.py`
- `src/fakejobs/extension.py`
- `src/fakejobs/robustness.py`
- `tests/test_extension.py`
- `tests/test_robustness.py`

## Nhật ký nghiệm thu cần tự điền

| Ngày thực tế | Việc đã tự làm / lệnh chạy | Kết quả và hạn chế | File/log/commit/PR | Người kiểm tra chéo |
|---|---|---|---|---|
| Chưa ghi nhận | | | | |

Lưu log mới tại `contributions/duc/evidence/` khi đã chạy; không chép log lịch sử rồi đổi tên thành log cá nhân. Không ghi đã viết hoặc chạy lại nếu chỉ nhập file.
