# Lê Bá Khánh Bình — Dữ liệu và tiền xử lý

Trạng thái: **được phân công, chưa xác nhận hoàn thành**.

Nhánh: `work/binh`. Kiểm tra chéo: Nguyễn Minh Đức.

## Các việc tự thực hiện

1. Đối chiếu shape, số nhãn thật/giả, missing và cách loại trùng với notebook và báo cáo.
2. Đọc features.py, giải thích fit/transform, TF-IDF, log1p và cách ngăn leakage.
3. Kiểm tra split_manifest/cv_manifest: giao nhóm bằng 0; từng fold có đủ hai lớp; ghi số lượng từng lớp.
4. Chạy tests/test_pipeline.py sau khi ghép đủ repo; giải trình những kiểm tra thuộc dữ liệu.
5. Gửi nhận xét nội dung mục 2–3 báo cáo cho Nhân tích hợp.

## File nhập vào repo

Copy **nội dung bên trong** ZIP cá nhân vào gốc repo; giữ nguyên đường dẫn. Các file kết quả/log là đầu ra đã có trước khi tạo repo, không phải minh chứng lần chạy của cá nhân.

- `data/cv_manifest.csv`
- `data/data_dictionary.csv`
- `data/fake_job_postings.csv`
- `data/split_manifest.csv`
- `evidence/Fake_Job_Postings_Midterm.ipynb`
- `evidence/Fake_Job_Postings_Midterm.py`
- `midterm_artifacts/01_info.txt`
- `midterm_artifacts/01_preview_raw_5rows.csv`
- `midterm_artifacts/02_missing_raw.csv`
- `midterm_artifacts/03_class_raw.csv`
- `midterm_artifacts/04_duplicate_audit.csv`
- `midterm_artifacts/04_quarantined_ids.csv`
- `midterm_artifacts/04_removed_duplicate_ids.csv`
- `midterm_artifacts/05_split_manifest.csv`
- `midterm_artifacts/05_split_summary.csv`
- `midterm_artifacts/06_missing_train.csv`
- `midterm_artifacts/06_missing_train.png`
- `midterm_artifacts/06_word_count_describe_train.csv`
- `midterm_artifacts/07_iqr_train.csv`
- `midterm_artifacts/07_outlier_distributions_train.png`
- `midterm_artifacts/08_class_distribution_raw.png`
- `midterm_artifacts/09_correlation_train.csv`
- `midterm_artifacts/09_correlation_train.png`
- `midterm_artifacts/10_feature_names.csv`
- `midterm_artifacts/11_categorical_fitted_statistics.csv`
- `midterm_artifacts/11_matrix_summary.csv`
- `midterm_artifacts/11_numeric_fitted_statistics.csv`
- `midterm_artifacts/12_run_summary.json`
- `midterm_artifacts/13_cv_fold_audit.csv`
- `midterm_artifacts/13_cv_manifest.csv`
- `midterm_artifacts/14_submission_checks.json`
- `results/fold_audit.csv`
- `src/fakejobs/features.py`
- `tests/test_pipeline.py`

## Nhật ký nghiệm thu cần tự điền

| Ngày thực tế | Việc đã tự làm / lệnh chạy | Kết quả và hạn chế | File/log/commit/PR | Người kiểm tra chéo |
|---|---|---|---|---|
| Chưa ghi nhận | | | | |

Lưu log mới tại `contributions/binh/evidence/` khi đã chạy; không chép log lịch sử rồi đổi tên thành log cá nhân. Không ghi đã viết hoặc chạy lại nếu chỉ nhập file.
