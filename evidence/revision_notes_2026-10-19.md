# Đối chiếu bản chấm 18/10 và cập nhật 19/10

| Nhận xét | Xử lý trong bản mới | Minh chứng |
|---|---|---|
| Đóng góp và metadata bìa | Để nhóm bổ sung theo yêu cầu; không tự xác nhận | CONTRIBUTIONS.md, mục 7.1 và bìa |
| Chưa tách hồ sơ doanh nghiệp | Bổ sung 3 fold train, nhóm liên thông từ nội dung hoặc profile không rỗng | scripts/audit_generalization.py; results/generalization_2026-10-19/profile_* |
| Benchmark chỉ một lượt | Năm lần mỗi pipeline, thay thứ tự có seed; median và Q1/Q3, tách fit/predict | benchmark_repetitions.csv, benchmark_summary.csv, hardware.json |
| Precision 80% chưa đạt | Giữ 74,53% ở đầu tóm tắt và kết quả chính; không đổi ngưỡng | results/test_metrics.json, báo cáo |
| Văn phong dài và kết quả khó tìm | Rút gọn tóm tắt và mục 5.8, đặt kết quả trước diễn giải; thay bảng benchmark cũ trong phần trình bày bằng bảng lặp mới | Báo cáo Word/PDF |
| Giải trình calibration và nested | Bổ sung câu hỏi 9–12 kèm cách tự kiểm tra, bao gồm nhóm profile và benchmark | HUONG_DAN_GIAI_TRINH.md |

## Giới hạn còn thực chất

- Nhóm profile là đại diện văn bản, không thay company ID hoặc xác minh độc lập doanh nghiệp.
- AP mới 0,650 ± 0,177 và dao động fold 0,446–0,768 phải được công bố đầy đủ. Không coi điểm thấp là lỗi để tìm seed thuận lợi hơn.
- SVM C=1 đã chọn trước trên train; phép kiểm tra profile không phải nested theo doanh nghiệp và không xác nhận ngoài nguồn.
- Nested hiện vẫn chỉ LR/SVM; RF/NB/GB chưa được nested chung. Mở rộng tuning sau khi đã biết kết quả không tự khôi phục test độc lập. Bộ dữ liệu ngoài nguồn có nhãn xác minh vẫn chưa có.
- Năm lượt benchmark cùng máy và ma trận không phản ánh phân bố thời gian đa máy, lưu lượng người dùng hoặc toàn pipeline từ đọc file đến API.
- Các thí nghiệm chính và bổ sung 18/10 được giữ làm nguồn lịch sử; cập nhật 19/10 có thư mục và protocol riêng. Không thay hồi tố hardware, thời gian hoặc log lịch sử.

## Kiểm tra tái lập

`python scripts/verify_generalization.py` tính lại AP từ từng dự đoán, đối chiếu nhãn train, nhóm liên thông và thống kê benchmark. `python scripts/check_submission.py` chặn thiếu fold/lượt chạy, giao nhóm hoặc thay đổi các file đã khóa. Kết quả kiểm thử và thực thi notebook ngày 19/10 được lưu riêng trong evidence.

AI Codex hỗ trợ thực hiện đợt này. Không quy các log hỗ trợ này thành minh chứng cá nhân của thành viên. Điểm chính thức do giảng viên quyết định; bản sửa không cam kết điểm số.
