# Đối chiếu các chỉnh sửa ngày 14–15/10/2026

| Vấn đề | Chỉnh sửa và minh chứng | Trạng thái |
|---|---|---|
| Hướng dẫn cài package bị gián đoạn | Bổ sung pyproject.toml, thử pip install -e . --no-deps trong virtualenv mới; cài đúng requirements.txt, pip check đạt | Đã kiểm tra |
| README nhắc file không có | Bổ sung MODEL_CARD.md và export_pdf.ps1; bỏ hướng dẫn builder/requirements-report không có, xác định Word là nguồn biên tập | Đã sửa |
| Thời gian không khớp JSON | Bảng Word dùng GB 66,59 s, RF 42,65 s và số tương ứng của các pipeline còn lại; check_submission.py đối chiếu tự động | Đã kiểm tra |
| Tên fit_seconds không đúng phạm vi đo lịch sử | Nêu rõ lịch sử gộp fit + chấm validation + AP, không gồm preprocessing; mã mới tách fit/predict và có kiểm thử hồi quy | Đã sửa mã và diễn giải; không giả lập số fit thuần cho lượt cũ |
| Kiểm định bị diễn giải quá mức | Sửa báo cáo, notebook, model card và metadata: phụ thuộc fold, selection bias, p-value thăm dò, khác biểu diễn đầu vào | Đã sửa; chưa có thí nghiệm nested CV mới |
| Profile lặp và đánh giá doanh nghiệp mới | Đưa kết quả lát cắt vào tóm tắt; giải thích thiếu profile khác với doanh nghiệp mới, không coi profile là company ID | Đã sửa |
| Notebook chưa chứa phân tích bổ sung | Thêm bảng profile, so sánh pipeline, nguồn thời gian và giới hạn thống kê; thực thi 15 ô code | Đã kiểm tra |
| Bộ tái lập thiếu script phân tích profile | reproduce.py sao chép cả scripts/; README chỉ rõ thứ tự quy trình chính rồi mở rộng trong thư mục mới | Đã sửa |
| Bản Word/PDF chưa đồng bộ | Xuất PDF bằng Microsoft Word và kiểm tra hình ảnh đủ 26 trang, không cắt nội dung | Đã kiểm tra |
| Tổng kết đóng góp cá nhân | Tổng kết sản phẩm cấp nhóm; CONTRIBUTIONS.md phân công nghiệm thu và nhật ký bằng chứng thực tế | Chờ từng thành viên thực hiện và xác nhận, theo kế hoạch GitHub của nhóm |

Kiểm tra mới: môi trường sạch cài đủ thư viện; package editable; 14/14 unit tests; 15 ô notebook không lỗi; 6 kiểm tra API/demo; đối chiếu bảng thời gian; checksum mô hình và protocol không đổi. Log và JSON cùng ngày trong evidence.

Không chạy lại toàn bộ 120 lượt tuning và 80 lượt mở rộng trong đợt sửa tài liệu này. Kết quả và log chạy cũ vẫn giữ nguồn, ngày và điều kiện đo của chúng. Test, ngưỡng và mô hình đã khóa không được dùng để chọn lại. Chênh AP SVM giữa lượt refit bổ sung và lượt khóa được ghi rõ, chưa xác định nguyên nhân bằng thí nghiệm kiểm soát.

Các điểm còn hạn chế: thiếu chứng cứ đóng góp cá nhân, chưa nested CV, chưa đánh giá ngoài nguồn/doanh nghiệp độc lập, precision test chưa đạt 80%. Đợt sửa làm bộ nộp nhất quán và có thể kiểm tra tốt hơn, không biến các hạn chế nghiên cứu thành kết quả đã đạt và không bảo đảm điểm chấm.
