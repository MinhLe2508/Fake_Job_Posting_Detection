# Đối chiếu nhận xét bản chấm 19/10

| Nhận xét | Cập nhật | Nguồn kiểm chứng |
|---|---|---|
| Chưa giải thích fold 3 thấp | Phân tích tỷ lệ lớp, thành phần nhóm, nhóm 589 tin, các nhóm nhiều tin giả, thứ hạng và phân rã AP có xử lý tie | PHAN_TICH_FOLD_PROFILE.md; results/fold_diagnostics_2026-10-20/ |
| Nên tin AP 0,878 hay 0,650 | Nêu câu hỏi, thiết kế và giới hạn của từng số; không lấy trung bình, không thay kết quả test | Tóm tắt báo cáo, MODEL_CARD.md, hướng dẫn giải trình câu 14 |
| Rủi ro viva | Bổ sung câu 13–14 và lệnh tự kiểm chứng; không ghi nhận thay nghiệm thu của thành viên | HUONG_DAN_GIAI_TRINH.md; scripts/verify_fold_diagnostics.py |
| Đóng góp cá nhân và bìa | Giữ nguyên, chờ nhóm bổ sung theo yêu cầu | CONTRIBUTIONS.md và báo cáo |

Ngày 20/10 không fit mô hình mới, không đổi split/seed/ngưỡng, không bỏ nhóm khó. Mọi dự đoán ngày 19/10 và kết quả test đã khóa được giữ nguyên. Chỉ tính lại thống kê từ dự đoán lưu; phân tích thành phần nhóm mang tính hậu kiểm mô tả, không chứng minh cơ chế nhân quả.

Giới hạn nghiên cứu vẫn được ghi rõ: nested mới bao phủ LR/SVM, kiểm tra profile là SVM cố định đã chọn trước, không có company ID hoặc dữ liệu ngoài nguồn; benchmark cùng máy/ma trận; precision test chưa đạt 80%. Không thêm thí nghiệm để che kết quả thấp hoặc tuyên bố những giới hạn này đã được giải quyết.

Ba kiểm thử mới kiểm tra xử lý tie, tính cộng được của đóng góp AP và đầu vào không hợp lệ. Các log ngày 20/10 là minh chứng hỗ trợ Codex; không quy thành đóng góp cá nhân. Bản cuối được kiểm tra notebook và bố cục Word/PDF; checksum phải tạo lại sau khi nhóm điền thông tin còn thiếu.
