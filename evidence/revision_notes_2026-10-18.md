# Hoàn thiện theo phản hồi bản chấm ngày 15/10

## Công việc thực hiện

- Nested CV 3 outer × 3 inner cho LR/SVM: tám cấu hình mỗi họ, cùng biểu diễn, 144 inner fit và sáu outer fit; lưu cấu hình, dự đoán ngoài fold, cảnh báo và kiểm tra nhóm.
- Ablation SVM: 25 lượt fit cho năm biến thể trên năm fold gốc, giữ C=1 và class_weight=None.
- Calibration logistic sigmoid: học từ OOF train, đánh giá Brier/log loss/AP trên validation; lưu điểm OOF và bảng phân khoảng. Không đưa vào demo.
- Phân tích độ nhạy chi phí FP/FN và tải kiểm tra cảnh báo theo giả định công khai, không tối ưu ngưỡng trên test.
- Đo lại năm pipeline cố định, tách thời gian fit và chấm; ghi môi trường và cấu hình phần cứng của lượt mới. Không gán cho dữ liệu lịch sử.
- Tính lại lát cắt profile, xác nhận 654 tin thiếu, 187 tin chưa thấy không rỗng và 11 tin giả; nhấn mạnh khoảng tin cậy rộng của riêng lát cắt 187 tin.
- Cập nhật báo cáo, notebook, model card, README, kiểm tra đóng gói và tài liệu tự giải trình.

Nguồn kết quả mới: results/robustness và nhật ký tương ứng trong evidence. Protocol được ghi trước khi chạy; mô hình, ngưỡng và kết quả test chính giữ nguyên checksum.

## Phạm vi chưa thay đổi

Nested CV mới chỉ bao phủ LR/SVM và nhóm nội dung; chưa đánh giá toàn bộ năm pipeline hoặc nhà tuyển dụng độc lập. Calibration và nested CV đều là bổ sung sau khi đã xem kết quả cũ, không phải đánh giá xác nhận ngoài nguồn. Benchmark chỉ một lượt mỗi cấu hình. Chi phí là kịch bản giả định, chưa được doanh nghiệp cung cấp. Precision test vẫn 74,53%, chưa đạt mục tiêu thiết kế 80%.

Theo yêu cầu người dùng, tên trường, tên giảng viên và tổng kết đóng góp cá nhân để nhóm bổ sung sau. Không tạo lịch sử commit, tác giả hoặc peer assessment thay cho thành viên. Tài liệu hướng dẫn giải trình giúp nhóm tự kiểm chứng trước khi nghiệm thu, không xác nhận thay mức độ hiểu bài.
