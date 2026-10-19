# Hướng dẫn tự kiểm tra kiến thức trước khi bảo vệ

Tài liệu để cả nhóm tự luyện giải thích và chạy lại minh chứng. Nội dung này không xác nhận ai đã thực hiện công việc và không thay peer assessment.

## 1 Vì sao precision test thấp hơn validation

Validation có precision 82,89%, test 74,53%. Ngưỡng được chọn để thỏa mục tiêu trên validation nên số đo tại đó có thiên lệch chọn lựa. Tỷ lệ tin giả cũng giảm từ 5,45% xuống 4,25%; nội dung và tỷ lệ báo nhầm thay đổi. Không thể quy toàn bộ chênh lệch cho một nguyên nhân.

Tự kiểm tra: đọc TP, FP, FN, TN trong results/validation_metrics.json và test_metrics.json; tính precision = TP/(TP+FP). Nếu giữ TPR/FPR nhưng thay prevalence, dùng precision = TPR × prevalence / [TPR × prevalence + FPR × (1−prevalence)]. Không dùng phép tính này như chứng minh nhân quả.

## 2 Vì sao giữ SVM dù Gradient Boosting có AP CV cao hơn

GB được thử sau khi đã khóa mô hình và mở test. Nếu có trong lưới ban đầu thì quy tắc AP CV cao nhất sẽ chọn pipeline GB. Giữ SVM bảo toàn quyết định đánh giá ban đầu, không khẳng định SVM tốt nhất. Kết quả GB là đề xuất cho protocol tương lai. NB/GB dùng biểu diễn khác, nên đối chiếu mở rộng là so sánh pipeline.

Tự kiểm tra: đối chiếu selection.json, models/lock.json và results/extension/protocol_extension.json. Xác định trường nào chứng minh ngưỡng không đổi.

## 3 Luật tra cứu 30 TP 0 FP 0 FN nghĩa là gì

Chỉ xét 2.548 tin test có profile đã thấy ở train. Đánh dấu giả khi profile từng gắn nhãn giả trong train: tìm được 30 tin giả, không có lỗi trên lát cắt này. Ngoài lát cắt có 114 tin giả, luật không nhận ra; vì vậy không nói luật đạt recall 100% trên toàn test. Profile lặp có thể làm đánh giá gộp thuận lợi hơn cho bài toán doanh nghiệp đã biết.

Tự kiểm tra: chạy scripts/company_profile_slices.py; đọc cả dòng tat_ca và profile_da_thay_trong_train.

## 4 Profile chưa thấy khác doanh nghiệp mới như thế nào

Profile là văn bản mô tả, không phải mã doanh nghiệp. Một công ty có thể viết nhiều bản; nhiều công ty có thể dùng chung mẫu. Trong 841 tin thiếu/chưa thấy profile có 654 tin thiếu hẳn profile; chỉ 187 tin có profile không rỗng chưa thấy, gồm 11 tin giả. Với 9 TP, 5 FP, 2 FN: precision 9/14=64,29%, recall 9/11=81,82%. Khoảng bootstrap theo nhóm rất rộng, không đủ xác nhận khả năng tổng quát hóa sang nhà tuyển dụng mới.

## 5 Nested CV giải quyết điều gì

Inner CV chọn siêu tham số chỉ trong outer-train. Sau đó fit cấu hình thắng trên outer-train và chấm outer-eval chưa tham gia lựa chọn đó. Preprocessing được fit lại bên trong từng inner split và outer fit. Tách vai trò này giảm thiên lệch do chọn cấu hình trên chính fold được dùng báo cáo điểm.

Thí nghiệm bổ sung dùng 3 outer × 3 inner, train hiện có, hai họ LR/SVM và cùng biểu diễn. Nó không kiểm tra toàn bộ năm họ, không cô lập doanh nghiệp và không tạo ra test ngoài nguồn mới. Vì thực hiện sau khi đã xem kết quả cũ, vẫn là phân tích bổ sung; không dùng p-value để khẳng định ưu thế.

Tự kiểm tra: results/robustness/nested_inner.csv, nested_outer.csv và nested_fold_audit.csv. Giải thích khác nhau giữa điểm inner tốt nhất và AP outer.

## 6 Ablation có phải cứ bỏ đặc trưng mà điểm giảm là đặc trưng đó quan trọng

Chỉ có thể nói điểm thay đổi trong cấu hình và cách chia đã thử. Các đặc trưng tương quan, tương tác với regularization và thay đổi biểu diễn. Bỏ log1p hoặc logo có thể tạo chênh nhỏ không ổn định. Ablation SVM cố định C=1, class_weight=None và fit lại preprocessing mỗi fold; không dùng test để chọn lại đặc trưng.

## 7 Calibration có làm tăng precision mà không giảm recall không

Không có bảo đảm đó. Ánh xạ sigmoid đơn điệu chủ yếu biến score thành xác suất ước lượng; thứ tự xếp hạng và AP giữ nguyên khi hệ số dương. Brier/log loss đánh giá chất lượng xác suất; Brier phản ánh cả calibration và phân biệt. Cần giữ riêng dữ liệu học calibrator và đánh giá. Phân tích mới học calibrator từ OOF train rồi chấm validation, không thay demo hay ngưỡng đã khóa.

## 8 Phân tích chi phí có chứng minh hiệu quả kinh tế không

Không, nếu chưa có chi phí nghiệp vụ. Bảng độ nhạy đặt chi phí FP=1 và FN=r là các kịch bản giả định, cho thấy quyết định phụ thuộc r. Với mô hình khóa, chi phí lỗi là 41+24r; dự đoán tất cả thật là 144r. Chi phí kiểm duyệt là thành phần khác: 161 cảnh báo × thời gian xem một tin, chưa tính kiểm tra mẫu âm tính hoặc xử lý khiếu nại. Không gọi các con số này là lợi nhuận thực tế.

## Tự nghiệm thu

Mỗi thành viên chọn ít nhất một phân tích, tự chạy lệnh, giải thích bảng kết quả và nêu một giới hạn. Ghi công việc thực tế, file hoặc commit, kết quả kiểm tra và người đối chiếu vào CONTRIBUTIONS.md khi nhóm thực hiện. Không học thuộc số liệu mà không biết nguồn.

## 9 Vì sao calibration không làm AP tăng

AP phụ thuộc thứ tự xếp hạng. Trong thí nghiệm này, hệ số sigmoid dương (khoảng 3,95), nên điểm lớn vẫn nhận xác suất lớn: AP validation giữ 0,903553. Brier và log loss đổi vì chúng xét giá trị xác suất. Ví dụ hai score -1 và 1 vẫn giữ thứ tự sau sigmoid. Có thể kiểm tra hệ số trong calibration.json và AP trong calibration_validation.csv. Với ánh xạ khác tạo nhiều điểm bằng nhau, không mặc định AP bất biến.

## 10 Vì sao không gọi chênh lệch nested CV là cải thiện

AP CV cũ dùng 5 fold vừa chọn cấu hình vừa báo cáo. Nested dùng 3 outer × 3 inner với seed, kích thước tập fit và phân bố khác. SVM 0,831 và 0,839 không phải hai hệ thống được so sánh trên cùng thiết kế đánh giá. Nested tách vai trò lựa chọn và đánh giá trong train; không chứng minh mô hình triển khai tốt hơn, và cũng không tạo test mới.

## 11 Tại sao nối cả nhóm nội dung lẫn profile

Chỉ nhóm theo profile có thể để nội dung trùng qua fold; chỉ nhóm nội dung có thể để profile trùng. Nối các tin có chung một trong hai quan hệ và lấy thành phần liên thông xử lý cả chuỗi nối bắc cầu. Profile rỗng không tạo cạnh nối. Kiểm tra profile_folds.csv: cả group_overlap, content_overlap, profile_overlap_nonempty phải bằng 0; kiểm tra số tin giả từng fold. Profile vẫn là proxy, không phải company ID. C=1 đã chọn trước trên train nên đây không phải đánh giá ngoài mẫu hoàn toàn độc lập.

## 12 Năm lượt benchmark có phải năm lần xác nhận chất lượng mô hình

Không. Cùng dữ liệu, seed và cấu hình nên đây là lặp phép đo thời gian. Báo cáo trung vị và Q1–Q3 của fit và chấm validation; thứ tự mô hình đổi theo seed cố định. Các lượt cùng máy/tiến trình, không thay phép thử tải API hay đánh giá trên nhiều máy. Không dùng độ lệch thời gian để kiểm định chất lượng dự đoán.

