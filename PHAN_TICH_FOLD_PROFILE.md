# Giải thích kết quả thấp của fold 3 và phạm vi sử dụng số đo

## Kết luận chính

Fold 3 có AP 0,4462, tỷ lệ tin giả thấp hơn và nhiều tin giả bị xếp sau tin thật hơn hai fold còn lại. Thành phần nhóm cho thấy sự tập trung các tin giả khó xếp hạng. Đây là mô tả dự đoán đã lưu, chưa xác định cơ chế nhân quả hoặc chứng minh riêng tác động của rò rỉ hồ sơ.

Không fit lại mô hình, không chọn seed khác, không bỏ nhóm khó và không thay ngưỡng đã khóa. Nguồn là dự đoán OOF train ngày 19/10, gồm 10.836 tin; không trộn score của các fold để tạo một AP gộp vì đó là các mô hình fit khác nhau.

## 1 Thành phần từng fold

| Fold | Tin / giả | Tỷ lệ giả | AP | ROC AUC | Trung vị thứ hạng tin giả | Trung vị số tin thật đứng trước hoặc bằng điểm |
|---|---|---|---|---|---|---|
| 1 | 3829 / 215 | 5.62% | 0.7361 | 0.9387 | 118.0 | 10.0 |
| 2 | 2714 / 145 | 5.34% | 0.7678 | 0.9442 | 79.0 | 6.0 |
| 3 | 4293 / 170 | 3.96% | 0.4462 | 0.8644 | 322.5 | 237.0 |

Fold 3 có 170/4.293 tin giả (3,96%), so với 145/2.714 (5,34%) ở fold 2. AP phụ thuộc tỷ lệ lớp, nhưng không đủ cơ sở quy toàn bộ chênh lệch cho mất cân bằng: ROC-AUC cũng thấp hơn (0,8644 so với 0,9442), và số tin thật đứng trước hoặc bằng điểm một tin giả có trung vị 237 so với 6. Các mô hình được fit trên những tập khác nhau, nên đây không phải thử nghiệm giữ mọi yếu tố cố định.

rank_end là số tin có score lớn hơn hoặc bằng score của tin được xét; dùng điểm cuối nhóm đồng điểm để không phá tie theo job_id. Trung vị thứ hạng chỉ mô tả vị trí, không phải ngưỡng nghiệp vụ.

## 2 Nhóm lớn nhất nằm ở đâu

Nhóm 17 gồm 589 tin nằm hoàn toàn trong fold 3 và không có tin giả. Các tiêu đề mẫu là tuyển giáo viên tiếng Anh; dữ liệu gốc có logo ở toàn bộ nhóm. Nhóm 9 cũng ở fold 3, gồm 359 tin thật, toàn bộ không có logo. Các mô tả này phản ánh nhãn và cột trong CSV, không xác minh danh tính hoặc uy tín doanh nghiệp.

Không thể kết luận nhóm 589 tin là nguyên nhân chính chỉ vì nó lớn nhất. Với mọi cặp tin thật–tin giả của fold 3, đếm trường hợp score tin thật >= score tin giả: nhóm 17 chiếm 5,26% số cặp đó, trong khi nhóm 9 chiếm 28,84%. Đây là thống kê đảo thứ tự/bằng điểm, KHÔNG phải FP tại ngưỡng triển khai và không phân rã nhân quả của AP.

## 3 Tin giả tập trung trong nhóm lớn hay nhỏ

| fold | group_size | groups | fake | fake_share | AP_contribution |
|---|---|---|---|---|---|
| 1 | 1 | 752 | 93 | 0.432558 | 0.350954 |
| 1 | 2-9 | 266 | 41 | 0.190698 | 0.133881 |
| 1 | >=10 | 67 | 81 | 0.376744 | 0.251299 |
| 2 | 1 | 772 | 91 | 0.627586 | 0.501612 |
| 2 | 2-9 | 254 | 34 | 0.234483 | 0.128302 |
| 2 | >=10 | 60 | 20 | 0.137931 | 0.137931 |
| 3 | 1 | 755 | 85 | 0.5 | 0.327006 |
| 3 | 2-9 | 264 | 33 | 0.194118 | 0.055192 |
| 3 | >=10 | 66 | 52 | 0.305882 | 0.063996 |

Trong fold 3, 85/170 tin giả thuộc nhóm đơn; 33 thuộc nhóm 2–9 tin; 52 thuộc nhóm từ 10 tin trở lên. Hai nhóm lớn chứa tin giả là 3389 (41/41 giả) và 2901 (11/11 giả), tổng cộng 30,59% tin giả của fold. Thứ hạng trung vị tương ứng là 559 và 1.170; trung vị số tin thật đứng trước/bằng điểm là 454 và 1.027.

Precision trung bình tại score của các tin giả trong nhóm 3389 chỉ 0,2251; nhóm 2901 là 0,1500. Chúng góp 0,05429 và 0,00971 vào AP fold 3, tổng cộng khoảng 0,0640. Ngược lại, nhóm 2001 trong fold 2 có 20 tin giả với precision tại score đều bằng 1, góp 20/145 = 0,13793 vào AP fold 2. Đây là khác biệt quan sát được về độ khó của các nhóm.

Với P là tổng tin giả của fold, AP = tổng precision tại score của từng tin giả / P (tính tie theo điểm cuối). Do đó có thể cộng đóng góp của từng nhóm để khôi phục AP. “Đóng góp AP” ở đây là phân rã số học của kết quả, không phải importance của đặc trưng, không phải AP riêng khi đánh giá chỉ nhóm đó.

Ba nhóm có nhiều tin giả nhất và hai nhóm lớn nhất mỗi fold được chọn theo quy tắc cố định, hòa theo ID nhóm tăng dần; ví dụ job_id/tiêu đề nằm trong selected_group_examples.json. Hai nhóm giả 3389 và 2901 đều có logo theo CSV: có logo không đảm bảo tin thật. Không dùng thông tin này để đổi mô hình hoặc ngưỡng.

## 4 Nên tin con số nào

| Số đo | Câu hỏi được trả lời | Không được suy ra |
|---|---|---|
| Test AP 0,878; precision 74,53%; recall 83,33% | Pipeline/ngưỡng đã khóa hoạt động thế nào trên 3.389 tin test giữ riêng theo nhóm nội dung của bộ dữ liệu này? | Hiệu năng trên doanh nghiệp mới, dữ liệu Việt Nam hoặc ngoài nguồn; đạt precision 80% |
| Profile CV AP trung bình 0,650 ± 0,177 | Cấu hình SVM cố định nhạy thế nào khi fit/chấm lại trên 3 fold train tách nhóm nội dung và profile khớp chính xác? | Một test mới của bundle triển khai; nested theo doanh nghiệp; mức giảm do duy nhất một nguyên nhân |

Cả hai số đều được tính thật từ dự đoán lưu và phục vụ hai câu hỏi khác nhau. Test đánh giá một pipeline đã khóa; số 0,650 là trung bình AP của ba mô hình được fit lại trên các phần train khác nhau. Không lấy trung bình hai số, không thay số test bằng số CV và không gọi chênh lệch là mức cải thiện/suy giảm nhân quả.

Nếu quyết định triển khai cho nhà tuyển dụng mới, hiện chưa đủ bằng chứng để dùng 0,878 làm kỳ vọng vận hành. Cần company ID hoặc nhóm doanh nghiệp được xác minh, protocol định trước và tập ngoài nguồn chưa dùng để phát triển. Kết quả profile là tín hiệu cảnh báo độ bền, không thay được đánh giá đó.

## 5 Tái lập và giới hạn

Chạy `python scripts/verify_fold_diagnostics.py` để đối chiếu dự đoán gốc, từng nhóm, AP, tie, ROC-AUC và số cặp đảo thứ tự. Chạy `python scripts/analyze_profile_folds.py` để sinh lại ở bản sao dự án đã chuyển thư mục kết quả ngày 20/10 ra ngoài; script từ chối ghi đè.

Phân tích được bổ sung sau khi đã biết fold 3 thấp; không phải giả thuyết xác nhận định trước. Cấu hình C=1 từng được chọn trên train, profile không phải company ID, chưa có nhãn mới ngoài nguồn. Còn các yếu tố chưa cô lập: từ vựng/miền nghề nghiệp, cấu trúc nhóm, thành phần tập fit, chất lượng nhãn và tỷ lệ lớp.
