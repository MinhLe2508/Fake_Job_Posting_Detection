# Biên bản kiểm tra chéo phần Đức — 20/10/2026

## Phạm vi và nguồn kiểm tra

Thực hiện theo yêu cầu của Nguyễn Minh Đức bằng Codex trên checkout thực tế. Đây là kiểm tra tự động và rà soát tài liệu có hỗ trợ AI; không ghi nhận Đức tự chạy thủ công, không thay xác nhận của người review PR.

Commit được kiểm tra: `7179145c1e6ab2ae86c276637d3b69a19319dc6d` (main sau PR #11 của Minh và PR #12 của Bình). Nhánh ghi nhận: `work/duc`. Đã đọc mã phân tích/verify, protocol, bảng kết quả, PHAN_TICH_FOLD_PROFILE.md, MODEL_CARD.md và HUONG_DAN_GIAI_TRINH.md. Đối chiếu 13 file phân tích/tài liệu/test với ZIP cập nhật 20/10: khớp byte. Không nhập thêm mô hình, huấn luyện, thay seed, dự đoán hoặc ngưỡng.

## Môi trường

- macOS; Python 3.14.7.
- numpy 2.5.1, pandas 3.0.3, scipy 1.18.0, scikit-learn 1.9.0.
- Khác môi trường nguồn (Python 3.12.14, scikit-learn 1.7.2); kết quả dưới đây chỉ xác nhận các phép kiểm tra đã thực sự chạy trong môi trường này.

## Lệnh và kết quả thực tế

`python3 scripts/verify_fold_diagnostics.py` — exit 0:

```json
{
  "passed": true,
  "prediction_rows": 10836,
  "groups": 3256,
  "folds": 3,
  "source_predictions_unchanged": true,
  "AP_decomposition_verified": true
}
```

`PYTHONPATH=src python3 -m unittest discover -s tests -p test_fold_diagnostics.py -v` — 3/3 tests đạt; kiểm tra tie, cộng đóng góp AP và từ chối nhãn/score không hợp lệ.

`PYTHONPATH=src python3 -m unittest discover -s tests -v` — 23 tests chạy, 20 đạt, 3 errors. Cả ba thuộc test_pipeline: test_missing_and_unknown_category_supported, test_serialized_prediction_parity, test_transform_does_not_change_model. Lỗi quan sát được: `AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'`, kèm InconsistentVersionWarning khi nạp estimator 1.7.2 bằng scikit-learn 1.9.0. Chưa xác nhận toàn bộ suite đạt trên đúng môi trường dependency của dự án; không sửa model để né lỗi. Lần chạy tests đổi xuống dòng results/fold_audit.csv (CRLF sang LF); đã khôi phục đúng byte từ commit kiểm tra.

Ngoài script verify, đã tính lại độc lập AP bằng cách gom score bằng nhau, sắp giảm dần, lấy precision tích lũy ở cuối mỗi nhóm tie và cộng theo số mẫu dương. Kết quả khớp sklearn và hai bảng profile_folds/fold_composition; không dùng ranking_details cho phép tính độc lập này.

## Số liệu đã đối chiếu

10.836 dự đoán OOF train, 530 tin giả, 3.256 nhóm. job_id duy nhất; nhãn khớp CSV gốc; manifest đầy đủ; một nhóm chỉ nằm trong một fold. ranked_predictions giữ nguyên job_id/fold/group/nhãn/score của dự đoán ngày 19/10 (dung sai số thực 1e-12). Protocol trỏ đúng hash nguồn. Phân rã AP, cặp âm–dương đảo thứ tự/bằng điểm và thống kê nhóm được script verify xác nhận.

| Fold | Số dòng | Tin giả | AP tính lại | ROC-AUC tính lại |
|---|---:|---:|---:|---:|
| 1 | 3829 | 215 | 0.736133733497 | 0.938699630635 |
| 2 | 2714 | 145 | 0.767845563961 | 0.944185984081 |
| 3 | 4293 | 170 | 0.446192938016 | 0.864397711546 |

AP trung bình ba fold = **0.650057411825**, độ lệch chuẩn mẫu = **0.177262384078**. Không trộn score các mô hình/fold để tính AP gộp.

Tính lại từ test_predictions: 3.389 dòng, 144 tin giả; AP **0.878145871039**; precision **0.745341614907**, recall **0.833333333333**; TN/FP/FN/TP = 3204/41/24/120. Cột predicted khớp score >= ngưỡng khóa **-0.37317585243277074**.

## Rà soát diễn giải

Ba tài liệu phân biệt rõ test AP 0,878 của pipeline khóa với profile CV AP 0,650 ± 0,177 của ba mô hình fit trên các phần train. Đây là hai thiết kế đánh giá khác nhau; profile là proxy văn bản, không phải company ID. Fold 3 thấp được diễn giải mô tả, không tuyên bố nguyên nhân nhân quả. “Đóng góp AP” là phân rã số học, không phải feature importance hoặc AP tính riêng trong nhóm. Cặp tin thật–giả đảo thứ tự không phải FP tại ngưỡng triển khai. Không có yêu cầu đổi model/ngưỡng theo kết quả hậu kiểm.

## Bảo toàn nguồn và artifact đã khóa

Các SHA256 sau khớp protocol 19/10 và/hoặc 20/10, được kiểm tra trước và sau các lệnh. Model, lock/ngưỡng, protocol lựa chọn, selection và kết quả test giữ nguyên:

| File | SHA256 |
|---|---|
| `results/generalization_2026-10-19/profile_predictions.csv` | `70983917d53b360a35043508d31ac5b31908d6e7151df31af10ac4e14d51a93d` |
| `results/generalization_2026-10-19/profile_group_manifest.csv` | `2d5921a66bdb14686cf7f471481ddd1b15cf8aba6a55accac4fa6c823e567656` |
| `results/generalization_2026-10-19/profile_folds.csv` | `007ea9a3ca7784516cd1e91ad46f86dcbfcad2c956512ab408fda1821d6ea70e` |
| `data/fake_job_postings.csv` | `59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60` |
| `models/fake_job_detector.joblib` | `3592e1ffa4766ddd7755c88b6d1e42f51abb26d216ebccfa03035b30e156093b` |
| `models/lock.json` | `e38ded604e4947b88ffbb752489761a1e2a8f8e52542934741f7af3e7f073da9` |
| `results/test_metrics.json` | `dbcc068f7a4b2a9b31bae61222fd44aa62f11d968637d1d29278662bc54310fb` |
| `results/test_predictions.csv` | `91537b4b2767cf31d2cbf91418da2e3b61446ef3692ad4965ac35258ba085086` |
| `results/protocol.json` | `b3984cfa66afb697f4c4e2a745b172daadd6a1030db52c415d9bb18a555194ab` |
| `results/selection.json` | `9250d186bec6e63b80311078ac412017ccbbe4a539fcdbffa72440f6d6c77b41` |

## Kết luận và việc còn lại

Các yêu cầu đối chiếu của Đức đạt: đúng dự đoán train đã lưu, số dòng/tin giả/AP khớp, tài liệu phân biệt test và profile CV, model/ngưỡng/test bất biến. Không xác nhận 23/23 tests trên máy này: cần chạy lại suite trong môi trường khóa trước nghiệm thu tích hợp cuối. PR này chỉ bổ sung biên bản thực tế; Nhân tích hợp báo cáo/notebook và tạo lại checksum sau các PR theo phân công.
