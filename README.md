# Phát hiện tin tuyển dụng giả mạo

Báo cáo cuối kỳ Data Mining IT701220 — Nhóm 7 — Track B — CRISP-DM.
Lê Bá Khánh Bình; Nguyễn Minh Đức; Lê Nhật Minh; Chu Đức Thành Nhân.

## Cài đặt và chạy demo

Giải nén toàn bộ ZIP. Mở PowerShell tại thư mục chứa README và pyproject.toml. Môi trường chính: Python 3.12.14 trên Windows; dùng Python 3.12 64-bit. Cài Python trước nếu lệnh `py -3.12` chưa có. Các thư viện khoa học được ghim trong requirements.txt.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe demo/app.py
```

Mở http://127.0.0.1:8765; Ctrl+C để dừng. Cần Internet để cài thư viện và build backend, không cần Internet để suy luận. `pyproject.toml` cấu hình package trong `src/`; cài dạng editable để mã tìm được dữ liệu/mô hình trong chính thư mục đã giải nén. Không di chuyển/xóa thư mục sau khi cài. Bộ này không được thiết kế như wheel độc lập chứa dữ liệu. Chỉ nạp joblib tin cậy đi kèm.

## Nội dung bộ nộp

- `report/Bao_cao_Cuoi_ky_Nhom7_CRISP_DM.docx` và `.pdf`: báo cáo đã đồng bộ.
- `Final_CRISP_DM.ipynb`: notebook đã thực thi, đọc kết quả đã khóa, có cả chẩn đoán profile và so sánh pipeline bổ sung.
- `src/fakejobs/`: tiền xử lý, thực nghiệm, phân tích, suy luận và mở rộng.
- `data/`: CSV gốc, từ điển, manifest split và CV.
- `results/`: kết quả chính; `results/extension/`: thí nghiệm thăm dò sau khóa.
- `models/`: mô hình và checksum khóa; `demo/`: giao diện và API localhost.
- `tests/`: kiểm thử pipeline và phần mở rộng. `scripts/check_submission.py` kiểm tra nhất quán tài liệu/số liệu riêng.
- `evidence/`: minh chứng, log kiểm tra. Log cũ vẫn ghi ngày chạy cũ; không được xem là lần chạy mới.
- `MODEL_CARD.md`: mục đích, số đo, giới hạn và điều kiện sử dụng.
- `CONTRIBUTIONS.md`: đầu ra cấp nhóm đã có và kế hoạch nghiệm thu từng thành viên.
- `PEER_ASSESSMENT_MAU_RIENG.md`: mẫu trống, mỗi người tự điền và nộp riêng.
- `midterm_artifacts/` và các file Midterm trong evidence: chứng cứ checkpoint lịch sử.

## Notebook và kiểm tra tài liệu

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
.\.venv\Scripts\python.exe -m jupyterlab Final_CRISP_DM.ipynb
.\.venv\Scripts\python.exe scripts/check_submission.py
```

Chọn kernel Python của `.venv`. Notebook không tự huấn luyện lại và không ghi đè log/bảng kết quả của bộ nộp. Muốn dựng lại notebook nguồn: `python scripts/build_notebook.py` (thay nội dung notebook và xóa outputs cũ), sau đó thực thi từ đầu. Sử dụng đúng interpreter `.venv` như các lệnh trên.

## Tái lập thí nghiệm chính từ đầu

```powershell
.\.venv\Scripts\python.exe scripts/reproduce.py --output ../Nhom7_reproduced
```

Thư mục đích phải chưa tồn tại và ở ngoài bộ nộp. Lệnh fit 120 cấu hình-fold, 5 Dummy, 20 ablation, chọn ngưỡng validation, khóa rồi đánh giá test theo giao thức cố định. Nó không chọn phương án nghiên cứu mới trên test. Thời gian phụ thuộc máy, có thể hàng chục phút. Thư mục mới có mã và kết quả; không thay bản đã nộp.

Nếu bị gián đoạn, chuyển đến thư mục tái lập, đặt `PYTHONPATH` thành `src` của nó và dùng đường dẫn tuyệt đối đến interpreter `.venv` đã cài để chạy:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
# Thay đường dẫn dưới bằng interpreter của bạn; không dùng Python hệ thống ngẫu nhiên.
& 'C:\duong_dan_bo_nop\.venv\Scripts\python.exe' -m fakejobs.experiment --stage all
& 'C:\duong_dan_bo_nop\.venv\Scripts\python.exe' -m fakejobs.analysis
```

Cơ chế resume bỏ qua kết quả có sẵn; chạy `--stage all` trên bộ nộp không chứng minh tái lập từ đầu. Dự đoán/thước đo so sánh với dung sai số học; thời gian chạy và hash nhị phân có thể khác môi trường. Bằng chứng tái lập toàn bộ lần đầu nằm trong `evidence/reproduction_full.log` và `reproduction_check.json`; không gán ngày mới cho lần chạy cũ.

## Tái lập các phân tích bổ sung

Sau khi tái lập quy trình chính ở thư mục mới, tiếp tục trong thư mục đó:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
& 'C:\duong_dan_bo_nop\.venv\Scripts\python.exe' scripts/company_profile_slices.py
& 'C:\duong_dan_bo_nop\.venv\Scripts\python.exe' -m fakejobs.extension --stage all
```

Phân tích profile dùng dự đoán validation/test đã khóa để chẩn đoán, không fit. Extension đọc CSV chung để kiểm tra dữ liệu, chỉ đưa train/validation vào các bước fit/chấm; không dùng test để chọn hoặc đánh giá các pipeline mới. Cần chạy chính trước vì extension đọc các kết quả CV chính. 16 cấu hình bổ sung × 5 fold = 80 fit, có resume. Không ghi đè models, selection hoặc kết quả test chính.

Kết quả bổ sung lịch sử dùng Python 3.12.3, scikit-learn 1.7.2; kết quả chính dùng Python 3.12.14. `results/extension/measurement_provenance.json` giải thích khác biệt. Số `fit_seconds` lịch sử trong validation_compare thực tế gộp classifier.fit + dự đoán + AP, không gồm preprocessing; báo cáo ghi đúng phạm vi này. Mã từ 14/10/2026 tách fit/predict, đánh dấu timing_schema v2 khi chạy mới. Không so sánh hai kiểu đo như cùng một đại lượng. RF dùng n_jobs=2, BLAS/OpenMP giới hạn 1.

## Giao thức và giới hạn

Lớp dương là 1. Chọn cấu hình theo AP CV trên train; chọn ngưỡng trên validation với precision ≥0,80 và recall cao nhất. Test chỉ dùng đánh giá sau khóa; không đổi ngưỡng theo test. Precision test 74,53% chưa đạt mục tiêu thiết kế. AP là Average Precision không nội suy của scikit-learn; điểm SVM không phải xác suất.

Các kiểm định trên năm fold dùng chung dữ liệu fit và được chọn siêu tham số trên chính fold đó chỉ có giá trị thăm dò. Nadeau–Bengio/Holm không loại bỏ thiên lệch chọn mô hình. NB dùng text-only, GB dùng feature selection: đây là so sánh pipeline. Không diễn giải p>0,05 thành tương đương. Kết quả profile trùng cho thấy giới hạn của holdout theo nội dung; cần đánh giá mới theo doanh nghiệp/thời gian và nguồn ngoài.

CSV từ [Kaggle Real/Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction/data), liên quan EMSCAD. Dữ liệu tiếng Anh lịch sử, nhãn kế thừa, không đại diện mọi nền tảng. Xem MODEL_CARD và báo cáo.

## Chỉnh báo cáo và đóng gói

Bản Word là nguồn biên tập. Chỉnh trực tiếp bằng Word, đối chiếu JSON/CSV và chạy `scripts/check_submission.py`. Không có builder tái tạo toàn bộ nội dung Word từ số liệu trong gói này; không cần requirements-report.txt. Xuất PDF trên Windows có Microsoft Word:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File report/export_pdf.ps1
.\.venv\Scripts\python.exe scripts/check_submission.py
.\.venv\Scripts\python.exe scripts/make_sha256sums.py
.\.venv\Scripts\python.exe scripts/verify_bundle.py
```

Sau chỉnh sửa hoặc chạy lệnh ghi file, tạo lại hash cuối cùng. ZIP không chứa virtualenv, cache, egg-info, Git hoặc peer assessment đã chấm.

## Minh bạch đóng góp

AI Codex hỗ trợ mã, thí nghiệm, demo và tài liệu. Sản phẩm cấp nhóm đã có minh chứng; nghiệm thu cá nhân qua GitHub là kế hoạch tiếp theo được nhóm xác nhận. Mỗi người cần thực sự kiểm tra/sửa phần nhận rồi ghi file, commit/PR, kết quả và kiểm tra chéo trong CONTRIBUTIONS.md; sau đó cập nhật mục 7.1 báo cáo. Việc chia file để commit không tự chứng minh tác giả hay đóng góp quá khứ. Không tạo điểm peer, số giờ hay lịch sử commit giả.

## Thực nghiệm bổ sung ngày 18/10

Chạy trong một bản sao bộ nộp hoặc thư mục tái lập đã hoàn tất thực nghiệm chính, để giữ số đo của bản nộp:

```powershell
$env:PYTHONPATH = (Resolve-Path ./src).Path
python -m fakejobs.robustness --stage all
python scripts/benchmark_extension.py
python scripts/company_profile_slices.py
```

Dùng đúng interpreter virtualenv theo hướng dẫn đầu README. Chạy benchmark sau khi robustness kết thúc; không chạy song song hai tác vụ vì sẽ làm sai điều kiện đo CPU. `robustness` ghi ra `results/robustness/`, không có resume từng fit: chạy lại một stage sẽ thay kết quả stage đó. Giao thức được ghi trước khi fit; thay mã nguồn đòi hỏi thư mục kết quả mới. `benchmark_extension.py` ghi lại benchmark.json của lượt mới.

Nested CV gồm 3 outer × 3 inner, tám cấu hình/họ cho LR và SVM (144 inner fit + 6 outer fit). Nó không bao phủ RF/NB/GB hoặc chia theo doanh nghiệp. SVM ablation gồm 25 fit, giữ cấu hình C=1, class_weight=None. Calibration học logistic sigmoid từ OOF train, chấm validation; không thay mô hình triển khai. Chi phí là kịch bản FP/FN giả định từ confusion matrix đã khóa, không chọn ngưỡng theo test. Phần cứng và môi trường lượt mới được ghi riêng; không gán cho thí nghiệm lịch sử.

Báo cáo mục 5.10–5.12 và notebook trình bày kết quả thực. `HUONG_DAN_GIAI_TRINH.md` hỗ trợ tự kiểm tra kiến thức; không thay nhật ký đóng góp. Theo yêu cầu nhóm, tên trường, giảng viên và xác nhận đóng góp cá nhân được nhóm bổ sung sau.
