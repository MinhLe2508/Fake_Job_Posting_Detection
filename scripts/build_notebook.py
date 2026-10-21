"""Tạo notebook dẫn dắt quy trình từ các module và artifact có thể kiểm chứng."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# Phát hiện tin tuyển dụng giả mạo — Cuối kỳ
**Nhóm 7 · Track B · CRISP-DM**

Lê Bá Khánh Bình; Nguyễn Minh Đức; Lê Nhật Minh; Chu Đức Thành Nhân.

Notebook trình bày dữ liệu, quy trình và kết quả thực đã lưu. Mặc định không huấn luyện lại hoặc điều chỉnh theo test. Để tái lập toàn bộ thí nghiệm ở thư mục mới, xem khối cuối và README. AI Codex hỗ trợ mã, thực nghiệm và tài liệu; nhóm phải kiểm chứng và tự giải trình.''')
md('''## 1. Business Understanding
Bài toán phân lớp có giám sát, đơn vị là một tin đăng, lớp dương `fraudulent=1`. Mục đích hỗ trợ hàng đợi kiểm duyệt, không tự động xác minh hay khóa tin. Chọn mô hình theo AP trên train CV, chọn ngưỡng trên validation để recall cao nhất khi precision ≥ 0,80. Đây là mục tiêu thiết kế của đồ án, chưa phải SLA doanh nghiệp.''')
code('''from pathlib import Path
import sys, json, io, subprocess, os
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd().resolve()
if not (ROOT / 'src/fakejobs').exists():
    raise RuntimeError('Mở notebook tại thư mục chứa README và src/.')
sys.path.insert(0, str(ROOT / 'src'))
from fakejobs.experiment import load_data, load_json, sha, metrics
from fakejobs.features import normalize_and_validate, make_features
pd.set_option('display.max_colwidth', 65)
display(load_json(ROOT / 'results/environment.json'))''')
md('''## 2. Data Understanding
CSV từ Kaggle/EMSCAD, 17.880 dòng và 18 cột. Bảng mẫu là dòng thật từ dữ liệu, không phải ví dụ tạo thủ công. `salary_range` là chuỗi; không coi là biến lương số đã chuẩn hóa.''')
code('''raw = pd.read_csv(ROOT / 'data/fake_job_postings.csv', low_memory=False)
print('Shape:', raw.shape)
raw.info()
display(raw[['job_id','title','location','salary_range','has_company_logo','fraudulent']].head())
display(raw.describe(include='all').T)''')
code('''missing = pd.DataFrame({'missing_count': raw.isna().sum(), 'missing_pct': raw.isna().mean()*100})
display(missing.sort_values('missing_pct', ascending=False))
counts = raw.fraudulent.value_counts().sort_index()
display(pd.DataFrame({'count':counts,'pct':counts/len(raw)*100}))
display(Image(filename=str(ROOT/'midterm_artifacts/08_class_distribution_raw.png')))''')
md('''## 3. Chuẩn bị dữ liệu và kiểm toán leakage
Loại bản trùng hoàn toàn theo predictor sau kiểm tra nhãn mâu thuẫn. Tách nhóm bằng nội dung chuẩn hóa của title/description/requirements. Manifest kế thừa giữa kỳ giữ nguyên test. Các hàm fit/transform nằm trong `src/fakejobs/features.py`; không fit bộ biến đổi trên toàn dữ liệu.''')
code('''parts, groups, folds = load_data()
display(pd.DataFrame([{'split':name,'rows':len(df),'fake':int(df.fraudulent.sum()),
    'fake_pct':100*df.fraudulent.mean()} for name,df in parts.items()]))
display(pd.read_csv(ROOT/'results/fold_audit.csv'))
print('Các kiểm tra nhóm, số tin giả và hash dữ liệu đã vượt qua.')''')
code('''train_features = make_features(parts['train'])
display(train_features.filter(like='_word_count').describe().T)
display(pd.read_csv(ROOT/'midterm_artifacts/07_iqr_train.csv'))
display(Image(filename=str(ROOT/'midterm_artifacts/07_outlier_distributions_train.png')))
display(Image(filename=str(ROOT/'midterm_artifacts/09_correlation_train.png')))''')
md('''Không tự xóa tin dài theo IQR; log1p giảm đuôi phân phối. Median/mode/scaler/OHE/TF-IDF được học riêng trong mỗi phần fit. 5-fold StratifiedGroupKFold có 106 tin giả và 81–84 nhóm giả ở mỗi fold đánh giá; nhóm train/eval không giao nhau. Phân tầng hỗ trợ giữ tỷ lệ nhưng không bảo đảm mọi tập dữ liệu đều cân bằng hoàn hảo.''')
md('''## 4. Modeling — 3 thuật toán và baseline
8 cấu hình/họ × 3 họ × 5 fold = 120 fit, thêm 5 Dummy. CV dùng AP để chọn cấu hình. F1 trong bảng CV ở ngưỡng mặc định chỉ dùng mô tả, không phải ngưỡng vận hành cuối. Toàn bộ fit có log trong CSV; cấu hình chưa hội tụ được hiển thị minh bạch.''')
code('''protocol = load_json(ROOT/'results/protocol.json')
display(pd.DataFrame(protocol['configurations']))
cv = pd.read_csv(ROOT/'results/cv_results.csv')
assert len(cv)==125
display(pd.read_csv(ROOT/'results/best_per_model.csv'))
display(cv.loc[cv.warnings.notna(), ['config_id','fold','warnings']])
display(Image(filename=str(ROOT/'results/figures/01_cv_ap.png')))
display(load_json(ROOT/'results/selection.json'))''')
md('''### Ablation
LR với C và class_weight đã chọn, giữ cùng 5 fold. `metadata_only` gồm cả độ dài văn bản nhưng không TF-IDF. Các ablation là phân tích tác động trên train; không dùng để đổi mô hình sau khi xem test.''')
code('''display(pd.read_csv(ROOT/'results/ablation_summary.csv'))
display(Image(filename=str(ROOT/'results/figures/02_ablation.png')))''')
md('''## 5. Evaluation — ngưỡng và test đã khóa
Chọn ngưỡng chỉ trên validation, sau đó khóa checksum mô hình/protocol. Điểm SVM là decision score, không phải xác suất. AP không nội suy của scikit-learn khác diện tích PR tính hình thang.''')
code('''display(load_json(ROOT/'results/validation_metrics.json'))
lock = load_json(ROOT/'models/lock.json')
assert sha(ROOT/'models/fake_job_detector.joblib')==lock['model_sha256']
assert sha(ROOT/'results/protocol.json')==lock['protocol_sha256']
display(lock)
display(Image(filename=str(ROOT/'results/figures/05_validation_threshold.png')))''')
code('''test = load_json(ROOT/'results/test_metrics.json')
pred = pd.read_csv(ROOT/'results/test_predictions.csv')
recomputed = metrics(pred.fraudulent, pred.score, test['threshold'])
for key,value in recomputed.items():
    assert np.isclose(value,test[key]), key
display(pd.Series(test).drop('dummy_prior').to_frame('selected_model'))
display(pd.Series(test['dummy_prior']).to_frame('dummy_prior'))
display(Image(filename=str(ROOT/'results/figures/04_confusion.png')))
display(Image(filename=str(ROOT/'results/figures/03_test_curves.png')))''')
code('''bootstrap = load_json(ROOT/'results/test_bootstrap.json')
display(pd.DataFrame(bootstrap['intervals']).T)
print('Bootstrap theo nhóm:', bootstrap['replicates'], 'lần; nhóm:', bootstrap['groups'])
display(pd.read_csv(ROOT/'results/error_slices.csv'))
display(pd.DataFrame(load_json(ROOT/'results/error_cases.json')))
display(Image(filename=str(ROOT/'results/figures/06_feature_weights.png')))''')
md('''Khoảng tin cậy có điều kiện trên mô hình đã fit, không bao gồm toàn bộ biến động huấn luyện hoặc dịch chuyển miền. Phân tích lỗi test là hậu kiểm, không dùng để chỉnh ngưỡng/thuật toán. Tin gần trùng hoặc cùng doanh nghiệp chưa được cô lập hoàn toàn; cần test ngoài nguồn trước pilot.''')
md('''### 5.7 Chẩn đoán theo company_profile
Đây là phân tích hậu kiểm ở ngưỡng đã khóa. Profile trùng không phải company ID. Nhóm không có/chưa thấy profile có 841 tin, trong đó 654 tin thiếu profile; không gọi toàn bộ nhóm này là doanh nghiệp mới. Precision 72,22%, recall 79,82% trên lát cắt này cần được đọc cùng kết quả gộp.''')
code('''display(pd.read_csv(ROOT/'results/company_profile_slices.csv'))
display(load_json(ROOT/'results/company_profile_slices.json'))''')
md('''### 5.8 So sánh pipeline bổ sung và giới hạn kiểm định
NB dùng TF-IDF text-only; HistGradientBoosting dùng combined + SelectKBest 2.000 đặc trưng. Khác biểu diễn nên không cô lập tác động thuật toán.

Năm fold dùng tập fit chồng lấn và cũng được dùng chọn cấu hình. Nadeau–Bengio/Holm không loại bỏ selection bias; Friedman trên fold không thay thế thiết kế nhiều dataset độc lập. P-value chỉ để truy nguyên phân tích thăm dò, không xác nhận ưu thế/tương đương. Validation đã dùng chọn ngưỡng, bootstrap validation cũng chỉ là chẩn đoán.

Bảng thời gian dưới lấy trực tiếp JSON lịch sử: fit + chấm validation + AP, không gồm preprocessing. Mã mới tách fit/predict cho lượt chạy mới; không đổi hồi tố số đo. RF n_jobs=2, các native pool giới hạn 1.''')
code('''comparison = load_json(ROOT/'results/extension/validation_compare.json')
timing = pd.DataFrame(comparison['models']).T.rename(columns={'fit_seconds':'fit_plus_validation_seconds_historical'})
display(timing)
display(pd.read_csv(ROOT/'results/extension/stats_cv_pairwise.csv'))
display(comparison['consistency_check'])
display(load_json(ROOT/'results/extension/measurement_provenance.json'))''')
md('''### 5.10 Phân tích bổ sung ngày 18/10
Nested CV trên train: 3 outer × 3 inner, tám cấu hình mỗi họ LR/SVM, cùng biểu diễn; inner chọn, outer đánh giá. Không bao phủ RF/NB/GB và không chia theo doanh nghiệp. Ablation SVM giữ C=1, class_weight=None. Calibration sigmoid học từ OOF train, chấm validation đã dùng trước đây nên chỉ là chẩn đoán. Chi phí là giả định chuẩn hóa, không phải chi phí doanh nghiệp. Không đổi bundle/ngưỡng hoặc lựa chọn theo test.''')
code('''display(pd.read_csv(ROOT/'results/robustness/nested_summary.csv'))
display(pd.read_csv(ROOT/'results/robustness/nested_outer.csv'))
display(pd.read_csv(ROOT/'results/robustness/nested_fold_audit.csv'))
display(pd.read_csv(ROOT/'results/robustness/svm_ablation_summary.csv'))
display(pd.read_csv(ROOT/'results/robustness/calibration_validation.csv'))
display(pd.read_csv(ROOT/'results/robustness/calibration_bins.csv'))
display(pd.read_csv(ROOT/'results/robustness/cost_sensitivity.csv'))
display(pd.DataFrame(load_json(ROOT/'results/robustness/benchmark.json')['rows']))
display(load_json(ROOT/'results/robustness/protocol.json'))''')
md('''### 5.13 Kiểm tra nhóm profile và benchmark lặp ngày 19/10
Train-only, SVM C=1 cố định đã chọn trước; nhóm liên thông từ nội dung hoặc profile không rỗng. Đây là kiểm tra độ bền sau khóa, chưa phải nested doanh nghiệp hay đánh giá ngoài nguồn. Benchmark 5 lượt mỗi pipeline phản ánh biến động thời gian trên cùng máy, không phải 5 mẫu chất lượng độc lập.''')
code('''extra = ROOT/'results/generalization_2026-10-19'
display(load_json(extra/'profile_summary.json'))
display(pd.read_csv(extra/'profile_folds.csv'))
display(pd.read_csv(extra/'benchmark_summary.csv'))
display(load_json(extra/'benchmark_scope.json'))
display(load_json(extra/'completion.json'))''')
md('''### 5.14 Thành phần fold và cách dùng số đo
Phân tích dự đoán train đã lưu ngày 19/10; không fit lại, không sửa fold/ngưỡng. AP thấp ở fold 3 đi kèm tỷ lệ giả thấp, thứ hạng các nhóm giả thấp và nhiều tin thật đứng trước. Đây là mô tả, không phải kết luận nhân quả. Test AP 0,878 và profile CV 0,650 trả lời hai câu hỏi khác nhau; chưa số nào xác nhận hiệu quả trên doanh nghiệp mới.''')
code('''diag = ROOT/'results/fold_diagnostics_2026-10-20'
display(pd.read_csv(diag/'fold_composition.csv'))
display(pd.read_csv(diag/'group_size_distribution.csv'))
group_diag = pd.read_csv(diag/'group_diagnostics.csv')
display(group_diag[(group_diag.fold==3) & group_diag.group.isin([17,9,3389,2901])])
sys.path.insert(0, str(ROOT/'scripts'))
from verify_fold_diagnostics import verify as verify_fold_diagnostics
display(verify_fold_diagnostics())''')
md('''## 6. Deployment và kiểm thử
Demo chạy bằng `python demo/app.py`. Dữ liệu nhập chỉ ở bộ nhớ, không tự huấn luyện. Mẫu bên dưới thuộc validation đã dùng chọn ngưỡng, không phải đánh giá độc lập mới.''')
code('''from fakejobs.inference import Detector
detector = Detector()
examples = load_json(ROOT/'evidence/demo_examples.json')
example = examples[0]
print('Mẫu validation:',example['job_id'],'nhãn gốc:',example['actual_label'])
display(detector.predict(example['fields']))
display(detector.predict({'title':'Data analyst','industry':'UNSEEN_CATEGORY_999'}))''')
code('''# Tiến trình con cần PYTHONPATH riêng; sys.path trong kernel không tự truyền xuống.
test_env = os.environ.copy()
test_env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + test_env.get('PYTHONPATH','')
test_env['PYTHONIOENCODING'] = 'utf-8'
result = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
    cwd=ROOT, env=test_env, capture_output=True, text=True, encoding='utf-8')
test_log = result.stdout + result.stderr
print(test_log)
assert result.returncode==0, 'Kiểm thử không đạt; xem log trước khi sử dụng kết quả.' ''')
md('''## 7. Tái lập toàn bộ từ đầu
Đổi `RUN_FULL_REPRODUCTION=True` khi muốn chạy lại toàn bộ ở một thư mục mới. Không chạy lại trên test rồi tiếp tục chọn mô hình/ngưỡng; bộ test hiện tại đã được mở. Nếu nghiên cứu hướng mới, cần protocol và tập đánh giá mới.

Sản phẩm cấp nhóm được tổng kết trong báo cáo. CONTRIBUTIONS.md phân định nghiệm thu thực tế sắp tới; commit file có sẵn không tự xác nhận tác giả. Peer assessment do từng thành viên tự điền và nộp riêng.''')
code('''RUN_FULL_REPRODUCTION = False
if RUN_FULL_REPRODUCTION:
    destination = ROOT.parent/'Nhom7_reproduced_from_notebook'
    subprocess.run([sys.executable,'scripts/reproduce.py','--output',str(destination)],cwd=ROOT,check=True)
else:
    print('Đang đọc kết quả đã khóa. Xem README để tái lập từ đầu ở thư mục mới.')''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12.14'}})
nbf.write(nb,ROOT/'Final_CRISP_DM.ipynb');print('Notebook created:',len(cells),'cells')
