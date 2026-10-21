"""Kiểm tra lỗi đóng gói/số liệu đã từng gặp; chỉ đọc, dùng thư viện chuẩn Python."""
from pathlib import Path
import hashlib, json, sys, tomllib, zipfile
import csv
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NS = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def audit(root=ROOT):
    issues = []
    required = ['pyproject.toml', 'README.md', 'MODEL_CARD.md', 'CONTRIBUTIONS.md',
                'requirements.txt', 'requirements-notebook.txt', 'report/export_pdf.ps1',
                'report/Bao_cao_Cuoi_ky_Nhom7_CRISP_DM.docx',
                'report/Bao_cao_Cuoi_ky_Nhom7_CRISP_DM.pdf',
                'Final_CRISP_DM.ipynb', 'scripts/reproduce.py']
    for name in required:
        if not (root/name).is_file(): issues.append('Missing: '+name)
    if issues: return issues
    config=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))
    if config['tool']['setuptools']['packages']['find']['where'] != ['src']:
        issues.append('Package discovery does not point to src')
    with zipfile.ZipFile(root/required[7]) as z:
        tree=ET.fromstring(z.read('word/document.xml'))
    tables=tree.findall('.//w:body/w:tbl',NS)
    def txt(el): return ''.join(t.text or '' for t in el.findall('.//w:t',NS))
    target=[t for t in tables if 'AP validation' in txt(t) and 'Cấu hình tốt nhất' in txt(t)]
    if len(target)!=1: issues.append('Expected exactly one extension comparison table')
    else:
        rows=target[0].findall('w:tr',NS)
        data=json.loads((root/'results/extension/validation_compare.json').read_text(encoding='utf-8'))
        for row,name in zip(rows[1:],['LinearSVM','LogisticRegression','RandomForest','NaiveBayes','HistGradientBoosting']):
            value=txt(row.findall('w:tc',NS)[4]).replace(',','.')
            if abs(float(value)-data['models'][name]['fit_seconds']) > .00501:
                issues.append('Report timing mismatch: '+name)
        if 'Fit + chấm' not in txt(rows[0]): issues.append('Historical timing scope not labeled')
    report=txt(tree)
    for old in ['cao gấp khoảng 80 lần','Friedman cho thấy các họ không cùng chất lượng','báo cáo không cam kết điểm rubric']:
        if old in report: issues.append('Stale report claim: '+old)
    lock=json.loads((root/'models/lock.json').read_text(encoding='utf-8'))
    for path,key in [('models/fake_job_detector.joblib','model_sha256'),('results/protocol.json','protocol_sha256')]:
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=lock[key]: issues.append('Lock mismatch: '+path)
    nb=json.loads((root/'Final_CRISP_DM.ipynb').read_text(encoding='utf-8'))
    src='\n'.join(''.join(c['source']) for c in nb['cells'])
    for term in ['company_profile_slices.csv','measurement_provenance.json']:
        if term not in src: issues.append('Notebook omits '+term)
    for cell in nb['cells']:
        for output in cell.get('outputs',[]):
            if output.get('output_type')=='error':issues.append('Notebook execution error')
    # Chặn đóng gói thiếu lượt thí nghiệm hoặc vô tình đưa test vào nested CV.
    rr=root/'results/robustness'
    for name,count in [('nested_inner.csv',144),('nested_outer.csv',6),('nested_fold_audit.csv',12),('svm_ablation.csv',25)]:
        with (rr/name).open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))
        if len(rows)!=count:issues.append('Incomplete robustness run: '+name)
        if name=='nested_fold_audit.csv' and any(int(row['group_overlap']) or int(row['eval_fake'])<1 for row in rows):issues.append('Invalid nested groups/classes')
    with (rr/'nested_predictions.csv').open(encoding='utf-8-sig',newline='') as handle: predictions=list(csv.DictReader(handle))
    with (root/'data/split_manifest.csv').open(encoding='utf-8-sig',newline='') as handle: train_ids={row['job_id'] for row in csv.DictReader(handle) if row['split']=='train'}
    for model in ['LinearSVM','LogisticRegression']:
        ids=[row['job_id'] for row in predictions if row['model']==model]
        if len(ids)!=len(set(ids)) or set(ids)!=train_ids:issues.append('Outer predictions must cover train exactly once: '+model)
    receipt=json.loads((rr/'completion_all.json').read_text())
    if not receipt['completed'] or not receipt['frozen_hashes_unchanged']:issues.append('Robustness completion failed')
    protocol=json.loads((rr/'protocol.json').read_text(encoding='utf-8'))
    if hashlib.sha256((root/'src/fakejobs/robustness.py').read_bytes()).hexdigest()!=protocol['code_sha256']:issues.append('Robustness source differs from executed protocol')
    benchmark=json.loads((rr/'benchmark.json').read_text(encoding='utf-8'))
    if len(benchmark['rows'])!=5 or not benchmark['frozen_hashes_unchanged']:issues.append('Incomplete benchmark')
    if hashlib.sha256((root/'scripts/benchmark_extension.py').read_bytes()).hexdigest()!=benchmark['source_sha256']:issues.append('Benchmark source differs from executed run')
    for path,expected in protocol['frozen_hashes'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=expected:issues.append('Frozen experiment changed during robustness: '+path)
    gg=root/'results/generalization_2026-10-19'
    protocol_new=json.loads((gg/'protocol.json').read_text(encoding='utf-8'))
    if hashlib.sha256((root/'scripts/audit_generalization.py').read_bytes()).hexdigest()!=protocol_new['code_sha256']:issues.append('Generalization source mismatch')
    for path,expected in protocol_new['frozen_hashes'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=expected:issues.append('Frozen file changed: '+path)
    with (gg/'profile_folds.csv').open(encoding='utf-8',newline='') as f: folds=list(csv.DictReader(f))
    if len(folds)!=3 or any(int(r[k]) for r in folds for k in ['group_overlap','content_overlap','profile_overlap_nonempty']):issues.append('Profile fold leakage or missing folds')
    if any(not 0<int(r['eval_fake'])<int(r['eval_n']) or not 0<int(r['fit_fake'])<int(r['fit_n']) for r in folds):issues.append('Profile fold missing class')
    with (gg/'profile_predictions.csv').open(encoding='utf-8',newline='') as f: ids=[r['job_id'] for r in csv.DictReader(f)]
    if len(ids)!=len(set(ids)) or set(ids)!=train_ids:issues.append('Profile predictions must cover train only exactly once')
    with (gg/'benchmark_repetitions.csv').open(encoding='utf-8',newline='') as f: reps=list(csv.DictReader(f))
    expected={(str(i),m) for i in range(1,6) for m in ['LinearSVM','LogisticRegression','RandomForest','NaiveBayes','HistGradientBoosting']}
    if len(reps)!=25 or {(r['repeat'],r['model']) for r in reps}!=expected:issues.append('Incomplete repeated benchmark')
    comp=json.loads((gg/'completion.json').read_text())
    if not comp['complete'] or not comp['frozen_hashes_unchanged']:issues.append('Generalization run not complete')
    dd=root/'results/fold_diagnostics_2026-10-20'
    pp=json.loads((dd/'protocol.json').read_text(encoding='utf-8'))
    for path,h in pp['source_sha256'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=h:issues.append('Diagnostic source changed: '+path)
    if hashlib.sha256((root/'scripts/analyze_profile_folds.py').read_bytes()).hexdigest()!=pp['script_sha256']:issues.append('Fold diagnostic code mismatch')
    with (dd/'ranked_predictions.csv').open(encoding='utf-8',newline='') as f: rr=list(csv.DictReader(f))
    if len(rr)!=len(train_ids) or {row['job_id'] for row in rr}!=train_ids:issues.append('Invalid diagnostic prediction coverage')
    for fold in ['1','2','3']:
        subset=[row for row in rr if row['fold']==fold]
        if not subset:issues.append('Missing fold diagnostic: '+fold)
    done=json.loads((dd/'completion.json').read_text())
    if not done['complete'] or not done['source_unchanged']:issues.append('Incomplete diagnostic run')
    return issues

if __name__=='__main__':
    problems=audit()
    print(json.dumps({'passed':not problems,'issues':problems},ensure_ascii=False,indent=2))
    sys.exit(bool(problems))
