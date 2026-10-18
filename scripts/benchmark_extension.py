"""Đo lại cấu hình cố định của 5 pipeline trong môi trường hiện tại; không chọn mô hình.
Chạy SAU robustness, tránh hai thí nghiệm cạnh tranh CPU. Không đọc dự đoán test.
"""
from pathlib import Path
import sys,time,json,importlib.metadata as md,platform,os,ctypes
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from threadpoolctl import threadpool_limits
from sklearn.metrics import average_precision_score
from fakejobs.extension import load_train_validation,feature_views,view_for,make_estimator,extension_configurations
from fakejobs.experiment import ROOT,configurations,scores,dump,sha

out=ROOT/'results/robustness'
out.mkdir(parents=True,exist_ok=True)
# Ghi phần cứng cho chính lượt đo hiện tại; không tái sử dụng metadata máy khác.
hardware={'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'memory_bytes':None}
if sys.platform=='win32':
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            hardware['processor']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended']]
        status=MemoryStatus();status.length=ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):hardware['memory_bytes']=int(status.total_phys)
    except OSError as exc:hardware['hardware_note']=str(exc)
dump(out/'hardware.json',hardware)
locked={p:sha(ROOT/p) for p in ['models/fake_job_detector.joblib','models/lock.json','results/test_metrics.json']}
best=json.loads((ROOT/'results/extension/stats_cv.json').read_text(encoding='utf-8'))['families_best_config']
cfg={c['config_id']:c for c in configurations()+extension_configurations()}
parts,_,_=load_train_validation();tr=parts['train'];va=parts['validation']
rows=[]
with threadpool_limits(limits=1):
    t=time.perf_counter();views=feature_views(tr.drop(columns=['job_id','fraudulent']),tr.fraudulent,va.drop(columns=['job_id','fraudulent']))
    prep=time.perf_counter()-t
    for family,cid in best.items():
        c=cfg[cid];A,B=views[view_for(family)];model=make_estimator(family,c['params'])
        t=time.perf_counter();model.fit(A,tr.fraudulent);fit=time.perf_counter()-t
        t=time.perf_counter();s=scores(model,B);predict=time.perf_counter()-t
        rows.append({'model':family,'config_id':cid,'fit_seconds':fit,'predict_seconds':predict,'AP_validation':average_precision_score(va.fraudulent,s),'features':A.shape[1]})
        print(rows[-1],flush=True)
assert locked=={p:sha(ROOT/p) for p in locked}
dump(out/'benchmark.json',{'date':'2026-10-18','rows':rows,'shared_feature_views_seconds':prep,
 'environment':{'python':platform.python_version(),'platform':platform.platform(),'versions':{p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','threadpoolctl']}},
 'hardware_file':'hardware.json','native_threads':1,'RF_n_jobs':2,'repetitions':1,
 'scope':'classifier.fit and score measured separately; preprocessing excluded; shared preprocessing includes all three views; order fixed, no speedup or significance claim; no model selection',
 'source_sha256':sha(__file__),'frozen_hashes_unchanged':True})
