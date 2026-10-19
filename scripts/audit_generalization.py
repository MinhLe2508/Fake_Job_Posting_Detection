"""Chẩn đoán sau khóa: nhóm nội dung/profile trên train và benchmark 5 lần.

Không chọn lại mô hình hoặc ngưỡng; không chấm phương án mới trên test.
Đầu ra riêng, không ghi đè thực nghiệm 18/10. Chạy một tiến trình duy nhất.
"""
from pathlib import Path
import sys, json, time, platform, warnings, importlib.metadata as md
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits, threadpool_info
from fakejobs.experiment import ROOT, configurations, scores, dump, sha
from fakejobs.extension import load_train_validation, feature_views, view_for, make_estimator, extension_configurations
from fakejobs.features import build_preprocessor
from fakejobs.robustness import split_audit

OUT = ROOT/'results/generalization_2026-10-19'

def connected_groups(content, profiles):
    """Nối mọi cặp trùng nội dung HOẶC profile không rỗng, kể cả nối bắc cầu.

    Không gom tất cả profile thiếu thành một công ty. Chỉ chuẩn hóa khoảng trắng
    và chữ thường; đây là proxy văn bản, không phải mã doanh nghiệp thật.
    """
    n=len(content); parent=list(range(n)); seen={}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    for i,(c,p) in enumerate(zip(content,profiles)):
        profile=' '.join(str(p).lower().split()) if pd.notna(p) else ''
        keys=[('content',str(c))]+([('profile',profile)] if profile else [])
        for key in keys:
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    return pd.Series([find(i) for i in range(n)]),pd.Series([' '.join(str(p).lower().split()) if pd.notna(p) else '' for p in profiles])

def profile_diagnostic(parts,groups):
    f=parts['train']; X=f.drop(columns=['job_id','fraudulent']); y=f.fraudulent
    g,profile=connected_groups(groups['train'],f.company_profile)
    split=StratifiedGroupKFold(3,shuffle=True,random_state=20261019)
    records=[];pred=[];assigned=np.zeros(len(f),int)
    for k,(tr,va) in enumerate(split.split(X,y,g),1):
        audit=split_audit(y,g,tr,va,f'profile_outer_{k}')
        cp=set(profile.iloc[tr]) & set(profile.iloc[va]);cp.discard('')
        assert not cp and not set(groups['train'].iloc[tr]) & set(groups['train'].iloc[va])
        prep=build_preprocessor();t=time.perf_counter();A=prep.fit_transform(X.iloc[tr]);B=prep.transform(X.iloc[va]);prepsec=time.perf_counter()-t
        model=make_estimator('LinearSVM',{'C':1,'class_weight':None})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');t=time.perf_counter();model.fit(A,y.iloc[tr]);fit=time.perf_counter()-t
        s=scores(model,B);assigned[va]+=1
        records.append({**audit,'fold':k,'AP':average_precision_score(y.iloc[va],s),'profile_overlap_nonempty':0,'content_overlap':0,'eval_missing_profile':int(profile.iloc[va].eq('').sum()),'preprocess_seconds':prepsec,'fit_seconds':fit,'warnings':' | '.join(str(w.message) for w in caught)})
        pred.extend({'job_id':int(f.iloc[i].job_id),'fold':k,'group':int(g.iloc[i]),'fraudulent':int(y.iloc[i]),'score':float(sc)} for i,sc in zip(va,s))
        print('profile',records[-1],flush=True)
    assert (assigned==1).all()
    df=pd.DataFrame(records);df.to_csv(OUT/'profile_folds.csv',index=False)
    pd.DataFrame(pred).to_csv(OUT/'profile_predictions.csv',index=False)
    pd.DataFrame({'job_id':f.job_id,'group':g,'profile_missing':profile.eq('')}).to_csv(OUT/'profile_group_manifest.csv',index=False)
    dump(OUT/'profile_summary.json',{'AP_mean':df.AP.mean(),'AP_std':df.AP.std(),'train_rows':len(f),'fake_rows':int(y.sum()),'groups':int(g.nunique()),'largest_group':int(g.value_counts().max()),'missing_profile_rows':int(profile.eq('').sum()),'fixed_config':{'C':1,'class_weight':None},'scope':'train-only post-lock diagnostic; configuration previously chosen on train CV, not nested or unbiased external evaluation; exact profile is not employer ID; no threshold selection'})

def repeated_benchmark(parts):
    best=json.loads((ROOT/'results/extension/stats_cv.json').read_text(encoding='utf-8'))['families_best_config']
    cfg={c['config_id']:c for c in configurations()+extension_configurations()}
    tr=parts['train'];va=parts['validation'];t=time.perf_counter()
    views=feature_views(tr.drop(columns=['job_id','fraudulent']),tr.fraudulent,va.drop(columns=['job_id','fraudulent']))
    prep=time.perf_counter()-t;rows=[];rng=np.random.default_rng(20261019)
    for repeat in range(1,6):
        for order,family in enumerate(rng.permutation(sorted(best)),1):
            c=cfg[best[family]];A,B=views[view_for(family)];m=make_estimator(family,c['params'])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');t=time.perf_counter();m.fit(A,tr.fraudulent);fit=time.perf_counter()-t
            t=time.perf_counter();s=scores(m,B);pred=time.perf_counter()-t
            rows.append({'repeat':repeat,'order':order,'model':family,'config_id':c['config_id'],'fit_seconds':fit,'predict_ms':pred*1000,'AP_validation':average_precision_score(va.fraudulent,s),'warnings':' | '.join(str(w.message) for w in caught)})
            pd.DataFrame(rows).to_csv(OUT/'benchmark_repetitions.csv',index=False)
            print('benchmark',repeat,family,round(fit,3),flush=True)
    df=pd.DataFrame(rows)
    df.groupby('model').agg(n=('fit_seconds','size'),fit_median_s=('fit_seconds','median'),fit_q1_s=('fit_seconds',lambda s:s.quantile(.25)),fit_q3_s=('fit_seconds',lambda s:s.quantile(.75)),predict_median_ms=('predict_ms','median'),predict_q1_ms=('predict_ms',lambda s:s.quantile(.25)),predict_q3_ms=('predict_ms',lambda s:s.quantile(.75)),AP_min=('AP_validation','min'),AP_max=('AP_validation','max')).reset_index().to_csv(OUT/'benchmark_summary.csv',index=False)
    dump(OUT/'benchmark_scope.json',{'repetitions':5,'seed_order':20261019,'shared_preprocessing_seconds':prep,'note':'same matrices, process and machine; newly instantiated estimator each repetition; no warmup excluded; randomized order; no preprocessing/disk/network in timed fit or prediction; repetitions assess runtime, not independent statistical accuracy samples; RF n_jobs=2, native threads=1'})

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    frozen=['models/fake_job_detector.joblib','models/lock.json','results/protocol.json','results/selection.json','results/test_metrics.json','results/test_predictions.csv']
    before={p:sha(ROOT/p) for p in frozen}
    dump(OUT/'protocol.json',{'date':'2026-10-19','code_sha256':sha(__file__),'seed':20261019,'profile_folds':3,'profile_config':'LinearSVM C=1 class_weight=None; fixed, no tuning','group_rule':'connected components of content_group OR nonempty normalized company_profile; train only','benchmark_repetitions':5,'benchmark_configs':'historical fixed best per family in results/extension/stats_cv.json','selection':'none; post-lock diagnostic only','frozen_hashes':before,'python':platform.python_version(),'platform':platform.platform(),'versions':{p:md.version(p) for p in ['numpy','pandas','scipy','scikit-learn','threadpoolctl']}})
    # Đọc metadata phần cứng hiện tại thay vì gán thông tin lịch sử.
    import os,ctypes
    hardware={'logical_cpus':os.cpu_count(),'processor':platform.processor(),'memory_bytes':None}
    if sys.platform=='win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:hardware['processor']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended']]
        status=MemoryStatus();status.length=ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):hardware['memory_bytes']=int(status.total_phys)
    dump(OUT/'hardware.json',hardware)
    t=time.perf_counter()
    with threadpool_limits(limits=1):
        dump(OUT/'threads.json',threadpool_info());parts,groups,_=load_train_validation()
        profile_diagnostic(parts,groups);repeated_benchmark(parts)
    assert before=={p:sha(ROOT/p) for p in frozen}
    dump(OUT/'completion.json',{'complete':True,'seconds':time.perf_counter()-t,'frozen_hashes_unchanged':True})

if __name__=='__main__':main()
