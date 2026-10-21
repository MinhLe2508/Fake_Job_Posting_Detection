"""Phân tích bổ sung sau khóa, ngày 18/10. Không lựa chọn lại mô hình triển khai.

Nested CV dùng train; calibration học từ OOF train và chỉ chấm validation.
Chi phí chỉ tính lại từ confusion matrix đã khóa, không tối ưu trên test.
"""
from pathlib import Path
import argparse, json, os, platform, time, warnings, importlib.metadata as md
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from threadpoolctl import threadpool_limits, threadpool_info
from .experiment import ROOT, dump, configurations, estimator, scores, sha
from .extension import load_train_validation
from .features import build_preprocessor

OUT=ROOT/'results/robustness'
FROZEN=['models/fake_job_detector.joblib','models/lock.json','results/protocol.json','results/selection.json','results/test_metrics.json','results/test_predictions.csv']

def split_audit(y,g,tr,va,label):
    assert not set(g.iloc[tr]) & set(g.iloc[va]), label
    assert set(y.iloc[tr])==set(y.iloc[va])=={0,1},label
    return {'split':label,'fit_n':len(tr),'eval_n':len(va),'fit_fake':int(y.iloc[tr].sum()),
            'eval_fake':int(y.iloc[va].sum()),'group_overlap':0}

def fit_score(c,A,y,B):
    m=estimator(c['model'],c['params']);t=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');m.fit(A,y)
    fit=time.perf_counter()-t;t=time.perf_counter();s=scores(m,B)
    return s,fit,time.perf_counter()-t,' | '.join(str(w.message) for w in caught)

def nested(parts,groups):
    frame=parts['train'];X=frame.drop(columns=['job_id','fraudulent']);y=frame.fraudulent;g=groups['train']
    configs=[c for c in configurations() if c['model'] in ['LogisticRegression','LinearSVM']]
    outer=StratifiedGroupKFold(3,shuffle=True,random_state=20261018)
    records=[];results=[];audit=[];predictions=[]
    for k,(tr,va) in enumerate(outer.split(X,y,g),1):
        audit.append(split_audit(y,g,tr,va,f'outer_{k}'))
        xt=X.iloc[tr];yt=y.iloc[tr];gt=g.iloc[tr]
        inner=StratifiedGroupKFold(3,shuffle=True,random_state=20261018+k)
        for j,(it,iv) in enumerate(inner.split(xt,yt,gt),1):
            audit.append(split_audit(yt,gt,it,iv,f'outer_{k}_inner_{j}'))
            prep=build_preprocessor();t=time.perf_counter()
            A=prep.fit_transform(xt.iloc[it]);B=prep.transform(xt.iloc[iv]);prepsec=time.perf_counter()-t
            for c in configs:
                s,fs,ps,w=fit_score(c,A,yt.iloc[it],B)
                records.append({'outer':k,'inner':j,**c,'AP':average_precision_score(yt.iloc[iv],s),
                                'preprocess_seconds':prepsec,'fit_seconds':fs,'predict_seconds':ps,'warnings':w})
            print(f'nested outer={k} inner={j} complete',flush=True)
        selected=pd.DataFrame([r for r in records if r['outer']==k]).groupby(['model','config_id']).AP.mean().reset_index().sort_values(['AP','config_id'],ascending=[False,True]).drop_duplicates('model')
        prep=build_preprocessor();t=time.perf_counter();A=prep.fit_transform(xt);B=prep.transform(X.iloc[va]);prepsec=time.perf_counter()-t
        for row in selected.itertuples():
            c=next(c for c in configs if c['config_id']==row.config_id)
            s,fs,ps,w=fit_score(c,A,yt,B)
            results.append({'outer':k,**c,'inner_AP_mean':row.AP,'outer_AP':average_precision_score(y.iloc[va],s),
                            'eval_n':len(va),'eval_fake':int(y.iloc[va].sum()),'preprocess_seconds':prepsec,'fit_seconds':fs,'predict_seconds':ps,'warnings':w})
            predictions.extend({'job_id':int(frame.iloc[idx].job_id),'outer':k,'model':c['model'],'fraudulent':int(y.iloc[idx]),'score':float(score)} for idx,score in zip(va,s))
        pd.DataFrame(records).to_csv(OUT/'nested_inner.csv',index=False)
        pd.DataFrame(results).to_csv(OUT/'nested_outer.csv',index=False)
        pd.DataFrame(audit).to_csv(OUT/'nested_fold_audit.csv',index=False)
        pd.DataFrame(predictions).to_csv(OUT/'nested_predictions.csv',index=False)
    summary=pd.DataFrame(results).groupby('model').agg(AP_mean=('outer_AP','mean'),AP_std=('outer_AP','std'),inner_AP_mean=('inner_AP_mean','mean')).reset_index()
    summary.to_csv(OUT/'nested_summary.csv',index=False)

def ablation_calibration(parts,folds):
    frame=parts['train'];X=frame.drop(columns=['job_id','fraudulent']);y=frame.fraudulent
    c={'model':'LinearSVM','params':{'C':1,'class_weight':None}}
    records=[];oof=np.full(len(frame),np.nan);assigned=np.zeros(len(frame),int)
    for variant in ['combined','text_only','metadata_only','no_logo','no_log']:
        for k,(tr,va) in enumerate(folds,1):
            prep=build_preprocessor(variant);t=time.perf_counter();A=prep.fit_transform(X.iloc[tr]);B=prep.transform(X.iloc[va]);prepsec=time.perf_counter()-t
            s,fs,ps,w=fit_score(c,A,y.iloc[tr],B)
            records.append({'variant':variant,'fold':k,'AP':average_precision_score(y.iloc[va],s),'preprocess_seconds':prepsec,'fit_seconds':fs,'predict_seconds':ps,'warnings':w})
            if variant=='combined':oof[va]=s;assigned[va]+=1
        pd.DataFrame(records).to_csv(OUT/'svm_ablation.csv',index=False)
        print('SVM ablation',variant,'complete',flush=True)
    assert np.isfinite(oof).all() and (assigned==1).all()
    pd.DataFrame(records).groupby('variant').AP.agg(['mean','std']).to_csv(OUT/'svm_ablation_summary.csv')
    pd.DataFrame({'job_id':frame.job_id,'fraudulent':y,'oof_score':oof}).to_csv(OUT/'calibration_train_oof.csv',index=False)
    # Logistic sigmoid học từ điểm OOF; không đưa nhãn validation vào fit.
    calibrator=LogisticRegression(C=1e6,solver='lbfgs',max_iter=2000,random_state=42)
    calibrator.fit(oof.reshape(-1,1),y)
    vp=pd.read_csv(ROOT/'results/validation_predictions.csv')
    assert np.array_equal(vp.job_id,parts['validation'].job_id)
    p=calibrator.predict_proba(vp.score.to_numpy().reshape(-1,1))[:,1]
    probabilities={'prior_train':np.full(len(vp),y.mean()),'sigmoid_score_heuristic':expit(vp.score.to_numpy()),'oof_logistic_sigmoid':p}
    results=[];bins=[]
    for name,prob in probabilities.items():
        results.append({'method':name,'Brier':brier_score_loss(vp.fraudulent,prob),'log_loss':log_loss(vp.fraudulent,prob),'AP':average_precision_score(vp.fraudulent,prob)})
        ids=np.minimum((prob*10).astype(int),9)
        for b in range(10):
            mask=ids==b
            if mask.any():bins.append({'method':name,'bin':b,'n':int(mask.sum()),'mean_probability':float(prob[mask].mean()),'observed_fake_rate':float(vp.fraudulent[mask].mean())})
    pd.DataFrame(results).to_csv(OUT/'calibration_validation.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'calibration_bins.csv',index=False)
    dump(OUT/'calibration.json',{'slope':float(calibrator.coef_[0,0]),'intercept':float(calibrator.intercept_[0]),'oof_rows':len(oof),
        'validation_rows':len(vp),'results':results,'note':'Post-lock diagnostic. C=1 already selected on train CV. No validation fit; validation previously used for threshold. Not an unbiased new evaluation. sigmoid(score) is a heuristic, not SVM probability. Brier combines calibration and discrimination.'})

def cost():
    m=json.loads((ROOT/'results/test_metrics.json').read_text());rows=[]
    # Đơn vị chuẩn hóa giả định: một FP = 1, một FN = r; không phải tiền doanh nghiệp.
    for r in [1,5,10,20,50]:
        rows.append({'FN_over_FP_cost':r,'locked_model_cost':m['FP']+r*m['FN'],
                     'all_real_cost':r*(m['TP']+m['FN']),'all_flagged_cost':m['TN']+m['FP'],
                     'alerts':m['TP']+m['FP'],'review_minutes_at_2min_each':2*(m['TP']+m['FP'])})
    pd.DataFrame(rows).to_csv(OUT/'cost_sensitivity.csv',index=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['all','nested','ablation','cost'],default='all');args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    before={p:sha(ROOT/p) for p in FROZEN}
    protocol={'date':'2026-10-18','status':'post-lock exploratory extension; no deployment selection',
      'nested':'train only; 3 outer x 3 inner StratifiedGroupKFold; seeds 20261018 and +outer; 8 configs each LR/SVM, original grid; 144 inner fits + 6 outer fits; same combined representation',
      'scope':'Nested comparison covers LR/SVM only, not RF/NB/GB; content grouping, not employer independence; no confirmatory p-values',
      'svm_ablation':'fixed C=1, weight=None; 5 variants x original 5 folds; chosen configuration is not reselected',
      'calibration':'logistic sigmoid C=1e6 learned from train OOF combined SVM scores; evaluate validation only; heuristic sigmoid and train prior comparisons',
      'cost':'fixed confusion matrix only, hypothetical unit FP cost=1 and FN ratios 1/5/10/20/50; no threshold search',
      'frozen_hashes':before,'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
      'logical_cpus':os.cpu_count(),'native_thread_limit':1,'libraries':{p:md.version(p) for p in ['numpy','scipy','pandas','scikit-learn','joblib','threadpoolctl']},'code_sha256':sha(__file__)}
    target=OUT/'protocol.json'
    if not target.exists():dump(target,protocol)
    else:assert json.loads(target.read_text())['code_sha256']==protocol['code_sha256'],'Source changed: use a new result directory'
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        dump(OUT/'runtime_threads.json',threadpool_info())
        if args.stage!='cost':parts,groups,folds=load_train_validation()
        if args.stage in ['all','nested']:nested(parts,groups)
        if args.stage in ['all','ablation']:ablation_calibration(parts,folds)
        if args.stage in ['all','cost']:cost()
    assert before=={p:sha(ROOT/p) for p in FROZEN}
    dump(OUT/f'completion_{args.stage}.json',{'completed':True,'elapsed_seconds':time.perf_counter()-start,'frozen_hashes_unchanged':True})
    print('Completed',args.stage,flush=True)

if __name__=='__main__':main()
