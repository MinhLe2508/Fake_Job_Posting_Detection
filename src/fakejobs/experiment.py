"""Thực nghiệm cuối kỳ có khóa giao thức, CV theo nhóm và test độc lập.

CV dùng lại ma trận của *cùng phần train trong cùng fold* giữa các cấu hình.
Đây là cache phép biến đổi xác định, tương đương fit Pipeline trong mỗi fold;
không dùng từ vựng hoặc thống kê của toàn train khi đánh giá fold đó.
"""
from pathlib import Path
import argparse, hashlib, importlib.metadata as md, json, os, platform, time, warnings
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score, confusion_matrix)
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
from .features import normalize_and_validate, audit_duplicates, content_groups, build_preprocessor

ROOT=Path(__file__).resolve().parents[2]
SEED=42
EXPECTED_SHA='59253295129d8b6a27607f9904f7d4ddf4dbb0ee35e079c7eb51b58b5e62bf60'
EXPECTED={'pandas':'2.3.3','numpy':'2.5.3','scipy':'1.16.3','scikit-learn':'1.7.2',
          'matplotlib':'3.11.2','seaborn':'0.13.2'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def csv_save(rows,path):pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')

def configurations():
    """Ngân sách đã đề xuất giữa kỳ: 8 cấu hình mỗi mô hình, tổng 120 fit CV."""
    result=[]
    for model in ['LogisticRegression','LinearSVM','RandomForest']:
        grid=({'C':[.1,1,10,100],'class_weight':[None,'balanced']} if model!='RandomForest'
            else {'max_depth':[20,None],'min_samples_leaf':[1,5],'class_weight':[None,'balanced_subsample']})
        for i,p in enumerate(ParameterGrid(grid),1):
            result.append({'config_id':f'{model}_{i:02d}','model':model,'params':p})
    return result

def estimator(name,params):
    if name=='LogisticRegression':return LogisticRegression(solver='liblinear',max_iter=2000,random_state=SEED,**params)
    if name=='LinearSVM':return LinearSVC(dual='auto',max_iter=10000,random_state=SEED,**params)
    if name=='RandomForest':return RandomForestClassifier(n_estimators=200,max_features='sqrt',n_jobs=2,random_state=SEED,**params)
    if name=='Dummy':return DummyClassifier(strategy='prior',random_state=SEED)
    raise ValueError(name)

def scores(model,X):
    """Decision score SVM không được gắn nhãn xác suất."""
    if hasattr(model,'predict_proba'):return model.predict_proba(X)[:,1]
    return model.decision_function(X)

def metrics(y,s,threshold,weights=None):
    pred=np.asarray(s)>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1],sample_weight=weights).ravel()
    return {'AP':float(average_precision_score(y,s,sample_weight=weights)),
      'ROC_AUC':float(roc_auc_score(y,s,sample_weight=weights)),
      'precision':float(precision_score(y,pred,sample_weight=weights,zero_division=0)),
      'recall':float(recall_score(y,pred,sample_weight=weights,zero_division=0)),
      'F1':float(f1_score(y,pred,sample_weight=weights,zero_division=0)),
      'accuracy':float(accuracy_score(y,pred,sample_weight=weights)),
      'alert_rate':float(np.average(pred,weights=weights)),
      'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def load_data():
    """Kiểm tra đúng dữ liệu/manifest đã nộp; tuyệt đối không chia lại test."""
    path=ROOT/'data/fake_job_postings.csv'
    if sha(path)!=EXPECTED_SHA:raise ValueError('CSV hash mismatch')
    raw=pd.read_csv(path,low_memory=False)
    frame,audit,quarantine,removed=audit_duplicates(normalize_and_validate(raw))
    assert len(raw)==17880 and len(frame)==17565 and len(removed)==315 and quarantine.empty
    manifest=pd.read_csv(ROOT/'data/split_manifest.csv',index_col=0).set_index('job_id').loc[frame.job_id].reset_index()
    groups=content_groups(frame)
    assert np.array_equal(groups.to_numpy(),manifest.content_group.to_numpy())
    parts={name:frame.loc[manifest.split.eq(name)].reset_index(drop=True) for name in ['train','validation','test']}
    group_parts={name:manifest.loc[manifest.split.eq(name),'content_group'].reset_index(drop=True) for name in parts}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:assert set(group_parts[a]).isdisjoint(group_parts[b])
    train=parts['train'];cv=pd.read_csv(ROOT/'data/cv_manifest.csv',index_col=0).set_index('job_id').loc[train.job_id]
    assert np.array_equal(cv.content_group.to_numpy(),group_parts['train'].to_numpy())
    folds=[];audits=[]
    for fold in range(1,6):
        va=np.flatnonzero(cv.evaluation_fold.to_numpy()==fold);tr=np.flatnonzero(cv.evaluation_fold.to_numpy()!=fold)
        y=train.fraudulent.to_numpy();g=group_parts['train']
        assert set(y[tr])==set(y[va])=={0,1}
        overlap=len(set(g.iloc[tr])&set(g.iloc[va]));fake_groups=g.iloc[va][y[va]==1].nunique()
        assert overlap==0 and y[va].sum()==106 and y[tr].sum()==424 and fake_groups>=20
        assert abs(y[va].mean()-y.mean())<=.02
        folds.append((tr,va));audits.append({'fold':fold,'fit_real':int((y[tr]==0).sum()),'fit_fake':int(y[tr].sum()),
            'eval_real':int((y[va]==0).sum()),'eval_fake':int(y[va].sum()),'fake_groups':int(fake_groups),'group_overlap':overlap})
    csv_save(audits,ROOT/'results/fold_audit.csv')
    return parts,group_parts,folds

def freeze_protocol():
    current={name:md.version(name) for name in EXPECTED}
    if current!=EXPECTED:raise RuntimeError(f'Install requirements.txt first: {current}')
    p={'seed':SEED,'dataset_sha256':EXPECTED_SHA,'split_sha256':sha(ROOT/'data/split_manifest.csv'),
       'cv_sha256':sha(ROOT/'data/cv_manifest.csv'),'selection':'mean CV Average Precision, deterministic config_id tie-break',
       'threshold':'maximum validation recall subject to precision >= 0.80; ties higher precision then higher threshold',
       'fallback':'If infeasible, declare business objective unmet; max-F1 threshold only a disclosed fallback',
       'test_policy':'Open once after lock; selected combined model and fixed prior baseline only',
       'ablation_policy':'Logistic Regression selected C/class_weight, same CV; diagnostic only, does not replace main winner',
       'variants':['text_only','metadata_only','no_logo','no_log'],'class_resampling':'none',
       'configurations':configurations(),'versions':current,'threads_per_forest':2,'blas_threads':1,
       'bootstrap':{'unit':'content_group','replicates':1000,'seed':SEED,'interval':'percentile 2.5% / 97.5%'},
       'training_code_sha256':sha(__file__),'feature_code_sha256':sha(ROOT/'src/fakejobs/features.py')}
    target=ROOT/'results/protocol.json'
    if target.exists():assert load_json(target)==p,'Protocol changed: start a new results directory'
    else:dump(target,p)
    return p

def run_cv(parts,folds):
    target=ROOT/'results/cv_results.csv'
    records=pd.read_csv(target).fillna('').to_dict('records') if target.exists() else []
    done={(r['config_id'],int(r['fold'])) for r in records}
    frame=parts['train'];X=frame.drop(columns=['fraudulent','job_id']);y=frame.fraudulent
    configs=configurations()+[{'config_id':'Dummy_01','model':'Dummy','params':{}}]
    for fold,(tr,va) in enumerate(folds,1):
        needed=[c for c in configs if (c['config_id'],fold) not in done]
        if not needed:continue
        t=time.perf_counter();prep=build_preprocessor();Xt=sparse.csr_matrix(prep.fit_transform(X.iloc[tr]));Xv=sparse.csr_matrix(prep.transform(X.iloc[va]));prep_time=time.perf_counter()-t
        print(f'CV fold {fold}: preprocessing {prep_time:.1f}s, {Xt.shape[1]} features',flush=True)
        for config in needed:
            model=estimator(config['model'],config['params']);t=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(Xt,y.iloc[tr])
            fit_time=time.perf_counter()-t;t=time.perf_counter();s=scores(model,Xv);predict_time=time.perf_counter()-t
            threshold=0. if config['model']=='LinearSVM' else .5
            row={'config_id':config['config_id'],'model':config['model'],'fold':fold,
                 'params':json.dumps(config['params'],sort_keys=True),'n_features':Xt.shape[1],
                 'preprocess_seconds':prep_time,'fit_seconds':fit_time,'predict_seconds':predict_time,
                 'warnings':' | '.join(str(w.message) for w in caught),**metrics(y.iloc[va],s,threshold)}
            records.append(row);csv_save(records,target)
            print(f"{row['config_id']} fold={fold} AP={row['AP']:.4f}, fit={fit_time:.1f}s",flush=True)
    df=pd.DataFrame(records);assert len(df)==125
    summary=df.groupby(['config_id','model','params'],dropna=False).agg(AP_mean=('AP','mean'),AP_std=('AP','std'),
        ROC_AUC_mean=('ROC_AUC','mean'),F1_default_mean=('F1','mean'),recall_default_mean=('recall','mean'),
        precision_default_mean=('precision','mean'),fit_seconds_mean=('fit_seconds','mean'),n_folds=('fold','nunique')).reset_index()
    assert summary.n_folds.eq(5).all()
    summary=summary.sort_values(['AP_mean','config_id'],ascending=[False,True]);summary.to_csv(ROOT/'results/cv_summary.csv',index=False,encoding='utf-8-sig')
    best=summary.drop_duplicates('model');best.to_csv(ROOT/'results/best_per_model.csv',index=False,encoding='utf-8-sig')
    winner=summary.loc[summary.model.ne('Dummy')].iloc[0]
    dump(ROOT/'results/selection.json',{'config_id':winner.config_id,'model':winner.model,'params':json.loads(winner.params),'AP_mean':winner.AP_mean,'AP_std':winner.AP_std,'selected_using':'train CV only'})

def run_ablation(parts,folds):
    best=pd.read_csv(ROOT/'results/best_per_model.csv');lr=best.loc[best.model.eq('LogisticRegression')].iloc[0];params=json.loads(lr.params)
    target=ROOT/'results/ablation.csv';records=pd.read_csv(target).to_dict('records') if target.exists() else []
    done={(r['variant'],int(r['fold'])) for r in records}
    X=parts['train'].drop(columns=['fraudulent','job_id']);y=parts['train'].fraudulent
    for variant in ['text_only','metadata_only','no_logo','no_log']:
        for fold,(tr,va) in enumerate(folds,1):
            if (variant,fold) in done:continue
            t=time.perf_counter();pipe=Pipeline([('preprocess',build_preprocessor(variant)),('classifier',estimator('LogisticRegression',params))])
            pipe.fit(X.iloc[tr],y.iloc[tr]);s=scores(pipe,X.iloc[va]);m=metrics(y.iloc[va],s,.5)
            records.append({'variant':variant,'fold':fold,'seconds':time.perf_counter()-t,**m});csv_save(records,target)
            print(f'Ablation {variant} fold {fold} AP={m["AP"]:.4f}',flush=True)
    base=pd.read_csv(ROOT/'results/cv_results.csv');base=base.loc[base.config_id.eq(lr.config_id)].copy();base['variant']='combined'
    all_rows=pd.concat([pd.DataFrame(records),base],ignore_index=True)
    summary=all_rows.groupby('variant').agg(AP_mean=('AP','mean'),AP_std=('AP','std'),F1_default_mean=('F1','mean'),recall_default_mean=('recall','mean')).reset_index()
    summary.to_csv(ROOT/'results/ablation_summary.csv',index=False,encoding='utf-8-sig')

def choose_threshold(y,s,min_precision=.8):
    p,r,t=precision_recall_curve(y,s);p=p[:-1];r=r[:-1]
    eligible=np.flatnonzero(p>=min_precision);feasible=len(eligible)>0
    if feasible:index=max(eligible,key=lambda i:(r[i],p[i],t[i]))
    else:
        f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0);index=int(np.argmax(f))
    return float(t[index]),bool(feasible),pd.DataFrame({'threshold':t,'precision':p,'recall':r})

def fit_and_lock(parts):
    if (ROOT/'models/lock.json').exists():return
    selected=load_json(ROOT/'results/selection.json')
    train=parts['train'];val=parts['validation'];X=train.drop(columns=['fraudulent','job_id']);V=val.drop(columns=['fraudulent','job_id'])
    pipe=Pipeline([('preprocess',build_preprocessor()),('classifier',estimator(selected['model'],selected['params']))])
    t=time.perf_counter();pipe.fit(X,train.fraudulent);fit_seconds=time.perf_counter()-t
    t=time.perf_counter();s=scores(pipe,V);predict_seconds=time.perf_counter()-t
    threshold,feasible,curve=choose_threshold(val.fraudulent,s)
    curve.to_csv(ROOT/'results/validation_threshold_curve.csv',index=False,encoding='utf-8-sig')
    results={'threshold':threshold,'precision_target':.8,'target_feasible_on_validation':feasible,
             'fit_seconds':fit_seconds,'batch_predict_seconds':predict_seconds,**metrics(val.fraudulent,s,threshold)}
    dump(ROOT/'results/validation_metrics.json',results)
    pred=val[['job_id','fraudulent']].copy();pred['score']=s;pred['predicted']=(s>=threshold).astype(int);pred.to_csv(ROOT/'results/validation_predictions.csv',index=False)
    score_type='decision_score' if selected['model']=='LinearSVM' else 'probability'
    bundle={'pipeline':pipe,'threshold':threshold,'score_type':score_type,'model_name':selected['model'],'selection':selected,'input_columns':X.columns.tolist(),'dataset_sha256':EXPECTED_SHA}
    joblib.dump(bundle,ROOT/'models/fake_job_detector.joblib',compress=3)
    lock={'selection':selected,'threshold':threshold,'score_type':score_type,'validation_precision_goal_met':feasible,
          'model_sha256':sha(ROOT/'models/fake_job_detector.joblib'),'protocol_sha256':sha(ROOT/'results/protocol.json'),
          'locked_before_test':True,'test_used_for_selection':False,'n_features':len(pipe.named_steps['preprocess'].named_steps['columns'].get_feature_names_out())}
    dump(ROOT/'models/lock.json',lock)
    print(f'Model locked: {selected["model"]}, threshold={threshold:.6f}, validation={results}',flush=True)

def bootstrap_groups(y,s,groups,threshold,n=1000):
    """Lấy mẫu lại *nhóm* test, giữ nguyên mô hình/ngưỡng. Không dùng để tinh chỉnh."""
    codes,unique=pd.factorize(groups);rng=np.random.default_rng(SEED);records=[];skipped=0
    for b in range(n):
        counts=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique));weights=counts[codes]
        if len(np.unique(y[weights>0]))<2:skipped+=1;continue
        m=metrics(y,s,threshold,weights);records.append(m)
    result={k:{'low':float(np.percentile([r[k] for r in records],2.5)),
               'high':float(np.percentile([r[k] for r in records],97.5))} for k in ['AP','ROC_AUC','precision','recall','F1','alert_rate']}
    return {'replicates':n,'valid':len(records),'skipped_single_class':skipped,'groups':len(unique),'intervals':result}

def evaluate_test(parts,groups):
    target=ROOT/'results/test_metrics.json'
    if target.exists():
        print('Test already evaluated: using frozen results; not fitting or tuning.',flush=True);return
    lock=load_json(ROOT/'models/lock.json');assert sha(ROOT/'models/fake_job_detector.joblib')==lock['model_sha256']
    assert sha(ROOT/'results/protocol.json')==lock['protocol_sha256']
    bundle=joblib.load(ROOT/'models/fake_job_detector.joblib');test=parts['test'];X=test.drop(columns=['fraudulent','job_id']);y=test.fraudulent.to_numpy()
    t=time.perf_counter();s=scores(bundle['pipeline'],X);seconds=time.perf_counter()-t
    m={'model':bundle['model_name'],'threshold':bundle['threshold'],'n_samples':len(y),'n_fake':int(y.sum()),
       'batch_predict_seconds':seconds,'milliseconds_per_post_batch':1000*seconds/len(y),**metrics(y,s,bundle['threshold'])}
    # Ghi dự đoán ngay; bootstrap và phân tích phía sau không gọi fit hoặc thay ngưỡng.
    pred=test[['job_id','fraudulent']].copy();pred['score']=s;pred['predicted']=(s>=bundle['threshold']).astype(int);pred['content_group']=groups['test']
    pred.to_csv(ROOT/'results/test_predictions.csv',index=False)
    baseline=np.full(len(y),parts['train'].fraudulent.mean());m['dummy_prior']=metrics(y,baseline,.5)
    dump(target,m);dump(ROOT/'results/test_bootstrap.json',bootstrap_groups(y,s,groups['test'],bundle['threshold']))
    print(f'FROZEN TEST: {m}',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['cv','ablation','lock','test','all'],default='all');args=parser.parse_args()
    for d in ['results','models']:(ROOT/d).mkdir(exist_ok=True)
    freeze_protocol();parts,groups,folds=load_data()
    dump(ROOT/'results/environment.json',{'python':platform.python_version(),'platform':platform.platform(),'logical_cpus':os.cpu_count(),
         'versions':{k:md.version(k) for k in list(EXPECTED)+['joblib','threadpoolctl']}})
    with threadpool_limits(limits=1):
        if args.stage in ['cv','all']:run_cv(parts,folds)
        if args.stage in ['ablation','all']:run_ablation(parts,folds)
        if args.stage in ['lock','all']:fit_and_lock(parts)
        if args.stage in ['test','all']:evaluate_test(parts,groups)

if __name__=='__main__':main()
