"""Tính lại số đo bổ sung từ dự đoán và log thời gian; không fit mô hình."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from audit_generalization import connected_groups
from fakejobs.extension import load_train_validation
from fakejobs.experiment import ROOT,sha

def verify():
    r=ROOT/'results/generalization_2026-10-19'
    protocol=json.loads((r/'protocol.json').read_text(encoding='utf-8'))
    assert sha(ROOT/'scripts/audit_generalization.py')==protocol['code_sha256']
    for p,h in protocol['frozen_hashes'].items():assert sha(ROOT/p)==h,p
    parts,groups,_=load_train_validation();tr=parts['train']
    g,profile=connected_groups(groups['train'],tr.company_profile)
    pred=pd.read_csv(r/'profile_predictions.csv').set_index('job_id')
    assert pred.index.is_unique and set(pred.index)==set(tr.job_id)
    pred=pred.loc[tr.job_id].reset_index()
    assert np.array_equal(pred.fraudulent,tr.fraudulent)
    assert np.array_equal(pred.group,g)
    folds=pd.read_csv(r/'profile_folds.csv').set_index('fold');aps=[]
    for k in [1,2,3]:
        ev=pred.fold.eq(k);fit=~ev
        assert not set(g[ev])&set(g[fit])
        assert not (set(profile[ev])- {''}) & (set(profile[fit])- {''})
        assert not set(groups['train'][ev])&set(groups['train'][fit])
        ap=average_precision_score(pred.loc[ev,'fraudulent'],pred.loc[ev,'score']);aps.append(ap)
        assert np.isclose(ap,folds.loc[k,'AP'],rtol=0,atol=1e-12)
        assert ev.sum()==folds.loc[k,'eval_n'] and tr.loc[ev,'fraudulent'].sum()==folds.loc[k,'eval_fake']
    summary=json.loads((r/'profile_summary.json').read_text())
    assert np.isclose(np.mean(aps),summary['AP_mean']) and np.isclose(np.std(aps,ddof=1),summary['AP_std'])
    raw=pd.read_csv(r/'benchmark_repetitions.csv');bs=pd.read_csv(r/'benchmark_summary.csv').set_index('model')
    assert len(raw)==25 and not raw.duplicated(['repeat','model']).any()
    for model,rows in raw.groupby('model'):
        assert set(rows.repeat)=={1,2,3,4,5}
        for rawcol,prefix,suffix in [('fit_seconds','fit','s'),('predict_ms','predict','ms')]:
            for label,q in [('median',.5),('q1',.25),('q3',.75)]:
                assert np.isclose(rows[rawcol].quantile(q),bs.loc[model,f'{prefix}_{label}_{suffix}'])
    return {'passed':True,'profile_rows':len(pred),'folds':3,'benchmark_rows':len(raw),'recomputed_AP_mean':float(np.mean(aps)),'frozen_hashes_unchanged':True}

if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
