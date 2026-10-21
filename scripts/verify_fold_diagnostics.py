"""Đối chiếu phân tích fold với dự đoán gốc, không fit hoặc đổi ngưỡng."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from analyze_profile_folds import ranking_details

ROOT=Path(__file__).resolve().parents[1]
def verify():
    r=ROOT/'results/fold_diagnostics_2026-10-20'
    protocol=json.loads((r/'protocol.json').read_text(encoding='utf-8'))
    for p,h in protocol['source_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    assert hashlib.sha256((ROOT/'scripts/analyze_profile_folds.py').read_bytes()).hexdigest()==protocol['script_sha256']
    source=pd.read_csv(ROOT/'results/generalization_2026-10-19/profile_predictions.csv').sort_values('job_id').reset_index(drop=True)
    ranked=pd.read_csv(r/'ranked_predictions.csv').sort_values('job_id').reset_index(drop=True)
    pd.testing.assert_frame_equal(source,ranked[source.columns],check_exact=False,rtol=1e-12)
    groups=pd.read_csv(r/'group_diagnostics.csv')
    foldtable=pd.read_csv(r/'fold_composition.csv').set_index('fold')
    size=pd.read_csv(r/'group_size_distribution.csv')
    for k,f in ranked.groupby('fold'):
        end,pr,neg,c=ranking_details(f.fraudulent,f.score)
        for expected,col in [(end,'rank_end'),(pr,'precision_at_score'),(neg,'negatives_above_or_tied'),(c,'AP_contribution')]:
            np.testing.assert_allclose(expected,f[col],rtol=1e-12,atol=1e-12)
        g=groups[groups.fold.eq(k)]
        assert g.n.sum()==len(f) and g.fake.sum()==f.fraudulent.sum()
        assert np.isclose(g.AP_contribution.sum(),average_precision_score(f.fraudulent,f.score))
        assert np.isclose(foldtable.loc[k,'ROC_AUC'],roc_auc_score(f.fraudulent,f.score))
        assert np.isclose(foldtable.loc[k,'AP'],g.AP_contribution.sum())
        pos=f[f.fraudulent.eq(1)]
        for row in g.itertuples():
            subset=f[f.group.eq(row.group)]
            assert len(subset)==row.n and subset.fraudulent.sum()==row.fake
            assert np.isclose(subset.AP_contribution.sum(),row.AP_contribution)
            pairs=sum(np.searchsorted(np.sort(pos.score),s,side='right') for s in subset.loc[subset.fraudulent.eq(0),'score'])
            assert pairs==row.negative_fake_pairs
        assert np.isclose(size[size.fold.eq(k)].AP_contribution.sum(),foldtable.loc[k,'AP'])
    comp=json.loads((r/'completion.json').read_text())
    assert comp['complete'] and comp['source_unchanged']
    return {'passed':True,'prediction_rows':len(ranked),'groups':len(groups),'folds':3,'source_predictions_unchanged':True,'AP_decomposition_verified':True}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
