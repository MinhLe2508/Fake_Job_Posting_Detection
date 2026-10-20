"""Phân tích hậu kiểm các fold đã lưu; không fit, không đổi seed/ngưỡng."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fold_diagnostics_2026-10-20'

def ranking_details(y,score):
    """Precision tại mỗi score tính cả tie; tổng trên tin dương / P đúng bằng AP.

    rank_end là số tin score >= score của tin đang xét, không phá tie theo ID.
    negatives_above_or_tied là số tin âm có score >= score của tin dương.
    """
    y=np.asarray(y);score=np.asarray(score,dtype=float)
    if set(np.unique(y))!={0,1} or not np.isfinite(score).all():
        raise ValueError('Cần đủ hai lớp 0/1 và score hữu hạn')
    allsorted=np.sort(score);possorted=np.sort(score[y==1])
    end=len(y)-np.searchsorted(allsorted,score,side='left')
    pos=len(possorted)-np.searchsorted(possorted,score,side='left')
    precision=pos/end
    contribution=np.where(y==1,precision/y.sum(),0.)
    assert np.isclose(contribution.sum(),average_precision_score(y,score))
    return end,precision,end-pos,contribution

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    paths=['results/generalization_2026-10-19/profile_predictions.csv',
           'results/generalization_2026-10-19/profile_group_manifest.csv',
           'results/generalization_2026-10-19/profile_folds.csv',
           'data/fake_job_postings.csv','models/fake_job_detector.joblib',
           'models/lock.json','results/test_metrics.json','results/test_predictions.csv']
    def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
    hashes={p:sha(p) for p in paths}
    pred=pd.read_csv(ROOT/paths[0])
    manifest=pd.read_csv(ROOT/paths[1])
    assert pred.job_id.is_unique and set(pred.job_id)==set(manifest.job_id)
    pred=pred.merge(manifest,on=['job_id','group'],validate='one_to_one')
    assert len(pred)==10836 and pred.fraudulent.sum()==530
    assert pred.groupby('group').fold.nunique().max()==1
    raw=pd.read_csv(ROOT/'data/fake_job_postings.csv',low_memory=False).set_index('job_id')
    assert np.array_equal(raw.loc[pred.job_id,'fraudulent'],pred.fraudulent)
    fold_rows=[];group_rows=[];rank_rows=[];buckets=[]
    for fold,df in pred.groupby('fold'):
        df=df.copy()
        end,pr,neg,apc=ranking_details(df.fraudulent,df.score)
        df['rank_end']=end;df['precision_at_score']=pr
        df['negatives_above_or_tied']=neg
        df['AP_contribution']=apc
        size=df.groupby('group').size()
        df['group_size']=df.group.map(size)
        positives=df[df.fraudulent.eq(1)]
        n=len(df);p=int(df.fraudulent.sum());total_pairs=float(positives.negatives_above_or_tied.sum())
        largest=size.idxmax()
        fold_rows.append({'fold':int(fold),'n':n,'fake':p,'fake_rate':p/n,'groups':len(size),
            'groups_with_fake':int(df.groupby('group').fraudulent.sum().gt(0).sum()),
            'largest_group':int(largest),'largest_group_n':int(size.max()),
            'largest_group_fake':int(df.loc[df.group.eq(largest),'fraudulent'].sum()),
            'missing_profile':int(df.profile_missing.sum()),
            'fake_in_groups_ge10':int(df.loc[df.group_size.ge(10),'fraudulent'].sum()),
            'AP':average_precision_score(df.fraudulent,df.score),
            'ROC_AUC':roc_auc_score(df.fraudulent,df.score),
            'positive_rank_median':float(positives.rank_end.median()),
            'negative_above_fake_median':float(positives.negatives_above_or_tied.median()),
            'positive_precision_mean':float(positives.precision_at_score.mean())})
        for group,g in df.groupby('group'):
            pg=g[g.fraudulent.eq(1)];ng=g[g.fraudulent.eq(0)]
            # Đóng góp của nhóm tin thật vào tổng cặp âm có score >= tin giả.
            wrong_pairs=int(sum(np.searchsorted(np.sort(positives.score),s,side='right') for s in ng.score))
            group_rows.append({'fold':int(fold),'group':int(group),'n':len(g),'fake':len(pg),
                'fake_share_of_fold':len(pg)/p,'AP_contribution':float(pg.AP_contribution.sum()),
                'positive_precision_mean':float(pg.precision_at_score.mean()) if len(pg) else None,
                'positive_rank_median':float(pg.rank_end.median()) if len(pg) else None,
                'negative_above_fake_median':float(pg.negatives_above_or_tied.median()) if len(pg) else None,
                'score_median':float(g.score.median()),'missing_profile':int(g.profile_missing.sum()),
                'negative_fake_pairs':wrong_pairs,'share_negative_fake_pairs':wrong_pairs/total_pairs if total_pairs else 0})
        for label,mask in [('1',df.group_size.eq(1)),('2-9',df.group_size.between(2,9)),('>=10',df.group_size.ge(10))]:
            g=df[mask];pg=g[g.fraudulent.eq(1)]
            buckets.append({'fold':int(fold),'group_size':label,'n':len(g),'groups':g.group.nunique(),'fake':len(pg),'fake_share':len(pg)/p,'AP_contribution':float(pg.AP_contribution.sum())})
        rank_rows.extend(df.to_dict('records'))
    folds=pd.DataFrame(fold_rows);groups=pd.DataFrame(group_rows)
    folds.to_csv(OUT/'fold_composition.csv',index=False)
    groups.to_csv(OUT/'group_diagnostics.csv',index=False)
    pd.DataFrame(buckets).to_csv(OUT/'group_size_distribution.csv',index=False)
    ranked=pd.DataFrame(rank_rows)
    ranked.to_csv(OUT/'ranked_predictions.csv',index=False)
    # Chọn theo quy tắc công bố: 3 nhóm nhiều tin giả nhất mỗi fold và 2 nhóm lớn nhất.
    examples=[]
    for fold,g in groups.groupby('fold'):
        chosen=set(g.sort_values(['fake','group'],ascending=[False,True]).head(3).group)
        chosen.update(g.sort_values(['n','group'],ascending=[False,True]).head(2).group)
        for group in sorted(chosen):
            rows=ranked[(ranked.fold==fold)&(ranked.group==group)]
            rawrows=raw.loc[rows.job_id]
            examples.append({'fold':int(fold),'group':int(group),'n':len(rows),
                'sample_job_ids':rows.job_id.head(3).tolist(),
                'sample_titles':rawrows.title.fillna('').head(3).tolist(),
                'logo_rate':float(rawrows.has_company_logo.mean()),
                'locations_top3':rawrows.location.fillna('(missing)').value_counts().head(3).to_dict()})
    assert all(sha(p)==h for p,h in hashes.items())
    def write(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write('selected_group_examples.json',examples)
    write('protocol.json',{'date':'2026-10-20','source_sha256':hashes,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Post-hoc descriptive decomposition of fixed train OOF predictions; no fitting, no split or threshold selection; no causal attribution',
        'group_size_bins':['1','2-9','>=10'],'tie_rule':'score >=; AP contribution uses precision at score endpoint',
        'negative_pair_rule':'number of negative-positive pairs with negative score >= positive score; not FP at a deployment threshold',
        'selected_examples':'top 3 fake-count groups and top 2 size groups per fold; ties by group ID ascending'})
    write('completion.json',{'complete':True,'source_unchanged':True,'rows':len(ranked),'folds':3,'groups':len(groups)})
    print(folds.to_string(index=False))
    print(groups[groups.fold.eq(3)].sort_values('fake',ascending=False).head(5).to_string(index=False))

if __name__=='__main__':main()
