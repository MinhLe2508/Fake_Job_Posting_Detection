"""Mở rộng SAU KHÓA: Naive Bayes, Gradient Boosting và kiểm định thống kê giữa các họ mô hình.

Quy tắc (ghi trong results/extension/protocol_extension.json):
- Chạy sau khi đã khóa mô hình/ngưỡng và đã mở test. Không đọc, không chấm, không nạp tập test.
- Không ghi đè results/selection.json, models/, protocol.json hay bất kỳ kết quả đã khóa nào.
- Dùng đúng 5 fold của train (data/cv_manifest.csv) và cùng tiền xử lý fit trong từng fold.
- Kết quả chỉ để so sánh/giải trình; không thay mô hình đã khóa.
"""
from pathlib import Path
import argparse, json, platform, time, warnings, importlib.metadata as md
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from threadpoolctl import threadpool_limits
from .experiment import ROOT, SEED, EXPECTED_SHA, sha, csv_save, dump, load_json, estimator, scores, metrics, configurations
from .features import normalize_and_validate, audit_duplicates, content_groups, build_preprocessor

OUT = ROOT / 'results' / 'extension'
K_FEATURES = 2000          # chọn đặc trưng trong fold cho Gradient Boosting (dữ liệu dày cần ma trận đặc)
BOOT = 1000

def extension_configurations():
    result = []
    for i, p in enumerate(ParameterGrid({'variant': ['MultinomialNB', 'ComplementNB'], 'alpha': [.01, .1, .5, 1.0]}), 1):
        result.append({'config_id': f'NaiveBayes_{i:02d}', 'model': 'NaiveBayes', 'params': p})
    grid = {'learning_rate': [.05, .1], 'max_leaf_nodes': [15, 31], 'class_weight': [None, 'balanced']}
    for i, p in enumerate(ParameterGrid(grid), 1):
        result.append({'config_id': f'HistGradientBoosting_{i:02d}', 'model': 'HistGradientBoosting', 'params': p})
    return result

def make_estimator(model, params):
    if model == 'NaiveBayes':
        cls = MultinomialNB if params['variant'] == 'MultinomialNB' else ComplementNB
        return cls(alpha=params['alpha'])
    if model == 'HistGradientBoosting':
        return HistGradientBoostingClassifier(max_iter=200, early_stopping=False, l2_regularization=1.0,
                                              random_state=SEED, **params)
    return estimator(model, params)

def load_train_validation():
    """Chỉ nạp train và validation. Tập test không được tạo ra trong module này."""
    path = ROOT / 'data/fake_job_postings.csv'
    if sha(path) != EXPECTED_SHA: raise ValueError('CSV hash mismatch')
    frame, _, quarantine, removed = audit_duplicates(normalize_and_validate(pd.read_csv(path, low_memory=False)))
    assert len(frame) == 17565 and len(removed) == 315 and quarantine.empty
    manifest = pd.read_csv(ROOT / 'data/split_manifest.csv', index_col=0).set_index('job_id').loc[frame.job_id].reset_index()
    assert np.array_equal(content_groups(frame).to_numpy(), manifest.content_group.to_numpy())
    parts = {n: frame.loc[manifest.split.eq(n)].reset_index(drop=True) for n in ['train', 'validation']}
    groups = {n: manifest.loc[manifest.split.eq(n), 'content_group'].reset_index(drop=True) for n in parts}
    assert set(groups['train']).isdisjoint(groups['validation'])
    cv = pd.read_csv(ROOT / 'data/cv_manifest.csv', index_col=0).set_index('job_id').loc[parts['train'].job_id]
    assert np.array_equal(cv.content_group.to_numpy(), groups['train'].to_numpy())
    folds = [(np.flatnonzero(cv.evaluation_fold.to_numpy() != f), np.flatnonzero(cv.evaluation_fold.to_numpy() == f)) for f in range(1, 6)]
    return parts, groups, folds

def feature_views(X_fit, y_fit, X_eval):
    """Ba góc nhìn đặc trưng, tất cả fit CHỈ trên phần huấn luyện: combined thưa, text-only, combined chọn K đặc trưng (dày)."""
    prep = build_preprocessor()
    A = sparse.csr_matrix(prep.fit_transform(X_fit)); Av = sparse.csr_matrix(prep.transform(X_eval))
    tprep = build_preprocessor('text_only')
    T = sparse.csr_matrix(tprep.fit_transform(X_fit)); Tv = sparse.csr_matrix(tprep.transform(X_eval))
    sel = SelectKBest(f_classif, k=min(K_FEATURES, A.shape[1])).fit(A, y_fit)
    D = sel.transform(A).toarray().astype(np.float32); Dv = sel.transform(Av).toarray().astype(np.float32)
    return {'sparse': (A, Av), 'text': (T, Tv), 'dense': (D, Dv)}

def view_for(model):
    return {'NaiveBayes': 'text', 'HistGradientBoosting': 'dense'}.get(model, 'sparse')

def run_cv():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'cv_results_extension.csv'
    records = pd.read_csv(target).fillna('').to_dict('records') if target.exists() else []
    done = {(r['config_id'], int(r['fold'])) for r in records}
    parts, _, folds = load_train_validation()
    X = parts['train'].drop(columns=['fraudulent', 'job_id']); y = parts['train'].fraudulent
    configs = extension_configurations()
    for fold, (tr, va) in enumerate(folds, 1):
        needed = [c for c in configs if (c['config_id'], fold) not in done]
        if not needed: continue
        t = time.perf_counter(); views = feature_views(X.iloc[tr], y.iloc[tr], X.iloc[va]); prep_time = time.perf_counter() - t
        print(f'CV fold {fold}: views {prep_time:.1f}s', flush=True)
        for c in needed:
            Xt, Xv = views[view_for(c['model'])]
            model = make_estimator(c['model'], c['params']); t = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); model.fit(Xt, y.iloc[tr])
            fit_time = time.perf_counter() - t; s = scores(model, Xv)
            row = {'config_id': c['config_id'], 'model': c['model'], 'fold': fold, 'params': json.dumps(c['params'], sort_keys=True),
                   'n_features': Xt.shape[1], 'preprocess_seconds': prep_time, 'fit_seconds': fit_time,
                   'warnings': ' | '.join(str(w.message) for w in caught), **metrics(y.iloc[va], s, .5)}
            records.append(row); csv_save(records, target)
            print(f"{row['config_id']} fold={fold} AP={row['AP']:.4f} fit={fit_time:.1f}s", flush=True)
    df = pd.DataFrame(records); assert len(df) == len(configs) * 5
    summary = df.groupby(['config_id', 'model', 'params']).agg(AP_mean=('AP', 'mean'), AP_std=('AP', 'std'), ROC_AUC_mean=('ROC_AUC', 'mean'),
        F1_default_mean=('F1', 'mean'), recall_default_mean=('recall', 'mean'), precision_default_mean=('precision', 'mean'),
        fit_seconds_mean=('fit_seconds', 'mean')).reset_index().sort_values(['AP_mean', 'config_id'], ascending=[False, True])
    summary.to_csv(OUT / 'cv_summary_extension.csv', index=False, encoding='utf-8-sig')

# ---------- kiểm định thống kê ----------
def nadeau_bengio(d, test_over_train):
    """t-test ghép cặp có hiệu chỉnh phương sai cho CV lặp/chồng lấn (Nadeau & Bengio, 2003)."""
    d = np.asarray(d, float); k = len(d); var = d.var(ddof=1)
    if var == 0: return float('nan'), float('nan')
    t = d.mean() / np.sqrt((1 / k + test_over_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))

def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0.
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i]); adj[i] = min(1., running)
    return adj.tolist()

def best_per_family(cv_all):
    summ = cv_all.groupby(['model', 'config_id']).AP.mean().reset_index().sort_values(['AP', 'config_id'], ascending=[False, True])
    return summ.drop_duplicates('model').set_index('model').config_id.to_dict()

def run_stats():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(ROOT / 'results/cv_results.csv'); ext = pd.read_csv(OUT / 'cv_results_extension.csv')
    cv_all = pd.concat([base[base.model != 'Dummy'], ext[ext.columns.intersection(base.columns)]], ignore_index=True)
    best = best_per_family(cv_all)
    winner = load_json(ROOT / 'results/selection.json')['config_id']
    assert best['LinearSVM'] == winner
    ap = {m: cv_all[cv_all.config_id == c].sort_values('fold').AP.to_numpy() for m, c in best.items()}
    assert all(len(v) == 5 for v in ap.values())
    rows = []; others = [m for m in ap if m != 'LinearSVM']
    for m in others:
        d = ap['LinearSVM'] - ap[m]; tn, pn = nadeau_bengio(d, 1 / 4); tp, pp = stats.ttest_rel(ap['LinearSVM'], ap[m])
        rows.append({'comparison': f'LinearSVM vs {m}', 'config_other': best[m], 'AP_svm_mean': ap['LinearSVM'].mean(), 'AP_other_mean': ap[m].mean(),
                     'mean_diff': d.mean(), 'sd_diff': d.std(ddof=1), 'folds_svm_better': int((d > 0).sum()),
                     't_plain': float(tp), 'p_plain': float(pp), 't_nadeau_bengio': tn, 'p_nadeau_bengio': pn,
                     'p_wilcoxon_exact': float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else float('nan')})
    adj = holm([r['p_nadeau_bengio'] for r in rows])
    for r, a in zip(rows, adj): r['p_nadeau_bengio_holm'] = a
    fr = stats.friedmanchisquare(*[ap[m] for m in ap])
    ranks = pd.DataFrame(ap).rank(axis=1, ascending=False).mean().to_dict()
    result = {'families_best_config': best, 'ap_per_fold': {m: v.tolist() for m, v in ap.items()}, 'pairwise_vs_selected': rows,
              'friedman': {'statistic': float(fr.statistic), 'p': float(fr.pvalue), 'mean_rank': ranks, 'n_folds': 5, 'n_models': len(ap)},
              'inference_status': 'exploratory_only; dependent folds and post-selection; different feature representations',
              'notes': ['Nadeau-Bengio: phương sai nhân (1/k + n_test/n_train) với k=5, n_test/n_train=1/4, df=4.',
                        'Wilcoxon exact với 5 cặp có p nhỏ nhất 0,0625, không thể đạt 0,05.',
                        'Chọn cấu hình trên cùng fold gây thiên lệch chọn lựa; mọi kiểm định chỉ có giá trị thăm dò.',
                        'Friedman trên fold không phải thiết kế nhiều dataset độc lập; không dùng p-value khẳng định khác biệt hoặc tương đương.']}
    dump(OUT / 'stats_cv.json', result)
    pd.DataFrame(rows).to_csv(OUT / 'stats_cv_pairwise.csv', index=False, encoding='utf-8-sig')
    return best

def group_bootstrap_ap_diff(y, s_a, s_b, groups, n=BOOT):
    codes, uniq = pd.factorize(groups); rng = np.random.default_rng(SEED); diffs = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
        if len(np.unique(y[w > 0])) < 2: continue
        diffs.append(average_precision_score(y, s_a, sample_weight=w) - average_precision_score(y, s_b, sample_weight=w))
    d = np.array(diffs)
    return {'valid': len(d), 'mean': float(d.mean()), 'ci_low': float(np.percentile(d, 2.5)), 'ci_high': float(np.percentile(d, 97.5)),
            'share_svm_better': float((d > 0).mean())}

def run_validation_compare(best):
    parts, groups, _ = load_train_validation()
    X = parts['train'].drop(columns=['fraudulent', 'job_id']); y = parts['train'].fraudulent
    Xe = parts['validation'].drop(columns=['fraudulent', 'job_id']); ye = parts['validation'].fraudulent.to_numpy()
    cfg = {c['config_id']: c for c in configurations() + extension_configurations()}
    views = feature_views(X, y, Xe); out = {}; scores_by = {}
    for fam, cid in best.items():
        c = cfg[cid]; Xt, Xv = views[view_for(c['model'])]; m = make_estimator(c['model'], c['params'])
        # Đo riêng classifier.fit; không gộp dự đoán/tính AP vào thời gian fit.
        t = time.perf_counter(); m.fit(Xt, y); fit_seconds = time.perf_counter() - t
        t = time.perf_counter(); s = scores(m, Xv); predict_seconds = time.perf_counter() - t
        scores_by[fam] = s
        out[fam] = {'config_id': cid, 'AP_validation': float(average_precision_score(ye, s)),
                    'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds}
        print(fam, cid, out[fam], flush=True)
    locked = load_json(ROOT / 'results/validation_metrics.json')['AP']
    consistency = {'locked_validation_AP': locked, 'refit_validation_AP': out['LinearSVM']['AP_validation'],
                   'abs_diff': abs(locked - out['LinearSVM']['AP_validation']),
                   'environment_scikit_learn': md.version('scikit-learn')}
    paired = {m: group_bootstrap_ap_diff(ye, scores_by['LinearSVM'], scores_by[m], groups['validation']) for m in best if m != 'LinearSVM'}
    dump(OUT / 'validation_compare.json', {'models': out, 'consistency_check': consistency, 'paired_group_bootstrap_AP_svm_minus_other': paired,
         'timing_schema': 'v2: fit_seconds = classifier.fit only; predict_seconds = score only; preprocessing and AP calculation excluded',
         'note': 'Phân tích thăm dò giữa các pipeline. Chỉ dùng train (fit) và validation (chấm); không dùng test để fit, chọn hoặc chấm. Validation đã dùng chọn ngưỡng nên không phải holdout xác nhận.'})

def write_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT / 'protocol_extension.json', {
        'status': 'supplementary, created after the model/threshold lock and after test was opened',
        'does_not_modify': ['results/selection.json', 'results/protocol.json', 'models/lock.json', 'results/test_metrics.json'],
        'test_policy': 'Reads the common CSV for schema/deduplication, materializes only train/validation; never fits, selects or scores on test rows',
        'cv': 'same 5 folds as data/cv_manifest.csv, preprocessing fit inside each fold',
        'metric': 'Average Precision (same as main protocol)',
        'added_families': {'NaiveBayes': '8 configs: {MultinomialNB, ComplementNB} x alpha {0.01,0.1,0.5,1.0}; text TF-IDF only (non-negative input)',
                           'HistGradientBoosting': '8 configs: learning_rate {0.05,0.1} x max_leaf_nodes {15,31} x class_weight {None,balanced}; max_iter=200, l2=1.0; SelectKBest(f_classif,k=2000) fit in-fold on the combined features'},
        'statistics': ['paired t-test and Nadeau-Bengio corrected t-test on fold AP (Holm over 4 comparisons)', 'exact Wilcoxon (informational)',
                       'Friedman test across families', 'paired content_group bootstrap of validation AP difference (1000 reps)'],
        'seed': SEED, 'sklearn_version_used': md.version('scikit-learn'), 'python': platform.python_version(),
        'caveat': 'Exploratory pipeline comparison: overlapping training folds, hyperparameter selection on the same folds, and differing feature representations; p-values are not confirmatory.',
        'timing': 'CV fit_seconds excludes preprocessing and prediction; validation v2 separates fit and prediction. RF uses n_jobs=2; native BLAS/OpenMP limited to 1.',
        'platform': platform.platform()})

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['cv', 'stats', 'validation', 'all'], default='all'); a = ap.parse_args()
    write_protocol()
    with threadpool_limits(limits=1):
        if a.stage in ['cv', 'all']: run_cv()
        if a.stage in ['stats', 'validation', 'all']: best = run_stats()
        if a.stage in ['validation', 'all']: run_validation_compare(best)

if __name__ == '__main__': main()
