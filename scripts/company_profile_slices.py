"""Chẩn đoán sau khóa: chia kết quả theo company_profile đã/chưa xuất hiện trong train.

Chỉ đọc dữ liệu, manifest và dự đoán đã lưu. KHÔNG fit lại, KHÔNG đổi mô hình/ngưỡng.
Chạy: python scripts/company_profile_slices.py
Ra: results/company_profile_slices.csv, results/company_profile_slices.json
"""
from pathlib import Path
import json, re
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED, N_BOOT = 42, 1000

def norm(x):
    s = re.sub(r"\s+", " ", str(x)).strip().lower() if pd.notna(x) else ""
    return s or None

def counts(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    tp = float(w[(y == 1) & (p == 1)].sum()); fp = float(w[(y == 0) & (p == 1)].sum()); fn = float(w[(y == 1) & (p == 0)].sum())
    return tp, fp, fn

def prf(tp, fp, fn):
    return (tp / (tp + fn) if tp + fn else np.nan, tp / (tp + fp) if tp + fp else np.nan)

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

def build(split_name, pred_file, raw, manifest, train_profiles, train_fake_profiles):
    pred = pd.read_csv(ROOT / "results" / pred_file)
    data = raw.merge(manifest[["job_id", "split", "content_group"]], on="job_id")
    data = data[data.split == split_name][["job_id", "profile"]]
    d = pred.merge(data, on="job_id", how="left")
    if "content_group" not in d:
        d = d.merge(manifest[["job_id", "content_group"]], on="job_id")
    d["slice"] = np.where(d.profile.isna(), "khong_co_profile",
                 np.where(d.profile.isin(train_profiles), "profile_da_thay_trong_train", "profile_chua_thay"))
    d["lookup_pred"] = d.profile.isin(train_fake_profiles).astype(int)  # luật tra cứu: profile từng gắn tin giả trong train
    return d

def summarize(d, split_name):
    rng = np.random.default_rng(SEED); rows = []
    groups = pd.factorize(d.content_group)[0]; ng = groups.max() + 1
    boots = [np.bincount(rng.integers(0, ng, ng), minlength=ng)[groups] for _ in range(N_BOOT)]
    seen_mask = d.slice == "profile_da_thay_trong_train"
    for name, m in [("tat_ca", np.ones(len(d), bool)), ("profile_da_thay_trong_train", seen_mask),
                    ("khong_co_hoac_chua_thay", ~seen_mask), ("khong_co_profile", d.slice == "khong_co_profile"),
                    ("profile_chua_thay", d.slice == "profile_chua_thay")]:
        g = d[m]; y = g.fraudulent.to_numpy(); p = g.predicted.to_numpy()
        tp, fp, fn = counts(y, p); r, pr = prf(tp, fp, fn)
        rs, ps = [], []
        for w in boots:
            ww = w[np.asarray(m)].astype(float)
            if ((y == 1) * ww).sum() == 0: continue
            t_, f_, n_ = counts(y, p, ww); a, b = prf(t_, f_, n_); rs.append(a); ps.append(b)
        ltp, lfp, lfn = counts(y, g.lookup_pred.to_numpy()); lr, lp = prf(ltp, lfp, lfn)
        lo, hi = wilson(tp, tp + fn)
        rows.append(dict(split=split_name, slice=name, n=len(g), fake=int(y.sum()), fake_rate=y.mean(),
            TP=int(tp), FP=int(fp), FN=int(fn), recall=r, recall_wilson_low=lo, recall_wilson_high=hi,
            recall_boot_low=np.nanpercentile(rs, 2.5), recall_boot_high=np.nanpercentile(rs, 97.5),
            precision=pr, precision_boot_low=np.nanpercentile(ps, 2.5), precision_boot_high=np.nanpercentile(ps, 97.5),
            lookup_TP=int(ltp), lookup_FP=int(lfp), lookup_FN=int(lfn)))
    return rows

def main():
    raw = pd.read_csv(ROOT / "data/fake_job_postings.csv")
    manifest = pd.read_csv(ROOT / "data/split_manifest.csv")
    raw["profile"] = raw.company_profile.map(norm)
    tr = raw.merge(manifest[["job_id", "split"]], on="job_id"); tr = tr[tr.split == "train"]
    train_profiles = set(tr.profile.dropna()); train_fake_profiles = set(tr[tr.fraudulent == 1].profile.dropna())
    out = []
    for split_name, f in [("test", "test_predictions.csv"), ("validation", "validation_predictions.csv")]:
        out += summarize(build(split_name, f, raw, manifest, train_profiles, train_fake_profiles), split_name)
    df = pd.DataFrame(out); df.to_csv(ROOT / "results/company_profile_slices.csv", index=False, encoding="utf-8-sig")
    (ROOT / "results/company_profile_slices.json").write_text(json.dumps(df.round(4).to_dict("records"), ensure_ascii=False, indent=1), encoding="utf-8")
    print(df.round(3).to_string())

if __name__ == "__main__":
    main()
