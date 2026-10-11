"""Post-hoc analyses computed from the saved run outputs (no training).

Usage:  python scripts/analysis.py      (a few minutes on a CPU)

Inputs : results/runs (groups A-C) and results/group_d/runs (group D)
Outputs: results/analysis/*.csv and analysis_summary.json

Analyses
  pairs        paired differences (A minus B) with a patient-level cluster bootstrap, for
               harder tasks, positive controls, annotation budgets, selection variants,
               channel drop and preprocessing without the DAPI mask
  budget       unique labelled training tiles per condition and fold
  swap         test-time bank swap: logit margins, fused-feature change, decision flips,
               agreement and patient-level outcomes
  null         label/top-1 retrieval mutual information for the full model and for
               replacement-bank runs (a null that keeps the class information of the
               pathology features), plus per-patient retrieval diagnostics
  jackknife    leave-one-patient-out sensitivity of the full-versus-pathology-only difference
"""
import os
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import mutual_info_score, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, '..', 'results')                # groups A-C
V3 = os.path.join(HERE, '..', 'results', 'group_d')     # group D
OUT = os.path.join(HERE, '..', 'results', 'analysis')
os.makedirs(OUT, exist_ok=True)
B = 2000
SUMMARY = {}


# ------------------------------------------------------------------ loading
def load(root):
    res = {}
    for p in glob.glob(os.path.join(root, 'runs', '*', 'seed*', 'fold*', 'result.json')):
        r = json.load(open(p))
        r['_dir'] = os.path.dirname(p)
        res[(r['condition'], r['seed'], r['fold'])] = r
    return res


R3, R2 = load(V3), load(V2)


def oof(res, cond):
    """{seed: DataFrame of pooled out-of-fold records}"""
    out = defaultdict(list)
    for (c, s, f), r in res.items():
        if c == cond:
            out[s] += r['records']
    return {s: pd.DataFrame(v) for s, v in out.items()}


# ------------------------------------------------------------------ metrics
def auc(y, p):
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bal(y, p):
    pred = p >= 0.5
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return np.nan
    return 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())


PATIENTS = [f'Case{i}' for i in range(1, 9)]
rng = np.random.default_rng(0)
DRAWS = [rng.choice(PATIENTS, size=8, replace=True) for _ in range(B)]


def per_patient_arrays(d):
    return {p: (g['y'].values.astype(int), g['prob'].values) for p, g in d.groupby('patient')}


def point(o, seeds):
    vals = np.array([[bal(o[s]['y'].values, o[s]['prob'].values), auc(o[s]['y'].values, o[s]['prob'].values)]
                     for s in seeds])
    return vals.mean(0), vals.std(0, ddof=1) if len(seeds) > 1 else np.full(2, np.nan)


def boot(o, seeds):
    arr = {s: per_patient_arrays(o[s]) for s in seeds}
    v = np.full((B, 2), np.nan)
    for b, dr in enumerate(DRAWS):
        accs, aucs = [], []
        for s in seeds:
            parts = [arr[s][p] for p in dr if p in arr[s]]
            y = np.concatenate([a for a, _ in parts]); pr = np.concatenate([q for _, q in parts])
            accs.append(bal(y, pr)); aucs.append(auc(y, pr))
        v[b] = [np.nanmean(accs), np.nanmean(aucs)]
    return v


def pair(resA, a, resB, b):
    oa, ob = oof(resA, a), oof(resB, b)
    seeds = sorted(set(oa) & set(ob))
    pa, sa = point(oa, sorted(oa)); pb, sb = point(ob, sorted(ob))
    dv = boot(oa, seeds) - boot(ob, seeds)
    d = point(oa, seeds)[0] - point(ob, seeds)[0]
    ci = np.nanpercentile(dv, [2.5, 97.5], axis=0)
    return dict(A=a, B=b, seeds=len(seeds), A_bal=pa[0], A_bal_sd=sa[0], A_auc=pa[1], A_auc_sd=sa[1],
                B_bal=pb[0], B_bal_sd=sb[0], B_auc=pb[1], B_auc_sd=sb[1],
                d_bal=d[0], d_bal_lo=ci[0, 0], d_bal_hi=ci[1, 0], d_auc=d[1], d_auc_lo=ci[0, 1], d_auc_hi=ci[1, 1])


def fmt_ms(m, s, pct):
    if pct:
        return f"{100 * m:.1f}" + ("" if np.isnan(s) else f" $\\pm$ {100 * s:.1f}")
    return f"{m:.3f}" + ("" if np.isnan(s) else f" $\\pm$ {s:.3f}")


def fmt_d(d, lo, hi, pct):
    if pct:
        return f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def save(df, name):
    df.to_csv(os.path.join(OUT, name + '.csv'), index=False)


# ------------------------------------------------------------------ 1. paired comparisons
PAIRS = [
    ('reference', 'ref_main', 'ref_path_only'),
    ('hard', 'hard_main', 'hard_path_only'),
    ('hard', 'hard_bank_gaussian', 'hard_path_only'),
    ('hard', 'hard_main', 'hard_bank_gaussian'),
    ('tm', 'tm_main', 'tm_path_only'),
    ('positive', 'pos_panck_context', 'hard_path_only'),
    ('positive', 'pos_panck_context', 'hard_main'),
    ('positive', 'pos_oracle_context', 'hard_path_only'),
    ('selection', 'selprev_main', 'selprev_path_only'),
    ('selection', 'sellast_main', 'sellast_path_only'),
    ('mask', 'nomask_main', 'nomask_path_only'),
    ('mask', 'nomask_main', 'ref_main'),
    ('mask', 'nomask_drop_DAPI', 'nomask_main'),
] + [('budget', f'budget{b}_main', f'budget{b}_path_only') for b in (2, 4, 8, 16)] \
  + [('channel', f'drop_{c}', 'ref_main') for c in ('CD3', 'CD8', 'FoxP3', 'PanCK', 'DAPI')]
rows = [dict(family=fam, **pair(R3, a, R3, b)) for fam, a, b in PAIRS]
pairs = pd.DataFrame(rows)
save(pairs, 'paired_differences')
SUMMARY['paired_differences'] = pairs.to_dict(orient='records')

# deterministic baselines (single run)
triv = []
for c in ('tm_trivial_channel_stats', 'nopanck_trivial_channel_stats'):
    o = oof(R3, c)[0]
    triv.append(dict(condition=c, bal=bal(o['y'].values, o['prob'].values), auc=auc(o['y'].values, o['prob'].values)))
save(pd.DataFrame(triv), 'baselines_group_d')
SUMMARY['baselines'] = triv

# ------------------------------------------------------------------ 2. unique labelled tiles
lab = []
for c in ['ref_main'] + [f'budget{b}_main' for b in (2, 4, 8, 16)]:
    n = [r['unique_labelled_train_tiles'] for (cc, s, f), r in R3.items() if cc == c]
    pool = [r['sizes']['train_pool'] for (cc, s, f), r in R3.items() if cc == c]
    lab.append(dict(condition=c, unique_min=min(n), unique_max=max(n), unique_mean=float(np.mean(n)),
                    pool_min=min(pool), pool_max=max(pool), val_min=min(r['sizes']['val'] for (cc, s, f), r in R3.items() if cc == c),
                    val_max=max(r['sizes']['val'] for (cc, s, f), r in R3.items() if cc == c)))
save(pd.DataFrame(lab), 'labelled_tiles')
SUMMARY['labelled_tiles'] = lab

# ------------------------------------------------------------------ 3. bank-swap diagnostics
swap_rows, pat_rows = [], []
for (c, s, f), r in sorted(R3.items()):
    if c != 'ref_main':
        continue
    z = np.load(os.path.join(r['_dir'], 'features.npz'))
    a0 = torch.tensor(z['aligned'])
    y = np.array([q['y'] for q in r['records']])
    p0 = np.array([q['prob'] for q in r['records']])
    mc0 = np.array([q['margin_cls'] for q in r['records']]); mg0 = np.array([q['margin_glob'] for q in r['records']])
    # scale reference: distance between fused features of different tiles of the same held-out patient
    dist = torch.cdist(a0, a0).numpy(); iu = np.triu_indices(len(a0), 1)
    tile_scale = float(np.median(dist[iu]))
    for alt, d in r['bank_swap_diagnostics'].items():
        p1 = np.array(d['prob']); mc1 = np.array(d['margin_cls']); mg1 = np.array(d['margin_glob'])
        swap_rows.append(dict(seed=s, fold=f, bank=alt, n=len(y),
                              abs_dprob=np.abs(p1 - p0).mean(),
                              abs_dmargin_total=np.abs((mc1 + mg1) - (mc0 + mg0)).mean(),
                              rel_dmargin_total=np.abs((mc1 + mg1) - (mc0 + mg0)).mean() / np.abs(mc0 + mg0).mean(),
                              abs_dmargin_cls=np.abs(mc1 - mc0).mean(), abs_dmargin_glob=np.abs(mg1 - mg0).mean(),
                              fused_cos=np.mean(d['aligned_cos']), fused_l2=np.mean(d['aligned_l2']),
                              fused_l2_rel=np.mean(d['aligned_l2']) / tile_scale,
                              flips=int(np.sum(d['decision_flip'])),
                              bal0=bal(y, p0), bal1=bal(y, p1)))
sw = pd.DataFrame(swap_rows)
save(sw, 'bank_swap_extended_per_fold')
g = sw.groupby('bank')
swap_tab = pd.DataFrame({
    'bank': g.size().index,
    'abs_dprob': g['abs_dprob'].mean().values,
    'abs_dmargin_total': g['abs_dmargin_total'].mean().values,
    'rel_dmargin_total': g['rel_dmargin_total'].mean().values,
    'fused_cos_mean': g['fused_cos'].mean().values,
    'fused_cos_min': g['fused_cos'].min().values,
    'fused_l2_rel': g['fused_l2_rel'].mean().values,
    'flips': g['flips'].sum().values,
    'tiles': g['n'].sum().values,
    'patients_changed': g.apply(lambda d: int((np.abs(d['bal1'] - d['bal0']) > 1e-9).sum())).values,
    'patient_folds': g.size().values,
    'max_patient_dbal': g.apply(lambda d: float(np.nanmax(np.abs(d['bal1'] - d['bal0'])))).values,
})
# agreement of decisions pooled over folds and seeds
kap = []
for alt in swap_tab['bank']:
    d0, d1 = [], []
    for (c, s, f), r in R3.items():
        if c == 'ref_main':
            d0 += [q['prob'] >= 0.5 for q in r['records']]
            d1 += [pp >= 0.5 for pp in r['bank_swap_diagnostics'][alt]['prob']]
    kap.append(cohen_kappa_score(d0, d1))
swap_tab['decision_kappa'] = kap
save(swap_tab, 'bank_swap_extended')
SUMMARY['bank_swap_extended'] = swap_tab.to_dict(orient='records')

# ------------------------------------------------------------------ 4. retrieval null and per-patient diagnostics
def mi_top1(res, cond):
    out = []
    for s, d in oof(res, cond).items():
        top1 = d['topk'].apply(lambda t: t[0]).values
        if (top1 < 0).all():
            continue
        out.append(mutual_info_score(d['y'].values, top1))
    return out


null = []
for root, res, cond, label in [
        ('groups A-C', R2, 'main', 'Full model (SSL bank)'),
        ('groups A-C', R2, 'bank_random_init', 'Untrained-encoder bank'),
        ('groups A-C', R2, 'bank_gaussian', 'Gaussian bank'),
        ('groups A-C', R2, 'bank_dim_shuffle', 'Dimension-shuffled bank'),
        ('groups A-C', R2, 'retrieval_random', 'Random retrieval'),
        ('groups A-C', R2, 'bank_mean', 'Constant bank'),
        ('group D', R3, 'ref_main', 'Full model (reference runs)')]:
    m = mi_top1(res, cond)
    null.append(dict(source=root, condition=cond, label=label, mi_mean=float(np.mean(m)), mi_sd=float(np.std(m, ddof=1)) if len(m) > 1 else np.nan, n_seeds=len(m)))
nl = pd.DataFrame(null)
save(nl, 'retrieval_null')
SUMMARY['retrieval_null'] = null

# per-patient: within-patient overlap of tumour and stroma retrieval sets, and LOPO context probe per patient
pp_rows = []
for s, d in oof(R2, 'main').items():
    for p, g in d.groupby('patient'):
        t = set(x for row in g[g.y == 1]['topk'] for x in row); st = set(x for row in g[g.y == 0]['topk'] for x in row)
        mi = mutual_info_score(g['y'].values, g['topk'].apply(lambda t: t[0]).values)
        pp_rows.append(dict(seed=s, patient=p, n=len(g), jaccard_T_S=len(t & st) / max(1, len(t | st)), within_patient_mi=mi))
ctx_rows = []
for s in (0, 1, 2):
    feats = [np.load(os.path.join(R2[('main', s, f)]['_dir'], 'features.npz')) for f in range(8)]
    X = np.concatenate([z['context'] for z in feats]); y = np.concatenate([z['y'] for z in feats])
    pats = np.concatenate([[R2[('main', s, f)]['test_patient']] * len(z['y']) for f, z in enumerate(feats)])
    for pt in PATIENTS:
        te = pats == pt
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)).fit(X[~te], y[~te])
        ctx_rows.append(dict(seed=s, patient=pt, probe_auc=auc(y[te], clf.predict_proba(X[te])[:, 1])))
ppd = pd.DataFrame(pp_rows).merge(pd.DataFrame(ctx_rows), on=['seed', 'patient'])
ppa = ppd.groupby('patient').agg(n=('n', 'first'), jaccard_T_S=('jaccard_T_S', 'mean'),
                                within_patient_mi=('within_patient_mi', 'mean'), probe_auc=('probe_auc', 'mean'),
                                probe_auc_sd=('probe_auc', 'std')).reset_index()
save(ppa, 'retrieval_per_patient')
SUMMARY['retrieval_per_patient'] = ppa.to_dict(orient='records')

# ------------------------------------------------------------------ 5. leave-one-patient-out jackknife
def jack(res, a, b, tag):
    oa, ob = oof(res, a), oof(res, b)
    seeds = sorted(set(oa) & set(ob))
    out = []
    for drop in [None] + PATIENTS:
        da, db, aa, ab = [], [], [], []
        for s in seeds:
            A = oa[s] if drop is None else oa[s][oa[s].patient != drop]
            Bd = ob[s] if drop is None else ob[s][ob[s].patient != drop]
            da.append(bal(A.y.values, A.prob.values)); db.append(bal(Bd.y.values, Bd.prob.values))
            aa.append(auc(A.y.values, A.prob.values)); ab.append(auc(Bd.y.values, Bd.prob.values))
        out.append(dict(set=tag, left_out=drop or 'none', d_bal=np.mean(da) - np.mean(db), d_auc=np.mean(aa) - np.mean(ab),
                        A_bal=np.mean(da), B_bal=np.mean(db)))
    return out


jk = pd.DataFrame(jack(R2, 'main', 'path_only', 'main analysis') + jack(R3, 'ref_main', 'ref_path_only', 'reference runs'))
save(jk, 'jackknife')
SUMMARY['jackknife'] = jk.to_dict(orient='records')

# per-patient difference full minus pathology-only (main analysis)
pdiff = []
for s in (0, 1, 2):
    a = oof(R2, 'main')[s]; b = oof(R2, 'path_only')[s]
    for p in PATIENTS:
        A = a[a.patient == p]; Bd = b[b.patient == p]
        pdiff.append(dict(seed=s, patient=p, d_bal=bal(A.y.values, A.prob.values) - bal(Bd.y.values, Bd.prob.values)))
pdf = pd.DataFrame(pdiff).groupby('patient')['d_bal'].agg(['mean', 'std']).reset_index()
save(pdf, 'per_patient_difference')
SUMMARY['per_patient_difference'] = pdf.to_dict(orient='records')

json.dump(SUMMARY, open(os.path.join(OUT, 'analysis_summary.json'), 'w'), indent=1, default=float)
print('written to', OUT)
pd.set_option('display.width', 220)
print(pairs[['family', 'A', 'B', 'A_bal', 'B_bal', 'd_bal', 'd_bal_lo', 'd_bal_hi', 'd_auc', 'd_auc_lo', 'd_auc_hi']].round(3).to_string())
print(swap_tab.round(3).to_string())
print(nl.round(3).to_string())
print(ppa.round(3).to_string())
print(jk.round(3).to_string())
print(pdf.round(3).to_string())
print(pd.DataFrame(lab).to_string())
