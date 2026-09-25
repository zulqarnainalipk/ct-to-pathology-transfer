"""Regenerate the data figures of the paper from the saved run outputs in results/.

Usage:  python scripts/make_figures.py      (writes PDF + PNG to docs/figures/)

The computations mirror main.py (paper_extras, retrieval_diagnostics, ssl_report,
reliability_figure and the aggregated condition summary), so every plotted value is
identical to the corresponding table in results/aggregate/tables/. Fonts: Latin
Modern Roman (GUST Font License), bundled in scripts/fonts/."""
import os, json, ast, warnings
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')
import logging; logging.getLogger('fontTools').setLevel(logging.ERROR)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', 'results')
AGG = os.path.join(ROOT, 'aggregate')
OUT = os.path.join(HERE, '..', 'docs', 'figures')
CHECK = {}

LM = os.path.join(HERE, 'fonts') + os.sep
for f in ('lmroman10-regular.ttf', 'lmroman10-bold.ttf', 'lmroman10-italic.ttf'):
    fm.fontManager.addfont(LM + f)
FS = 8
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Latin Modern Roman'], 'mathtext.fontset': 'cm',
    'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS, 'xtick.labelsize': FS, 'ytick.labelsize': FS,
    'legend.fontsize': FS, 'pdf.fonttype': 42, 'axes.unicode_minus': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 2,
    'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.handlelength': 1.6, 'legend.borderaxespad': 0.3,
    'savefig.dpi': 300, 'figure.dpi': 150, 'lines.linewidth': 1.2,
})
CM = 1 / 2.54
COL, TXT = 8.4 * CM, 17.2 * CM

# ---- one palette for the whole paper (Okabe-Ito, colour-blind safe) ----
FULL = '#0072B2'      # full model (CT context)
PATH = '#E69F00'      # pathology-only
STATS = '#4d4d4d'     # channel statistics + LR
CTC = '#8c8c8c'       # CT bank
TUM = '#D55E00'       # tumour core
STR = '#56B4E9'       # stroma
SIG = '#D55E00'       # interval excludes zero
NEUT = '#333333'


def panel(ax, s, x=-0.02, y=1.02, ha='right'):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va='bottom', fontsize=FS, fontweight='bold')


def save(fig, stem, png=True):
    fig.savefig(os.path.join(OUT, stem + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    if png:
        fig.savefig(os.path.join(OUT, stem + '.png'), bbox_inches='tight', pad_inches=0.02, dpi=300)
    plt.close(fig)


def fast_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float('nan')


def ece(y, p, n_bins=10):          # same as expected_calibration_error in main.py
    pred = (p >= 0.5).astype(int); conf = np.where(pred == 1, p, 1 - p); correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def load(cond):
    out = {}
    d = os.path.join(ROOT, 'runs', cond)
    for s in sorted(os.listdir(d)):
        seed = int(s.replace('seed', ''))
        folds = {}
        for f in sorted(os.listdir(os.path.join(d, s))):
            r = json.load(open(os.path.join(d, s, f, 'result.json')))
            r['_dir'] = os.path.join(d, s, f)
            folds[r['fold']] = r
        out[seed] = folds
    return out


def oof(cond):
    return {s: pd.DataFrame([rec for f in sorted(fo) for rec in fo[f]['records']]) for s, fo in load(cond).items()}


# =================================================================== forest
def forest():
    cs = pd.read_csv(os.path.join(AGG, 'tables', 'condition_summary.csv')).set_index('condition')
    groups = [('Comparators', [('path_only', 'Pathology-only (no CT)'), ('trivial_channel_stats', 'Channel statistics + LR')]),
              ('CT-bank controls', [('bank_random_init', 'Untrained CT encoder'), ('bank_gaussian', 'Gaussian-noise bank'),
                                    ('bank_dim_shuffle', 'Dimension-shuffled bank'), ('bank_mean', 'Constant (mean) bank'),
                                    ('retrieval_random', 'Random retrieval')]),
              ('Circularity / forced use', [('ctr_random_pos', 'InfoNCE, random positives'),
                                            ('abl_free_gate', 'No gate floor / usage penalty')]),
              ('Component ablations', [('abl_no_cdam', '\u2212Fusion (CDAM, gate)'), ('abl_no_adv', '\u2212Adversarial'),
                                       ('abl_no_ctr', '\u2212InfoNCE'), ('abl_no_proto', '\u2212Prototypes')]),
              ('Support size', [('main_k8', 'Full model, $K=8$')]),
              ('Channel drop (seed 0)', [(f'drop_{c}', f'\u2212{c}') for c in ('CD3', 'CD8', 'FoxP3', 'PanCK', 'DAPI')])]
    rows, y, ypos, heads = [], 0, [], []
    for g, items in groups:
        heads.append((y, g)); y += 1
        for c, lab in items:
            rows.append((c, lab, y)); y += 1
        y += 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14.35 * CM, 11.0 * CM), sharey=True, gridspec_kw={'wspace': 0.08})
    for ax, key, scale, xl, pl in ((axes[0], 'bal_acc', 100, '$\\Delta$ balanced accuracy (pp)', '(a)'),
                                   (axes[1], 'auc', 1, '$\\Delta$ AUC', '(b)')):
        for c, lab, yy in rows:
            d = cs.loc[c, f'delta_{key}'] * scale
            lo, hi = [v * scale for v in ast.literal_eval(cs.loc[c, f'delta_{key}_ci'])]
            sig = lo > 0 or hi < 0
            col = SIG if sig else NEUT
            ax.hlines(yy, lo, hi, color=col, lw=1.3 if sig else 1.0)
            ax.plot(d, yy, 'o', ms=4.2, color=col, mec='white', mew=0.5, zorder=3)
            CHECK[f'forest {c} {key}'] = (round(d, 4), round(lo, 4), round(hi, 4), sig)
        ax.axvline(0, color='#666666', lw=0.7, ls='--', zorder=1)
        ax.set_xlabel(xl); ax.grid(axis='y', visible=False)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        panel(ax, pl, x=0.0, y=1.005, ha='left')
    for i, (yy, g) in enumerate(heads):
        for ax in axes:
            if i % 2 == 0:
                nxt = heads[i + 1][0] - 0.35 if i + 1 < len(heads) else y - 0.35
                ax.axhspan(yy - 0.5, nxt - 0.5, color='#f4f4f4', zorder=0, lw=0)
    ticks = [yy for _, _, yy in rows] + [yy for yy, _ in heads]
    labs = [lab for _, lab, _ in rows] + [g for _, g in heads]
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(labs)
    for t, (yy) in zip(axes[0].get_yticklabels(), ticks):
        if yy in [h for h, _ in heads]:
            t.set_fontweight('bold')
    axes[0].set_ylim(y - 0.6, -0.8)
    axes[1].set_xlim(-0.30, 0.06)
    save(fig, 'forest_controls')


# =================================================================== SSL convergence
def ssl():
    d = os.path.join(ROOT, 'ssl')
    style = {  # (label, colour, linestyle, marker)
        'tau0.2_seed0_max50': ('$\\tau=0.2$, seed 0, 50 ep (main)', FULL, '-', 'o'),
        'tau0.2_seed1_max50': ('$\\tau=0.2$, seed 1, 50 ep (main)', FULL, '--', 's'),
        'tau0.2_seed2_max50': ('$\\tau=0.2$, seed 2, 50 ep (main)', FULL, ':', '^'),
        'tau0.2_seed0_max100': ('$\\tau=0.2$, seed 0, 100 ep (sweep)', '#000000', '-', 'D'),
        'tau0.05_seed0_max50': ('$\\tau=0.05$, seed 0, 50 ep', '#009E73', '-', 'v'),
        'tau0.1_seed0_max50': ('$\\tau=0.1$, seed 0, 50 ep', '#CC79A7', '-', 'P'),
        'tau0.5_seed0_max50': ('$\\tau=0.5$, seed 0, 50 ep', TUM, '-', 'X'),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18.6 * CM, 5.6 * CM), gridspec_kw={'wspace': 0.32})
    handles = []
    for sub, (lab, col, ls, mk) in style.items():
        h = json.load(open(os.path.join(d, sub, 'history.json')))
        H = pd.DataFrame(h['history'])
        assert h['tau'] == float(sub.split('_')[0][3:]) and h['max_epochs'] == int(sub.split('max')[1])
        L = H.dropna(subset=['loss'])
        axes[0].plot(L['epoch'], L['loss'], color=col, ls=ls, lw=0.9)
        Hm = H.dropna(subset=['view_top1'])
        kw = dict(color=col, ls=ls, lw=0.9, marker=mk, ms=2.8, mew=0)
        axes[1].plot(Hm['epoch'], Hm['view_top1'], **kw)
        axes[2].plot(Hm['epoch'], Hm['effective_rank'], **kw)
        handles.append(Line2D([], [], label=lab, **kw))
        for e in (0, 10, 25, 50, 100):
            r = Hm[Hm.epoch == e]
            if len(r):
                CHECK[f'ssl {sub} ep{e}'] = (None if pd.isna(r['loss'].iloc[0]) else round(float(r['loss'].iloc[0]), 3),
                                             round(float(r['view_top1'].iloc[0]), 3), round(float(r['effective_rank'].iloc[0]), 1))
    chance = 1 / 61
    axes[1].axhline(chance, color='#666666', lw=0.7, ls='--')
    axes[1].text(100, chance + 0.02, 'chance (1/61)', ha='right', va='bottom', fontsize=FS, color='#444444')
    axes[1].set_ylim(0, 1.04)
    axes[0].set_ylabel('NT-Xent loss'); axes[1].set_ylabel('View-retrieval top-1 accuracy'); axes[2].set_ylabel('Effective rank of bank')
    for a, p in zip(axes, ('(a)', '(b)', '(c)')):
        a.set_xlabel('Epoch'); a.set_xlim(-3, 103); a.set_xticks([0, 25, 50, 75, 100])
        panel(a, p, x=-0.02, y=1.02, ha='right')
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=3, columnspacing=1.2,
               handlelength=2.2)
    save(fig, 'ssl_convergence')


# =================================================================== t-SNE
def tsne():
    from sklearn.manifold import TSNE
    folds = load('main')[0]
    feats = [np.load(os.path.join(folds[f]['_dir'], 'features.npz')) for f in sorted(folds)]
    bank = torch.load(os.path.join(ROOT, 'banks', 'main_seed0.pt')).numpy()
    y = np.concatenate([z['y'] for z in feats])
    fig, axes = plt.subplots(1, 2, figsize=(9.42 * CM, 4.9 * CM), gridspec_kw={'wspace': 0.06})
    for ax, key, title, pl in ((axes[0], 'path', 'Before fusion', '(a)'), (axes[1], 'aligned', 'After gated fusion', '(b)')):
        X = np.concatenate([bank, np.concatenate([z[key] for z in feats])])
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        E = TSNE(n_components=2, perplexity=min(30, len(X) // 4), init='pca', random_state=0).fit_transform(X)
        nb = len(bank)
        ax.scatter(E[:nb, 0], E[:nb, 1], s=9, marker='^', c=CTC, lw=0, label='CT bank (61 volumes)')
        ax.scatter(E[nb:][y == 1, 0], E[nb:][y == 1, 1], s=7, c=TUM, lw=0, label='Tumour core')
        ax.scatter(E[nb:][y == 0, 0], E[nb:][y == 0, 1], s=7, marker='s', c=STR, lw=0, label='Stroma')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color('#bbbbbb'); sp.set_linewidth(0.5)
        ax.text(0.0, 1.015, pl, transform=ax.transAxes, ha='left', va='bottom', fontweight='bold')
        ax.text(0.15, 1.015, title, transform=ax.transAxes, ha='left', va='bottom')
    CHECK['tsne n bank / tiles / tumour'] = (len(bank), len(y), int(y.sum()))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, columnspacing=0.8, handletextpad=0.1,
               markerscale=1.5)
    save(fig, 'tsne_main')


# =================================================================== bank usage + attention/gate
def usage():
    folds = load('main')
    recs0 = [rec for f in sorted(folds[0]) for rec in folds[0][f]['records']]
    topk = np.array([r['topk'] for r in recs0])
    n_bank = len(torch.load(os.path.join(ROOT, 'banks', 'main_seed0.pt')))
    counts = np.bincount(topk.ravel(), minlength=n_bank)
    pr = counts / counts.sum()
    CHECK['usage entries ever retrieved'] = int((counts > 0).sum())
    CHECK['usage entropy norm'] = float(-(pr[pr > 0] * np.log(pr[pr > 0])).sum() / np.log(n_bank))
    CHECK['usage top5 hub share'] = float(np.sort(pr)[::-1][:5].sum())
    CHECK['usage total / max'] = (int(counts.sum()), int(counts.max()))
    fig, ax = plt.subplots(figsize=(9.06 * CM, 3.9 * CM))
    ax.bar(range(n_bank), counts, width=0.8, color=FULL, lw=0)
    ax.axhline(counts.sum() / n_bank, color='#444444', ls='--', lw=0.8)
    ax.legend([Line2D([], [], color='#444444', ls='--', lw=0.8)], [f'uniform usage ({counts.sum() / n_bank:.1f})'],
              loc='upper left', bbox_to_anchor=(0.46, 1.02), handlelength=2.0)
    ax.set_xlim(-1, n_bank); ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xlabel('CT bank entry (volume index)'); ax.set_ylabel('Times retrieved'); ax.grid(axis='x', visible=False)
    panel(ax, '(a)', x=-0.10, y=1.0, ha='right')
    save(fig, 'bank_usage_main')

    recs = [rec for s in folds for f in folds[s] for rec in folds[s][f]['records'] if 'attn' in rec]
    a = np.clip(np.array([r['attn'] for r in recs]), 1e-12, 1)
    ent = -(a * np.log(a)).sum(1) / np.log(a.shape[1])
    gate = np.array([r['gate_mean'] for r in recs])
    CHECK['attn n / mean ent / mean gate'] = (len(recs), float(ent.mean()), float(gate.mean()), float(gate.min()), float(gate.max()))
    fig, axes = plt.subplots(1, 2, figsize=(9.42 * CM, 3.9 * CM), gridspec_kw={'wspace': 0.3})
    axes[0].hist(ent, bins=30, color=FULL, lw=0.3, ec='white')
    axes[0].set_xlabel('Attention entropy / $\\log k$\n(1 = uniform)')
    axes[1].hist(gate, bins=30, color=FULL, lw=0.3, ec='white')
    axes[1].set_xlabel('Mean transfer-gate\nvalue')
    axes[0].set_ylabel('Tiles')
    for ax, p in zip(axes, ('(b)', '(c)')):
        ax.grid(axis='x', visible=False)
        panel(ax, p, x=-0.02, y=1.0, ha='right')
    axes[0].set_xlim(0, 1.02)
    save(fig, 'attention_gate_hist')


# =================================================================== ROC
def roc():
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(4.13 * CM, 4.13 * CM))
    for c, lab, col, ls in (('main', 'Full', FULL, '-'), ('path_only', 'Path.-only', PATH, '-'),
                            ('trivial_channel_stats', 'Ch. stats', STATS, (0, (3, 1.2)))):
        d = pd.concat(oof(c).values())
        fpr, tpr, _ = roc_curve(d['y'], d['prob'])
        auc = fast_auc(d['y'].values, d['prob'].values)
        CHECK[f'roc pooled AUC {c} (n={len(d)})'] = round(auc, 4)
        ax.plot(fpr, tpr, color=col, lw=1.2, ls=ls, label=f'{lab} {auc:.3f}')
    ax.plot([0, 1], [0, 1], ls=':', color='#888888', lw=0.7)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect('equal')
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_xticklabels(['0', '0.5', '1']); ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate')
    ax.legend(loc='lower right', title='AUC', frameon=True, facecolor='white', edgecolor='none', framealpha=1, handlelength=1.3, handletextpad=0.4, borderaxespad=0.1, labelspacing=0.25,
              title_fontsize=FS, alignment='right')
    save(fig, 'roc_pooled')


# =================================================================== reliability
def reliability():
    d = pd.concat(oof('main').values())
    y, p = d['y'].values, d['prob'].values
    pred = (p >= 0.5).astype(int); conf = np.where(pred == 1, p, 1 - p); corr = (pred == y).astype(float)
    bins = np.linspace(0.5, 1, 11)
    xs, ys, ns, ctr = [], [], [], []
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)  # same bins as ECE
        if m.any():
            xs.append(conf[m].mean()); ys.append(corr[m].mean()); ns.append(int(m.sum())); ctr.append((lo + hi) / 2)
    e = ece(y, p)
    CHECK['reliability n tiles'] = len(y); CHECK['reliability bins (conf,acc,n)'] = [(round(a, 3), round(b, 3), n) for a, b, n in zip(xs, ys, ns)]
    CHECK['reliability pooled ECE'] = round(e, 4)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(4.04 * CM, 4.5 * CM), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1.25], 'hspace': 0.12})
    ax.plot([0.5, 1], [0.5, 1], ls=':', color='#888888', lw=0.7)
    sz = 6 + 40 * np.sqrt(np.array(ns) / max(ns))
    ax.scatter(xs, ys, s=sz, color=FULL, ec='white', lw=0.4, zorder=3)
    ax.set_ylim(-0.05, 1.08); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_ylabel('Accuracy')
    ax.text(1.0, 1.03, f'ECE = {e:.3f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=FS)
    bx.bar(ctr, ns, width=0.042, color='#9ecae1', lw=0)
    bx.set_yscale('log'); bx.set_ylim(0.6, 2000); bx.set_yticks([1, 10, 100, 1000]); bx.set_yticklabels(['1', '', '100', ''])
    bx.minorticks_off()
    for c_, n in zip(ctr, ns):
        pass
    bx.set_ylabel('Tiles')
    bx.set_xlim(0.48, 1.02); bx.set_xticks([0.5, 0.75, 1]); bx.set_xticklabels(['0.5', '0.75', '1'])
    bx.set_xlabel('Confidence'); bx.grid(axis='x', visible=False)
    fig.align_ylabels([ax, bx])
    save(fig, 'reliability_main')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['forest', 'ssl', 'tsne', 'usage', 'roc', 'reliability']
    for w in which:
        globals()[w]()
    for k, v in CHECK.items():
        print(k, '=>', v)
