"""Unpaired cross-dimensional transfer from 3D CT to few-shot multiplex pathology.

Single-file experiment pipeline for the study "Does Unpaired 3D CT Context Help
Few-Shot Multiplex Pathology Classification? A Controlled Feasibility Study".

Pipeline
  Phase 1  contrastive (NT-Xent) pretraining of a 3D CT encoder on unlabelled
           CT volumes; the frozen encoder builds a bank of whole-volume descriptors.
  Phase 2  episodic training of the pathology model: top-k retrieval from the
           CT bank, cross-dimensional attention (CDAM), gated residual fusion,
           prototypical heads, a domain discriminator (GRL) and InfoNCE.
  Protocol leave-one-patient-out cross-validation (8 patients), three seeds,
           pooled out-of-fold predictions, patient-level cluster bootstrap.
  Controls pathology-only comparator, four replacement CT banks, random
           retrieval, test-time bank swap, InfoNCE with random positives,
           component ablations, SSL epoch/temperature sweeps, channel drop and a
           hand-crafted channel-statistics baseline.

Usage
  python main.py --list                                   # conditions and run counts per group
  python main.py --run smoke --ct_dir D1 --path_dir D2    # tiny CPU smoke test
  python main.py --run full  --ct_dir D1 --path_dir D2    # all groups, one worker per GPU
  python main.py --run A     --ct_dir D1 --path_dir D2    # one independent group (A, B or C)
  python main.py --aggregate RESULTS_DIR --agg_out RESULTS_DIR/aggregate

Run without arguments inside a notebook (``%run main.py``) to use the settings
below and get live progress bars. Every run writes
<output_dir>/runs/<condition>/seed<s>/fold<f>/result.json and is skipped if that
file already exists, so interrupted sessions resume and separate groups can be
merged afterwards with --aggregate. See README.md and docs/REPRODUCING.md.
"""
import os
import sys
import copy
import json
import time
import math
import argparse
import warnings
import platform
from dataclasses import dataclass, field, asdict, replace
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Function

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict

warnings.filterwarnings('ignore')

rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'], 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# =============================================================================
# USER SETTINGS  -- used when the file is run without arguments, e.g. in a
# notebook cell:   %run main.py     (live progress bars per GPU)
# =============================================================================
RUN = 'full'                        # 'full' = groups A+B+C | 'A' | 'B' | 'C' | 'smoke' (tiny CPU test)
OUTPUT_ROOT = './results_run'      # finished runs stored here are skipped when you run again
AGGREGATE_AFTER = True              # build tables/figures into OUTPUT_ROOT + '_aggregate' at the end
ZIP_AFTER = True                    # write OUTPUT_ROOT + '_<RUN>_outputs.zip' for download
WORKERS_PER_GPU = 2                 # parallel workers per GPU (~4 GB each) so the GPU never idles
# Data paths: ResearchConfig.ct_data_dir / pathology_data_dir below, or --ct_dir / --path_dir.


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ResearchConfig:
    # ---- paths (overridden by --ct_dir / --path_dir / --out); see docs/DATA.md ----
    ct_data_dir: str = './data/LungCT-Diagnosis/LungCT-Diagnosis'
    pathology_data_dir: str = './data/PKG - HNSCC-mIF-mIHC'
    output_dir: str = './results_run'

    # ---- target task ----
    label_mode: str = 'roi_t_vs_s'          # roi_t_vs_s | roi_tm_vs_s | legacy_mask
    legacy_mask_threshold: float = 0.05
    pathology_channels: Tuple[str, ...] = ('CD3', 'CD8', 'FoxP3', 'PanCK', 'DAPI')
    use_dapi_roi_mask: bool = True          # original label-blind DAPI ROI masking
    drop_channel: Optional[str] = None      # channel zeroed at train+test (R2.5)
    max_patients: Optional[int] = None      # smoke only
    max_tiles_per_class_per_patient: Optional[int] = None  # smoke only

    # ---- CT preprocessing ----
    ct_volume_depth: int = 16
    ct_slice_size: int = 128
    ct_hu_window: Tuple[float, float] = (-1000.0, 400.0)
    ct_max_volumes: Optional[int] = None    # smoke only
    ct_in_channels: int = 1
    path_image_size: int = 224
    path_in_channels: int = 5

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 0
    batch_size: int = 4

    # ---- Phase 1: CT SSL ----
    ssl_epochs: int = 50
    ssl_temperature: float = 0.2
    ssl_max_epochs: int = 50                # length of the SSL run the snapshot comes from
    ssl_lr: float = 1e-4
    ssl_batch_size: int = 8
    ssl_augmentation_strength: float = 0.5
    ssl_eval_interval: int = 5

    # ---- Phase 2 ----
    ct_bank: str = 'ssl'                    # ssl | random_init | gaussian | dim_shuffle | mean
    retrieval_mode: str = 'topk'            # topk | random
    ctr_positive: str = 'retrieved'         # retrieved | random_bank
    transfer_epochs: int = 30
    transfer_lr: float = 1e-4
    n_way: int = 2
    k_shot: int = 2
    n_query: int = 5
    min_episodes_per_epoch: int = 20
    max_episodes_per_epoch: int = 50
    adversarial_lambda: float = 0.3
    contrastive_lambda: float = 0.1
    prototype_lambda: float = 0.5
    uncertainty_threshold: float = 0.85
    feature_dim: int = 512
    projection_dim: int = 256
    num_prototypes: int = 10
    gradient_reversal_alpha: float = 1.0
    cdam_top_k: int = 5
    cdam_temperature: float = 0.1
    use_cdam: bool = True
    use_domain_adversarial: bool = True
    use_prototypical_branch: bool = True
    use_contrastive_alignment: bool = True
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True                    # fp16 autocast for the 2D encoder on GPU (speed on T4)
    patience: int = 10
    min_delta: float = 1e-4
    transfer_gate_floor: float = 0.15
    retrieval_residual_weight: float = 0.20
    alignment_lambda: float = 0.20
    transfer_usage_lambda: float = 0.05
    alignment_margin: float = 0.05
    min_transfer_shift: float = 0.03
    global_proto_momentum: float = 0.85
    episodic_proto_logit_weight: float = 0.45
    global_proto_logit_weight: float = 0.30
    classifier_logit_weight: float = 0.55
    support_consistency_lambda: float = 0.10

    # ---- evaluation ----
    bootstrap_iterations: int = 2000
    ece_bins: int = 10


CFG = ResearchConfig()

SEEDS = (0, 1, 2)

# Every experimental condition = overrides applied to the base config.
# 'ssl' = (snapshot epoch, temperature, length of the SSL run it is taken from).
MAIN_SSL = (50, 0.2, 50)
PATH_ONLY = dict(use_cdam=False, use_domain_adversarial=False, use_contrastive_alignment=False)
CONDITIONS: Dict[str, Dict[str, Any]] = {
    # headline + comparator (R1.2, R2.1)
    'main':             dict(),
    'path_only':        dict(PATH_ONLY),
    'main_k8':          dict(k_shot=8),
    'path_only_k8':     dict(PATH_ONLY, k_shot=8),
    # CT-bank controls (R1.2, R1.5)
    'bank_random_init': dict(ct_bank='random_init'),
    'bank_gaussian':    dict(ct_bank='gaussian'),
    'bank_dim_shuffle': dict(ct_bank='dim_shuffle'),
    'bank_mean':        dict(ct_bank='mean'),
    'retrieval_random': dict(retrieval_mode='random'),
    # InfoNCE circularity + forced-usage controls (R1.5)
    'ctr_random_pos':   dict(ctr_positive='random_bank'),
    'abl_free_gate':    dict(transfer_gate_floor=0.0, alignment_lambda=0.0, transfer_usage_lambda=0.0),
    # component ablations (R1.5, R2.1)
    'abl_no_cdam':      dict(use_cdam=False),
    'abl_no_adv':       dict(use_domain_adversarial=False),
    'abl_no_ctr':       dict(use_contrastive_alignment=False),
    'abl_no_proto':     dict(use_prototypical_branch=False),
    # SSL sensitivity (R1.3): epochs at tau=0.2 (snapshots of one 100-epoch run) and tau at 50 epochs
    'ssl_e0':           dict(ssl=(0, 0.2, 100)),
    'ssl_e10':          dict(ssl=(10, 0.2, 100)),
    'ssl_e25':          dict(ssl=(25, 0.2, 100)),
    'ssl_e50':          dict(ssl=(50, 0.2, 100)),
    'ssl_e100':         dict(ssl=(100, 0.2, 100)),
    'ssl_t0.05':        dict(ssl=(50, 0.05, 50)),
    'ssl_t0.1':         dict(ssl=(50, 0.1, 50)),
    'ssl_t0.5':         dict(ssl=(50, 0.5, 50)),
}
for _ch in ('CD3', 'CD8', 'FoxP3', 'PanCK', 'DAPI'):
    CONDITIONS[f'drop_{_ch}'] = dict(drop_channel=_ch)

# Conditions whose runs additionally store features + test-time bank swaps.
EXTRAS_CONDITIONS = {'main'}

# Independent run groups (each can run in its own session / GPU).
GROUPS: Dict[str, List[Tuple[str, Tuple[int, ...]]]] = {
    'A': [('main', SEEDS), ('path_only', SEEDS), ('main_k8', SEEDS), ('path_only_k8', SEEDS),
          ('ctr_random_pos', SEEDS)],
    'B': [('bank_random_init', SEEDS), ('bank_gaussian', SEEDS), ('bank_dim_shuffle', SEEDS),
          ('bank_mean', SEEDS), ('retrieval_random', SEEDS), ('abl_free_gate', SEEDS)],
    'C': [('abl_no_cdam', SEEDS), ('abl_no_adv', SEEDS), ('abl_no_ctr', SEEDS), ('abl_no_proto', SEEDS)]
         + [(c, (0,)) for c in ('ssl_e0', 'ssl_e10', 'ssl_e25', 'ssl_e50', 'ssl_e100',
                                'ssl_t0.05', 'ssl_t0.1', 'ssl_t0.5')]
         + [(f'drop_{c}', (0,)) for c in ('CD3', 'CD8', 'FoxP3', 'PanCK', 'DAPI')],
}


def condition_config(base: ResearchConfig, name: str, seed: int) -> ResearchConfig:
    ov = dict(CONDITIONS[name])
    ssl = ov.pop('ssl', MAIN_SSL)
    cfg = replace(base, **ov, seed=seed)
    cfg.ssl_epochs, cfg.ssl_temperature, cfg.ssl_max_epochs = ssl
    return cfg


# =============================================================================
# UTILITIES
# =============================================================================
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # speed over bit-exact reproducibility (seeds still fix data order, splits and init)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class Logger:
    def __init__(self, path: Optional[str]):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.path:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


class Progress:
    """Worker -> launcher progress channel (one small JSON file per worker)."""
    path: Optional[str] = None
    state: Dict[str, Any] = {}
    _last = 0.0

    @classmethod
    def setup(cls, path: str) -> None:
        cls.path, cls.state = path, {}

    @classmethod
    def update(cls, force: bool = False, **kw) -> None:
        cls.state.update(kw)
        if cls.path and (force or time.time() - cls._last > 2):
            cls._last = time.time()
            save_json(cls.state, cls.path)


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f'.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, default=_json_default)
    for attempt in range(20):          # Windows refuses to replace a file another process has open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5)
    os.remove(tmp)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


class FileLock:
    """Cross-process lock via O_EXCL lock files, so parallel shards never train
    the same SSL encoder twice at the same time."""
    def __init__(self, path: str, stale_seconds: float = 3 * 3600):
        self.path, self.stale = path, stale_seconds

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.path) > self.stale:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                time.sleep(5)

    def __exit__(self, *exc):
        try:
            os.remove(self.path)
        except OSError:
            pass


def info_nce_loss(features_2d: torch.Tensor, features_3d: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    features_2d = F.normalize(features_2d, dim=1)
    features_3d = F.normalize(features_3d, dim=1)
    logits = torch.matmul(features_2d, features_3d.T) / temperature
    labels = torch.arange(features_2d.size(0), device=features_2d.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def retrieve_ct_context(query: torch.Tensor, bank: torch.Tensor, cfg: ResearchConfig,
                        gen: Optional[torch.Generator] = None):
    """Top-k cosine retrieval (as in the original), or k random bank entries
    when cfg.retrieval_mode == 'random' (control, R1.2/R1.5)."""
    bank = bank.to(query.device)
    qn, bn = F.normalize(query, dim=1), F.normalize(bank, dim=1)
    k = min(cfg.cdam_top_k, bank.size(0))
    if cfg.retrieval_mode == 'random':
        idx = torch.rand(query.size(0), bank.size(0), generator=gen).argsort(1)[:, :k]
        idx = idx.to(query.device)
        sims = torch.gather(qn @ bn.T, 1, idx)
    else:
        sims, idx = torch.topk(qn @ bn.T, k=k, dim=1)
    neighbours = bank[idx]
    weights = F.softmax(sims / cfg.cdam_temperature, dim=1).unsqueeze(-1)
    retrieved = (neighbours * weights).sum(dim=1)
    return retrieved, idx, neighbours


def compute_model_selection_score(m: Dict[str, float]) -> float:
    return float(0.50 * m.get('balanced_accuracy', 0.0) + 0.30 * m.get('f1_score_macro', 0.0)
                 + 0.20 * m.get('accuracy', 0.0))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Binary ECE on the predicted-class confidence."""
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def binary_metrics(y, p, threshold: float = 0.5, n_bins: int = 10) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    two = len(np.unique(y)) > 1
    return {
        'n': int(len(y)),
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)) if two else float('nan'),
        'f1_score_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
        'auc_roc': float(roc_auc_score(y, p)) if two else float('nan'),
        'cohen_kappa': float(cohen_kappa_score(y, pred)) if two else float('nan'),
        'mcc': float(matthews_corrcoef(y, pred)) if two else float('nan'),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) else float('nan'),
        'specificity': float(tn / (tn + fp)) if (tn + fp) else float('nan'),
        'ppv': float(tp / (tp + fp)) if (tp + fp) else float('nan'),
        'npv': float(tn / (tn + fn)) if (tn + fn) else float('nan'),
        'ece': expected_calibration_error(y, p, n_bins),
        'brier': float(brier_score_loss(y, p)),
        'confusion_matrix': cm.tolist(),
    }


# =============================================================================
# DATA
# =============================================================================
class PathologyData:
    """All pathology tiles held in memory. Patient = 'CaseN' prefix; label from
    the ROI type in the file name ()."""

    def __init__(self, cfg: ResearchConfig, log: Logger):
        self.cfg = cfg
        root = cfg.pathology_data_dir
        seg_dir = os.path.join(root, 'Segmentation')
        if not os.path.isdir(seg_dir):
            raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")
        names = sorted(f[:-4] for f in os.listdir(seg_dir) if f.endswith('.png'))

        rows = []
        for base in names:
            parts = base.split('_')
            patient, roi = parts[0], parts[1]
            roi_type = roi[0].upper()
            mask = np.array(Image.open(os.path.join(seg_dir, base + '.png')).convert('L'))
            legacy_ratio = float((mask > 127).mean())
            legacy_label = int(legacy_ratio > cfg.legacy_mask_threshold)
            if cfg.label_mode == 'roi_t_vs_s':
                label = {'T': 1, 'S': 0}.get(roi_type)
            elif cfg.label_mode == 'roi_tm_vs_s':
                label = {'T': 1, 'M': 1, 'S': 0}.get(roi_type)
            elif cfg.label_mode == 'legacy_mask':
                label = legacy_label
            else:
                raise ValueError(cfg.label_mode)
            paths = self._channel_paths(base)
            rows.append(dict(name=base, patient=patient, roi=roi, roi_type=roi_type, label=label,
                             legacy_ratio=legacy_ratio, legacy_label=legacy_label,
                             complete=paths is not None, paths=paths))
        self.all_rows = rows

        use = [r for r in rows if r['label'] is not None and r['complete']]
        if cfg.max_patients:
            keep = sorted({r['patient'] for r in use}, key=_case_key)[:cfg.max_patients]
            use = [r for r in use if r['patient'] in keep]
        if cfg.max_tiles_per_class_per_patient:
            cnt, sub = Counter(), []
            for r in use:
                key = (r['patient'], r['label'])
                if cnt[key] < cfg.max_tiles_per_class_per_patient:
                    cnt[key] += 1
                    sub.append(r)
            use = sub
        self.rows = use
        self.names = [r['name'] for r in use]
        self.patients = np.array([r['patient'] for r in use])
        self.labels = np.array([r['label'] for r in use], dtype=int)
        self.patient_list = sorted(set(self.patients), key=_case_key)

        imgs = np.stack([self._load(r['paths']) for r in use]).astype(np.float32)
        self.raw_channel_stats = self._channel_stats(imgs)          # before ROI masking
        if cfg.use_dapi_roi_mask:
            imgs = np.stack([self._dapi_roi(x) for x in imgs])
        self.images = torch.from_numpy(imgs)                        # [N, C, H, W] in [0, 1]
        log(f"Pathology: {len(use)} tiles, {len(self.patient_list)} patients, "
            f"label_mode={cfg.label_mode}, class counts={np.bincount(self.labels, minlength=2).tolist()}")

    def _channel_paths(self, base: str) -> Optional[Dict[str, str]]:
        out = {}
        for ch in self.cfg.pathology_channels:
            for sub in ('mIF_Data', 'mIHC_Data'):
                p = os.path.join(self.cfg.pathology_data_dir, sub, f"{base}_{ch}.png")
                if os.path.isfile(p):
                    out[ch] = p
                    break
            else:
                return None
        return out

    def _load(self, paths: Dict[str, str]) -> np.ndarray:
        s = self.cfg.path_image_size
        return np.stack([
            np.asarray(Image.open(paths[ch]).convert('L').resize((s, s), Image.BILINEAR), dtype=np.float32) / 255.0
            for ch in self.cfg.pathology_channels
        ])

    def _dapi_roi(self, image: np.ndarray) -> np.ndarray:
        # unchanged from the original: label-blind DAPI 75th-percentile ROI mask
        chans = self.cfg.pathology_channels
        ref = image[chans.index('DAPI') if 'DAPI' in chans else 0]
        roi = (ref > np.percentile(ref, 75.0)).astype(np.float32)
        if roi.mean() < 0.01:
            roi = (ref > ref.mean() + 0.3 * ref.std()).astype(np.float32)
        return image * roi[None]

    def _channel_stats(self, imgs: np.ndarray) -> np.ndarray:
        feats = []
        for c in range(imgs.shape[1]):
            x = imgs[:, c].reshape(len(imgs), -1)
            feats += [x.mean(1), x.std(1), (x > 0.2).mean(1)]
        return np.stack(feats, 1)

    def summary(self) -> Dict[str, Any]:
        rows = self.all_rows
        per = defaultdict(lambda: defaultdict(int))
        for r in rows:
            per[r['patient']][r['roi_type']] += 1
        cross = Counter((r['roi_type'], r['legacy_label']) for r in rows)
        used = pd.DataFrame(self.rows)
        # R2.5: how well does one hand-crafted channel statistic predict each label?
        chans = self.cfg.pathology_channels
        stat_auc = {}
        for i, ch in enumerate(chans):
            stat_auc[f'{ch}_mean'] = {
                'task_label': _safe_auc(self.labels, self.raw_channel_stats[:, 3 * i]),
                'legacy_label': _safe_auc(used['legacy_label'].values, self.raw_channel_stats[:, 3 * i]),
            }
        return {
            'label_mode': self.cfg.label_mode,
            'n_tiles_on_disk': len(rows),
            'n_tiles_used': len(self.rows),
            'n_patients': len(self.patient_list),
            'patients': self.patient_list,
            'tiles_per_patient_by_roi_type': {p: dict(v) for p, v in sorted(per.items(), key=lambda kv: _case_key(kv[0]))},
            'used_class_counts': {int(k): int(v) for k, v in zip(*np.unique(self.labels, return_counts=True))},
            'used_per_patient_class_counts': {
                p: np.bincount(self.labels[self.patients == p], minlength=2).tolist() for p in self.patient_list},
            'legacy_label_counts': dict(Counter(int(r['legacy_label']) for r in rows)),
            'roi_type_x_legacy_label': {f'{k[0]}_{k[1]}': v for k, v in sorted(cross.items())},
            'legacy_ratio_quantiles': np.quantile([r['legacy_ratio'] for r in rows], [0, .1, .25, .5, .75, .9, 1]).tolist(),
            'single_channel_mean_auc': stat_auc,
        }


def _case_key(p: str):
    digits = ''.join(ch for ch in p if ch.isdigit())
    return (int(digits) if digits else 0, p)


def _safe_auc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float('nan')


def load_ct_volumes(cfg: ResearchConfig, cache_dir: str, log: Logger) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """ True HU, z-sorted slices, `depth` slices spread uniformly over the series."""
    import pydicom
    lo, hi = cfg.ct_hu_window
    key = f"ct_d{cfg.ct_volume_depth}_s{cfg.ct_slice_size}_w{int(lo)}_{int(hi)}_n{cfg.ct_max_volumes}"
    cache = os.path.join(cache_dir, key + '.npz')
    with FileLock(cache + '.lock'):     # parallel shards: only one reads the DICOMs
        return _load_ct_volumes_locked(cfg, cache, log)


def _load_ct_volumes_locked(cfg: ResearchConfig, cache: str, log: Logger):
    import pydicom
    lo, hi = cfg.ct_hu_window
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        log(f"CT: loaded cached volumes {z['volumes'].shape} from {cache}")
        return z['volumes'], list(z['ids']), json.loads(str(z['info']))

    root = cfg.ct_data_dir
    if not os.path.isdir(root):
        raise FileNotFoundError(f"CT data directory not found: {root}")
    patients = sorted(d for d in os.listdir(root) if d.startswith('R_') and os.path.isdir(os.path.join(root, d)))
    vols, ids, n_slices, thick = [], [], [], []
    for pid in patients:
        for date in sorted(os.listdir(os.path.join(root, pid))):
            dpath = os.path.join(root, pid, date)
            if not os.path.isdir(dpath):
                continue
            for series in sorted(os.listdir(dpath)):
                spath = os.path.join(dpath, series)
                if not os.path.isdir(spath):
                    continue
                files = [os.path.join(spath, f) for f in os.listdir(spath) if f.endswith('.dcm')]
                if len(files) < cfg.ct_volume_depth:
                    continue
                heads = []
                for fp in files:
                    try:
                        h = pydicom.dcmread(fp, stop_before_pixels=True)
                    except Exception:
                        continue
                    ipp = getattr(h, 'ImagePositionPatient', None)
                    zpos = float(ipp[2]) if ipp is not None else float(getattr(h, 'InstanceNumber', 0) or 0)
                    heads.append((zpos, fp, float(getattr(h, 'SliceThickness', 0) or 0)))
                if len(heads) < cfg.ct_volume_depth:
                    continue
                heads.sort(key=lambda t: t[0])
                pick = np.linspace(0, len(heads) - 1, cfg.ct_volume_depth).round().astype(int)
                slices = []
                for i in pick:
                    ds = pydicom.dcmread(heads[i][1])
                    hu = ds.pixel_array.astype(np.float32) * float(getattr(ds, 'RescaleSlope', 1) or 1) \
                        + float(getattr(ds, 'RescaleIntercept', 0) or 0)
                    hu = (np.clip(hu, lo, hi) - lo) / (hi - lo)
                    img = Image.fromarray(hu).resize((cfg.ct_slice_size, cfg.ct_slice_size), Image.BILINEAR)
                    slices.append(np.asarray(img, dtype=np.float32))
                vols.append(np.stack(slices))
                ids.append(f"{pid}/{series}")
                n_slices.append(len(heads))
                thick.append(heads[0][2])
                if cfg.ct_max_volumes and len(vols) >= cfg.ct_max_volumes:
                    break
            if cfg.ct_max_volumes and len(vols) >= cfg.ct_max_volumes:
                break
        if cfg.ct_max_volumes and len(vols) >= cfg.ct_max_volumes:
            break
    volumes = np.stack(vols).astype(np.float32)
    info = {'n_volumes': len(vols), 'n_patients': len({i.split('/')[0] for i in ids}),
            'slices_per_series_min_max_mean': [int(min(n_slices)), int(max(n_slices)), float(np.mean(n_slices))],
            'slice_thickness_values_mm': sorted({round(t, 2) for t in thick}),
            'hu_window': [lo, hi], 'depth_sampled': cfg.ct_volume_depth, 'slice_size': cfg.ct_slice_size}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + f'.tmp{os.getpid()}.npz'
    np.savez_compressed(tmp, volumes=volumes, ids=np.array(ids), info=json.dumps(info))
    os.replace(tmp, cache)
    log(f"CT: {volumes.shape[0]} volumes {volumes.shape[1:]} from {info['n_patients']} patients; info={info}")
    return volumes, ids, info


# ---- GPU augmentations (same operations/probabilities as the original numpy versions) ----
def augment_ct(v: torch.Tensor, g: torch.Generator, strength: float) -> torch.Tensor:
    """v: [B, 1, D, H, W]"""
    out = []
    for x in v:
        r = torch.rand(7, generator=g).tolist()
        if r[0] < 0.5: x = x.flip(1)
        if r[1] < 0.5: x = x.flip(2)
        if r[2] < 0.5: x = x.flip(3)
        if r[3] < strength:
            x = (x * (0.9 + 0.2 * torch.rand(1, generator=g).item())).clamp(0, 1)
        if r[4] < strength:
            x = (x + (-0.1 + 0.2 * torch.rand(1, generator=g).item())).clamp(0, 1)
        if r[5] < strength * 0.5:
            x = (x + 0.02 * torch.randn(x.shape, generator=g).to(x.device)).clamp(0, 1)
        out.append(x)
    return torch.stack(out)


def augment_path(v: torch.Tensor, g: torch.Generator, strength: float = 0.5,
                 gdev: Optional[torch.Generator] = None) -> torch.Tensor:
    """Batched version of the original per-tile augmentation (same operations and
    probabilities): h/v flips, random 90-degree rotation, per-channel intensity scaling,
    Gaussian noise. v: [B, C, H, W]. Random decisions on the CPU generator `g`, the noise
    tensor on the device generator `gdev` (no host->device noise copies)."""
    B, C = v.shape[:2]
    r = torch.rand(B, 5, generator=g)
    k = torch.randint(1, 4, (B,), generator=g)
    scale = 0.9 + 0.2 * torch.rand(B, C, generator=g)
    x = v.clone()
    m = (r[:, 0] < 0.5).to(v.device)
    x = torch.where(m[:, None, None, None], x.flip(3), x)
    m = (r[:, 1] < 0.5).to(v.device)
    x = torch.where(m[:, None, None, None], x.flip(2), x)
    for kk in (1, 2, 3):
        sel = ((r[:, 2] < 0.5) & (k == kk)).nonzero().flatten().to(v.device)
        if len(sel):
            x[sel] = torch.rot90(x[sel], kk, dims=(2, 3))
    sc = torch.where((r[:, 3] < strength)[:, None], scale, torch.ones_like(scale)).to(v.device)
    x = (x * sc[:, :, None, None]).clamp(0, 1)
    noisy = (r[:, 4] < strength * 0.5).to(v.device)
    if bool(noisy.any()):
        gen = gdev if (gdev is not None and gdev.device.type == v.device.type) else None
        noise = torch.randn(x.shape, generator=gen, device=v.device) if gen is not None else \
            torch.randn(x.shape, generator=g).to(v.device)
        x = torch.where(noisy[:, None, None, None], (x + 0.02 * noise).clamp(0, 1), x)
    return x


# =============================================================================
# MODEL: 3D and 2D encoders, GRL discriminator, CDAM, prototypical head
# =============================================================================
class MultiScale3DEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32, feature_dim: int = 512):
        super().__init__()
        
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool3d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        
        global_feat = self.global_pool(x4).flatten(1)
        global_feat = self.fc(global_feat)
        
        multiscale = [x1, x2, x3, x4]
        
        return global_feat, multiscale

class MultiChannel2DEncoder(nn.Module):
    def __init__(self, in_channels: int = 5, base_channels: int = 32, feature_dim: int = 512):
        super().__init__()
        
        self.channel_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, base_channels // 2, 3, padding=1),
                nn.BatchNorm2d(base_channels // 2),
                nn.ReLU(inplace=True)
            ) for _ in range(in_channels)
        ])
        
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels),
            nn.Sigmoid()
        )
        
        self.enc1 = self._conv_block(base_channels // 2 * in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(base_channels * 8, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        channel_feats = []
        for i in range(C):
            feat = self.channel_convs[i](x[:, i:i+1, :, :])
            channel_feats.append(feat)
        
        channel_global = torch.stack([
            F.adaptive_avg_pool2d(f, 1).flatten(1)
            for f in channel_feats
        ], dim=1)
        
        attention_weights = self.channel_attention(channel_global.mean(dim=2))
        
        weighted_feats = []
        for i in range(C):
            weighted = channel_feats[i] * attention_weights[:, i:i+1, None, None]
            weighted_feats.append(weighted)
        
        x = torch.cat(weighted_feats, dim=1)
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.pool(x)
        x = self.enc4(x)
        
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        
        return x

class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        
        self.grl = GradientReversalLayer()
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        self.grl.alpha = alpha
        reversed_features = self.grl(features)
        domain_logits = self.discriminator(reversed_features)
        return domain_logits

class CrossDimensionalAttention(nn.Module):
    def __init__(self, feature_dim: int = 512, num_heads: int = 8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.last_attention_weights: Optional[torch.Tensor] = None
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        batch_size = query.size(0)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        
        query_proj = self.query_proj(query)
        key_proj = self.key_proj(key_value)
        value_proj = self.value_proj(key_value)
        
        query = query_proj.view(batch_size, 1, self.feature_dim)
        key = key_proj
        value = value_proj
        
        attn_output, attn_weights = self.attention(query, key, value, need_weights=True)
        self.last_attention_weights = attn_weights.detach()
        attn_output = attn_output.squeeze(1)
        
        x1 = self.norm1(query_proj + attn_output)
        ffn_output = self.ffn(x1)
        x2 = self.norm2(x1 + ffn_output)
        
        combined = torch.cat([query_proj, x2], dim=-1)
        gate_weights = self.gate(combined)
        
        aligned_features = gate_weights * x2 + (1 - gate_weights) * query_proj
        
        output = self.output_proj(aligned_features)
        output = self.norm3(output + aligned_features)
        
        return output

class PrototypicalNetwork(nn.Module):
    def __init__(self, feature_dim: int = 512, num_prototypes: int = 10):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
    
    def compute_prototypes(self, support_features: torch.Tensor, support_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        prototypes = []
        
        for c in range(num_classes):
            mask = support_labels == c
            class_features = support_features[mask]
            
            if len(class_features) > 0:
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
            else:
                prototypes.append(torch.zeros(self.feature_dim, device=support_features.device))
        
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes
    
    def compute_distances(self, query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        queries = query_features.unsqueeze(1)
        protos = prototypes.unsqueeze(0)
        
        distances = torch.sum((queries - protos) ** 2, dim=2)
        
        return distances
    
    def forward(self, support_features: torch.Tensor, support_labels: torch.Tensor, query_features: torch.Tensor, num_classes: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        prototypes = self.compute_prototypes(support_features, support_labels, num_classes)
        distances = self.compute_distances(query_features, prototypes)
        logits = -distances
        
        return logits, prototypes

class UncertaintyAwarePseudoLabeler(nn.Module):
    def __init__(self, feature_dim: int = 512, num_classes: int = 2, num_mc_samples: int = 10, confidence_threshold: float = 0.85):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_mc_samples = num_mc_samples
        self.confidence_threshold = confidence_threshold
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, num_classes)
        )
    
    def forward_with_uncertainty(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        
        logits_samples = []
        for _ in range(self.num_mc_samples):
            logits = self.classifier(features)
            logits_samples.append(logits)
        
        logits_samples = torch.stack(logits_samples, dim=0)
        mean_logits = logits_samples.mean(dim=0)
        
        probs_samples = F.softmax(logits_samples, dim=2)
        mean_probs = probs_samples.mean(dim=0)
        
        uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        return mean_logits, uncertainty
    
    def generate_pseudo_labels(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, uncertainty = self.forward_with_uncertainty(features)
        probs = F.softmax(logits, dim=1)
        
        confidences, pseudo_labels = probs.max(dim=1)
        
        normalized_uncertainty = uncertainty / math.log(self.num_classes)
        confidence_score = 1.0 - normalized_uncertainty
        
        mask = confidence_score > self.confidence_threshold
        
        return pseudo_labels, confidence_score, mask

class CrossDomainTransferModel(nn.Module):
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def __init__(self, config: ResearchConfig):
        super().__init__()
        
        self.config = config
        
        self.ct_encoder = MultiScale3DEncoder(
            in_channels=config.ct_in_channels,
            feature_dim=config.feature_dim
        )
        
        self.path_encoder = MultiChannel2DEncoder(
            in_channels=config.path_in_channels,
            feature_dim=config.feature_dim
        )
        
        self.domain_discriminator = DomainDiscriminator(config.feature_dim)
        
        self.cross_attention = CrossDimensionalAttention(
            feature_dim=config.feature_dim
        )
        
        self.prototypical_net = PrototypicalNetwork(
            feature_dim=config.feature_dim,
            num_prototypes=config.num_prototypes
        )
        
        self.pseudo_labeler = UncertaintyAwarePseudoLabeler(
            feature_dim=config.feature_dim,
            num_classes=config.n_way,
            confidence_threshold=config.uncertainty_threshold
        )
        
        self.ct_projector = nn.Sequential(
            nn.Linear(config.feature_dim, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, config.projection_dim)
        )
        
        self.path_projector = nn.Sequential(
            nn.Linear(config.feature_dim, config.projection_dim),
            nn.ReLU(),
            nn.Linear(config.projection_dim, config.projection_dim)
        )
        self.retrieval_adapter = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.LayerNorm(config.feature_dim),
            nn.GELU()
        )
        self.transfer_gate = nn.Sequential(
            nn.Linear(config.feature_dim * 3, config.feature_dim),
            nn.GELU(),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Linear(config.feature_dim, config.n_way)
        self.last_attention_weights: Optional[torch.Tensor] = None
        self.last_transfer_gate: Optional[torch.Tensor] = None
        self.register_buffer('global_prototypes', torch.zeros(config.n_way, config.feature_dim))
        self.register_buffer('global_proto_initialized', torch.zeros(config.n_way, dtype=torch.bool))
        
        self.apply(self._init_weights)
    
    def forward_ct(self, ct_volumes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features, multiscale = self.ct_encoder(ct_volumes)
        projection = self.ct_projector(features)
        return features, projection
    
    def forward_path(self, path_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.path_encoder(path_images)
        projection = self.path_projector(features)
        return features, projection
    
    def align_features(
        self,
        path_features: torch.Tensor,
        ct_features: torch.Tensor,
        ct_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if ct_tokens is None:
            ct_tokens = ct_features.unsqueeze(1)
        attended = self.cross_attention(path_features, ct_tokens)
        self.last_attention_weights = self.cross_attention.last_attention_weights
        retrieved_summary = self.retrieval_adapter(ct_features)
        gate_input = torch.cat([path_features, attended, retrieved_summary], dim=1)
        gate = self.transfer_gate(gate_input)
        gate = self.config.transfer_gate_floor + (1.0 - self.config.transfer_gate_floor) * gate
        self.last_transfer_gate = gate.detach()

        fused = path_features + gate * (attended - path_features)
        fused = fused + self.config.retrieval_residual_weight * retrieved_summary
        fused = F.layer_norm(fused, (fused.size(-1),))
        return fused

    def compute_transfer_regularization(
        self,
        path_features: torch.Tensor,
        aligned_features: torch.Tensor,
        ct_features: torch.Tensor
    ) -> torch.Tensor:
        raw_ct_sim = F.cosine_similarity(path_features, ct_features, dim=1)
        aligned_ct_sim = F.cosine_similarity(aligned_features, ct_features, dim=1)
        shift = torch.norm(aligned_features - path_features, dim=1)

        alignment_penalty = F.relu(raw_ct_sim - aligned_ct_sim + self.config.alignment_margin).mean()
        usage_penalty = F.relu(self.config.min_transfer_shift - shift).mean()
        return (
            self.config.alignment_lambda * alignment_penalty +
            self.config.transfer_usage_lambda * usage_penalty
        )
    
    def update_global_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.no_grad():
            for c in range(self.config.n_way):
                mask = labels == c
                if mask.any():
                    class_mean = features[mask].mean(dim=0)
                    if self.global_proto_initialized[c]:
                        self.global_prototypes[c] = (
                            self.config.global_proto_momentum * self.global_prototypes[c] +
                            (1.0 - self.config.global_proto_momentum) * class_mean
                        )
                    else:
                        self.global_prototypes[c] = class_mean
                        self.global_proto_initialized[c] = True

    def compute_global_proto_logits(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.global_proto_initialized.any()):
            return torch.zeros(features.size(0), self.config.n_way, device=features.device)

        prototypes = self.global_prototypes.to(features.device)
        distances = torch.cdist(features, prototypes, p=2) ** 2
        logits = -distances

        if (~self.global_proto_initialized).any():
            invalid = (~self.global_proto_initialized).to(features.device)
            logits[:, invalid] = -1e4
        return logits

    def combine_logits(
        self,
        classifier_logits: torch.Tensor,
        episodic_proto_logits: Optional[torch.Tensor] = None,
        global_proto_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        combined = self.config.classifier_logit_weight * classifier_logits
        if episodic_proto_logits is not None:
            combined = combined + self.config.episodic_proto_logit_weight * episodic_proto_logits
        if global_proto_logits is not None:
            combined = combined + self.config.global_proto_logit_weight * global_proto_logits
        return combined

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)



# =============================================================================
# PHASE 1: CT SELF-SUPERVISED PRETRAINING
# =============================================================================
SSL_SNAPSHOTS = (0, 10, 25, 50, 100)


def encode_ct(encoder: nn.Module, vols: torch.Tensor, bs: int = 8) -> torch.Tensor:
    encoder.eval()
    with torch.no_grad():
        return torch.cat([encoder(vols[i:i + bs])[0] for i in range(0, len(vols), bs)]).cpu()


def ssl_intrinsic_metrics(encoder: nn.Module, vols: torch.Tensor, cfg: ResearchConfig) -> Dict[str, float]:
    """Label-free quality indicators of the CT representation (R1.3)."""
    g = torch.Generator().manual_seed(12345)
    clean = F.normalize(encode_ct(encoder, vols), dim=1)
    v1 = F.normalize(encode_ct(encoder, augment_ct(vols, g, cfg.ssl_augmentation_strength)), dim=1)
    v2 = F.normalize(encode_ct(encoder, augment_ct(vols, g, cfg.ssl_augmentation_strength)), dim=1)
    n = len(clean)
    sim = v1 @ v2.T
    view_top1 = float((sim.argmax(1) == torch.arange(n)).float().mean())
    alignment = float(((v1 - v2) ** 2).sum(1).mean())
    d2 = torch.cdist(clean, clean) ** 2
    off = ~torch.eye(n, dtype=torch.bool)
    uniformity = float(torch.log(torch.exp(-2 * d2[off]).mean()))
    cos = clean @ clean.T
    mean_cos = float(cos[off].mean())
    s = torch.linalg.svdvals(clean - clean.mean(0, keepdim=True))
    p = s / s.sum()
    eff_rank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
    return {'view_top1': view_top1, 'chance_top1': 1.0 / n, 'alignment': alignment,
            'uniformity': uniformity, 'mean_pairwise_cos': mean_cos, 'effective_rank': eff_rank}


def ssl_dir(root: str, tau: float, seed: int, max_epochs: int) -> str:
    return os.path.join(root, 'ssl', f'tau{tau}_seed{seed}_max{max_epochs}')


def get_ssl_encoder_state(cfg: ResearchConfig, vols: torch.Tensor, root: str, log: Logger) -> Dict[str, torch.Tensor]:
    d = ssl_dir(root, cfg.ssl_temperature, cfg.seed, cfg.ssl_max_epochs)
    target = os.path.join(d, f'encoder_epoch{cfg.ssl_epochs}.pt')
    if not os.path.exists(target):
        with FileLock(d + '.lock'):
            if not os.path.exists(target):
                train_ssl(cfg, vols, d, log)
    return torch.load(target, map_location='cpu')


def train_ssl(cfg: ResearchConfig, vols: torch.Tensor, out_dir: str, log: Logger) -> None:
    max_ep = cfg.ssl_max_epochs
    snaps = sorted({e for e in SSL_SNAPSHOTS if e <= max_ep} | {cfg.ssl_epochs, max_ep})
    log(f"SSL: tau={cfg.ssl_temperature} seed={cfg.seed} epochs={max_ep} snapshots={snaps} -> {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.seed)
    model = CrossDomainTransferModel(cfg).to(cfg.device)
    enc, proj = model.ct_encoder, model.ct_projector
    # projection head is trained too (the original optimiser held only the encoder)
    opt = AdamW(list(enc.parameters()) + list(proj.parameters()), lr=cfg.ssl_lr, weight_decay=cfg.weight_decay)
    g = torch.Generator().manual_seed(cfg.seed + 17)
    n = len(vols)
    bs = min(cfg.ssl_batch_size, n)
    history = []
    t0 = time.time()

    def record(epoch, loss):
        row = {'epoch': epoch, 'loss': loss, 'time_s': time.time() - t0}
        row.update(ssl_intrinsic_metrics(enc, vols, cfg))
        history.append(row)
        log(f"  SSL epoch {epoch}: " + ', '.join(f"{k}={v:.4f}" for k, v in row.items()
                                                  if isinstance(v, float) and k != 'time_s'))

    def snapshot(epoch):
        torch.save({k: v.detach().cpu() for k, v in enc.state_dict().items()},
                   os.path.join(out_dir, f'encoder_epoch{epoch}.pt'))

    record(0, None)
    if 0 in snaps:
        snapshot(0)
    for epoch in range(1, max_ep + 1):
        enc.train(); proj.train()
        perm = torch.randperm(n, generator=g)
        losses = []
        for b in range(n // bs):
            x = vols[perm[b * bs:(b + 1) * bs]]
            v1 = augment_ct(x, g, cfg.ssl_augmentation_strength)
            v2 = augment_ct(x, g, cfg.ssl_augmentation_strength)
            z = F.normalize(torch.cat([proj(enc(v1)[0]), proj(enc(v2)[0])]), dim=1)
            sim = z @ z.T
            sim.masked_fill_(torch.eye(2 * len(x), dtype=torch.bool, device=z.device), -float('inf'))
            tgt = torch.arange(len(x), device=z.device)
            tgt = torch.cat([tgt + len(x), tgt])
            loss = F.cross_entropy(sim / cfg.ssl_temperature, tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(proj.parameters()), cfg.grad_clip)
            opt.step()
            losses.append(loss.item())
        mean_loss = float(np.mean(losses)) if losses else float('nan')
        Progress.update(epoch=f"SSL tau={cfg.ssl_temperature} ep {epoch}/{max_ep}")
        if epoch % cfg.ssl_eval_interval == 0 or epoch in snaps:
            record(epoch, mean_loss)
        else:
            history.append({'epoch': epoch, 'loss': mean_loss, 'time_s': time.time() - t0})
        if epoch in snaps:
            snapshot(epoch)
    save_json({'tau': cfg.ssl_temperature, 'seed': cfg.seed, 'max_epochs': max_ep,
               'batch_size': bs, 'n_volumes': n, 'history': history}, os.path.join(out_dir, 'history.json'))


class BankFactory:
    """CT feature banks: SSL bank and the controls (R1.2/R1.5)."""

    def __init__(self, vols: torch.Tensor, root: str, log: Logger):
        self.vols, self.root, self.log = vols, root, log
        self.cache: Dict[Tuple, torch.Tensor] = {}

    def _encoder_bank(self, cfg: ResearchConfig, random_init: bool) -> torch.Tensor:
        key = ('rand', cfg.seed) if random_init else ('ssl', cfg.ssl_epochs, cfg.ssl_temperature, cfg.ssl_max_epochs, cfg.seed)
        if key not in self.cache:
            set_seed(cfg.seed)
            model = CrossDomainTransferModel(cfg).to(cfg.device)     # same init as SSL epoch 0
            if not random_init:
                model.ct_encoder.load_state_dict(get_ssl_encoder_state(cfg, self.vols, self.root, self.log))
            self.cache[key] = encode_ct(model.ct_encoder, self.vols)
        return self.cache[key]

    def get(self, cfg: ResearchConfig) -> torch.Tensor:
        if cfg.ct_bank == 'random_init':
            return self._encoder_bank(cfg, True)
        bank = self._encoder_bank(cfg, False)
        g = torch.Generator().manual_seed(cfg.seed + 101)
        if cfg.ct_bank == 'ssl':
            return bank
        if cfg.ct_bank == 'gaussian':      # per-dimension mean/std matched noise
            return bank.mean(0) + bank.std(0) * torch.randn(bank.shape, generator=g)
        if cfg.ct_bank == 'dim_shuffle':   # destroys structure, keeps each entry's value distribution
            return torch.stack([row[torch.randperm(len(row), generator=g)] for row in bank])
        if cfg.ct_bank == 'mean':          # identical context for every query
            return bank.mean(0, keepdim=True).repeat(len(bank), 1)
        raise ValueError(cfg.ct_bank)


# =============================================================================
# PHASE 2: EPISODIC TRANSFER TRAINING
# =============================================================================
def encode_path(model: CrossDomainTransferModel, x: torch.Tensor, cfg: ResearchConfig):
    """2D encoder forward, fp16 autocast on GPU; outputs returned in fp32 so prototypes,
    distances and losses are computed in full precision."""
    amp = cfg.use_amp and 'cuda' in str(cfg.device)
    if amp:
        x = x.contiguous(memory_format=torch.channels_last)   # tensor-core friendly layout
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        f, z = model.forward_path(x)
    return f.float(), z.float()


def forward_eval(model: CrossDomainTransferModel, x: torch.Tensor, bank: torch.Tensor, cfg: ResearchConfig,
                 gen: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    model.eval()
    out = defaultdict(list)
    with torch.no_grad():
        for i in range(0, len(x), 64):          # eval mode (BN frozen): batch size is free
            xb = x[i:i + 64]
            f, _ = encode_path(model, xb, cfg)
            r, idx, tok = retrieve_ct_context(f, bank, cfg, gen)
            if cfg.use_cdam:
                a = model.align_features(f, r, tok)
                out['attn'].append(model.last_attention_weights.squeeze(1).cpu())
                out['gate'].append(model.last_transfer_gate.mean(1).cpu())
            else:
                a = f
            glob = model.compute_global_proto_logits(a) if cfg.use_prototypical_branch else None
            cls = model.classify(a)
            logits = model.combine_logits(cls, global_proto_logits=glob)
            # R1.4: weighted tumour-vs-stroma logit margin contributed by each term
            out['m_cls'].append((cfg.classifier_logit_weight * (cls[:, 1] - cls[:, 0])).cpu())
            if glob is not None:
                out['m_glob'].append((cfg.global_proto_logit_weight * (glob[:, 1] - glob[:, 0])).cpu())
            out['prob'].append(F.softmax(logits, 1)[:, 1].cpu())
            out['idx'].append(idx.cpu())
            out['f'].append(f.cpu()); out['a'].append(a.cpu()); out['r'].append(r.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def train_fold(cfg: ResearchConfig, images: torch.Tensor, labels: torch.Tensor, bank: torch.Tensor,
               tr: np.ndarray, va: np.ndarray, run_seed: int, log: Logger) -> Tuple[CrossDomainTransferModel, Dict]:
    dev = cfg.device
    set_seed(run_seed)
    model = CrossDomainTransferModel(cfg).to(dev)
    if cfg.use_amp and 'cuda' in str(dev):
        model.path_encoder.to(memory_format=torch.channels_last)
    for p in model.ct_encoder.parameters():          # stage 2 uses the precomputed (frozen) bank
        p.requires_grad = False
    # every other module is optimised (discriminator, gate, adapter were missing before)
    params = [p for n, p in model.named_parameters() if not n.startswith('ct_encoder.')]
    opt = AdamW(params, lr=cfg.transfer_lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.use_amp and 'cuda' in str(dev))
    bank = bank.to(dev)
    g = torch.Generator().manual_seed(run_seed)
    gdev = torch.Generator(device=dev).manual_seed(run_seed) if 'cuda' in str(dev) else None

    y_tr = labels[tr].cpu().numpy()
    cls_idx = [tr[y_tr == c] for c in range(cfg.n_way)]
    counts = np.bincount(y_tr, minlength=cfg.n_way).astype(np.float32)
    w = 1.0 / np.maximum(counts, 1)
    w = torch.tensor(w / w.sum() * cfg.n_way, dtype=torch.float32, device=dev)
    ce, dce = nn.CrossEntropyLoss(weight=w), nn.CrossEntropyLoss()

    k, q = cfg.k_shot, cfg.n_query
    smallest = min(len(c) for c in cls_idx)
    if smallest < k + q:                             # only reachable in smoke mode / tiny patients
        k = max(1, min(k, smallest - 1)); q = max(1, smallest - k)
        log(f"    [warn] class size {smallest} < K+Q; using K={k}, Q={q}")
    n_ep = min(cfg.max_episodes_per_epoch, max(cfg.min_episodes_per_epoch, len(tr) // (k * cfg.n_way)))

    best_score, best_state, best_epoch, patience = -float('inf'), None, 0, 0
    history = []
    for epoch in range(1, cfg.transfer_epochs + 1):
        model.train()
        ep_loss, ep_acc = [], []                    # device tensors; one host sync per epoch
        for _ in range(n_ep):
            sup, qry = [], []
            for c in range(cfg.n_way):
                sel = cls_idx[c][torch.randperm(len(cls_idx[c]), generator=g)[:k + q].numpy()]
                sup += sel[:k].tolist(); qry += sel[k:].tolist()
            sup = [sup[i] for i in torch.randperm(len(sup), generator=g)]
            qry = [qry[i] for i in torch.randperm(len(qry), generator=g)]
            ys, yq = labels[sup], labels[qry]
            # support + query in one encoder pass (one batch-norm batch, as in standard ProtoNets)
            x_all = augment_path(images[sup + qry], g, gdev=gdev)
            f_all, z_all = encode_path(model, x_all, cfg)
            fs, fq, zq = f_all[:len(sup)], f_all[len(sup):], z_all[len(sup):]
            rq, _, tq = retrieve_ct_context(fq.detach(), bank, cfg, g)
            rs, _, ts = retrieve_ct_context(fs.detach(), bank, cfg, g)
            aq = model.align_features(fq, rq, tq) if cfg.use_cdam else fq
            as_ = model.align_features(fs, rs, ts) if cfg.use_cdam else fs

            cls_q, cls_s = model.classify(aq), model.classify(as_)
            if cfg.use_prototypical_branch:
                proto_logits, _ = model.prototypical_net(as_, ys, aq, cfg.n_way)
                proto_loss = ce(proto_logits, yq)
                model.update_global_prototypes(as_.detach(), ys)
                comb = model.combine_logits(cls_q, proto_logits, model.compute_global_proto_logits(aq))
                sup_comb = model.combine_logits(cls_s, global_proto_logits=model.compute_global_proto_logits(as_))
            else:
                comb, sup_comb = model.combine_logits(cls_q), model.combine_logits(cls_s)

            loss = ce(comb, yq) + 0.5 * ce(cls_q, yq)
            if cfg.use_prototypical_branch:
                loss = loss + cfg.prototype_lambda * proto_loss
            if cfg.use_domain_adversarial:
                dl_p = dce(model.domain_discriminator(aq, alpha=cfg.gradient_reversal_alpha),
                           torch.ones(len(aq), dtype=torch.long, device=dev))
                dl_c = dce(model.domain_discriminator(rq, alpha=cfg.gradient_reversal_alpha),
                           torch.zeros(len(rq), dtype=torch.long, device=dev))
                loss = loss + cfg.adversarial_lambda * 0.5 * (dl_p + dl_c)
            if cfg.use_contrastive_alignment:
                if cfg.ctr_positive == 'random_bank':    # query-independent positives (R1.5 control)
                    ct_side = bank[torch.randint(len(bank), (len(zq),), generator=g).to(dev)]
                else:
                    ct_side = rq
                loss = loss + cfg.contrastive_lambda * info_nce_loss(zq, model.ct_projector(ct_side), cfg.cdam_temperature)
            loss = loss + cfg.support_consistency_lambda * ce(sup_comb, ys)
            if cfg.use_cdam:
                loss = loss + model.compute_transfer_regularization(fq, aq, rq) \
                    + model.compute_transfer_regularization(fs, as_, rs)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            ep_loss.append(loss.detach())
            ep_acc.append((comb.argmax(1) == yq).float().mean())
        sched.step()
        Progress.update(epoch=f"ep {epoch}/{cfg.transfer_epochs}")

        ev = forward_eval(model, images[va], bank, cfg, torch.Generator().manual_seed(999))
        vm = binary_metrics(labels[va].cpu().numpy(), ev['prob'].numpy())
        score = compute_model_selection_score(vm)
        history.append({'epoch': epoch, 'train_loss': float(torch.stack(ep_loss).mean()), 'train_acc': float(torch.stack(ep_acc).mean()),
                        'val_acc': vm['accuracy'], 'val_bal_acc': vm['balanced_accuracy'],
                        'val_auc': vm['auc_roc'], 'val_score': score})
        if score > best_score + cfg.min_delta:
            best_score, best_epoch, patience = score, epoch, 0
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, {'history': history, 'best_epoch': best_epoch, 'best_val_score': best_score,
                   'epochs_run': len(history), 'episodes_per_epoch': n_ep, 'k_shot_used': k, 'n_query_used': q}


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================
def fold_plan(patients: List[str]) -> List[Tuple[int, str, str]]:
    """Leave-one-patient-out; validation patient = next patient in the rotation."""
    P = len(patients)
    return [(f, patients[f], patients[(f + 1) % P]) for f in range(P)]


def run_dir(root: str, cond: str, seed: int, fold: int) -> str:
    return os.path.join(root, 'runs', cond, f'seed{seed}', f'fold{fold}')


def planned_folds(patients: List[str], smoke: bool) -> List[Tuple[int, str, str]]:
    plan = fold_plan(patients)
    return plan[:2] if smoke else plan


def build_runs(group: str, patients: List[str], smoke: bool) -> List[Tuple[str, int, int, str, str]]:
    groups = ['A', 'B', 'C'] if group == 'all' else [group]
    plan = planned_folds(patients, smoke)
    runs = []
    for gname in groups:
        for cond, seeds in GROUPS[gname]:
            for s in (seeds[:1] if smoke else seeds):
                for f, tp, vp in plan:
                    runs.append((cond, s, f, tp, vp))
    return runs


def smoke_config(base: ResearchConfig) -> ResearchConfig:
    return replace(base, max_patients=3, max_tiles_per_class_per_patient=3, ct_max_volumes=4,
                   ct_volume_depth=8, ct_slice_size=32, path_image_size=64, transfer_epochs=2,
                   n_query=1, k_shot=1, min_episodes_per_epoch=2, max_episodes_per_epoch=2,
                   ssl_batch_size=4, ssl_eval_interval=1, patience=2, bootstrap_iterations=50)


def condition_config_for(base: ResearchConfig, cond: str, seed: int, smoke: bool) -> ResearchConfig:
    cfg = condition_config(base, cond, seed)
    if smoke:
        e, m = cfg.ssl_epochs, cfg.ssl_max_epochs
        cfg.ssl_epochs, cfg.ssl_max_epochs = (0 if e == 0 else min(e, 2)), 2
        if cfg.k_shot > 1:
            cfg.k_shot = 2
    return cfg


def run_trivial_baseline(data: PathologyData, root: str, log: Logger, smoke: bool) -> None:
    """Hand-crafted per-channel intensity statistics + logistic regression, LOPO (R2.5)."""
    X, y = data.raw_channel_stats, data.labels
    plan = planned_folds(data.patient_list, smoke)
    for f, tp, _ in plan:
        out = os.path.join(run_dir(root, 'trivial_channel_stats', 0, f), 'result.json')
        if os.path.exists(out):
            continue
        te = np.where(data.patients == tp)[0]
        tr = np.where(data.patients != tp)[0]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        save_json({'condition': 'trivial_channel_stats', 'seed': 0, 'fold': f, 'n_folds': len(plan), 'test_patient': tp,
                   'records': [{'name': data.names[i], 'patient': tp, 'y': int(y[i]), 'prob': float(pp)}
                               for i, pp in zip(te, p)]}, out)
    log("Trivial channel-statistics baseline done.")


def execute(args, base: ResearchConfig) -> None:
    root = base.output_dir
    os.makedirs(root, exist_ok=True)
    tag = f"{args.group}_shard{args.shard.replace('/', 'of')}"
    log = Logger(os.path.join(root, 'logs', f'run_{tag}.log'))
    Progress.setup(os.path.join(root, 'progress', f'{tag}.json'))
    Progress.update(force=True, status='loading data', epoch='', done=0, total=0, finished=False)
    log("=" * 80)
    log(f"REVISED PIPELINE | mode={args.mode} group={args.group} shard={args.shard} device={base.device}")
    log(f"python {platform.python_version()} torch {torch.__version__} "
        f"gpu={torch.cuda.get_device_name(base.device) if 'cuda' in base.device else 'none'}")
    log("=" * 80)
    save_json(asdict(base), os.path.join(root, 'base_config.json'))

    data = PathologyData(base, log)
    save_json(data.summary(), os.path.join(root, 'dataset_summary.json'))
    vols_np, ct_ids, ct_info = load_ct_volumes(base, os.path.join(root, 'cache'), log)
    save_json({**ct_info, 'ids': ct_ids}, os.path.join(root, 'ct_summary.json'))
    vols = torch.from_numpy(vols_np).unsqueeze(1).to(base.device)
    images = data.images.to(base.device)
    labels = torch.from_numpy(data.labels).long().to(base.device)
    ch_images: Dict[Optional[str], torch.Tensor] = {None: images}

    si, sn = (int(v) for v in args.shard.split('/'))
    n_folds = len(planned_folds(data.patient_list, args.mode == 'smoke'))
    runs = build_runs(args.group, data.patient_list, args.mode == 'smoke')
    if args.only:
        runs = [r for r in runs if r[0] in set(args.only.split(','))]
    all_runs = runs
    runs = [r for i, r in enumerate(runs) if i % sn == si]
    todo = [r for r in runs if not os.path.exists(os.path.join(run_dir(root, r[0], r[1], r[2]), 'result.json'))]
    log(f"{len(runs)} runs in this shard, {len(runs) - len(todo)} already done, {len(todo)} to do")

    Progress.update(force=True, total=len(runs), done=len(runs) - len(todo), status='starting')
    if si == 0 and args.group in ('A', 'all'):
        run_trivial_baseline(data, root, log, args.mode == 'smoke')

    # Split the SSL pretraining jobs of the whole group across workers up front, so no GPU
    # sits idle waiting for an encoder another worker is training.
    smoke = args.mode == 'smoke'
    ssl_jobs = {}
    for cond, seed, *_ in all_runs:
        c = condition_config_for(base, cond, seed, smoke)
        if c.ct_bank != 'random_init':
            ssl_jobs.setdefault((c.ssl_temperature, c.seed, c.ssl_max_epochs), c)
    pending = sorted((k for k, c in ssl_jobs.items() if not os.path.exists(
        os.path.join(ssl_dir(root, *k), f'encoder_epoch{c.ssl_max_epochs}.pt'))), key=lambda k: (-k[2], k))
    load, owner = [0] * sn, {}
    for k in pending:                      # greedy balance by SSL epochs (same result in every worker)
        w = load.index(min(load))
        owner[k], load[w] = w, load[w] + k[2]
    mine = [k for k in pending if owner[k] == si]
    log(f"SSL encoders: {len(ssl_jobs)} needed, {len(pending)} missing, this worker pretrains {mine}")
    for j, k in enumerate(mine):
        Progress.update(force=True, status=f"SSL pretraining {j + 1}/{len(mine)}", epoch='')
        c = ssl_jobs[k]
        get_ssl_encoder_state(replace(c, ssl_epochs=c.ssl_max_epochs), vols, root, log)

    banks = BankFactory(vols, root, log)
    t_start, done = time.time(), 0
    for cond, seed, fold, test_p, val_p in todo:
        t0 = time.time()
        cfg = condition_config_for(base, cond, seed, args.mode == 'smoke')
        te = np.where(data.patients == test_p)[0]
        va = np.where(data.patients == val_p)[0]
        tr = np.where((data.patients != test_p) & (data.patients != val_p))[0]
        if cfg.drop_channel not in ch_images:
            x = images.clone()
            x[:, list(cfg.pathology_channels).index(cfg.drop_channel)] = 0
            ch_images[cfg.drop_channel] = x
        imgs = ch_images[cfg.drop_channel]
        Progress.update(force=True, status=f"{cond} s{seed} f{fold}", epoch='building CT bank')
        bank = banks.get(cfg)
        run_seed = seed * 1000 + fold
        log(f"RUN {cond} seed={seed} fold={fold} test={test_p} val={val_p} "
            f"train/val/test={len(tr)}/{len(va)}/{len(te)} K={cfg.k_shot} ssl=({cfg.ssl_epochs},{cfg.ssl_temperature},{cfg.ssl_max_epochs})")

        model, info = train_fold(cfg, imgs, labels, bank, tr, va, run_seed, log)
        ev = forward_eval(model, imgs[te], bank, cfg, torch.Generator().manual_seed(999))
        y_te = data.labels[te]
        m = binary_metrics(y_te, ev['prob'].numpy())
        records = []
        for j, i in enumerate(te):
            rec = {'name': data.names[i], 'patient': test_p, 'y': int(y_te[j]), 'prob': float(ev['prob'][j]),
                   'topk': ev['idx'][j].tolist(), 'margin_cls': float(ev['m_cls'][j])}
            if 'm_glob' in ev:
                rec['margin_glob'] = float(ev['m_glob'][j])
            if 'attn' in ev:
                rec['attn'] = [round(float(a), 5) for a in ev['attn'][j]]
                rec['gate_mean'] = float(ev['gate'][j])
            records.append(rec)
        result = {'condition': cond, 'seed': seed, 'fold': fold, 'n_folds': n_folds, 'test_patient': test_p, 'val_patient': val_p,
                  'sizes': {'train': len(tr), 'val': len(va), 'test': len(te)},
                  'class_counts': {'train': np.bincount(data.labels[tr], minlength=2).tolist(),
                                   'val': np.bincount(data.labels[va], minlength=2).tolist(),
                                   'test': np.bincount(y_te, minlength=2).tolist()},
                  'overrides': CONDITIONS[cond], 'config': asdict(cfg), 'test_metrics': m,
                  'records': records, **info}

        d = run_dir(root, cond, seed, fold)
        if cond in EXTRAS_CONDITIONS:
            # test-time bank swap: same trained model, context replaced (R1.2)
            swaps = {}
            for alt in ('random_init', 'gaussian', 'dim_shuffle', 'mean'):
                acfg = replace(cfg, ct_bank=alt)
                e2 = forward_eval(model, imgs[te], banks.get(acfg), cfg, torch.Generator().manual_seed(999))
                swaps[alt] = e2['prob'].tolist()
            e2 = forward_eval(model, imgs[te], bank, replace(cfg, retrieval_mode='random'),
                              torch.Generator().manual_seed(999))
            swaps['random_retrieval'] = e2['prob'].tolist()
            result['bank_swap_probs'] = swaps
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, 'features.npz'), path=ev['f'].numpy(), aligned=ev['a'].numpy(),
                                context=ev['r'].numpy(), y=y_te, names=np.array([data.names[i] for i in te]))
            bank_path = os.path.join(root, 'banks', f'{cond}_seed{seed}.pt')
            if not os.path.exists(bank_path):
                os.makedirs(os.path.dirname(bank_path), exist_ok=True)
                torch.save(bank, bank_path)
        result['time_s'] = time.time() - t0
        save_json(result, os.path.join(d, 'result.json'))
        done += 1
        Progress.update(force=True, done=len(runs) - len(todo) + done, epoch='',
                        mean_run_s=(time.time() - t_start) / done)
        el = time.time() - t_start
        log(f"  -> acc={m['accuracy']:.3f} bal_acc={m['balanced_accuracy']:.3f} auc={m['auc_roc']:.3f} "
            f"best_epoch={info['best_epoch']} ({result['time_s']:.0f}s) | {done}/{len(todo)} done, "
            f"ETA {el / done * (len(todo) - done) / 60:.1f} min")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log(f"SHARD FINISHED in {(time.time() - t_start) / 60:.1f} min")
    Progress.update(force=True, status='finished', epoch='', finished=True)


# =============================================================================
# AGGREGATION (tables, figures, statistics for the paper)
# =============================================================================
KEY_METRICS = ['accuracy', 'balanced_accuracy', 'f1_score_macro', 'auc_roc', 'cohen_kappa', 'mcc',
               'sensitivity', 'specificity', 'ece', 'brier']


def _fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.stats import rankdata
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fast_bal_acc(y: np.ndarray, p: np.ndarray) -> float:
    pred = p >= 0.5
    t1, t0 = y == 1, y == 0
    if not t1.any() or not t0.any():
        return float('nan')
    return float(0.5 * (pred[t1].mean() + (~pred[t0]).mean()))


def _latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    cols = list(df.columns)
    lines = ['\\begin{table}[t]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
             '\\resizebox{\\columnwidth}{!}{%', '\\begin{tabular}{@{}' + 'l' * 1 + 'c' * (len(cols) - 1) + '@{}}',
             '\\toprule', ' & '.join(str(c).replace('_', '\\_').replace('%', '\\%') for c in cols) + ' \\\\', '\\midrule']
    for _, row in df.iterrows():
        lines.append(' & '.join(str(v).replace('_', '\\_').replace('%', '\\%') for v in row.values) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}%', '}', '\\end{table}']
    return '\n'.join(lines)


def _write_table(df: pd.DataFrame, out: str, name: str, caption: str) -> None:
    df.to_csv(os.path.join(out, 'tables', f'{name}.csv'), index=False)
    with open(os.path.join(out, 'tables', f'{name}.tex'), 'w', encoding='utf-8') as f:
        f.write(_latex_table(df, caption, f'tab:{name}'))


def _fmt(m, s=None, pct=False, nd=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return '--'
    if pct:
        return f"{100 * m:.1f}" + (f" $\\pm$ {100 * s:.1f}" if s is not None and not np.isnan(s) else '')
    return f"{m:.{nd}f}" + (f" $\\pm$ {s:.{nd}f}" if s is not None and not np.isnan(s) else '')


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, 'fro') ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, 'fro') * np.linalg.norm(Y.T @ Y, 'fro') + 1e-12))


def aggregate(roots: List[str], out: str) -> None:
    os.makedirs(os.path.join(out, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    log = Logger(os.path.join(out, 'aggregate.log'))
    summary: Dict[str, Any] = {'roots': roots}

    results, seen = {}, set()
    for root in roots:
        for dp, _, fs in os.walk(os.path.join(root, 'runs')):
            if 'result.json' in fs:
                with open(os.path.join(dp, 'result.json')) as f:
                    r = json.load(f)
                key = (r['condition'], r['seed'], r['fold'])
                if key not in seen:
                    seen.add(key)
                    r['_dir'] = dp
                    results[key] = r
    ds = None
    for root in roots:
        p = os.path.join(root, 'dataset_summary.json')
        if os.path.exists(p):
            ds = json.load(open(p)); break
    summary['dataset'] = ds
    for root in roots:
        p = os.path.join(root, 'ct_summary.json')
        if os.path.exists(p):
            summary['ct'] = {k: v for k, v in json.load(open(p)).items() if k != 'ids'}; break
    n_folds = ds['n_patients'] if ds else max(k[2] for k in results) + 1
    patients = ds['patients'] if ds else None

    # ---- pooled out-of-fold records per (condition, seed); only complete sets ----
    by_cs = defaultdict(dict)
    for (c, s, f), r in results.items():
        by_cs[(c, s)][f] = r
    complete, incomplete = {}, {}
    for cs, folds in by_cs.items():
        need = next(iter(folds.values())).get('n_folds', n_folds)
        (complete if len(folds) == need else incomplete)[cs] = folds
    summary['incomplete_condition_seeds'] = {f'{c}/seed{s}': len(v) for (c, s), v in incomplete.items()}
    if incomplete:
        log(f"WARNING incomplete (excluded): {summary['incomplete_condition_seeds']}")
    oof = {cs: pd.DataFrame([rec for f in sorted(folds) for rec in folds[f]['records']])
           for cs, folds in complete.items()}
    conds = sorted({c for c, _ in oof}, key=lambda c: list(CONDITIONS).index(c) if c in CONDITIONS else -1)

    rows = []
    for (c, s), df in oof.items():
        m = binary_metrics(df['y'].values, df['prob'].values)
        per_pat = df.groupby('patient').apply(lambda d: _fast_auc(d['y'].values, d['prob'].values))
        folds = complete[(c, s)]
        rows.append({'condition': c, 'seed': s, **{k: m[k] for k in KEY_METRICS},
                     'mean_per_patient_auc': float(np.nanmean(per_pat.values)),
                     'mean_best_epoch': float(np.mean([r.get('best_epoch', np.nan) for r in folds.values()])),
                     'total_time_min': float(np.nansum([r.get('time_s', np.nan) for r in folds.values()]) / 60),
                     'confusion_matrix': m['confusion_matrix']})
    per_seed = pd.DataFrame(rows).sort_values(['condition', 'seed'])
    per_seed.to_csv(os.path.join(out, 'tables', 'per_condition_seed_metrics.csv'), index=False)

    # ---- patient-level cluster bootstrap (seed-averaged metric) and paired deltas vs main ----
    rng = np.random.default_rng(0)
    B = 2000
    pats = sorted({p for df in oof.values() for p in df['patient']}, key=_case_key)
    draws = [rng.choice(pats, size=len(pats), replace=True) for _ in range(B)]
    arrays = {cs: {p: (d['y'].values.astype(int), d['prob'].values) for p, d in df.groupby('patient')}
              for cs, df in oof.items()}

    def boot(c, seeds):
        """Seed-averaged (bal. acc., AUC) for each patient-level bootstrap draw."""
        vals = np.full((B, 2), np.nan)
        for b, dr in enumerate(draws):
            accs, aucs = [], []
            for s in seeds:
                parts = [arrays[(c, s)][p] for p in dr if p in arrays[(c, s)]]
                if not parts:
                    continue
                y = np.concatenate([a for a, _ in parts]); p_ = np.concatenate([q for _, q in parts])
                accs.append(_fast_bal_acc(y, p_)); aucs.append(_fast_auc(y, p_))
            if accs:
                vals[b] = [np.nanmean(accs), np.nanmean(aucs)]
        return vals

    seeds_of = {c: sorted(s for cc, s in oof if cc == c) for c in conds}
    agg_rows = []
    for c in conds:
        sub = per_seed[per_seed.condition == c]
        row = {'condition': c, 'n_seeds': len(sub)}
        for k in KEY_METRICS + ['mean_per_patient_auc']:
            row[k + '_mean'] = float(sub[k].mean())
            row[k + '_sd'] = float(sub[k].std(ddof=1)) if len(sub) > 1 else float('nan')
        v = boot(c, seeds_of[c])
        row['bal_acc_ci'] = np.nanpercentile(v[:, 0], [2.5, 97.5]).tolist()
        row['auc_ci'] = np.nanpercentile(v[:, 1], [2.5, 97.5]).tolist()
        shared = sorted(set(seeds_of[c]) & set(seeds_of.get('main', [])))
        if c != 'main' and shared:
            # paired comparison with the full model on the shared seeds and the same patient draws
            dv = boot(c, shared) - boot('main', shared)
            pc = per_seed[(per_seed.condition == c) & per_seed.seed.isin(shared)][['balanced_accuracy', 'auc_roc']].mean().values
            pm = per_seed[(per_seed.condition == 'main') & per_seed.seed.isin(shared)][['balanced_accuracy', 'auc_roc']].mean().values
            row['delta_seeds'] = shared
            row['delta_bal_acc'] = float(pc[0] - pm[0])
            row['delta_bal_acc_ci'] = np.nanpercentile(dv[:, 0], [2.5, 97.5]).tolist()
            row['delta_auc'] = float(pc[1] - pm[1])
            row['delta_auc_ci'] = np.nanpercentile(dv[:, 1], [2.5, 97.5]).tolist()
            row['p_boot_delta_auc_ge0'] = float(np.nanmean(dv[:, 1] >= 0))
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(os.path.join(out, 'tables', 'condition_summary.csv'), index=False)
    summary['conditions'] = agg.to_dict(orient='records')
    A = {r['condition']: r for r in agg_rows}

    def trow(c, label):
        r = A[c]
        return {'Model': label, 'Seeds': r['n_seeds'],
                'Acc. (%)': _fmt(r['accuracy_mean'], r['accuracy_sd'], True),
                'Bal. acc. (%)': _fmt(r['balanced_accuracy_mean'], r['balanced_accuracy_sd'], True),
                'Bal. acc. 95% CI': f"[{100 * r['bal_acc_ci'][0]:.1f}, {100 * r['bal_acc_ci'][1]:.1f}]",
                'AUC': _fmt(r['auc_roc_mean'], r['auc_roc_sd']),
                'AUC 95% CI': f"[{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}]",
                'F1': _fmt(r['f1_score_macro_mean'], r['f1_score_macro_sd']),
                'MCC': _fmt(r['mcc_mean'], r['mcc_sd']),
                'Sens. (%)': _fmt(r['sensitivity_mean'], None, True), 'Spec. (%)': _fmt(r['specificity_mean'], None, True),
                'ECE': _fmt(r['ece_mean']), 'Brier': _fmt(r['brier_mean'])}

    main_rows = [trow(c, l) for c, l in [('main', 'Full model (CT context)'), ('path_only', 'Pathology-only'),
                                         ('main_k8', 'Full model, K=8'), ('path_only_k8', 'Pathology-only, K=8'),
                                         ('trivial_channel_stats', 'Channel statistics + LR')] if c in A]
    if main_rows:
        _write_table(pd.DataFrame(main_rows), out, 'main_results',
                     'Leave-one-patient-out results (pooled out-of-fold predictions; mean $\\pm$ SD over seeds; '
                     '95\\% CI from a patient-level cluster bootstrap).')

    def drow(c, label):
        r = A[c]
        d = {'Condition': label,
             'Bal. acc. (%)': _fmt(r['balanced_accuracy_mean'], r['balanced_accuracy_sd'], True),
             'AUC': _fmt(r['auc_roc_mean'], r['auc_roc_sd'])}
        if 'delta_auc' in r:
            d['$\\Delta$ bal. acc. (pp) [95% CI]'] = f"{100 * r['delta_bal_acc']:+.1f} [{100 * r['delta_bal_acc_ci'][0]:+.1f}, {100 * r['delta_bal_acc_ci'][1]:+.1f}]"
            d['$\\Delta$ AUC [95% CI]'] = f"{r['delta_auc']:+.3f} [{r['delta_auc_ci'][0]:+.3f}, {r['delta_auc_ci'][1]:+.3f}]"
        return d

    ctrl = [('main', 'Full model (SSL bank, top-k)'), ('path_only', 'No CT context (pathology-only)'),
            ('bank_random_init', 'Bank: untrained CT encoder'), ('bank_gaussian', 'Bank: Gaussian noise'),
            ('bank_dim_shuffle', 'Bank: dimension-shuffled'), ('bank_mean', 'Bank: constant (mean)'),
            ('retrieval_random', 'Random retrieval'), ('ctr_random_pos', 'InfoNCE: random positives'),
            ('abl_free_gate', 'No gate floor / usage penalty'), ('abl_no_cdam', '$-$CDAM'),
            ('abl_no_adv', '$-$adversarial'), ('abl_no_ctr', '$-$InfoNCE'), ('abl_no_proto', '$-$prototypes')]
    rows_ = [drow(c, l) for c, l in ctrl if c in A]
    if rows_:
        _write_table(pd.DataFrame(rows_), out, 'controls_ablations',
                     'CT-bank controls and component ablations on identical folds and seeds; $\\Delta$ relative to the full model '
                     '(paired patient-level bootstrap).')
    drops = [drow(f'drop_{ch}', f'Drop {ch}') for ch in ('CD3', 'CD8', 'FoxP3', 'PanCK', 'DAPI') if f'drop_{ch}' in A]
    if drops:
        _write_table(pd.DataFrame(drops), out, 'channel_drop', 'Channel-drop analysis (seed 0).')

    # ---- test-time bank swap on the trained full models ----
    swap_rows = []
    for (c, s), folds in complete.items():
        if c != 'main' or 'bank_swap_probs' not in folds[0]:
            continue
        y = np.concatenate([[rec['y'] for rec in folds[f]['records']] for f in sorted(folds)])
        base_p = np.concatenate([[rec['prob'] for rec in folds[f]['records']] for f in sorted(folds)])
        swap_rows.append({'seed': s, 'bank': 'ssl (as trained)', 'bal_acc': _fast_bal_acc(y, base_p), 'auc': _fast_auc(y, base_p)})
        for alt in folds[0]['bank_swap_probs']:
            p = np.concatenate([folds[f]['bank_swap_probs'][alt] for f in sorted(folds)])
            swap_rows.append({'seed': s, 'bank': alt, 'bal_acc': _fast_bal_acc(y, p), 'auc': _fast_auc(y, p),
                              'mean_abs_prob_change': float(np.mean(np.abs(p - base_p)))})
    if swap_rows:
        sw = pd.DataFrame(swap_rows)
        sw.to_csv(os.path.join(out, 'tables', 'bank_swap_per_seed.csv'), index=False)
        g = sw.groupby('bank', sort=False).agg(['mean', 'std'])
        tab = pd.DataFrame({'Evaluation bank': g.index,
                            'Bal. acc. (%)': [_fmt(g.loc[b, ('bal_acc', 'mean')], g.loc[b, ('bal_acc', 'std')], True) for b in g.index],
                            'AUC': [_fmt(g.loc[b, ('auc', 'mean')], g.loc[b, ('auc', 'std')]) for b in g.index],
                            'Mean |$\\Delta p$|': [_fmt(g.loc[b, ('mean_abs_prob_change', 'mean')]) if ('mean_abs_prob_change', 'mean') in g.columns else '--' for b in g.index]})
        _write_table(tab, out, 'bank_swap', 'Test-time replacement of the CT bank for models trained with the SSL bank.')
        summary['bank_swap'] = sw.to_dict(orient='records')

    # ---- per-patient results for the full model ----
    if any(c == 'main' for c, _ in oof):
        pp = []
        for (c, s), df in oof.items():
            if c != 'main':
                continue
            for p, d in df.groupby('patient'):
                pp.append({'patient': p, 'seed': s, 'n': len(d), 'n_tumor': int(d['y'].sum()),
                           'bal_acc': _fast_bal_acc(d['y'].values, d['prob'].values),
                           'auc': _fast_auc(d['y'].values, d['prob'].values)})
        pp = pd.DataFrame(pp)
        g = pp.groupby('patient').agg({'n': 'first', 'n_tumor': 'first', 'bal_acc': ['mean', 'std'], 'auc': ['mean', 'std']})
        tab = pd.DataFrame({'Patient': g.index, 'Tiles (T/S)': [f"{g.loc[p, ('n', 'first')]} ({g.loc[p, ('n_tumor', 'first')]}/{g.loc[p, ('n', 'first')] - g.loc[p, ('n_tumor', 'first')]})" for p in g.index],
                            'Bal. acc. (%)': [_fmt(g.loc[p, ('bal_acc', 'mean')], g.loc[p, ('bal_acc', 'std')], True) for p in g.index],
                            'AUC': [_fmt(g.loc[p, ('auc', 'mean')], g.loc[p, ('auc', 'std')]) for p in g.index]})
        tab = tab.iloc[sorted(range(len(tab)), key=lambda i: _case_key(tab['Patient'].iloc[i]))]
        _write_table(tab, out, 'per_patient', 'Per-patient (held-out) results of the full model, mean $\\pm$ SD over seeds.')

    def guarded(name, fn):
        """Each diagnostic/figure is optional: a failure is logged, never fatal."""
        try:
            return fn()
        except Exception as e:
            import traceback
            log(f"WARNING {name} failed: {e}\n{traceback.format_exc()}")

    # ---- retrieval / alignment diagnostics (R1.2) ----
    summary['retrieval_alignment_diagnostics'] = guarded('retrieval diagnostics',
                                                         lambda: retrieval_diagnostics(complete, roots, out, log))
    # ---- SSL convergence + sensitivity (R1.3) ----
    summary['ssl'] = guarded('SSL report', lambda: ssl_report(roots, per_seed, out, log))

    # ---- calibration figure ----
    if any(c == 'main' for c, _ in oof):
        df = pd.concat([d for (c, _), d in oof.items() if c == 'main'])
        guarded('reliability figure', lambda: reliability_figure(
            df['y'].values, df['prob'].values, os.path.join(out, 'figures', 'reliability_main.png')))

    # ---- forest plot of deltas (CI drawn as a segment; the point may sit outside a skewed CI) ----
    def forest():
        fr = [r for r in agg_rows if 'delta_auc' in r]
        if not fr:
            return
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(fr) + 1))
        for i, r in enumerate(fr):
            ax.hlines(i, r['delta_auc_ci'][0], r['delta_auc_ci'][1], color='#1f4e79', lw=1.5)
            ax.plot(r['delta_auc'], i, 'o', color='#1f4e79')
        ax.axvline(0, color='grey', lw=0.8, ls='--')
        ax.set_yticks(range(len(fr))); ax.set_yticklabels([r['condition'] for r in fr])
        ax.set_xlabel('$\\Delta$ AUC vs full model (95% patient-bootstrap CI)')
        fig.savefig(os.path.join(out, 'figures', 'forest_delta_auc.png'), bbox_inches='tight'); plt.close(fig)
    guarded('forest plot', forest)

    try:
        paper_extras(roots, complete, oof, per_seed, ds, summary, out, log)
    except Exception as e:                     # never lose the main tables because of a figure
        import traceback
        log(f"WARNING paper_extras failed: {e}\n{traceback.format_exc()}")
    save_json(summary, os.path.join(out, 'summary.json'))
    log(f"Aggregation written to {out}")


def paper_extras(roots, complete, oof, per_seed, ds, summary, out, log) -> None:
    """Dataset/label-audit/hyper-parameter/confusion/compute tables and ROC, t-SNE and
    training-curve figures for the revised manuscript and the response letter."""
    T, Fg = os.path.join(out, 'tables'), os.path.join(out, 'figures')
    # ---- dataset table (R2.2, R2.3) ----
    if ds:
        rows = []
        for p, d in ds['tiles_per_patient_by_roi_type'].items():
            used = ds['used_per_patient_class_counts'].get(p, [0, 0])
            rows.append({'Patient': p, 'Tumour core (T)': d.get('T', 0), 'Margin (M)': d.get('M', 0),
                         'Stroma (S)': d.get('S', 0), 'All tiles': sum(d.values()),
                         'Used: tumour / non-tumour': f"{used[1]} / {used[0]}"})
        tot = {k: sum(r[k] for r in rows) for k in ('Tumour core (T)', 'Margin (M)', 'Stroma (S)', 'All tiles')}
        cc = ds['used_class_counts']
        rows.append({'Patient': 'Total', **tot,
                     'Used: tumour / non-tumour': f"{cc.get('1', cc.get(1, 0))} / {cc.get('0', cc.get(0, 0))}"})
        _write_table(pd.DataFrame(rows), out, 'dataset_patients',
                     'HNSCC-mIF-mIHC tiles per patient and ROI type; the task uses T (tumour) vs S (stroma).')
        # label audit (R2.5): ROI type vs the original nuclei-mask threshold label
        x = ds['roi_type_x_legacy_label']
        audit = [{'ROI type': t, 'Old label = non-tumour': x.get(f'{t}_0', 0), 'Old label = tumour': x.get(f'{t}_1', 0)}
                 for t in ('T', 'M', 'S')]
        _write_table(pd.DataFrame(audit), out, 'label_audit',
                     'Original mask-ratio label (>5\\% nuclei-mask pixels) versus the pathologist ROI type.')
        ch = [{'Channel statistic': k.replace('_', ' '), 'AUC (T vs S label)': f"{v['task_label']:.3f}",
               'AUC (original label)': f"{v['legacy_label']:.3f}"} for k, v in ds['single_channel_mean_auc'].items()]
        _write_table(pd.DataFrame(ch), out, 'single_channel_auc',
                     'Discrimination by a single channel mean intensity (no learning).')
    # ---- CT table ----
    if summary.get('ct'):
        ct = summary['ct']
        _write_table(pd.DataFrame([{'Volumes': ct['n_volumes'], 'Patients': ct['n_patients'],
                                    'Slices per series (min/max/mean)': '/'.join(f"{v:.0f}" for v in ct['slices_per_series_min_max_mean']),
                                    'Slice thickness (mm)': ', '.join(str(v) for v in ct['slice_thickness_values_mm']),
                                    'HU window': f"[{ct['hu_window'][0]:.0f}, {ct['hu_window'][1]:.0f}]",
                                    'Model input': f"{ct['depth_sampled']} x {ct['slice_size']} x {ct['slice_size']}"}]),
                     out, 'ct_summary', 'LungCT-Diagnosis source volumes as used.')
    # ---- hyper-parameters and conditions ----
    for root in roots:
        bp = os.path.join(root, 'base_config.json')
        if os.path.exists(bp):
            cfg = json.load(open(bp))
            skip = {'ct_data_dir', 'pathology_data_dir', 'output_dir', 'device', 'seed', 'max_patients',
                    'max_tiles_per_class_per_patient', 'ct_max_volumes', 'drop_channel'}
            hp = [{'Parameter': k, 'Value': str(v)} for k, v in cfg.items() if k not in skip]
            _write_table(pd.DataFrame(hp), out, 'hyperparameters', 'Complete configuration of the revised experiments.')
            break
    cond_rows = []
    for c, ov in CONDITIONS.items():
        seeds = sorted({s_ for cc, s_ in oof if cc == c})
        o = dict(ov)
        ssl = o.pop('ssl', MAIN_SSL)
        cond_rows.append({'Condition': c, 'Changes vs full model': ', '.join(f'{k}={v}' for k, v in o.items()) or '--',
                          'SSL (epochs, tau, run length)': str(ssl), 'Seeds completed': ','.join(map(str, seeds)) or '--'})
    _write_table(pd.DataFrame(cond_rows), out, 'conditions', 'Experimental conditions.')
    # ---- pooled confusion matrices ----
    cm_rows = []
    for c in ('main', 'path_only', 'main_k8', 'path_only_k8', 'trivial_channel_stats'):
        sub = per_seed[per_seed.condition == c]
        if len(sub):
            cm = np.sum([np.array(m) for m in sub['confusion_matrix']], axis=0)
            cm_rows.append({'Model': c, 'Seeds': len(sub), 'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
                            'FN': int(cm[1, 0]), 'TP': int(cm[1, 1])})
    if cm_rows:
        _write_table(pd.DataFrame(cm_rows), out, 'confusion_pooled',
                     'Out-of-fold confusion matrices summed over seeds (positive class = tumour core).')
    # ---- compute cost ----
    comp = per_seed.groupby('condition').agg(runs_min=('total_time_min', 'mean'), best_epoch=('mean_best_epoch', 'mean')).reset_index()
    comp.columns = ['Condition', 'Train+eval time per seed (min, 8 folds)', 'Mean selected epoch']
    comp = comp.round(2)
    _write_table(comp, out, 'compute_cost', 'Compute cost per condition (GPU minutes per seed).')
    # ---- ROC curves ----
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    for c, lab, col in (('main', 'Full model', '#1f4e79'), ('path_only', 'Pathology-only', '#c55a11'),
                        ('trivial_channel_stats', 'Channel stats + LR', '#7f7f7f')):
        dfs = [d for (cc, _), d in oof.items() if cc == c]
        if not dfs:
            continue
        d = pd.concat(dfs)
        fpr, tpr, _ = roc_curve(d['y'], d['prob'])
        ax.plot(fpr, tpr, color=col, lw=1.4, label=f"{lab} (AUC {_fast_auc(d['y'].values, d['prob'].values):.3f})")
    ax.plot([0, 1], [0, 1], ls='--', color='grey', lw=0.8)
    ax.set_xlabel('False-positive rate'); ax.set_ylabel('True-positive rate'); ax.legend(fontsize=7, loc='lower right')
    fig.savefig(os.path.join(Fg, 'roc_pooled.png'), bbox_inches='tight'); plt.close(fig)
    # ---- validation training curves (main vs pathology-only) ----
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for c, col in (('main', '#1f4e79'), ('path_only', '#c55a11')):
        H = [pd.DataFrame(r['history']) for (cc, _, _), r in
             ((k, v) for k, v in _flatten(complete).items()) if cc == c and r.get('history')]
        if not H:
            continue
        H = pd.concat(H).groupby('epoch').agg(['mean', 'count'])
        axes[0].plot(H.index, H[('train_loss', 'mean')], color=col, label=c)
        axes[1].plot(H.index, H[('val_bal_acc', 'mean')], color=col, label=c)
    axes[0].set_title('Training loss'); axes[1].set_title('Validation balanced accuracy')
    for a in axes:
        a.set_xlabel('Epoch'); a.legend(fontsize=7)
    fig.savefig(os.path.join(Fg, 'training_curves.png'), bbox_inches='tight'); plt.close(fig)
    # ---- t-SNE of CT bank vs pathology features (main, first seed) ----
    from sklearn.manifold import TSNE
    main_seeds = sorted(s_ for c, s_ in complete if c == 'main')
    if main_seeds:
        s0 = main_seeds[0]
        folds = complete[('main', s0)]
        feats = [np.load(os.path.join(folds[f]['_dir'], 'features.npz')) for f in sorted(folds)
                 if os.path.exists(os.path.join(folds[f]['_dir'], 'features.npz'))]
        bank = None
        for root in roots:
            bp = os.path.join(root, 'banks', f'main_seed{s0}.pt')
            if os.path.exists(bp):
                bank = torch.load(bp).numpy()
        if feats and bank is not None:
            y = np.concatenate([z['y'] for z in feats])
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
            for ax, key, title in ((axes[0], 'path', 'Pathology features (before fusion)'),
                                   (axes[1], 'aligned', 'Fused features (after CDAM)')):
                X = np.concatenate([bank, np.concatenate([z[key] for z in feats])])
                X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
                E = TSNE(n_components=2, perplexity=min(30, len(X) // 4), init='pca', random_state=0).fit_transform(X)
                nb = len(bank)
                ax.scatter(E[:nb, 0], E[:nb, 1], s=10, marker='^', c='#7f7f7f', label='CT bank (61 volumes)')
                ax.scatter(E[nb:][y == 1, 0], E[nb:][y == 1, 1], s=8, c='#c00000', label='Tumour core')
                ax.scatter(E[nb:][y == 0, 0], E[nb:][y == 0, 1], s=8, c='#2e75b6', label='Stroma')
                ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            axes[0].legend(fontsize=6, loc='best')
            fig.savefig(os.path.join(Fg, 'tsne_main.png'), bbox_inches='tight'); plt.close(fig)
    # ---- attention entropy / gate distribution ----
    recs = [rec for (c, _), folds in complete.items() if c == 'main' for f in folds for rec in folds[f]['records'] if 'attn' in rec]
    if recs:
        a = np.clip(np.array([r['attn'] for r in recs]), 1e-12, 1)
        ent = -(a * np.log(a)).sum(1) / np.log(a.shape[1])
        gate = np.array([r['gate_mean'] for r in recs])
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
        axes[0].hist(ent, bins=30, color='#1f4e79'); axes[0].set_xlabel('Attention entropy / log k (1 = uniform)')
        axes[1].hist(gate, bins=30, color='#1f4e79'); axes[1].set_xlabel('Mean transfer-gate value')
        fig.savefig(os.path.join(Fg, 'attention_gate_hist.png'), bbox_inches='tight'); plt.close(fig)
    log("Paper extras written (dataset, label audit, hyper-parameters, confusion, compute, ROC, t-SNE, curves).")


def _flatten(complete):
    return {(c, s_, f): r for (c, s_), folds in complete.items() for f, r in folds.items()}


def reliability_figure(y, p, path, n_bins=10):
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    corr = (pred == y).astype(float)
    bins = np.linspace(0.5, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf <= hi)
        if m.any():
            xs.append(conf[m].mean()); ys.append(corr[m].mean()); ns.append(m.sum())
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0.5, 1], [0.5, 1], '--', color='grey', lw=0.8)
    ax.plot(xs, ys, 'o-', color='#1f4e79')
    for x, yv, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, yv), fontsize=7, textcoords='offset points', xytext=(3, -8))
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(f'Held-out (OOF) ECE = {expected_calibration_error(y, p, n_bins):.3f}', fontsize=9)
    fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def retrieval_diagnostics(complete, roots, out, log) -> Dict[str, Any]:
    from scipy.stats import chi2_contingency
    from sklearn.metrics import mutual_info_score
    res = {'per_seed': []}
    topk_by_seed = {}
    for (c, s), folds in sorted(complete.items()):
        if c != 'main':
            continue
        recs = [rec for f in sorted(folds) for rec in folds[f]['records']]
        y = np.array([r['y'] for r in recs])
        topk = np.array([r['topk'] for r in recs])
        topk_by_seed[s] = {r['name']: set(r['topk']) for r in recs}
        n_bank = None
        bank = None
        for root in roots:
            bp = os.path.join(root, 'banks', f'main_seed{s}.pt')
            if os.path.exists(bp):
                bank = torch.load(bp).numpy(); n_bank = len(bank); break
        n_bank = n_bank or int(topk.max()) + 1
        counts = np.bincount(topk.ravel(), minlength=n_bank)
        pr = counts / counts.sum()
        ent = float(-(pr[pr > 0] * np.log(pr[pr > 0])).sum() / np.log(n_bank))
        top1 = topk[:, 0]
        mi = float(mutual_info_score(y, top1))
        rng = np.random.default_rng(s)
        null = np.array([mutual_info_score(rng.permutation(y), top1) for _ in range(1000)])
        tab = pd.crosstab(y, top1).values
        chi_p = float(chi2_contingency(tab)[1]) if tab.shape[1] > 1 else float('nan')
        attn = np.array([r['attn'] for r in recs]) if 'attn' in recs[0] else None
        row = {'seed': s, 'n_bank': n_bank, 'n_entries_ever_retrieved': int((counts > 0).sum()),
               'usage_entropy_normalised': ent, 'top5_hub_share': float(np.sort(pr)[::-1][:5].sum()),
               'label_top1_mutual_info': mi, 'label_top1_mi_perm_p': float((null >= mi).mean()),
               'label_top1_chi2_p': chi_p}
        if 'margin_glob' in recs[0]:
            mc = np.array([r['margin_cls'] for r in recs]); mg = np.array([r['margin_glob'] for r in recs])
            tot = mc + mg
            row['median_abs_margin_classifier'] = float(np.median(np.abs(mc)))
            row['median_abs_margin_global_proto'] = float(np.median(np.abs(mg)))
            row['decision_agreement_classifier_only'] = float(np.mean(np.sign(mc) == np.sign(tot)))
            row['decision_agreement_global_proto_only'] = float(np.mean(np.sign(mg) == np.sign(tot)))
        if attn is not None:
            a = np.clip(attn, 1e-12, 1)
            row['attention_entropy_over_logk'] = float((-(a * np.log(a)).sum(1) / np.log(a.shape[1])).mean())
            row['attention_max_mean'] = float(a.max(1).mean())
            row['gate_mean'] = float(np.mean([r['gate_mean'] for r in recs]))
        # features: linear probe on the retrieved CT context, domain separability, CKA
        feats = []
        for f in sorted(folds):
            fp = os.path.join(folds[f]['_dir'], 'features.npz')
            if os.path.exists(fp):
                feats.append(np.load(fp))
        if len(feats) == len(folds):
            ctx = np.concatenate([z['context'] for z in feats]); fy = np.concatenate([z['y'] for z in feats])
            fpath = np.concatenate([z['path'] for z in feats]); fal = np.concatenate([z['aligned'] for z in feats])
            pats = np.array([r['patient'] for r in recs])
            row['context_probe_auc'], row['context_probe_perm_p'] = _lopo_probe(ctx, fy, pats, rng)
            row['cka_path_vs_context'] = _linear_cka(fpath, ctx)
            row['cka_aligned_vs_context'] = _linear_cka(fal, ctx)
            if bank is not None:
                for nm, X in (('raw_path', fpath), ('aligned', fal)):
                    Z = np.concatenate([bank, X]); dy = np.r_[np.zeros(len(bank)), np.ones(len(X))]
                    pr_ = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced')),
                                            Z, dy, cv=StratifiedKFold(5, shuffle=True, random_state=0))
                    row[f'domain_clf_bal_acc_ct_vs_{nm}'] = float(balanced_accuracy_score(dy, pr_))
                    bn_ = bank / np.linalg.norm(bank, axis=1, keepdims=True); xn = X / np.linalg.norm(X, axis=1, keepdims=True)
                    row[f'linear_mmd_ct_vs_{nm}'] = float(np.sum((bn_.mean(0) - xn.mean(0)) ** 2))
        res['per_seed'].append(row)
        # usage figure (first seed)
        if s == min(ss for cc, ss in complete if cc == 'main'):
            fig, ax = plt.subplots(figsize=(6, 2.4))
            ax.bar(range(n_bank), counts, color='#1f4e79')
            ax.axhline(counts.sum() / n_bank, color='grey', ls='--', lw=0.8, label='uniform usage')
            ax.set_xlabel('CT bank entry (volume index)'); ax.set_ylabel('Times retrieved'); ax.legend(fontsize=7)
            fig.savefig(os.path.join(out, 'figures', 'bank_usage_main.png'), bbox_inches='tight'); plt.close(fig)
    seeds = sorted(topk_by_seed)
    if len(seeds) > 1:
        jac = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                A_, B_ = topk_by_seed[seeds[i]], topk_by_seed[seeds[j]]
                jac += [len(A_[n] & B_[n]) / len(A_[n] | B_[n]) for n in A_ if n in B_]
        res['retrieval_jaccard_across_seeds'] = float(np.mean(jac))
    if res['per_seed']:
        df = pd.DataFrame(res['per_seed'])
        df.to_csv(os.path.join(out, 'tables', 'retrieval_diagnostics_per_seed.csv'), index=False)
        m = df.drop(columns=['seed']).agg(['mean', 'std']).T
        tab = pd.DataFrame({'Diagnostic': m.index, 'Mean': [f"{v:.3f}" for v in m['mean']],
                            'SD': [f"{v:.3f}" if not np.isnan(v) else '--' for v in m['std']]})
        _write_table(tab, out, 'retrieval_diagnostics', 'Retrieval and alignment diagnostics for the full model (mean over seeds).')
    return res


def _lopo_probe(X, y, pats, rng, n_perm=100) -> Tuple[float, float]:
    """Leave-one-patient-out logistic probe AUC and a label-permutation p-value."""
    def auc_of(yy):
        p = np.zeros(len(yy))
        for pt in np.unique(pats):
            te = pats == pt
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
            clf.fit(X[~te], yy[~te])
            p[te] = clf.predict_proba(X[te])[:, 1]
        return _fast_auc(yy, p)
    obs = auc_of(y)
    null = np.array([auc_of(rng.permutation(y)) for _ in range(n_perm)])
    return float(obs), float((null >= obs).mean())


def ssl_report(roots, per_seed: pd.DataFrame, out, log) -> Dict[str, Any]:
    hists = {}
    for root in roots:
        d = os.path.join(root, 'ssl')
        if not os.path.isdir(d):
            continue
        for sub in os.listdir(d):
            hp = os.path.join(d, sub, 'history.json')
            if os.path.exists(hp) and sub not in hists:
                hists[sub] = json.load(open(hp))
    if not hists:
        return {}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    for sub, h in sorted(hists.items()):
        H = pd.DataFrame(h['history'])
        lab = f"$\\tau$={h['tau']}, seed {h['seed']}, {h['max_epochs']} ep"
        axes[0].plot(H['epoch'], H['loss'], lw=1, label=lab)
        Hm = H.dropna(subset=['view_top1']) if 'view_top1' in H else H.iloc[0:0]
        axes[1].plot(Hm['epoch'], Hm['view_top1'], marker='.', lw=1)
        axes[2].plot(Hm['epoch'], Hm['effective_rank'], marker='.', lw=1)
    axes[0].set_title('NT-Xent loss'); axes[1].set_title('View-retrieval top-1 (61 volumes)'); axes[2].set_title('Effective rank of bank')
    for a in axes:
        a.set_xlabel('Epoch')
    axes[0].legend(fontsize=6)
    fig.savefig(os.path.join(out, 'figures', 'ssl_convergence.png'), bbox_inches='tight'); plt.close(fig)

    rows = []
    for c, ov in CONDITIONS.items():
        if not c.startswith('ssl_') and c != 'main':
            continue
        e, tau, mx = ov.get('ssl', MAIN_SSL)
        sub = f'tau{tau}_seed0_max{mx}'
        intr = {}
        if sub in hists:
            H = [r for r in hists[sub]['history'] if r['epoch'] == e and 'view_top1' in r]
            if H:
                intr = H[0]
        down = per_seed[(per_seed.condition == c) & (per_seed.seed == 0)]
        rows.append({'Condition': c, 'SSL epochs': e, '$\\tau$': tau,
                     'Loss': _fmt(intr.get('loss')) if intr.get('loss') is not None else '--',
                     'View top-1': _fmt(intr.get('view_top1')), 'Eff. rank': _fmt(intr.get('effective_rank'), nd=1),
                     'Mean cos': _fmt(intr.get('mean_pairwise_cos')),
                     'Bal. acc. (%)': _fmt(down['balanced_accuracy'].iloc[0], None, True) if len(down) else '--',
                     'AUC': _fmt(down['auc_roc'].iloc[0]) if len(down) else '--'})
    tab = pd.DataFrame(rows)
    _write_table(tab, out, 'ssl_sensitivity', 'SSL pretraining sensitivity (seed 0): label-free representation '
                 'metrics at the snapshot and downstream leave-one-patient-out performance.')
    return {'histories': {k: v['history'] for k, v in hists.items()}, 'table': rows}


# =============================================================================
# CLI
# =============================================================================
def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and 'IPKernelApp' in ip.config
    except Exception:
        return False


def zip_outputs(paths: List[str], zip_path: str) -> None:
    """Zip results; skip the DICOM cache, lock/tmp files and SSL encoder weights (~25 MB each)."""
    import zipfile
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for base in paths:
            if not os.path.isdir(base):
                continue
            parent = os.path.dirname(os.path.abspath(base))
            for dp, dns, fns in os.walk(base):
                dns[:] = [d for d in dns if d != 'cache']
                for fn in fns:
                    if fn.endswith('.lock') or '.tmp' in fn or fn.startswith('encoder_epoch'):
                        continue
                    fp = os.path.join(dp, fn)
                    z.write(fp, os.path.relpath(fp, parent))


def launch(run: str, out: str, ct_dir: Optional[str] = None, path_dir: Optional[str] = None,
           aggregate_after: bool = True, zip_after: bool = True) -> None:
    """Start one worker process per GPU, show live progress, then aggregate + zip."""
    import subprocess
    if run not in ('full', 'A', 'B', 'C', 'smoke'):
        raise ValueError(f"RUN must be 'full', 'A', 'B', 'C' or 'smoke', got {run!r}")
    mode = 'smoke' if run == 'smoke' else 'full'
    group = 'all' if run in ('full', 'smoke') else run
    devices = [f'cuda:{i}' for i in range(torch.cuda.device_count()) for _ in range(WORKERS_PER_GPU)] or ['cpu']
    n = len(devices)
    names = [f"{d}#{devices[:i].count(d) + 1}" if devices.count(d) > 1 else d for i, d in enumerate(devices)]
    os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
    prog = [os.path.join(out, 'progress', f"{group}_shard{i}of{n}.json") for i in range(n)]
    for p in prog:
        if os.path.exists(p):
            os.remove(p)
    # no worker is alive yet, so any lock file is left over from an interrupted session
    for dp, _, fns in os.walk(out):
        for fn in fns:
            if fn.endswith('.lock'):
                os.remove(os.path.join(dp, fn))
    script = os.path.abspath(__file__)
    procs, handles = [], []
    for i, dev in enumerate(devices):
        cmd = [sys.executable, script, '--worker', '--mode', mode, '--group', group,
               '--shard', f'{i}/{n}', '--device', dev, '--out', out]
        if ct_dir:
            cmd += ['--ct_dir', ct_dir]
        if path_dir:
            cmd += ['--path_dir', path_dir]
        fh = open(os.path.join(out, 'logs', f'worker{i}_console.log'), 'a')
        handles.append(fh)
        procs.append(subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT))
    print(f"RUN={run!r}: {n} worker(s) on {devices}\nOutput: {out}\nDetailed logs: {out}/logs/", flush=True)

    notebook = _in_notebook()
    if notebook:
        from tqdm.auto import tqdm
        fmt = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} runs {postfix}'
        bars = [tqdm(total=1, desc=f'{d:>9}', bar_format=fmt) for d in names]
        total_bar = tqdm(total=1, desc='    TOTAL', bar_format=fmt)
    t0, first_done, t_first, last_print = time.time(), None, None, 0.0
    try:
        while True:
            codes = [pr.poll() for pr in procs]     # before reading progress, so the final state is shown
            states = []
            for p in prog:
                try:
                    with open(p) as f:
                        states.append(json.load(f))
                except Exception:
                    states.append({})
            tot = sum(st.get('total', 0) for st in states)
            dn = sum(st.get('done', 0) for st in states)
            if first_done is None and all(st.get('total') for st in states):
                first_done, t_first = dn, time.time()
            eta = ''
            if all(st.get('mean_run_s') for st in states):
                # workers run in parallel -> finish time is the slowest worker's remaining work
                rem = max((st.get('total', 0) - st.get('done', 0)) * st['mean_run_s'] for st in states)
                eta = f"| ~{np.mean([st['mean_run_s'] for st in states]) / 60:.1f} min/run | ETA {rem / 3600:.1f} h"
            elapsed = f"elapsed {(time.time() - t0) / 3600:.2f} h"
            lines = []
            for i, (st, c) in enumerate(zip(states, codes)):
                if c not in (None, 0):
                    note = 'CRASHED (see log)'
                else:
                    note = f"{st.get('status', 'starting')} {st.get('epoch', '')}".strip()
                if notebook:
                    bars[i].total, bars[i].n = max(st.get('total', 0), 1), st.get('done', 0)
                    bars[i].set_postfix_str(note, refresh=True)
                lines.append(f"   {names[i]}: {st.get('done', 0)}/{st.get('total', '?')} | {note}")
            if notebook:
                total_bar.total, total_bar.n = max(tot, 1), dn
                total_bar.set_postfix_str(f"{elapsed} {eta}", refresh=True)
            elif time.time() - last_print > 60 or all(c is not None for c in codes):
                last_print = time.time()
                print(f"[{datetime.now():%H:%M:%S}] {dn}/{tot} runs | {elapsed} {eta}\n" + '\n'.join(lines), flush=True)
            if all(c is not None for c in codes):
                break
            time.sleep(5)
    except KeyboardInterrupt:
        for pr in procs:
            pr.terminate()
        print("Interrupted: workers stopped. Finished runs are saved; run again to resume.")
        raise
    finally:
        if notebook:
            for b in bars + [total_bar]:
                b.close()
        for fh in handles:
            fh.close()

    failed = [i for i, pr in enumerate(procs) if pr.returncode != 0]
    for i in failed:
        print(f"\n!!! worker {i} ({names[i]}) exited with code {procs[i].returncode}. Last log lines:")
        with open(os.path.join(out, 'logs', f'worker{i}_console.log'), errors='replace') as f:
            print(''.join(f.readlines()[-25:]))
    print(f"\nAll workers stopped after {(time.time() - t0) / 3600:.2f} h "
          f"({'WITH FAILURES - fix and run again to resume' if failed else 'all OK'}).", flush=True)

    agg_out = out.rstrip('/\\') + '_aggregate'
    if aggregate_after:
        print("Aggregating results (tables, figures, statistics)...", flush=True)
        aggregate([out], agg_out)
    if zip_after:
        zip_path = out.rstrip('/\\') + f'_{run}_outputs.zip'
        zip_outputs([out, agg_out], zip_path)
        print(f"Download this file: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None, help="override RUN: full | A | B | C | smoke")
    ap.add_argument('--worker', action='store_true', help='internal: run one shard directly')
    ap.add_argument('--mode', choices=['full', 'smoke'], default='full')
    ap.add_argument('--group', choices=['A', 'B', 'C', 'all'], default=None)
    ap.add_argument('--shard', default='0/1', help='i/n: run every n-th run starting at i')
    ap.add_argument('--only', default=None, help='comma-separated condition names')
    ap.add_argument('--device', default=None)
    ap.add_argument('--ct_dir', default=None)
    ap.add_argument('--path_dir', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--aggregate', nargs='*', default=None, help='output roots to merge')
    ap.add_argument('--agg_out', default='./results_run/aggregate')
    args, _ = ap.parse_known_args()

    if args.aggregate is not None:
        aggregate(args.aggregate or [OUTPUT_ROOT], args.agg_out)
        return
    if args.list:
        for gname in ['A', 'B', 'C']:
            n = sum(len(seeds) for _, seeds in GROUPS[gname])
            print(f"group {gname}: {len(GROUPS[gname])} conditions, {n} condition-seeds, "
                  f"{n * 8} runs (8 LOPO folds): {[c for c, _ in GROUPS[gname]]}")
        return
    if not (args.worker or args.group):
        launch(args.run or RUN, args.out or OUTPUT_ROOT, args.ct_dir, args.path_dir, AGGREGATE_AFTER, ZIP_AFTER)
        return

    args.group = args.group or 'all'
    base = CFG
    if args.mode == 'smoke':
        base = smoke_config(base)
    base = replace(base, **{k: v for k, v in dict(ct_data_dir=args.ct_dir, pathology_data_dir=args.path_dir,
                                                  output_dir=args.out or OUTPUT_ROOT, device=args.device).items() if v})
    execute(args, base)


if __name__ == '__main__':
    main()
