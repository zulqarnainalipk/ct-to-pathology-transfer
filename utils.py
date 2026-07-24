"""Shared helpers: seeding, logging, confidence intervals, the InfoNCE loss,
CT-feature retrieval, grouped cross-validation splitting, fold manifests,
and optional-dependency availability flags used across the pipeline."""
import os
import random
import json
import re
import warnings
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    STRATIFIED_GROUP_KFOLD_AVAILABLE = True
except ImportError:
    STRATIFIED_GROUP_KFOLD_AVAILABLE = False

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from config import ResearchConfig

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def log_message(message: str, log_file: str = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_msg + "\n")
            f.flush()

def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 2000,
    bounded: bool = False,
    seed: int = 42
) -> Tuple[float, float, float]:
    values_arr = np.asarray(values, dtype=np.float64)
    values_arr = values_arr[~np.isnan(values_arr)]
    if len(values_arr) == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(values_arr))
    if len(values_arr) == 1:
        if bounded:
            mean = float(np.clip(mean, 0.0, 1.0))
        return (mean, mean, mean)

    rng = np.random.default_rng(seed)
    n = len(values_arr)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = values_arr[rng.integers(0, n, size=n)]
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    if bounded:
        mean = float(np.clip(mean, 0.0, 1.0))
        lower = float(np.clip(lower, 0.0, 1.0))
        upper = float(np.clip(upper, 0.0, 1.0))
    return (mean, lower, upper)


def info_nce_loss(
    features_2d: torch.Tensor,
    features_3d: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    features_2d = F.normalize(features_2d, dim=1)
    features_3d = F.normalize(features_3d, dim=1)
    logits = torch.matmul(features_2d, features_3d.T) / temperature
    labels = torch.arange(features_2d.size(0), device=features_2d.device)
    loss_2d_to_3d = F.cross_entropy(logits, labels)
    loss_3d_to_2d = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_2d_to_3d + loss_3d_to_2d)


def retrieve_nearest_ct_features(
    query_features: torch.Tensor,
    ct_features_bank: torch.Tensor,
    top_k: int = 5,
    temperature: float = 0.1,
    use_ann: bool = False,
    ann_num_probes: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bank = ct_features_bank.to(query_features.device)
    query_norm = F.normalize(query_features, dim=1)
    bank_norm = F.normalize(bank, dim=1)
    k = min(top_k, bank_norm.size(0))

    if use_ann and not FAISS_AVAILABLE:
        raise RuntimeError("ANN retrieval requested but FAISS is not installed. Disable use_ann_retrieval or install faiss.")

    if use_ann and FAISS_AVAILABLE:
        bank_np = bank_norm.detach().cpu().numpy().astype(np.float32)
        query_np = query_norm.detach().cpu().numpy().astype(np.float32)
        index = faiss.IndexFlatIP(bank_np.shape[1])
        if ann_num_probes > 0 and hasattr(index, 'nprobe'):
            index.nprobe = ann_num_probes
        index.add(bank_np)
        topk_values_np, topk_indices_np = index.search(query_np, k)
        topk_values = torch.from_numpy(topk_values_np).to(query_features.device)
        topk_indices = torch.from_numpy(topk_indices_np).long().to(query_features.device)
    else:
        similarity = torch.matmul(query_norm, bank_norm.T)
        topk_values, topk_indices = torch.topk(similarity, k=k, dim=1)

    neighbor_features = bank[topk_indices]
    weights = F.softmax(topk_values / temperature, dim=1).unsqueeze(-1)
    retrieved = torch.sum(neighbor_features * weights, dim=1)
    return retrieved, topk_indices, neighbor_features


def build_run_tag(prefix: str, fold_idx: int, k_shot: Optional[int] = None) -> str:
    tag = f"{prefix}_fold{fold_idx + 1}"
    if k_shot is not None:
        tag = f"{tag}_k{k_shot}"
    return tag


def compute_model_selection_score(metrics: Dict[str, float]) -> float:
    return float(
        0.50 * metrics.get('balanced_accuracy', 0.0) +
        0.30 * metrics.get('f1_score_macro', 0.0) +
        0.20 * metrics.get('accuracy', 0.0)
    )


def load_external_feature_bank(feature_path: str, expected_dim: int) -> torch.Tensor:
    if feature_path is None:
        raise ValueError("external_ct_feature_bank_path must be set for non-SSL CT feature sources.")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"External CT feature bank not found: {feature_path}")

    if feature_path.endswith('.npy'):
        features = torch.from_numpy(np.load(feature_path)).float()
    else:
        loaded = torch.load(feature_path, map_location='cpu', weights_only=False)
        if isinstance(loaded, dict):
            if 'features' in loaded:
                features = loaded['features'].float()
            elif 'ct_features' in loaded:
                features = loaded['ct_features'].float()
            else:
                raise KeyError("Could not find 'features' or 'ct_features' in external CT feature bank file.")
        elif isinstance(loaded, torch.Tensor):
            features = loaded.float()
        else:
            raise TypeError("Unsupported external feature format. Expected Tensor, .npy, or dict with features.")

    if features.ndim != 2 or features.size(1) != expected_dim:
        raise ValueError(
            f"External CT features must have shape [N, {expected_dim}], got {tuple(features.shape)}."
        )
    return features


def extract_patient_id_from_name(base_name: str) -> str:
    tokenized = re.split(r'[_\- ]+', base_name)
    if len(tokenized) == 0:
        return base_name
    for token in tokenized:
        if re.search(r'\d', token):
            return token
    return tokenized[0]


def build_grouped_split(
    labels: List[int],
    groups: List[str],
    n_splits: int,
    seed: int
) -> Tuple[Any, int]:
    class_counts = np.bincount(labels)
    max_splits = int(class_counts.min()) if len(class_counts) > 0 else n_splits
    effective_splits = max(2, min(n_splits, max_splits))
    if STRATIFIED_GROUP_KFOLD_AVAILABLE:
        splitter = StratifiedGroupKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=seed
        )
    else:
        warnings.warn("StratifiedGroupKFold not available; using StratifiedKFold without grouping.")
        splitter = StratifiedKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=seed
        )
    return splitter, effective_splits


def save_fold_manifest(
    output_dir: str,
    fold_idx: int,
    config: ResearchConfig,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
    train_groups: List[str],
    val_groups: List[str],
    test_groups: List[str]
) -> None:
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'fold': int(fold_idx),
        'seed': int(config.seed),
        'task_setting': config.task_setting,
        'ablation_flags': {
            'use_cdam': config.use_cdam,
            'use_domain_adversarial': config.use_domain_adversarial,
            'use_prototypical_branch': config.use_prototypical_branch,
            'use_contrastive_alignment': config.use_contrastive_alignment
        },
        'split_sizes': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx)
        },
        'unique_groups': {
            'train': len(set(train_groups)),
            'val': len(set(val_groups)),
            'test': len(set(test_groups))
        }
    }
    fold_dir = os.path.join(output_dir, 'analysis', f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)


def clear_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
