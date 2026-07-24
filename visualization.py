"""Publication-quality plotting: global matplotlib/seaborn style setup, the
VisualizationSuite (attention maps, embedding alignment, prototype drift,
domain confusion, etc.), and the cross-validation summary plot."""
import os
import warnings
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from config import ResearchConfig

def setup_research_style():
    rcParams.update({
        # --- Figure & saving ---
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'figure.constrained_layout.use': True,
        'savefig.pad_inches': 0.15,

        # --- Fonts (journal-safe serif) ---
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'font.size': 10,

        # --- Titles & labels ---
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,

        # --- Axes & grid ---
        'axes.autolimit_mode': 'round_numbers',
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,

        # --- Spines (clean academic look) ---
        'axes.spines.top': False,
        'axes.spines.right': False,

        # --- Lines & markers ---
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'patch.linewidth': 0.5,

        # --- Color cycle (colorblind & print friendly) ---
        'axes.prop_cycle': plt.cycler('color', [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf'   # cyan
        ]),
    })

    # --- Seaborn: light touch, compatible with papers ---
    sns.set_style("whitegrid", {
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.color': '0.85',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5
    })

    # --- Research-oriented colormaps ---
    research_cmaps = {
        'attention': LinearSegmentedColormap.from_list(
            'attention',
            ['#2E4057', '#048A81', '#54C6EB', '#8FC93A', '#F18F01']
        ),
        'alignment': LinearSegmentedColormap.from_list(
            'alignment',
            ['#2C003E', '#512B58', '#6B5B95', '#8B78E6', '#C77DFF']
        ),
        'uncertainty': LinearSegmentedColormap.from_list(
            'uncertainty',
            ['#D32F2F', '#F57C00', '#FBC02D', '#388E3C', '#1976D2']
        )
    }

    return research_cmaps


RESEARCH_CMAPS = setup_research_style()

class VisualizationSuite:
    """
    Research-grade visualization suite for publication-quality figures.
    Implements best practices for academic papers:
    - High DPI (300+)
    - Vector formats (PDF/SVG)
    - Colorblind-friendly palettes
    - Proper font scaling and serif fonts
    - Statistical rigor (CIs, calibration)
    """
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.style_setup = False
        self._init_directories()

    def _init_directories(self):
        for subdir in ['embeddings', 'attention', 'curves', 'calibration', 'features', 'summary']:
            os.makedirs(os.path.join(self.output_dir, 'plots', subdir), exist_ok=True)

    def _setup_style(self):
        if self.style_setup: return
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 14,
            'savefig.dpi': 300,
            'figure.autolayout': True
        })
        sns.set_context("paper")
        self.style_setup = True

    def _safe_layout(self, fig):
        try:
            fig.tight_layout()
        except Exception as e:
            warnings.warn(f"tight_layout failed: {e}")

    def plot_tsne_umap_alignment(self, ct_features: torch.Tensor, path_features: torch.Tensor, epoch: int, labels: Optional[np.ndarray] = None) -> None:
        self._setup_style()
        ct_np = ct_features.detach().cpu().numpy()
        path_np = path_features.detach().cpu().numpy()
        
        from sklearn.preprocessing import StandardScaler
        all_features = np.vstack([ct_np, path_np])
        all_features = StandardScaler().fit_transform(all_features)
        
        domain_labels = np.array(['3D CT'] * len(ct_np) + ['2D Pathology'] * len(path_np))
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        
        if UMAP_AVAILABLE:
            try:
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
                proj = reducer.fit_transform(all_features)
                title_prefix = "UMAP"
            except Exception as e:
                # UMAP/sklearn version mismatches can fail at runtime (e.g., ensure_all_finite API changes).
                print(f"[WARN] UMAP failed ({e}). Falling back to t-SNE.")
                tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
                proj = tsne.fit_transform(all_features)
                title_prefix = "t-SNE"
        else:
            print("[WARN] UMAP unavailable. Falling back to t-SNE.")
            tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
            proj = tsne.fit_transform(all_features)
            title_prefix = "t-SNE"

        ax = axes[0]
        colors = ['#4C72B0', '#DD8452']
        for i, domain in enumerate(['3D CT', '2D Pathology']):
            mask = domain_labels == domain
            ax.scatter(proj[mask, 0], proj[mask, 1], label=domain, alpha=0.6, s=25, 
                      color=colors[i], edgecolors='none')
        
        ax.set_title(f'{title_prefix}: Domain Alignment', fontweight='bold')
        ax.legend(frameon=True, loc='best', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        
        ax = axes[1]
        if labels is not None:
            if len(labels) == len(path_np):
                ct_placeholder = np.full(len(ct_np), -1)
                combined_labels = np.concatenate([ct_placeholder, labels])
                scatter = ax.scatter(
                    proj[:, 0], proj[:, 1], c=combined_labels,
                    cmap='viridis', alpha=0.6, s=25, edgecolors='none'
                )
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Class Label', size=8)
                ax.set_title(f'{title_prefix}: Latent Semantics', fontweight='bold')
            else:
                ax.scatter(proj[:, 0], proj[:, 1], alpha=0.6, s=25, edgecolors='none')
                ax.set_title(f'{title_prefix}: Projection', fontweight='bold')
        else:
            ax.scatter(proj[:, 0], proj[:, 1], alpha=0.6, s=25, edgecolors='none')
            ax.set_title(f'{title_prefix}: Projection', fontweight='bold')
        
        ax.set_xticks([]); ax.set_yticks([])
        
        plt.savefig(os.path.join(self.output_dir, 'plots', 'embeddings', f'alignment_ep{epoch:03d}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'embeddings', f'alignment_ep{epoch:03d}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    def plot_attention_heatmaps(self, attention_weights: torch.Tensor, ct_volume: torch.Tensor, path_image: torch.Tensor, epoch: int, sample_id: str) -> None:
        self._setup_style()
        att = attention_weights.detach().cpu().numpy()
        ct = ct_volume.detach().cpu().numpy().squeeze()
        path = path_image.detach().cpu().numpy().squeeze()

        if ct.ndim >= 3:
            mid = ct.shape[0] // 2
            ct_slice = ct[mid]
            base_map = ct_slice
            base_title = 'Input CT (Mid-slice)'
        elif ct.ndim == 2:
            base_map = ct
            base_title = 'CT Feature Map'
        else:
            flat = ct.reshape(-1)
            side = int(np.ceil(np.sqrt(len(flat))))
            padded = np.zeros(side * side, dtype=np.float32)
            padded[:len(flat)] = flat
            base_map = padded.reshape(side, side)
            base_title = 'CT Feature Grid'

        if path.ndim >= 3:
            path_map = path[0]
        elif path.ndim == 2:
            path_map = path
        else:
            pflat = path.reshape(-1)
            pside = int(np.ceil(np.sqrt(len(pflat))))
            ppadded = np.zeros(pside * pside, dtype=np.float32)
            ppadded[:len(pflat)] = pflat
            path_map = ppadded.reshape(pside, pside)
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(path_map, cmap='gray')
        axes[0].set_title('Pathology Query', fontsize=10)
        axes[0].axis('off')
        
        from scipy.ndimage import zoom
        if len(att.shape) == 1:
            side = int(np.sqrt(len(att)))
            att_2d = att.reshape(side, side)
        else:
            att_2d = att[0] if len(att.shape) > 2 else att
        if att_2d.ndim == 0:
            att_2d = np.array([[float(att_2d)]], dtype=np.float32)
        elif att_2d.ndim == 1:
            att_2d = att_2d.reshape(1, -1)

        scale_h = max(1e-6, base_map.shape[0] / att_2d.shape[0])
        scale_w = max(1e-6, base_map.shape[1] / att_2d.shape[1])
        att_resized = zoom(att_2d, (scale_h, scale_w), order=1)
        
        im = axes[1].imshow(att_resized, cmap='inferno')
        axes[1].set_title('Attention Map', fontsize=10)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        axes[2].imshow(base_map, cmap='gray')
        axes[2].imshow(att_resized, cmap='inferno', alpha=0.5)
        axes[2].set_title(f'{base_title} + Attention', fontsize=10)
        axes[2].axis('off')
        
        plt.savefig(os.path.join(self.output_dir, 'plots', 'attention', f'att_{sample_id}_ep{epoch:03d}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'attention', f'att_{sample_id}_ep{epoch:03d}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_training_curves(self, history: Dict[str, List[float]], epoch: int) -> None:
        self._setup_style()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        ax = axes[0]
        if 'train_loss' in history:
            ax.plot(history['train_loss'], label='Train', color='#4C72B0', lw=1.5)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Val', color='#DD8452', lw=1.5, ls='--')
        ax.set_title('Learning Convergence', fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax = axes[1]
        if 'train_acc' in history:
            ax.plot(history['train_acc'], label='Train', color='#4C72B0', lw=1.5)
        if 'val_acc' in history:
            ax.plot(history['val_acc'], label='Val', color='#DD8452', lw=1.5, ls='--')
        ax.set_title('Model Performance', fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.savefig(os.path.join(self.output_dir, 'plots', 'curves', f'curves_ep{epoch:03d}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'curves', f'curves_ep{epoch:03d}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_uncertainty_calibration(self, confidences: np.ndarray, predictions: np.ndarray, true_labels: np.ndarray, epoch: int) -> None:
        self._setup_style()
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(true_labels, confidences, n_bins=10)
        
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        ax.plot(prob_pred, prob_true, marker='.', label='Model', color='#4C72B0')
        
        ax.set_xlabel('Mean Predicted Confidence')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Reliability Diagram', fontweight='bold')
        ax.legend(loc='lower right', frameon=False)
        
        ece = np.mean(np.abs(prob_pred - prob_true))
        ax.text(0.05, 0.9, f'ECE: {ece:.4f}', transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.savefig(os.path.join(self.output_dir, 'plots', 'calibration', f'calib_ep{epoch:03d}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'calibration', f'calib_ep{epoch:03d}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_distributions(self, ct_features: torch.Tensor, path_features: torch.Tensor, aligned_features: torch.Tensor, epoch: int) -> None:
        self._setup_style()
        ct_flat = ct_features.detach().cpu().numpy().flatten()
        path_flat = path_features.detach().cpu().numpy().flatten()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.kdeplot(ct_flat, label='3D CT Features', fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(path_flat, label='2D Path Features', fill=True, alpha=0.3, ax=ax)
        
        ax.set_title('Feature Activation Distributions', fontweight='bold')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.legend(frameon=False)
        
        plt.savefig(os.path.join(self.output_dir, 'plots', 'features', f'dist_ep{epoch:03d}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'features', f'dist_ep{epoch:03d}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prototype_evolution(self, prototype_history: List[torch.Tensor], epoch: int) -> None:
        if len(prototype_history) < 2:
            return
        self._setup_style()

        history = torch.stack([p.detach().cpu() for p in prototype_history], dim=0)  # [T, C, D]
        steps = np.arange(history.size(0))
        n_classes = history.size(1)

        fig, ax = plt.subplots(figsize=(7, 4))
        for c in range(n_classes):
            proto = history[:, c, :]  # [T, D]
            deltas = torch.norm(proto[1:] - proto[:-1], dim=1).numpy()
            drift = np.concatenate([[0.0], deltas])
            ax.plot(steps, drift, marker='o', linewidth=1.5, label=f'Class {c}')

        ax.set_title('Prototype Evolution Drift', fontweight='bold')
        ax.set_xlabel('Checkpoint Index')
        ax.set_ylabel('L2 Drift')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'plots', 'summary', f'prototype_drift_ep{epoch:03d}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'summary', f'prototype_drift_ep{epoch:03d}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_summary(fold_results: List[Dict], aggregated: Dict, output_dir: str):
    """Research-grade summary plot with confidence intervals and statistical significance."""
    metrics = ['accuracy', 'balanced_accuracy', 'f1_score_macro', 'auc_roc', 'cohen_kappa']
    display_names = ['Accuracy', 'Bal. Accuracy', 'F1 (Macro)', 'AUC-ROC', 'Cohen\'s Kappa']
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))

    for idx, (metric, name) in enumerate(zip(metrics, display_names)):
        if metric in aggregated:
            values = aggregated[metric]['values']
            mean = aggregated[metric]['mean']
            ci_lower = aggregated[metric]['ci_lower']
            ci_upper = aggregated[metric]['ci_upper']

            ax = axes[idx]
            sns.violinplot(data=values, ax=ax, color='#A1C9F4', inner=None, alpha=0.4)
            sns.swarmplot(data=values, ax=ax, color='#4C72B0', size=4)

            ax.errorbar(0, mean, yerr=[[mean-ci_lower], [ci_upper-mean]], fmt='o', color='black', capsize=5, label='95% CI')

            ax.set_title(name, fontweight='bold', fontsize=11)
            ax.set_xticks([])
            if idx == 0: ax.set_ylabel('Score Value')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'summary_results.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, 'summary_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
