"""Phase 1-10 comprehensive analysis modules: classification metrics, dataset
statistics, the ablation study runner, clinical/operating-point metrics,
training dynamics, robustness testing, interpretability, feature/embedding
analysis, and LaTeX/HTML report generation."""
import os
import random
import json
from typing import List, Tuple, Dict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef, precision_score, recall_score,
    balanced_accuracy_score, average_precision_score, jaccard_score, hamming_loss
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats

from config import ResearchConfig
from datasets import CT3DVolumeDataset, PathologyMultiChannelDataset
from models import CrossDomainTransferModel

# =============================================================================
# COMPREHENSIVE ANALYSIS MODULES FOR 150+ PAGE RESEARCH REPORT
# =============================================================================

class ComprehensiveMetricsCalculator:
    """Phase 1: Core Metrics Expansion - Calculate all classification metrics."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.n_classes = len(np.unique(y_true))

    def calculate_all_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive set of classification metrics."""
        metrics = {}

        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(self.y_true, self.y_pred)
        metrics['f1_macro'] = f1_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        metrics['f1_micro'] = f1_score(self.y_true, self.y_pred, average='micro', zero_division=0)
        metrics['precision_macro'] = precision_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)

        if self.y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(self.y_true, self.y_prob[:, 1] if self.y_prob.ndim > 1 else self.y_prob)
                metrics['average_precision'] = average_precision_score(self.y_true, self.y_prob[:, 1] if self.y_prob.ndim > 1 else self.y_prob)
            except:
                metrics['auc_roc'] = 0.5
                metrics['average_precision'] = 0.5

        metrics['cohen_kappa'] = cohen_kappa_score(self.y_true, self.y_pred)
        metrics['mcc'] = matthews_corrcoef(self.y_true, self.y_pred)
        metrics['jaccard_macro'] = jaccard_score(self.y_true, self.y_pred, average='macro', zero_division=0)
        metrics['hamming_loss'] = hamming_loss(self.y_true, self.y_pred)

        # Clinical metrics
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        metrics['fdr'] = fp / (fp + tp) if (fp + tp) > 0 else 0  # False Discovery Rate

        # Advanced metrics
        metrics['youden_index'] = metrics['sensitivity'] + metrics['specificity'] - 1
        metrics['informedness'] = metrics['sensitivity'] + metrics['specificity'] - 1
        metrics['markedness'] = metrics['ppv'] + metrics['npv'] - 1

        # Likelihood ratios
        metrics['lr_positive'] = metrics['sensitivity'] / (metrics['fpr'] + 1e-8)
        metrics['lr_negative'] = (1 - metrics['sensitivity']) / (metrics['specificity'] + 1e-8)

        # Diagnostic Odds Ratio
        metrics['dor'] = (metrics['sensitivity'] * metrics['specificity']) / ((1 - metrics['sensitivity']) * (1 - metrics['specificity']) + 1e-8)

        return metrics

    def per_class_metrics(self) -> pd.DataFrame:
        """Calculate per-class precision, recall, and F1."""
        classes = np.unique(self.y_true)
        per_class = []
        for cls in classes:
            y_true_binary = (self.y_true == cls).astype(int)
            y_pred_binary = (self.y_pred == cls).astype(int)

            prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

            per_class.append({
                'class': cls,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'support': np.sum(y_true_binary)
            })
        return pd.DataFrame(per_class)


class StatisticalAnalysisSuite:
    """Phase 2: Statistical Analysis & Significance Testing."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def bootstrap_confidence_interval(self, data: np.ndarray, n_iterations: int = 1000, ci: float = 0.95) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_means = []
        n_samples = len(data)

        for _ in range(n_iterations):
            sample = np.random.choice(data, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)
        alpha = (1 - ci) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

        return np.mean(bootstrap_means), lower, upper

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    def cliffs_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(group1), len(group2)
        dominance = 0

        for x in group1:
            for y in group2:
                if x > y:
                    dominance += 1
                elif x < y:
                    dominance -= 1

        return dominance / (n1 * n2)

    def permutation_test(self, group1: np.ndarray, group2: np.ndarray, n_permutations: int = 10000) -> float:
        """Perform permutation test."""
        observed_diff = np.mean(group1) - np.mean(group2)
        combined = np.concatenate([group1, group2])
        n1 = len(group1)

        count = 0
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
            if abs(perm_diff) >= abs(observed_diff):
                count += 1

        return count / n_permutations

    def mcnemar_test(self, y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> Tuple[float, float]:
        """McNemar's test for comparing two classifiers."""
        from statsmodels.stats.contingency_tables import mcnemar

        correct1 = (pred1 == y_true)
        correct2 = (pred2 == y_true)

        # Create contingency table
        both_correct = np.sum(correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        first_correct = np.sum(correct1 & ~correct2)
        second_correct = np.sum(~correct1 & correct2)

        table = [[both_correct, first_correct], [second_correct, both_wrong]]
        result = mcnemar(table, exact=True)

        return result.statistic, result.pvalue

    def compare_models(self, transfer_results: Dict, baseline_results: Dict, metrics: List[str]) -> pd.DataFrame:
        """Comprehensive statistical comparison of models."""
        comparisons = []

        for metric in metrics:
            if metric in transfer_results and metric in baseline_results:
                transfer_vals = transfer_results[metric]
                baseline_vals = baseline_results[metric]

                # Paired t-test
                t_stat, t_pval = stats.ttest_rel(transfer_vals, baseline_vals)

                # Wilcoxon signed-rank test
                try:
                    w_stat, w_pval = stats.wilcoxon(transfer_vals, baseline_vals)
                except:
                    w_stat, w_pval = np.nan, np.nan

                # Effect sizes
                cohens = self.cohens_d(transfer_vals, baseline_vals)
                cliffs = self.cliffs_delta(transfer_vals, baseline_vals)

                # Bootstrap CI for difference
                diff = np.array(transfer_vals) - np.array(baseline_vals)
                mean_diff, ci_low, ci_high = self.bootstrap_confidence_interval(diff)

                comparisons.append({
                    'metric': metric,
                    'transfer_mean': np.mean(transfer_vals),
                    'baseline_mean': np.mean(baseline_vals),
                    'difference': mean_diff,
                    'ci_lower': ci_low,
                    'ci_upper': ci_high,
                    't_statistic': t_stat,
                    't_pvalue': t_pval,
                    'wilcoxon_statistic': w_stat,
                    'wilcoxon_pvalue': w_pval,
                    'cohens_d': cohens,
                    'cliffs_delta': cliffs,
                    'significant': t_pval < 0.05
                })

        return pd.DataFrame(comparisons)


class DatasetAnalyzer:
    """Phase 3: Advanced Data Analysis - Dataset characterization."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.analysis_dir = os.path.join(output_dir, 'analysis', 'dataset')
        os.makedirs(self.analysis_dir, exist_ok=True)

    def analyze_ct_dataset(self, ct_dataset: CT3DVolumeDataset) -> Dict:
        """Comprehensive CT dataset analysis."""
        stats = {
            'total_volumes': len(ct_dataset),
            'dimensions': [],
            'spacings': [],
            'intensity_stats': []
        }

        for i in range(len(ct_dataset)):
            volume, label, patient_id = ct_dataset[i]
            if isinstance(volume, torch.Tensor):
                volume = volume.numpy()

            stats['dimensions'].append(volume.shape)
            stats['intensity_stats'].append({
                'mean': np.mean(volume),
                'std': np.std(volume),
                'min': np.min(volume),
                'max': np.max(volume),
                'median': np.median(volume)
            })

        df_dims = pd.DataFrame(stats['dimensions'], columns=['depth', 'height', 'width'])
        stats['dimension_summary'] = df_dims.describe().to_dict()

        intensity_df = pd.DataFrame(stats['intensity_stats'])
        stats['intensity_summary'] = intensity_df.describe().to_dict()

        return stats

    def analyze_pathology_dataset(self, path_dataset: PathologyMultiChannelDataset) -> Dict:
        """Comprehensive pathology dataset analysis."""
        stats = {
            'total_samples': len(path_dataset),
            'channel_stats': {ch: [] for ch in path_dataset.channels},
            'tumor_ratios': [],
            'labels': [],
            'patient_distribution': defaultdict(int)
        }

        for i in range(len(path_dataset)):
            image, label, patient_id = path_dataset[i]
            if isinstance(image, torch.Tensor):
                image = image.numpy()

            stats['labels'].append(label)
            stats['patient_distribution'][patient_id] += 1

            for idx, ch in enumerate(path_dataset.channels):
                ch_data = image[idx] if image.ndim == 3 else image
                stats['channel_stats'][ch].append({
                    'mean': np.mean(ch_data),
                    'std': np.std(ch_data),
                    'min': np.min(ch_data),
                    'max': np.max(ch_data),
                    'p5': np.percentile(ch_data, 5),
                    'p95': np.percentile(ch_data, 95)
                })

        # Calculate tumor ratio if available
        stats['class_distribution'] = pd.Series(stats['labels']).value_counts().to_dict()

        return stats

    def plot_dataset_distributions(self, ct_stats: Dict, path_stats: Dict):
        """Generate comprehensive dataset visualization."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Comprehensive Dataset Analysis', fontsize=14, fontweight='bold')

        # CT dimensions
        if ct_stats.get('dimensions'):
            df_dims = pd.DataFrame(ct_stats['dimensions'], columns=['depth', 'height', 'width'])
            axes[0, 0].hist(df_dims['depth'], bins=20, alpha=0.7, label='Depth')
            axes[0, 0].hist(df_dims['height'], bins=20, alpha=0.7, label='Height')
            axes[0, 0].set_title('CT Volume Dimensions')
            axes[0, 0].legend()

        # Class distribution
        if path_stats.get('class_distribution'):
            classes = list(path_stats['class_distribution'].keys())
            counts = list(path_stats['class_distribution'].values())
            axes[0, 1].bar(classes, counts)
            axes[0, 1].set_title('Pathology Class Distribution')
            axes[0, 1].set_xlabel('Class')
            axes[0, 1].set_ylabel('Count')

        # Patient sample distribution
        if path_stats.get('patient_distribution'):
            patient_counts = list(path_stats['patient_distribution'].values())
            axes[0, 2].hist(patient_counts, bins=20)
            axes[0, 2].set_title('Samples per Patient')
            axes[0, 2].set_xlabel('Number of Samples')
            axes[0, 2].set_ylabel('Number of Patients')

        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'dataset_distributions.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.analysis_dir, 'dataset_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()


class AblationStudyRunner:
    """Phase 4: Ablation Studies - Systematic component evaluation."""

    def __init__(self, config: ResearchConfig):
        self.config = config
        self.results = {}

    def run_ablation_suite(self, ct_dataset, path_train, path_val, logger_file: str) -> Dict:
        """Run comprehensive ablation studies."""
        ablation_configs = [
            {'name': 'Full Model', 'use_cdam': True, 'use_domain': True, 'use_prototype': True, 'use_contrastive': True},
            {'name': 'No CDAM', 'use_cdam': False, 'use_domain': True, 'use_prototype': True, 'use_contrastive': True},
            {'name': 'No Domain Adv', 'use_cdam': True, 'use_domain': False, 'use_prototype': True, 'use_contrastive': True},
            {'name': 'No Prototype', 'use_cdam': True, 'use_domain': True, 'use_prototype': False, 'use_contrastive': True},
            {'name': 'No Contrastive', 'use_cdam': True, 'use_domain': True, 'use_prototype': True, 'use_contrastive': False},
            {'name': 'Only Classifier', 'use_cdam': False, 'use_domain': False, 'use_prototype': False, 'use_contrastive': False},
        ]

        results = {}
        for ablation in ablation_configs:
            print(f"\nRunning ablation: {ablation['name']}")

            # Temporarily modify config
            original_config = {
                'use_cdam': self.config.use_cdam,
                'use_domain': self.config.use_domain_adversarial,
                'use_prototype': self.config.use_prototypical_branch,
                'use_contrastive': self.config.use_contrastive_alignment
            }

            self.config.use_cdam = ablation['use_cdam']
            self.config.use_domain_adversarial = ablation['use_domain']
            self.config.use_prototypical_branch = ablation['use_prototype']
            self.config.use_contrastive_alignment = ablation['use_contrastive']

            try:
                model = CrossDomainTransferModel(self.config)
                result = train_episodic_meta_learning(model, ct_dataset, path_train, path_val, self.config, logger_file)
                results[ablation['name']] = result
            except Exception as e:
                print(f"Ablation {ablation['name']} failed: {e}")

            # Restore config
            self.config.use_cdam = original_config['use_cdam']
            self.config.use_domain_adversarial = original_config['use_domain']
            self.config.use_prototypical_branch = original_config['use_prototype']
            self.config.use_contrastive_alignment = original_config['use_contrastive']

        self.results = results
        return results

    def plot_ablation_results(self, output_dir: str):
        """Visualize ablation study results."""
        if not self.results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Ablation Study Results', fontsize=14, fontweight='bold')

        ablation_names = list(self.results.keys())
        metrics_to_plot = ['accuracy', 'balanced_accuracy', 'f1_macro', 'auc_roc']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            values = []
            for name in ablation_names:
                if metric in self.results[name]:
                    values.append(self.results[name][metric])
                else:
                    values.append(0)

            ax.bar(ablation_names, values)
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_ylim([0, 1])
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation_study.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'ablation_study.png'), dpi=300, bbox_inches='tight')
        plt.close()


class ClinicalMetricsAnalyzer:
    """Phase 5: Clinical & Medical Analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, 'analysis', 'clinical')
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_clinical_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray = None) -> Dict:
        """Calculate clinical relevance metrics."""
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.99, 100)

        metrics = {'thresholds': thresholds}

        sensitivity = []
        specificity = []
        ppv = []
        npv = []
        f1_scores = []

        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                pos_pred_val = tp / (tp + fp) if (tp + fp) > 0 else 0
                neg_pred_val = tn / (tn + fn) if (tn + fn) > 0 else 0
                f1 = f1_score(y_true, y_pred, zero_division=0)

                sensitivity.append(sens)
                specificity.append(spec)
                ppv.append(pos_pred_val)
                npv.append(neg_pred_val)
                f1_scores.append(f1)

        metrics['sensitivity'] = np.array(sensitivity)
        metrics['specificity'] = np.array(specificity)
        metrics['ppv'] = np.array(ppv)
        metrics['npv'] = np.array(npv)
        metrics['f1'] = np.array(f1_scores)

        # Find optimal thresholds
        youden_idx = np.argmax(metrics['sensitivity'] + metrics['specificity'] - 1)
        metrics['optimal_youden_threshold'] = thresholds[youden_idx]

        f1_idx = np.argmax(metrics['f1'])
        metrics['optimal_f1_threshold'] = thresholds[f1_idx]

        return metrics

    def plot_clinical_curves(self, metrics: Dict, fold: int, k_shot: int):
        """Generate clinical metric curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Clinical Analysis Curves (Fold {fold}, K-Shot {k_shot})', fontsize=14, fontweight='bold')

        thresholds = metrics['thresholds']

        # Sensitivity vs Threshold
        axes[0, 0].plot(thresholds, metrics['sensitivity'], label='Sensitivity', linewidth=2)
        axes[0, 0].plot(thresholds, metrics['specificity'], label='Specificity', linewidth=2)
        axes[0, 0].axvline(metrics['optimal_youden_threshold'], color='red', linestyle='--', label=f"Youden's Threshold: {metrics['optimal_youden_threshold']:.3f}")
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].set_title("Sensitivity & Specificity vs Threshold")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # PPV vs NPV
        axes[0, 1].plot(thresholds, metrics['ppv'], label='PPV', linewidth=2)
        axes[0, 1].plot(thresholds, metrics['npv'], label='NPV', linewidth=2)
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Predictive Value')
        axes[0, 1].set_title('Positive & Negative Predictive Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # F1 vs Threshold
        axes[1, 0].plot(thresholds, metrics['f1'], linewidth=2, color='green')
        axes[1, 0].axvline(metrics['optimal_f1_threshold'], color='red', linestyle='--', label=f"Optimal F1: {metrics['optimal_f1_threshold']:.3f}")
        axes[1, 0].set_xlabel('Threshold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score vs Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Youden's Index
        youden_index = metrics['sensitivity'] + metrics['specificity'] - 1
        axes[1, 1].plot(thresholds, youden_index, linewidth=2, color='purple')
        axes[1, 1].axvline(metrics['optimal_youden_threshold'], color='red', linestyle='--', label=f"Max: {np.max(youden_index):.3f}")
        axes[1, 1].set_xlabel('Threshold')
        axes[1, 1].set_ylabel("Youden's Index")
        axes[1, 1].set_title("Youden's Index vs Threshold")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'clinical_curves_fold{fold}_k{k_shot}.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, f'clinical_curves_fold{fold}_k{k_shot}.png'), dpi=300, bbox_inches='tight')
        plt.close()


class TrainingDynamicsAnalyzer:
    """Phase 6: Temporal & Training Dynamics Analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, 'analysis', 'training_dynamics')
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = defaultdict(list)

    def log_epoch_metrics(self, epoch: int, metrics: Dict):
        """Log metrics for each epoch."""
        self.history['epoch'].append(epoch)
        for key, value in metrics.items():
            self.history[key].append(value)

    def plot_training_curves(self):
        """Generate comprehensive training curves."""
        if not self.history['epoch']:
            return

        df = pd.DataFrame(self.history)

        # Plot all metrics
        metric_cols = [c for c in df.columns if c != 'epoch']
        n_metrics = len(metric_cols)

        fig, axes = plt.subplots((n_metrics + 2) // 3, 3, figsize=(15, 4 * ((n_metrics + 2) // 3)))
        axes = axes.flatten() if n_metrics > 1 else [axes]

        for idx, metric in enumerate(metric_cols):
            axes[idx].plot(df['epoch'], df[metric], linewidth=2)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.replace('_', ' ').title())
            axes[idx].set_title(f'{metric.replace("_", " ").title()} vs Epoch')
            axes[idx].grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_convergence(self) -> Dict:
        """Analyze model convergence properties."""
        if not self.history:
            return {}

        analysis = {}
        for key, values in self.history.items():
            if key != 'epoch' and len(values) > 1:
                # Check for convergence (3 consecutive epochs with change < 1%)
                for i in range(2, len(values)):
                    changes = [abs(values[j] - values[j-1]) / (abs(values[j-1]) + 1e-8) for j in range(i-2, i+1)]
                    if all(c < 0.01 for c in changes):
                        analysis[f'{key}_convergence_epoch'] = self.history['epoch'][i]
                        break

                # Calculate smoothness (variance of differences)
                diffs = np.diff(values)
                analysis[f'{key}_smoothness'] = np.std(diffs)

        return analysis


class RobustnessAnalyzer:
    """Phase 8: Robustness & Generalization Analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, 'analysis', 'robustness')
        os.makedirs(self.output_dir, exist_ok=True)

    def test_noise_robustness(self, model, test_dataset, config, noise_levels=[0.01, 0.05, 0.1, 0.15, 0.2]):
        """Test model robustness to Gaussian noise."""
        results = {}
        model.eval()

        for noise_level in noise_levels:
            predictions = []
            labels = []

            with torch.no_grad():
                for i in range(len(test_dataset)):
                    img, label, _ = test_dataset[i]
                    img = img.unsqueeze(0).to(config.device)
                    noise = torch.randn_like(img) * noise_level
                    noisy_img = img + noise
                    feat, _ = model.forward_path(noisy_img)
                    logits = model.classify(feat)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    predictions.extend(pred)
                    labels.append(label)

            acc = accuracy_score(labels, predictions)
            results[f'noise_{noise_level}'] = acc

        fig, ax = plt.subplots(figsize=(8, 6))
        noise_vals = [float(k.split('_')[1]) for k in results.keys()]
        acc_vals = list(results.values())
        ax.plot(noise_vals, acc_vals, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Noise Level (std)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Robustness to Gaussian Noise')
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'noise_robustness.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'noise_robustness.png'), dpi=300, bbox_inches='tight')
        plt.close()
        return results

    def test_channel_dropout(self, model, test_dataset, config, dropout_rates=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """Test robustness to channel dropout."""
        results = {}
        model.eval()

        for dropout_rate in dropout_rates:
            predictions = []
            labels = []

            with torch.no_grad():
                for i in range(len(test_dataset)):
                    img, label, _ = test_dataset[i]
                    img = img.unsqueeze(0).to(config.device)
                    if torch.rand(1) < dropout_rate:
                        channel_idx = torch.randint(0, img.shape[1], (1,)).item()
                        img[:, channel_idx] = 0
                    feat, _ = model.forward_path(img)
                    logits = model.classify(feat)
                    pred = logits.argmax(dim=1).cpu().numpy()
                    predictions.extend(pred)
                    labels.append(label)

            acc = accuracy_score(labels, predictions)
            results[f'dropout_{dropout_rate}'] = acc

        fig, ax = plt.subplots(figsize=(8, 6))
        dropout_vals = [float(k.split('_')[1]) for k in results.keys()]
        acc_vals = list(results.values())
        ax.plot(dropout_vals, acc_vals, 's-', linewidth=2, markersize=8, color='orange')
        ax.set_xlabel('Channel Dropout Rate')
        ax.set_ylabel('Accuracy')
        ax.set_title('Robustness to Channel Dropout')
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_dir, 'channel_dropout_robustness.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'channel_dropout_robustness.png'), dpi=300, bbox_inches='tight')
        plt.close()
        return results


class InterpretabilityAnalyzer:
    """Phase 9: Interpretability & Explainability Analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, 'analysis', 'interpretability')
        os.makedirs(self.output_dir, exist_ok=True)

    def attention_visualization(self, model, save_name='attention'):
        """Visualize attention weights from CDAM."""
        if hasattr(model, 'last_attention_weights'):
            attn_weights = model.last_attention_weights.cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(attn_weights, ax=ax, cmap='viridis')
            ax.set_title('Cross-Dimensional Attention Weights')
            plt.savefig(os.path.join(self.output_dir, f'{save_name}_heatmap.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_dir, f'{save_name}_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def prototype_visualization(self, model, save_name='prototypes'):
        """Visualize learned prototypes."""
        if hasattr(model.prototypical_net, 'prototypes'):
            prototypes = model.prototypical_net.prototypes.cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(prototypes, ax=ax, cmap='coolwarm', center=0)
            ax.set_title('Learned Prototypes')
            ax.set_xlabel('Feature Dimension')
            ax.set_ylabel('Class Prototype')
            plt.savefig(os.path.join(self.output_dir, f'{save_name}.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_dir, f'{save_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()


class FeatureAnalyzer:
    """Phase 3 (cont): Feature and Embedding Analysis."""

    def __init__(self, output_dir: str):
        self.output_dir = os.path.join(output_dir, 'analysis', 'features')
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_embeddings(self, features: np.ndarray, labels: np.ndarray, patient_ids: List[str], fold: int, k_shot: int):
        """Comprehensive embedding analysis."""
        # Dimensionality reduction
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features

        # t-SNE
        if len(features) > 30:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            embeddings_2d = tsne.fit_transform(features_reduced)

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot by class
            for label in np.unique(labels):
                mask = labels == label
                axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Class {label}', alpha=0.6, s=50)
            axes[0].set_title('t-SNE Embedding (Colored by Class)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot density
            from scipy.stats import gaussian_kde
            xy = np.vstack([embeddings_2d[:, 0], embeddings_2d[:, 1]])
            density = gaussian_kde(xy)(xy)
            scatter = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=density, cmap='viridis', s=50, alpha=0.6)
            axes[1].set_title('t-SNE Embedding (Density)')
            plt.colorbar(scatter, ax=axes[1])
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'embeddings_fold{fold}_k{k_shot}.pdf'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(self.output_dir, f'embeddings_fold{fold}_k{k_shot}.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # Clustering analysis
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        if len(np.unique(labels)) > 1 and len(features) > 10:
            try:
                silhouette = silhouette_score(features, labels)
                calinski = calinski_harabasz_score(features, labels)
                davies_bouldin = davies_bouldin_score(features, labels)

                return {
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'davies_bouldin_score': davies_bouldin
                }
            except:
                return {}
        return {}


class ReportGenerator:
    """Phase 10: Automated Report Generation."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.report_sections = []

    def add_section(self, title: str, content: str, figures: List[str] = None):
        """Add a section to the report."""
        self.report_sections.append({
            'title': title,
            'content': content,
            'figures': figures or []
        })

    def generate_latex_report(self, filename: str = 'research_report.tex'):
        """Generate LaTeX report."""
        latex_content = []

        latex_content.append(r"\documentclass[11pt,a4paper]{article}")
        latex_content.append(r"\usepackage[utf8]{inputenc}")
        latex_content.append(r"\usepackage{graphicx}")
        latex_content.append(r"\usepackage{booktabs}")
        latex_content.append(r"\usepackage{amsmath}")
        latex_content.append(r"\usepackage{hyperref}")
        latex_content.append(r"\usepackage{geometry}")
        latex_content.append(r"\geometry{margin=1in}")
        latex_content.append(r"\begin{document}")
        latex_content.append(r"\title{Comprehensive Research Report: Cross-Domain Transfer Learning}")
        latex_content.append(r"\author{Automated Analysis System}")
        latex_content.append(r"\date{\today}")
        latex_content.append(r"\maketitle")
        latex_content.append(r"\tableofcontents")
        latex_content.append(r"\newpage")

        for section in self.report_sections:
            latex_content.append(f"\\section{{{section['title']}}}")
            latex_content.append(section['content'])

            if section['figures']:
                for fig in section['figures']:
                    latex_content.append(r"\begin{figure}[h]")
                    latex_content.append(r"\centering")
                    latex_content.append(f"\\includegraphics[width=0.8\\textwidth]{{{fig}}}")
                    latex_content.append(r"\end{figure}")

        latex_content.append(r"\end{document}")

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(latex_content))

        return filepath

    def generate_html_report(self, filename: str = 'research_report.html'):
        """Generate HTML report."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html><head><title>Research Report</title></head><body>")
        html.append("<h1>Comprehensive Research Report</h1>")

        for section in self.report_sections:
            html.append(f"<h2>{section['title']}</h2>")
            html.append(f"<p>{section['content']}</p>")

            if section['figures']:
                html.append("<div>")
                for fig in section['figures']:
                    html.append(f'<img src="{fig}" style="max-width:800px;"/><br/>')
                html.append("</div>")

        html.append("</body></html>")

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(html))

        return filepath


def run_comprehensive_analysis(
    model,
    test_dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    ct_dataset,
    path_train_dataset,
    config: ResearchConfig,
    output_dir: str,
    fold: int,
    k_shot: int,
    logger_file: str
):
    """
    Run all comprehensive analysis modules for 150+ page report.
    This function orchestrates all analysis phases.
    """
    analysis_results = {}

    print(f"\n{'='*80}")
    print(f"RUNNING COMPREHENSIVE ANALYSIS - Fold {fold}, K-Shot {k_shot}")
    print(f"{'='*80}")

    # Phase 1: Core Metrics
    print("\n[Phase 1] Calculating comprehensive metrics...")
    metrics_calc = ComprehensiveMetricsCalculator(y_true, y_pred, y_prob)
    all_metrics = metrics_calc.calculate_all_metrics()
    per_class = metrics_calc.per_class_metrics()
    analysis_results['metrics'] = all_metrics
    analysis_results['per_class_metrics'] = per_class

    # Save metrics to CSV
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(os.path.join(output_dir, 'analysis', f'metrics_fold{fold}_k{k_shot}.csv'), index=False)
    per_class.to_csv(os.path.join(output_dir, 'analysis', f'per_class_metrics_fold{fold}_k{k_shot}.csv'), index=False)

    # Phase 3: Dataset Analysis
    print("\n[Phase 3] Analyzing datasets...")
    dataset_analyzer = DatasetAnalyzer(output_dir)
    if ct_dataset is not None:
        ct_stats = dataset_analyzer.analyze_ct_dataset(ct_dataset)
        analysis_results['ct_stats'] = ct_stats
    if path_train_dataset is not None:
        path_stats = dataset_analyzer.analyze_pathology_dataset(path_train_dataset)
        analysis_results['path_stats'] = path_stats
        dataset_analyzer.plot_dataset_distributions(ct_stats if 'ct_stats' in locals() else {}, path_stats)

    # Phase 5: Clinical Analysis
    print("\n[Phase 5] Calculating clinical metrics...")
    clinical_analyzer = ClinicalMetricsAnalyzer(output_dir)
    if y_prob is not None:
        clinical_metrics = clinical_analyzer.calculate_clinical_metrics(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        clinical_analyzer.plot_clinical_curves(clinical_metrics, fold, k_shot)
        analysis_results['clinical_metrics'] = clinical_metrics

    # Phase 3 (cont): Feature Analysis
    print("\n[Phase 3] Analyzing features and embeddings...")
    feature_analyzer = FeatureAnalyzer(output_dir)

    # Extract features if model available
    if model is not None and test_dataset is not None:
        model.eval()
        features_list = []
        labels_list = []
        patient_ids_list = []

        with torch.no_grad():
            for i in range(len(test_dataset)):
                img, label, patient_id = test_dataset[i]
                img = img.unsqueeze(0).to(config.device)
                feat, _ = model.forward_path(img)
                features_list.append(feat.cpu().numpy().flatten())
                labels_list.append(label)
                patient_ids_list.append(patient_id)

        if features_list:
            features_array = np.vstack(features_list)
            clustering_metrics = feature_analyzer.analyze_embeddings(
                features_array, np.array(labels_list), patient_ids_list, fold, k_shot
            )
            analysis_results['clustering_metrics'] = clustering_metrics

    # Save comprehensive results
    results_file = os.path.join(output_dir, 'analysis', f'analysis_results_fold{fold}_k{k_shot}.json')
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                              for k, v in value.items()}
            elif isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict()
            else:
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS COMPLETED")
    print(f"Results saved to: {os.path.join(output_dir, 'analysis')}")
    print(f"{'='*80}")

    return analysis_results
