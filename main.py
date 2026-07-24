"""Entry point: pretrains the CT encoder, trains and evaluates the
cross-domain transfer model with grouped 3-fold cross-validation, runs
the comprehensive analysis suite per fold/k-shot, and finally runs
inference over the full dataset.

Run with: python main.py"""
import os
import sys
import json
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

warnings.filterwarnings('ignore')

from config import ResearchConfig, CFG
from utils import (
    set_seed, log_message, compute_confidence_interval, build_grouped_split,
    save_fold_manifest, clear_memory, STRATIFIED_GROUP_KFOLD_AVAILABLE, FAISS_AVAILABLE
)
from datasets import (
    CT3DVolumeDataset, PathologyMultiChannelDataset, CT3DAugmentation, Pathology2DAugmentation
)
from models import CrossDomainTransferModel
from training import train_ct_ssl, train_path_ssl_control, train_episodic_meta_learning
from evaluation import evaluate_model, run_final_inference
from visualization import plot_summary
from analysis import run_comprehensive_analysis, ReportGenerator

def main():
    set_seed(CFG.seed)
    
    logger_file = os.path.join(CFG.output_dir, 'logs', 'experiment.log')
    
    log_message("="*80, logger_file)
    log_message("RESEARCH-GRADE 3D-TO-2D CROSS-DOMAIN TRANSFER", logger_file)
    log_message("="*80, logger_file)
    log_message(f"Device: {CFG.device}", logger_file)
    log_message(f"Output: {CFG.output_dir}", logger_file)
    log_message(f"Task framing: {CFG.task_setting}", logger_file)
    log_message(f"CDAM retrieval keys (k): {CFG.cdam_top_k}", logger_file)
    log_message(
        f"Retrieval backend: {'FAISS-ANN' if (CFG.use_ann_retrieval and FAISS_AVAILABLE) else 'Exact top-k (matmul)'}",
        logger_file
    )
    log_message(
        "Ablation flags: "
        f"CDAM={CFG.use_cdam}, "
        f"DomainAdv={CFG.use_domain_adversarial}, "
        f"Proto={CFG.use_prototypical_branch}, "
        f"Contrastive={CFG.use_contrastive_alignment}",
        logger_file
    )
    
    try:
        ct_ssl_dataset = CT3DVolumeDataset(
            data_dir=CFG.ct_data_dir,
            depth=CFG.ct_volume_depth,
            slice_size=CFG.ct_slice_size,
            transform=CT3DAugmentation(CFG.ssl_augmentation_strength),
            seed=CFG.seed
        )
        ct_bank_dataset = CT3DVolumeDataset(
            data_dir=CFG.ct_data_dir,
            depth=CFG.ct_volume_depth,
            slice_size=CFG.ct_slice_size,
            transform=None,
            seed=CFG.seed
        )
        log_message(f"Loaded {len(ct_ssl_dataset)} CT volumes", logger_file)
    except Exception as e:
        log_message(f"CRITICAL: Failed to load CT data: {e}", logger_file)
        raise
    
    try:
        path_train_dataset_full = PathologyMultiChannelDataset(
            data_dir=CFG.pathology_data_dir,
            channels=CFG.pathology_channels,
            image_size=CFG.path_image_size,
            transform=Pathology2DAugmentation(),
            seed=CFG.seed
        )
        path_eval_dataset_full = PathologyMultiChannelDataset(
            data_dir=CFG.pathology_data_dir,
            channels=CFG.pathology_channels,
            image_size=CFG.path_image_size,
            transform=None,
            seed=CFG.seed
        )
        log_message(f"Loaded {len(path_eval_dataset_full)} pathology samples", logger_file)
    except Exception as e:
        log_message(f"CRITICAL: Failed to load pathology data: {e}", logger_file)
        raise
    
    labels = [path_eval_dataset_full.samples[i]['label'] for i in range(len(path_eval_dataset_full))]
    groups = [path_eval_dataset_full.samples[i]['patient_id'] for i in range(len(path_eval_dataset_full))]
    splitter, effective_splits = build_grouped_split(labels, groups, CFG.n_splits, CFG.seed)
    if effective_splits != CFG.n_splits:
        log_message(f"Adjusted n_splits from {CFG.n_splits} to {effective_splits}", logger_file)
    if not STRATIFIED_GROUP_KFOLD_AVAILABLE:
        log_message("WARNING: StratifiedGroupKFold unavailable; falling back to StratifiedKFold.", logger_file)

    log_message(f"\n{'='*80}", logger_file)
    log_message("PRETRAINING CT ENCODER (BEFORE CROSS-VALIDATION)", logger_file)
    log_message(f"{'='*80}", logger_file)

    ct_encoder_pretrained = False
    if CFG.ct_feature_source == 'ssl':
        pretrain_model = CrossDomainTransferModel(CFG)
        pretrain_model.to(CFG.device)
        ct_loader = DataLoader(
            ct_ssl_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            drop_last=True
        )
        ssl_history = train_ct_ssl(pretrain_model, ct_loader, CFG, logger_file)
        torch.save({
            'model_state_dict': pretrain_model.ct_encoder.state_dict(),
            'ssl_history': ssl_history,
            'config': CFG.__dict__,
            'dataset_info': {
                'n_ct_volumes': len(ct_ssl_dataset),
                'depth': CFG.ct_volume_depth,
                'slice_size': CFG.ct_slice_size
            }
        }, os.path.join(CFG.output_dir, 'models', 'ct_encoder_pretrained.pt'))
        log_message("CT encoder pretrained and saved successfully", logger_file)
        ct_encoder_pretrained = True
        clear_memory()
    elif CFG.ct_feature_source == '2d_ssl':
        log_message("2D SSL control mode: CT encoder will not be pretrained (using pathology self-supervision)", logger_file)

    log_message(f"\n{'='*80}", logger_file)
    log_message("STARTING CROSS-VALIDATION", logger_file)
    log_message(f"{'='*80}", logger_file)

    fold_results = []
    fold_best_kshots = []
    fold_best_kshot_accs = []

    split_iter = splitter.split(np.zeros(len(labels)), labels, groups) if STRATIFIED_GROUP_KFOLD_AVAILABLE else splitter.split(np.zeros(len(labels)), labels)
    for fold_idx, (train_val_idx, test_idx) in enumerate(split_iter):
        log_message(f"\n{'='*80}", logger_file)
        log_message(f"FOLD {fold_idx + 1}/{effective_splits}", logger_file)
        log_message(f"{'='*80}", logger_file)
        
        train_val_idx = np.array(train_val_idx)
        train_val_labels = np.array([labels[i] for i in train_val_idx])
        train_val_groups = np.array([groups[i] for i in train_val_idx])

        inner_splitter, _ = build_grouped_split(
            train_val_labels.tolist(),
            train_val_groups.tolist(),
            n_splits=max(2, min(5, len(np.unique(train_val_groups)))),
            seed=CFG.seed + fold_idx
        )
        inner_iter = inner_splitter.split(
            np.zeros(len(train_val_idx)),
            train_val_labels,
            train_val_groups
        ) if STRATIFIED_GROUP_KFOLD_AVAILABLE else inner_splitter.split(np.zeros(len(train_val_idx)), train_val_labels)
        inner_train_rel, inner_val_rel = next(inner_iter)
        train_idx = train_val_idx[inner_train_rel].tolist()
        val_idx = train_val_idx[inner_val_rel].tolist()
        test_idx = list(test_idx)

        train_dataset = Subset(path_train_dataset_full, train_idx)
        val_dataset = Subset(path_eval_dataset_full, val_idx)
        test_dataset = Subset(path_eval_dataset_full, test_idx)
        
        train_labels = [path_eval_dataset_full.samples[i]['label'] for i in train_idx]
        val_labels = [path_eval_dataset_full.samples[i]['label'] for i in val_idx]
        test_labels = [path_eval_dataset_full.samples[i]['label'] for i in test_idx]
        
        log_message(f"Train class distribution: {np.bincount(train_labels)}", logger_file)
        log_message(f"Val class distribution: {np.bincount(val_labels)}", logger_file)
        log_message(f"Test class distribution: {np.bincount(test_labels)}", logger_file)
        log_message(
            f"Unique groups - Train: {len(set(groups[i] for i in train_idx))}, "
            f"Val: {len(set(groups[i] for i in val_idx))}, "
            f"Test: {len(set(groups[i] for i in test_idx))}",
            logger_file
        )
        
        log_message(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}", logger_file)
        save_fold_manifest(
            CFG.output_dir,
            fold_idx,
            CFG,
            train_idx,
            val_idx,
            test_idx,
            [groups[i] for i in train_idx],
            [groups[i] for i in val_idx],
            [groups[i] for i in test_idx]
        )
        
        log_message(f"\n{'='*60}", logger_file)
        log_message("TRAINING: Cross-Domain Transfer Model (3D-to-2D)", logger_file)
        log_message(f"{'='*60}", logger_file)

        model = CrossDomainTransferModel(CFG)

        if CFG.ct_feature_source == 'ssl' and ct_encoder_pretrained:
            checkpoint = torch.load(os.path.join(CFG.output_dir, 'models', 'ct_encoder_pretrained.pt'), weights_only=False)
            model.ct_encoder.load_state_dict(checkpoint['model_state_dict'])
            log_message(f"Fold {fold_idx + 1}: Loaded pretrained CT encoder", logger_file)
        elif CFG.ct_feature_source == '2d_ssl':
            train_path_ssl_control(model, train_dataset, CFG, logger_file)

        transfer_result = train_episodic_meta_learning(
            model, ct_bank_dataset, train_dataset, val_dataset, CFG, logger_file, fold_idx=fold_idx
        )
        fold_best_kshots.append(transfer_result.get('best_k_shot'))
        fold_best_kshot_accs.append(transfer_result.get('best_k_shot_acc'))

        best_model_path = transfer_result['best_model_path']
        best_model_data = torch.load(best_model_path, weights_only=False)

        if isinstance(best_model_data, dict) and 'model_state_dict' in best_model_data:
            model.load_state_dict(best_model_data['model_state_dict'])
        else:
            model.load_state_dict(best_model_data)

        test_results = evaluate_model(
            model, test_dataset, transfer_result['ct_features_bank'],
            CFG, logger_file
        )

        fold_results.append(test_results)

        fold_dir = os.path.join(CFG.output_dir, 'analysis', f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, 'results.json'), 'w') as f:
            json.dump(test_results, f, indent=2)

        # Run comprehensive analysis for each k-shot configuration
        if 'all_predictions' in test_results and 'all_true_labels' in test_results:
            for k_shot in CFG.k_shot_support:
                y_true = np.array(test_results['all_true_labels'])
                y_pred = np.array(test_results['all_predictions'])
                y_prob = np.array(test_results.get('all_probabilities', []))

                if len(y_prob) > 0 and y_prob.ndim == 1:
                    y_prob = np.column_stack([1 - y_prob, y_prob])

                comprehensive_results = run_comprehensive_analysis(
                    model=model,
                    test_dataset=test_dataset,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_prob=y_prob if len(y_prob) > 0 else None,
                    ct_dataset=ct_dataset,
                    path_train_dataset=train_dataset,
                    config=CFG,
                    output_dir=CFG.output_dir,
                    fold=fold_idx,
                    k_shot=k_shot,
                    logger_file=logger_file
                )

        clear_memory()
    
    log_message(f"\n{'='*80}", logger_file)
    log_message("FINAL RESULTS: 3D-to-2D Transfer Model", logger_file)
    log_message(f"{'='*80}", logger_file)

    aggregated = {}
    for metric in ['accuracy', 'balanced_accuracy', 'f1_score_macro', 'auc_roc', 'cohen_kappa', 'mcc']:
        values = [r[metric] for r in fold_results if not np.isnan(r[metric])]
        if values:
            bounded_metric = metric in {'accuracy', 'balanced_accuracy', 'f1_score_macro', 'auc_roc'}
            mean, ci_lower, ci_upper = compute_confidence_interval(
                values,
                n_bootstrap=CFG.bootstrap_iterations,
                bounded=bounded_metric,
                seed=CFG.seed
            )
            aggregated[metric] = {
                'mean': mean,
                'std': np.std(values),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'values': values
            }

            log_message(
                f"{metric}: {mean:.4f} +/- {np.std(values):.4f} "
                f"[{ci_lower:.4f}, {ci_upper:.4f}]",
                logger_file
            )

    final_output = {
        'transfer_model': aggregated
    }

    with open(os.path.join(CFG.output_dir, 'final_results.json'), 'w') as f:
        json.dump(final_output, f, indent=2, default=str)

    plot_summary(fold_results, aggregated, os.path.join(CFG.output_dir, 'transfer_summary'))

    valid_best_k = [k for k in fold_best_kshots if k is not None]
    valid_best_acc = [a for a in fold_best_kshot_accs if a is not None]
    if valid_best_k:
        kshot_counts = {int(k): valid_best_k.count(k) for k in sorted(set(valid_best_k))}
        consensus_k = max(kshot_counts.items(), key=lambda item: (item[1], item[0]))[0]
        log_message(f"\n{'='*80}", logger_file)
        log_message("BEST K-SHOT CONFIGURATION", logger_file)
        log_message(f"{'='*80}", logger_file)
        log_message(f"Per-fold selected K-shot values: {valid_best_k}", logger_file)
        log_message(f"Consensus Best K-shot: {consensus_k}", logger_file)
        if valid_best_acc:
            log_message(
                f"Mean best validation accuracy across folds: {float(np.mean(valid_best_acc)):.4f}",
                logger_file
            )
        log_message(f"K-shot selection counts: {kshot_counts}", logger_file)
        log_message(f"{'='*80}", logger_file)
    
    log_message("\n" + "="*80, logger_file)
    log_message("EXPERIMENT COMPLETED SUCCESSFULLY", logger_file)
    log_message("="*80, logger_file)

    log_message("\n" + "="*80, logger_file)
    log_message("STARTING FINAL INFERENCE ON ALL SAMPLES", logger_file)
    log_message("="*80, logger_file)

    try:
        inference_df = run_final_inference(CFG, logger_file)
        log_message(f"\nInference completed on {len(inference_df)} samples", logger_file)
        log_message(f"Results saved to: {os.path.join(CFG.output_dir, 'inference', 'all_samples_predictions.csv')}", logger_file)
    except Exception as e:
        log_message(f"Inference failed: {e}", logger_file)
        import traceback
        traceback.print_exc()

    # Generate comprehensive report
    log_message("\n" + "="*80, logger_file)
    log_message("GENERATING COMPREHENSIVE RESEARCH REPORT", logger_file)
    log_message("="*80, logger_file)

    try:
        report_gen = ReportGenerator(CFG.output_dir)

        # Add sections to report
        report_gen.add_section(
            "Executive Summary",
            f"Cross-domain transfer learning from CT to pathology achieved {aggregated.get('accuracy', {}).get('mean', 0):.3f} accuracy across {CFG.n_splits} folds.",
            []
        )

        report_gen.add_section(
            "Methodology",
            f"Episodic meta-learning with {CFG.k_shot_support} k-shot configurations using {CFG.transfer_epochs} transfer epochs.",
            []
        )

        report_gen.add_section(
            "Results",
            f"Transfer model performance: Accuracy={aggregated.get('accuracy', {}).get('mean', 0):.3f}, AUC-ROC={aggregated.get('auc_roc', {}).get('mean', 0):.3f}, Cohen's Kappa={aggregated.get('cohen_kappa', {}).get('mean', 0):.3f}",
            ['transfer_summary/plots']
        )

        # Generate reports
        latex_path = report_gen.generate_latex_report()
        html_path = report_gen.generate_html_report()

        log_message(f"LaTeX report generated: {latex_path}", logger_file)
        log_message(f"HTML report generated: {html_path}", logger_file)

    except Exception as e:
        log_message(f"Report generation failed: {e}", logger_file)
        import traceback
        traceback.print_exc()

    log_message("="*80, logger_file)
    log_message("ALL ANALYSIS MODULES COMPLETED", logger_file)
    log_message(f"Output directory: {CFG.output_dir}", logger_file)
    log_message("="*80, logger_file)

    return aggregated

if __name__ == "__main__":
    try:
        results = main()
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nResults saved to: {CFG.output_dir}")
        print("\nKey Findings:")
        for metric, stats in results.items():
            if isinstance(stats, dict) and 'mean' in stats:
                print(f"  {metric}: {stats['mean']:.4f} Â± {stats['std']:.4f}")
    
    except FileNotFoundError as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("Real data not found. NO SYNTHETIC FALLBACK.")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
