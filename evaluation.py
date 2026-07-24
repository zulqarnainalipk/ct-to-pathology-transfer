"""Evaluation: per-fold test-set evaluation, full-dataset inference, and the
final-inference entry point that reloads the best saved checkpoint."""
import os
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score
)

from config import ResearchConfig, CFG
from datasets import CT3DVolumeDataset, PathologyMultiChannelDataset
from models import CrossDomainTransferModel
from utils import log_message, retrieve_nearest_ct_features

def evaluate_model(model: CrossDomainTransferModel, test_dataset: PathologyMultiChannelDataset, ct_features_bank: torch.Tensor, config: ResearchConfig, logger_file: str = None, show_progress: bool = True) -> Dict[str, Any]:
    model.to(config.device)
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    iterator = tqdm(test_loader, desc="Evaluating") if show_progress else test_loader
    
    with torch.no_grad():
        for batch_idx, (path_images, labels, _) in enumerate(iterator):
            path_images = path_images.to(config.device)
            
            path_feat, _ = model.forward_path(path_images)
            ct_feat_batch, _, ct_tokens = retrieve_nearest_ct_features(
                path_feat,
                ct_features_bank,
                top_k=config.cdam_top_k,
                temperature=config.cdam_temperature,
                use_ann=config.use_ann_retrieval,
                ann_num_probes=config.ann_num_probes
            )
            
            aligned = model.align_features(path_feat, ct_feat_batch, ct_tokens) if config.use_cdam else path_feat
            
            cls_logits = model.classify(aligned)
            global_proto_logits = model.compute_global_proto_logits(aligned)
            logits = model.combine_logits(cls_logits, global_proto_logits=global_proto_logits)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    if len(np.unique(all_labels)) > 1:
        auc_roc = roc_auc_score(all_labels, all_probs)
    else:
        auc_roc = np.nan
    
    cm = confusion_matrix(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    
    results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score_macro': f1_macro,
        'f1_score_weighted': f1_weighted,
        'auc_roc': auc_roc,
        'cohen_kappa': kappa,
        'mcc': mcc,
        'confusion_matrix': cm.tolist()
    }
    
    if logger_file:
        log_message(f"Evaluation Results:", logger_file)
        log_message(f"  Accuracy: {accuracy:.4f}", logger_file)
        log_message(f"  Balanced Accuracy: {balanced_acc:.4f}", logger_file)
        log_message(f"  F1 (Macro): {f1_macro:.4f}", logger_file)
        log_message(f"  AUC-ROC: {auc_roc:.4f}", logger_file)
        log_message(f"  Cohen's Kappa: {kappa:.4f}", logger_file)

    return results

def run_inference_on_all_samples(model: CrossDomainTransferModel, dataset: PathologyMultiChannelDataset, ct_features_bank: torch.Tensor, config: ResearchConfig, output_path: str, logger_file: str = None) -> pd.DataFrame:
    log_message("="*80, logger_file)
    log_message("RUNNING INFERENCE ON ALL SAMPLES", logger_file)
    log_message("="*80, logger_file)

    model.to(config.device)
    model.eval()

    inference_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    all_records = []

    with torch.no_grad():
        for batch_idx, (path_images, labels, sample_names) in enumerate(tqdm(inference_loader, desc="Inference")):
            path_images = path_images.to(config.device)

            path_feat, _ = model.forward_path(path_images)
            ct_feat_batch, topk_indices, ct_tokens = retrieve_nearest_ct_features(
                path_feat,
                ct_features_bank,
                top_k=config.cdam_top_k,
                temperature=config.cdam_temperature,
                use_ann=config.use_ann_retrieval,
                ann_num_probes=config.ann_num_probes
            )

            aligned = model.align_features(path_feat, ct_feat_batch, ct_tokens) if config.use_cdam else path_feat

            cls_logits = model.classify(aligned)
            global_proto_logits = model.compute_global_proto_logits(aligned)
            logits = model.combine_logits(cls_logits, global_proto_logits=global_proto_logits)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            confidences = probs.max(dim=1)[0].detach().cpu().numpy()
            pred_probs = probs[:, 1].detach().cpu().numpy()

            for i in range(len(sample_names)):
                pred_class = int(preds[i].item())
                confidence = float(confidences[i])
                tumor_prob = float(pred_probs[i])

                if pred_class == 1:
                    if tumor_prob >= 0.8:
                        severity = "High"
                    elif tumor_prob >= 0.6:
                        severity = "Moderate"
                    else:
                        severity = "Low"
                else:
                    severity = "None"

                record = {
                    'sample_name': str(sample_names[i]) if isinstance(sample_names[i], str) else str(sample_names[i].item()),
                    'predicted_class': pred_class,
                    'predicted_label': 'Tumor' if pred_class == 1 else 'Non-Tumor',
                    'confidence': round(confidence, 4),
                    'tumor_probability': round(tumor_prob, 4),
                    'severity': severity,
                    'true_class': int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i]),
                    'true_label': 'Tumor' if (int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])) == 1 else 'Non-Tumor',
                    'correct_prediction': bool((int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])) == pred_class)
                }
                all_records.append(record)

    df = pd.DataFrame(all_records)

    df.to_csv(output_path, index=False)
    log_message(f"Inference results saved to: {output_path}", logger_file)
    log_message(f"Total samples processed: {len(df)}", logger_file)
    log_message(f"Correct predictions: {df['correct_prediction'].sum()}/{len(df)} ({100*df['correct_prediction'].mean():.2f}%)", logger_file)

    severity_counts = df['severity'].value_counts().to_dict()
    log_message(f"Severity distribution: {severity_counts}", logger_file)

    summary_path = output_path.replace('.csv', '_summary.json')
    summary_stats = {
        'total_samples': int(len(df)),
        'accuracy': float(df['correct_prediction'].mean()),
        'tumor_samples': int((df['predicted_class'] == 1).sum()),
        'non_tumor_samples': int((df['predicted_class'] == 0).sum()),
        'severity_distribution': {k: int(v) for k, v in severity_counts.items()},
        'mean_confidence': float(df['confidence'].mean()),
        'mean_tumor_probability': float(df[df['predicted_class'] == 1]['tumor_probability'].mean()) if (df['predicted_class'] == 1).any() else 0.0
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    log_message(f"Summary stats saved to: {summary_path}", logger_file)

    return df


def run_final_inference(config: ResearchConfig = CFG, logger_file: str = None) -> pd.DataFrame:
    log_message("\n" + "="*80, logger_file)
    log_message("FINAL INFERENCE: Loading Best Transfer Model", logger_file)
    log_message("="*80, logger_file)

    try:
        path_dataset = PathologyMultiChannelDataset(
            data_dir=config.pathology_data_dir,
            channels=config.pathology_channels,
            image_size=config.path_image_size,
            transform=None,
            seed=config.seed
        )
        log_message(f"Loaded {len(path_dataset)} pathology samples for inference", logger_file)
    except Exception as e:
        log_message(f"Failed to load pathology data for inference: {e}", logger_file)
        raise

    try:
        ct_dataset = CT3DVolumeDataset(
            data_dir=config.ct_data_dir,
            depth=config.ct_volume_depth,
            slice_size=config.ct_slice_size,
            transform=None,
            seed=config.seed
        )
    except Exception as e:
        log_message(f"Failed to load CT data for inference: {e}", logger_file)
        raise

    model = CrossDomainTransferModel(config)

    best_model_path = os.path.join(config.output_dir, 'models', 'best_model.pt')
    if not os.path.exists(best_model_path):
        log_message(f"Best model not found at {best_model_path}", logger_file)
        alternative_paths = sorted(Path(os.path.join(config.output_dir, 'models')).glob('best_kshot_*_model.pt'))
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                best_model_path = str(alt_path)
                log_message(f"Using alternative model: {best_model_path}", logger_file)
                break

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"No trained model found for inference")

    best_model_data = torch.load(best_model_path, map_location=config.device, weights_only=False)
    if isinstance(best_model_data, dict) and 'model_state_dict' in best_model_data:
        model.load_state_dict(best_model_data['model_state_dict'])
        log_message(f"Loaded model from: {best_model_path}", logger_file)
        if 'k_shot' in best_model_data:
            log_message(f"Model was trained with k-shot: {best_model_data['k_shot']}", logger_file)
    else:
        model.load_state_dict(best_model_data)
        log_message(f"Loaded model state dict from: {best_model_path}", logger_file)

    model.to(config.device)
    model.eval()

    log_message("Building CT feature bank for inference...", logger_file)
    ct_loader = DataLoader(ct_dataset, batch_size=config.batch_size, shuffle=False)
    ct_features_bank = []
    with torch.no_grad():
        for ct_volumes, _, _ in tqdm(ct_loader, desc="Extracting CT features"):
            ct_volumes = ct_volumes.to(config.device)
            features, _ = model.forward_ct(ct_volumes)
            ct_features_bank.append(features.cpu())
    ct_features_bank = torch.cat(ct_features_bank, dim=0)
    log_message(f"CT feature bank size: {len(ct_features_bank)}", logger_file)

    os.makedirs(os.path.join(config.output_dir, 'inference'), exist_ok=True)
    output_path = os.path.join(config.output_dir, 'inference', 'all_samples_predictions.csv')

    results_df = run_inference_on_all_samples(
        model, path_dataset, ct_features_bank, config, output_path, logger_file
    )

    log_message("="*80, logger_file)
    log_message("INFERENCE COMPLETED", logger_file)
    log_message("="*80, logger_file)

    return results_df
