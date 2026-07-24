"""Training loops: CT self-supervised contrastive pretraining, the pathology
SSL control condition, and episodic meta-learning for cross-domain
transfer."""
import os
import random
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn.utils import clip_grad_norm_

from config import ResearchConfig
from datasets import CT3DVolumeDataset, PathologyMultiChannelDataset
from models import CrossDomainTransferModel
from utils import (
    log_message, retrieve_nearest_ct_features, info_nce_loss,
    compute_model_selection_score, build_run_tag, clear_memory
)
from visualization import VisualizationSuite

def train_ct_ssl(model: CrossDomainTransferModel, ct_loader: DataLoader, config: ResearchConfig, logger_file: str) -> Dict[str, List]:
    log_message("="*80, logger_file)
    log_message("Phase 1: 3D CT Self-Supervised Pre-training", logger_file)
    log_message("="*80, logger_file)
    
    model.to(config.device)
    model.train()
    
    optimizer = AdamW(
        model.ct_encoder.parameters(),
        lr=config.ssl_lr,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    history = {'epoch': [], 'loss': []}
    
    for epoch in range(1, config.ssl_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(ct_loader, desc=f"SSL Epoch {epoch}/{config.ssl_epochs}")
        
        for ct_volumes, labels, patient_ids in pbar:
            ct_volumes = ct_volumes.to(config.device)
            
            aug = CT3DAugmentation(strength=config.ssl_augmentation_strength)
            rng1 = np.random.RandomState(epoch + num_batches)
            rng2 = np.random.RandomState(epoch + num_batches + 1000)
            
            view1 = torch.stack([
                torch.from_numpy(aug(vol.detach().cpu().numpy().squeeze(0), rng1)).unsqueeze(0).float()
                for vol in ct_volumes
            ]).to(config.device)
            
            view2 = torch.stack([
                torch.from_numpy(aug(vol.detach().cpu().numpy().squeeze(0), rng2)).unsqueeze(0).float()
                for vol in ct_volumes
            ]).to(config.device)
            
            _, proj1 = model.forward_ct(view1)
            _, proj2 = model.forward_ct(view2)
            
            z1 = F.normalize(proj1, dim=1)
            z2 = F.normalize(proj2, dim=1)
            
            batch_size = z1.shape[0]
            
            representations = torch.cat([z1, z2], dim=0)
            similarity_matrix = torch.mm(representations, representations.t())
            
            labels = torch.arange(batch_size, device=config.device)
            labels = torch.cat([labels + batch_size, labels], dim=0)
            
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=config.device)
            similarity_matrix.masked_fill_(mask, -float('inf'))
            
            loss = F.cross_entropy(similarity_matrix / config.ssl_temperature, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        
        log_message(f"Epoch {epoch}/{config.ssl_epochs} - Loss: {avg_loss:.4f}", logger_file)
        
        if epoch % 5 == 0:
            ckpt_path = os.path.join(
                config.output_dir, 'checkpoints',
                f'ct_ssl_epoch_{epoch}.pt'
            )
            torch.save(model.ct_encoder.state_dict(), ckpt_path)
    
    return history

def train_path_ssl_control(
    model: CrossDomainTransferModel,
    path_dataset: Dataset,
    config: ResearchConfig,
    logger_file: str
) -> None:
    log_message("="*80, logger_file)
    log_message("Control: 2D Pathology SSL Pre-training", logger_file)
    log_message("="*80, logger_file)

    model.to(config.device)
    model.train()
    loader = DataLoader(path_dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = AdamW(
        list(model.path_encoder.parameters()) + list(model.path_projector.parameters()),
        lr=config.ssl_lr,
        weight_decay=config.weight_decay
    )
    aug = Pathology2DAugmentation(strength=config.ssl_augmentation_strength)

    for epoch in range(1, config.path_ssl_control_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for images, _, _ in loader:
            images = images.to(config.device)

            rng1 = np.random.RandomState(config.seed + epoch + n_batches)
            rng2 = np.random.RandomState(config.seed + epoch + n_batches + 1000)

            view1 = torch.stack([
                torch.from_numpy(aug(img.detach().cpu().numpy(), rng1)).float()
                for img in images
            ]).to(config.device)
            view2 = torch.stack([
                torch.from_numpy(aug(img.detach().cpu().numpy(), rng2)).float()
                for img in images
            ]).to(config.device)

            _, proj1 = model.forward_path(view1)
            _, proj2 = model.forward_path(view2)
            z1 = F.normalize(proj1, dim=1)
            z2 = F.normalize(proj2, dim=1)

            batch_size = z1.shape[0]
            reps = torch.cat([z1, z2], dim=0)
            similarity = torch.mm(reps, reps.t())
            labels = torch.arange(batch_size, device=config.device)
            labels = torch.cat([labels + batch_size, labels], dim=0)
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=config.device)
            similarity.masked_fill_(mask, -float('inf'))
            loss = F.cross_entropy(similarity / config.ssl_temperature, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        mean_loss = epoch_loss / max(1, n_batches)
        log_message(f"2D SSL Epoch {epoch}/{config.path_ssl_control_epochs} - Loss: {mean_loss:.4f}", logger_file)

def train_episodic_meta_learning(model: CrossDomainTransferModel, ct_dataset: CT3DVolumeDataset, path_train_dataset: PathologyMultiChannelDataset, path_val_dataset: PathologyMultiChannelDataset, config: ResearchConfig, logger_file: str, fold_idx: int = 0) -> Dict[str, Any]:
    log_message("="*80, logger_file)
    log_message("Phase 2: Episodic Meta-Learning for Cross-Domain Transfer", logger_file)
    log_message(f"K-shot configurations: {config.k_shot_support}", logger_file)
    log_message("="*80, logger_file)
    
    visualizer = VisualizationSuite(config)
    
    model.to(config.device)

    if config.stage2_freeze_ct_encoder:
        for param in model.ct_encoder.parameters():
            param.requires_grad = False
        model.ct_encoder.eval()
        log_message("Stage-2 policy: CT encoder frozen (feature-bank mode).", logger_file)
    else:
        for param in model.ct_encoder.parameters():
            param.requires_grad = True
        model.ct_encoder.train()
        log_message(
            f"Stage-2 policy: CT encoder fine-tuning enabled (LR x{config.ct_finetune_lr_multiplier:.3f}).",
            logger_file
        )

    if config.ct_feature_source == 'ssl':
        log_message("Building CT feature bank from SSL-pretrained encoder...", logger_file)
        ct_loader = DataLoader(ct_dataset, batch_size=config.batch_size, shuffle=False)
        ct_features_bank = []

        model.eval()
        with torch.no_grad():
            for ct_volumes, labels, _ in tqdm(ct_loader, desc="Extracting CT features"):
                ct_volumes = ct_volumes.to(config.device)
                features, _ = model.forward_ct(ct_volumes)
                ct_features_bank.append(features.cpu())

        ct_features_bank = torch.cat(ct_features_bank, dim=0)
        log_message(f"Extracted {len(ct_features_bank)} CT features from SSL encoder.", logger_file)
    elif config.ct_feature_source in {'ct_clip', 'spectre'}:
        ct_features_bank = load_external_feature_bank(
            config.external_ct_feature_bank_path,
            expected_dim=config.feature_dim
        )
        log_message(
            f"Loaded {len(ct_features_bank)} frozen CT features from {config.ct_feature_source} baseline bank.",
            logger_file
        )
    elif config.ct_feature_source == '2d_ssl':
        model.eval()
        with torch.no_grad():
            path_loader = DataLoader(path_train_dataset, batch_size=config.batch_size, shuffle=False)
            path_feature_bank = []
            for path_images, _, _ in tqdm(path_loader, desc="Building 2D control feature bank"):
                path_images = path_images.to(config.device)
                path_features, _ = model.forward_path(path_images)
                path_feature_bank.append(path_features.cpu())
        ct_features_bank = torch.cat(path_feature_bank, dim=0)
        log_message(
            f"Built {len(ct_features_bank)} pathology-derived control features (2D SSL baseline).",
            logger_file
        )
    else:
        raise ValueError(f"Unsupported ct_feature_source: {config.ct_feature_source}")

    trainable_param_groups = [
        {'params': list(model.path_encoder.parameters()), 'lr': config.transfer_lr},
        {'params': list(model.cross_attention.parameters()), 'lr': config.transfer_lr},
        {'params': list(model.prototypical_net.parameters()), 'lr': config.transfer_lr},
        {'params': list(model.pseudo_labeler.parameters()), 'lr': config.transfer_lr},
        {'params': list(model.classifier.parameters()), 'lr': config.transfer_lr},
        {'params': list(model.path_projector.parameters()), 'lr': config.transfer_lr},
        {'params': list(model.ct_projector.parameters()), 'lr': config.transfer_lr},
    ]
    if not config.stage2_freeze_ct_encoder:
        trainable_param_groups.append({
            'params': list(model.ct_encoder.parameters()),
            'lr': config.transfer_lr * config.ct_finetune_lr_multiplier
        })

    optimizer = AdamW(trainable_param_groups, weight_decay=config.weight_decay)
    
    all_results = {}
    best_k_shot = None
    best_k_shot_acc = 0.0
    
    for k_shot in config.k_shot_support:
        log_message(f"\n{'='*60}", logger_file)
        log_message(f"Training with K-Shot = {k_shot}", logger_file)
        log_message(f"{'='*60}", logger_file)
        
        optimizer = AdamW(trainable_param_groups, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        if hasattr(path_train_dataset, 'dataset'):
            train_labels = [path_train_dataset.dataset.samples[i]['label'] for i in path_train_dataset.indices]
        else:
            train_labels = [path_train_dataset.samples[i]['label'] for i in range(len(path_train_dataset))]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = class_weights.to(config.device)
        
        cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
        domain_criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        best_val_score = -float('inf')
        patience_counter = 0
        
        history = {
            'epoch': [], 'train_loss': [], 'train_acc': [],
            'val_acc': [], 'val_f1': [], 'domain_loss': [],
            'prototype_loss': [], 'contrastive_loss': [], 'lr': []
        }
        
        prototype_history = []
        
        all_confidences = []
        all_predictions = []
        all_true_labels = []
        
        for epoch in range(1, config.transfer_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_episodes = 0
            
            class_0_indices = []
            class_1_indices = []
            
            for idx in range(len(path_train_dataset)):
                _, label, _ = path_train_dataset[idx]
                if label == 0:
                    class_0_indices.append(idx)
                elif label == 1:
                    class_1_indices.append(idx)
            
            if len(class_0_indices) < k_shot + config.n_query or \
               len(class_1_indices) < k_shot + config.n_query:
                continue
            
            episodes_per_epoch = min(50, max(20, len(path_train_dataset) // (k_shot * config.n_way)))
            pbar = tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch}/{config.transfer_epochs}")
            for episode in pbar:
                support_indices = []
                query_indices = []
                
                for label, class_indices in enumerate([class_0_indices, class_1_indices]):
                    selected = random.sample(class_indices, k_shot + config.n_query)
                    support_indices.extend(selected[:k_shot])
                    query_indices.extend(selected[k_shot:])
                
                random.shuffle(support_indices)
                random.shuffle(query_indices)
                
                support_images = []
                support_labels = []
                query_images = []
                query_labels = []
                
                for idx in support_indices:
                    img, label, _ = path_train_dataset[idx]
                    support_images.append(img)
                    support_labels.append(label)
                
                for idx in query_indices:
                    img, label, _ = path_train_dataset[idx]
                    query_images.append(img)
                    query_labels.append(label)
                
                support_images = torch.stack(support_images).to(config.device)
                support_labels = torch.tensor(support_labels, device=config.device)
                query_images = torch.stack(query_images).to(config.device)
                query_labels = torch.tensor(query_labels, device=config.device)
                
                support_feat, _ = model.forward_path(support_images)
                query_feat, query_proj = model.forward_path(query_images)
                
                ct_feat_batch, _, ct_tokens_query = retrieve_nearest_ct_features(
                    query_feat.detach(),
                    ct_features_bank,
                    top_k=config.cdam_top_k,
                    temperature=config.cdam_temperature,
                    use_ann=config.use_ann_retrieval,
                    ann_num_probes=config.ann_num_probes
                )
                
                if config.use_cdam:
                    aligned_query = model.align_features(query_feat, ct_feat_batch, ct_tokens_query)
                else:
                    aligned_query = query_feat

                support_ct_batch, _, ct_tokens_support = retrieve_nearest_ct_features(
                    support_feat.detach(),
                    ct_features_bank,
                    top_k=config.cdam_top_k,
                    temperature=config.cdam_temperature,
                    use_ann=config.use_ann_retrieval,
                    ann_num_probes=config.ann_num_probes
                )
                if config.use_cdam:
                    aligned_support = model.align_features(support_feat, support_ct_batch, ct_tokens_support)
                else:
                    aligned_support = support_feat
                
                proto_logits, prototypes = model.prototypical_net(
                    aligned_support, support_labels, aligned_query, config.n_way
                )
                proto_loss = cls_criterion(proto_logits, query_labels)
                
                model.update_global_prototypes(aligned_support.detach(), support_labels.detach())
                global_proto_logits = model.compute_global_proto_logits(aligned_query)
                cls_logits = model.classify(aligned_query)
                combined_logits = model.combine_logits(
                    cls_logits,
                    episodic_proto_logits=proto_logits,
                    global_proto_logits=global_proto_logits
                )
                cls_loss = cls_criterion(cls_logits, query_labels)
                combined_loss = cls_criterion(combined_logits, query_labels)

                support_cls_logits = model.classify(aligned_support)
                support_global_logits = model.compute_global_proto_logits(aligned_support)
                support_combined_logits = model.combine_logits(
                    support_cls_logits,
                    global_proto_logits=support_global_logits
                )
                support_consistency_loss = cls_criterion(support_combined_logits, support_labels)
                
                if config.use_domain_adversarial:
                    domain_labels_path = torch.ones(len(aligned_query), dtype=torch.long, device=config.device)
                    domain_labels_ct = torch.zeros(len(ct_feat_batch), dtype=torch.long, device=config.device)
                    domain_logits_path = model.domain_discriminator(aligned_query, alpha=config.gradient_reversal_alpha)
                    domain_logits_ct = model.domain_discriminator(ct_feat_batch, alpha=config.gradient_reversal_alpha)
                    domain_loss = (domain_criterion(domain_logits_path, domain_labels_path) +
                                  domain_criterion(domain_logits_ct, domain_labels_ct)) / 2
                else:
                    domain_loss = torch.zeros(1, device=config.device).squeeze()

                if config.use_contrastive_alignment:
                    ct_proj_batch = model.ct_projector(ct_feat_batch)
                    contrastive_loss = info_nce_loss(
                        query_proj,
                        ct_proj_batch,
                        temperature=config.cdam_temperature
                    )
                else:
                    contrastive_loss = torch.zeros(1, device=config.device).squeeze()

                if config.use_cdam:
                    transfer_reg = model.compute_transfer_regularization(query_feat, aligned_query, ct_feat_batch)
                    transfer_reg = transfer_reg + model.compute_transfer_regularization(support_feat, aligned_support, support_ct_batch)
                else:
                    transfer_reg = torch.zeros(1, device=config.device).squeeze()

                total_loss = combined_loss + 0.5 * cls_loss
                if config.use_prototypical_branch:
                    total_loss = total_loss + config.prototype_lambda * proto_loss
                if config.use_domain_adversarial:
                    total_loss = total_loss + config.adversarial_lambda * domain_loss
                if config.use_contrastive_alignment:
                    total_loss = total_loss + config.contrastive_lambda * contrastive_loss
                total_loss = total_loss + config.support_consistency_lambda * support_consistency_loss
                total_loss = total_loss + transfer_reg
                
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                preds = combined_logits.argmax(dim=1)
                acc = (preds == query_labels).float().mean().item()

                epoch_loss += total_loss.item()
                epoch_acc += acc
                num_episodes += 1

                if episode % 10 == 0 and config.save_attention_maps:
                    if hasattr(model, 'last_attention_weights'):
                        sample_idx = query_indices[0]
                        visualizer.plot_attention_heatmaps(
                            model.last_attention_weights,
                            ct_feat_batch[0:1],
                            query_images[0:1],
                            epoch,
                            f"episode_{episode}"
                        )
                
                if episode % 5 == 0:
                    prototype_history.append(prototypes.detach().cpu().clone())
                
                if episode % 5 == 0:
                    with torch.no_grad():
                        probs = F.softmax(combined_logits, dim=1)
                        confidences = probs.max(dim=1)[0].detach().cpu().numpy()
                        predictions = preds.detach().cpu().numpy()
                        true_labels_batch = query_labels.detach().cpu().numpy()
                        
                        all_confidences.extend(confidences)
                        all_predictions.extend(predictions)
                        all_true_labels.extend(true_labels_batch)
                
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'acc': f'{acc:.3f}',
                    'episode': f'{episode+1}/{episodes_per_epoch}'
                })
            
            scheduler.step()
            
            if num_episodes > 0 and epoch % config.val_interval == 0:
                val_metrics = evaluate_model(
                    model, path_val_dataset, ct_features_bank,
                    config, logger_file, show_progress=False
                )
                
                val_acc = val_metrics['accuracy']
                val_f1 = val_metrics['f1_score_macro']
                
                history['epoch'].append(epoch)
                history['train_loss'].append(epoch_loss / num_episodes)
                history['train_acc'].append(epoch_acc / num_episodes)
                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)
                
                history['domain_loss'].append(domain_loss.item() if 'domain_loss' in locals() else 0)
                history['prototype_loss'].append(proto_loss.item() if 'proto_loss' in locals() else 0)
                history['contrastive_loss'].append(contrastive_loss.item() if 'contrastive_loss' in locals() else 0)
                history['lr'].append(optimizer.param_groups[0]['lr'])
                
                log_message(
                    f"Epoch {epoch} - Train Loss: {epoch_loss/num_episodes:.4f}, "
                    f"Train Acc: {epoch_acc/num_episodes:.3f}, "
                    f"Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}",
                    logger_file
                )
                
                if epoch % (config.val_interval * 2) == 0:
                    visualizer.plot_training_curves(history, epoch)
                    
                    if len(all_confidences) > 0:
                        visualizer.plot_uncertainty_calibration(
                            np.array(all_confidences),
                            np.array(all_predictions),
                            np.array(all_true_labels),
                            epoch
                        )
                    
                    if epoch % (config.val_interval * 4) == 0:
                        sample_size = min(100, len(ct_features_bank), len(path_train_dataset))
                        ct_sample = ct_features_bank[:sample_size].to(config.device)
                        
                        path_sample_indices = random.sample(range(len(path_train_dataset)), sample_size)
                        path_sample = []
                        path_sample_labels = []
                        
                        for idx in path_sample_indices:
                            img, label, _ = path_train_dataset[idx]
                            path_sample.append(img)
                            path_sample_labels.append(label)
                        
                        path_sample = torch.stack(path_sample).to(config.device)
                        
                        with torch.no_grad():
                            path_feat, _ = model.forward_path(path_sample)
                        
                        aligned_sample = model.align_features(path_feat, ct_sample, ct_sample.unsqueeze(1))
                        
                        visualizer.plot_tsne_umap_alignment(
                            ct_sample,
                            path_feat,
                            epoch,
                            np.array(path_sample_labels)
                        )
                        
                        visualizer.plot_feature_distributions(
                            ct_sample,
                            path_feat,
                            aligned_sample,
                            epoch
                        )
                
                val_score = compute_model_selection_score(val_metrics)
                if val_score > best_val_score + config.min_delta:
                    best_val_score = val_score
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_path = os.path.join(
                        config.output_dir,
                        'models',
                        f'{build_run_tag("transfer", fold_idx, k_shot)}_best_model.pt'
                    )
                    torch.save(
                        model.state_dict(),
                        best_model_path
                    )
                else:
                    patience_counter += 1
                
                if patience_counter >= config.patience:
                    log_message(f"Early stopping at epoch {epoch}", logger_file)
                    break
        
        if len(prototype_history) > 0:
            visualizer.plot_prototype_evolution(prototype_history, epoch)
        
        all_results[k_shot] = {
            'history': history,
            'best_val_acc': best_val_acc,
            'best_val_score': best_val_score,
            'ct_features_bank': ct_features_bank,
            'best_model_path': os.path.join(
                config.output_dir,
                'models',
                f'{build_run_tag("transfer", fold_idx, k_shot)}_best_model.pt'
            )
        }
        
        if best_val_acc > best_k_shot_acc:
            best_k_shot_acc = best_val_acc
            best_k_shot = k_shot
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'k_shot': k_shot,
                'best_val_acc': best_val_acc,
                'best_val_score': best_val_score,
                'config': config.__dict__
            }, os.path.join(config.output_dir, 'models', f'best_kshot_{k_shot}_model.pt'))
            
            log_message(f"New best K-shot model saved: K={k_shot}, Val Acc: {best_val_acc:.4f}", logger_file)
        
        clear_memory()
    
    return {
        'all_results': all_results,
        'best_k_shot': best_k_shot,
        'best_k_shot_acc': best_k_shot_acc,
        'best_model_path': all_results[best_k_shot]['best_model_path'] if best_k_shot in all_results else None,
        'ct_features_bank': ct_features_bank
    }
