"""Model architecture: the multi-scale 3D CT encoder, the multi-channel 2D
pathology encoder, the gradient-reversal domain discriminator, the
cross-dimensional attention bridge (CDAM), the prototypical network, and
the top-level CrossDomainTransferModel that composes them."""
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from config import ResearchConfig

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
