"""Central configuration for the CT-to-pathology cross-domain transfer pipeline."""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch

@dataclass
class ResearchConfig:
    ct_data_dir: str = '/kaggle/input/lungct/LungCT-Diagnosis/LungCT-Diagnosis'
    pathology_data_dir: str = '/kaggle/input/head-and-neck-carcinoma/PKG - HNSCC-mIF-mIHC'
    output_dir: str = './research_3d2d_transfer'
    pathology_channels: Tuple[str, ...] = ('CD3', 'CD8', 'FoxP3', 'PanCK', 'DAPI')
    ct_volume_depth: int = 16
    ct_slice_size: int = 128
    ct_in_channels: int = 1
    path_image_size: int = 224
    path_in_channels: int = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    n_splits: int = 3   ############# 5 for full Run 
    batch_size: int = 4
    num_workers: int = 0
    ssl_epochs: int = 10
    ssl_lr: float = 1e-4
    ssl_temperature: float = 0.2
    ssl_augmentation_strength: float = 0.5
    transfer_epochs: int = 30
    transfer_lr: float = 1e-4
    warmup_epochs: int = 3
    n_way: int = 2
    k_shot_support: List[int] = field(default_factory=lambda: [2,8])
    n_query: int = 5
    meta_batch_size: int = 4
    adversarial_lambda: float = 0.3
    contrastive_lambda: float = 0.1
    prototype_lambda: float = 0.5
    reconstruction_lambda: float = 0.05
    uncertainty_threshold: float = 0.85
    feature_dim: int = 512
    projection_dim: int = 256
    num_prototypes: int = 10
    gradient_reversal_alpha: float = 1.0
    cdam_top_k: int = 5
    cdam_temperature: float = 0.1
    stage2_freeze_ct_encoder: bool = True
    ct_finetune_lr_multiplier: float = 0.1
    task_setting: str = 'low_shot_supervised'
    ct_feature_source: str = 'ssl'  # one of: ssl, ct_clip, spectre, 2d_ssl
    external_ct_feature_bank_path: Optional[str] = None
    use_ann_retrieval: bool = False
    ann_num_probes: int = 8
    retrieval_token_count: int = 5
    path_ssl_control_epochs: int = 10
    bootstrap_iterations: int = 2000
    use_cdam: bool = True
    use_domain_adversarial: bool = True
    use_prototypical_branch: bool = True
    use_contrastive_alignment: bool = True
    run_ablation_suite: bool = False
    domain_confusion_epochs: int = 5
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    use_mixed_precision: bool = False
    enable_visualization: bool = True
    plot_interval: int = 5
    save_attention_maps: bool = True
    track_prototype_evolution: bool = True
    enable_tsne_umap: bool = True
    tsne_perplexity: int = 30
    umap_n_neighbors: int = 15
    track_domain_confusion: bool = True
    uncertainty_calibration_bins: int = 10
    feature_distribution_analysis: bool = True
    save_intermediate_features: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    val_interval: int = 1
    save_best_only: bool = True
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
    
    def __post_init__(self):
        for subdir in ['models', 'plots', 'logs', 'checkpoints', 'analysis']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

CFG = ResearchConfig()
