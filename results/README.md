# Saved outputs of the reported runs

Everything in the paper is computed from the files in this folder. There are two sets of runs, both made with `main.py` on two NVIDIA T4 GPUs (see [`../docs/REPRODUCING.md`](../docs/REPRODUCING.md)):

- **Groups A to C** (464 training runs, 7 SSL pretraining runs): the main comparison, CT-bank controls, ablations, SSL sweeps and the seed-0 channel drop. Their outputs are at the top level of this folder, and `aggregate/` was produced by `main.py --aggregate results --agg_out results/aggregate`.
- **Group D** (696 training runs, 3 SSL pretraining runs and two deterministic channel-statistics baselines): harder tasks, positive controls, annotation budgets, checkpoint-selection and preprocessing variants, channel drop with three seeds and the extended bank swap. It has its own reference full and pathology-only models (`ref_main`, `ref_path_only`) and is stored in [`group_d/`](group_d/) with the same layout. `group_d/aggregate/` was produced by `main.py --aggregate results/group_d --agg_out results/group_d/aggregate`.

`analysis/` holds the post-hoc analyses that combine both sets (`scripts/analysis.py`), and the figures in `../docs/figures/` come from `scripts/make_figures.py`.

```text
results/
  base_config.json        ResearchConfig of the run (defaults before condition overrides)
  dataset_summary.json    Tile counts per patient and region type, label audit, single-channel AUCs
  ct_summary.json         CT series used: count, slices per series, slice thickness, HU window
  runs/<condition>/seed<s>/fold<f>/
      result.json         One training run (see schema below)
      features.npz        Main model only: out-of-fold features of the held-out patient
  ssl/tau<t>_seed<s>_max<e>/history.json
                          SSL pretraining curves (loss and label-free representation metrics)
  banks/main_seed<s>.pt   CT feature bank of the main model (torch tensor, 61 x 512)
  logs/                   Launcher and worker logs of the full run
  aggregate/
      tables/*.csv, *.tex Tables of the paper and supporting tables
      summary.json        All aggregated statistics, bootstrap intervals and diagnostics
  group_d/                Group D, same layout as above, plus:
      dataset_summary_roi_t_vs_m.json
                          Tile counts for tumour core vs margin
      banks/ref_main_seed<s>.pt
                          CT feature bank of the group D reference model
                          (features.npz is stored for ref_main only)
  analysis/
      *.csv               Post-hoc analyses (see the last section)
      analysis_summary.json
                          The same numbers in one JSON file
```

Fold `f` holds out patient `Case{f+1}`, and the next patient in the rotation is the validation patient (the previous one for `selprev_*`). Seeds are 0, 1 and 2. In groups A to C the SSL sweeps and channel-drop conditions use seed 0 only; group D repeats the channel drop with all three seeds. `trivial_channel_stats`, `tm_trivial_channel_stats` and `nopanck_trivial_channel_stats` are hand-crafted logistic-regression baselines; they are deterministic (stored as seed 0) and have no training history.

## Conditions

| Condition | Meaning |
|---|---|
| `main`, `main_k8` | Full model with K = 2 (default) or K = 8 support tiles per class |
| `path_only`, `path_only_k8` | No CT context: no retrieval, fusion, adversarial or InfoNCE terms |
| `bank_random_init`, `bank_gaussian`, `bank_dim_shuffle`, `bank_mean` | CT bank replaced by an untrained encoder, per-dimension Gaussian noise, dimension-permuted entries, or the bank mean |
| `retrieval_random` | Real bank, k entries retrieved at random |
| `ctr_random_pos` | InfoNCE positives drawn from the bank independently of the query |
| `abl_free_gate` | Gate floor and usage penalty removed |
| `abl_no_cdam`, `abl_no_adv`, `abl_no_ctr`, `abl_no_proto` | Without the fusion pathway, the adversarial loss, InfoNCE, or the prototypes |
| `ssl_e{0,10,25,50,100}`, `ssl_t{0.05,0.1,0.5}` | SSL epoch and temperature sweeps (seed 0) |
| `drop_{CD3,CD8,FoxP3,PanCK,DAPI}` | One input channel set to zero in training and testing (seed 0) |
| `trivial_channel_stats` | Logistic regression on 15 per-channel intensity statistics |

Group D (`group_d/runs/`, three seeds each unless noted):

| Condition | Meaning |
|---|---|
| `ref_main`, `ref_path_only` | Reference full and pathology-only models of group D (same configuration as `main` and `path_only`); `ref_main` also stores the extended bank-swap diagnostics |
| `hard_main`, `hard_path_only`, `hard_bank_gaussian` | PanCK channel removed from the input: full model, pathology-only, Gaussian-noise bank |
| `tm_main`, `tm_path_only` | Tumour core versus tumour margin (`label_mode='roi_t_vs_m'`, 94 / 96 tiles) |
| `pos_panck_context` | Positive control with PanCK removed: each tile's own PanCK intensity statistics, randomly projected to 512 dimensions, replace the retrieved CT context |
| `pos_oracle_context` | Positive control with PanCK removed: a label-encoding context vector (fixed random vector per class plus noise), used only as a capacity check |
| `budget{2,4,8,16}_main`, `budget{2,4,8,16}_path_only` | Training pool limited to 2, 4, 8 or 16 labelled tiles per class (drawn per seed and fold) |
| `selprev_main`, `selprev_path_only` | Validation patient is the previous patient in the rotation |
| `sellast_main`, `sellast_path_only` | No validation-based selection: the model after the last epoch (30) is kept |
| `drop_{CD3,CD8,FoxP3,PanCK,DAPI}` | One input channel set to zero in training and testing, three seeds |
| `nomask_main`, `nomask_path_only`, `nomask_drop_DAPI` | Preprocessing without the DAPI-derived region mask |
| `tm_trivial_channel_stats` | Channel-statistics baseline for tumour core versus margin |
| `nopanck_trivial_channel_stats` | Channel-statistics baseline without the three PanCK statistics |

## `result.json` schema

| Key | Content |
|---|---|
| `condition`, `seed`, `fold`, `n_folds` | Run identity |
| `test_patient`, `val_patient` | Held-out and validation patient |
| `sizes`, `class_counts` | Tiles and class counts in train / validation / test; group D runs also store `sizes.train_pool` (training tiles before any budget limit) |
| `unique_labelled_train_tiles` | Group D only: number of distinct labelled tiles the model was trained on |
| `overrides`, `config` | Condition overrides and the full effective configuration |
| `test_metrics` | Metrics on the held-out patient (accuracy, balanced accuracy, macro-F1, AUC, kappa, MCC, sensitivity, specificity, PPV, NPV, ECE, Brier, confusion matrix) |
| `records` | One entry per held-out tile: `name`, `patient`, true label `y`, tumour probability `prob`, retrieved bank indices `topk`, attention weights `attn`, mean transfer gate `gate_mean`, and the weighted logit margins `margin_cls` and `margin_glob` |
| `history` | Per-epoch training loss and accuracy, validation accuracy, balanced accuracy, AUC and selection score |
| `best_epoch`, `best_val_score`, `epochs_run` | Checkpoint selection |
| `episodes_per_epoch`, `k_shot_used`, `n_query_used` | Episode set-up actually used |
| `bank_swap_probs` | Main model only: held-out tumour probabilities when the bank is swapped at test time |
| `bank_swap_diagnostics` | `ref_main` only: for each swapped bank, the tile probabilities, logit margins (`margin_cls`, `margin_glob`), cosine similarity and L2 distance between the fused features before and after the swap, and decision flips |
| `time_s` | Wall-clock time of the run |

`features.npz` contains the arrays `path` (pathology features before fusion), `aligned` (after gated fusion), `context` (retrieved CT context), `y` and `names` for the held-out tiles.

## Post-hoc analyses (`analysis/`)

Produced by `python scripts/analysis.py` from the per-run outputs of both sets; nothing is retrained. Intervals come from the same patient-level cluster bootstrap as the aggregation (2000 resamples, seed 0).

| File | Content |
|---|---|
| `paired_differences.csv` | Paired differences (A minus B) in balanced accuracy and AUC with 95% CIs: harder tasks, positive controls, budgets, selection and mask variants, three-seed channel drop |
| `baselines_group_d.csv` | Channel-statistics baselines for tumour core versus margin and without PanCK |
| `labelled_tiles.csv` | Unique labelled training tiles, training pool and validation tiles per fold |
| `bank_swap_extended.csv`, `bank_swap_extended_per_fold.csv` | Test-time bank swap of `ref_main`: changes of probability and logit margin, fused-feature cosine and relative shift, decision flips, Cohen's kappa, patient-level changes |
| `retrieval_null.csv` | Mutual information between label and top-1 retrieved entry for the full model and for models trained with replacement banks |
| `retrieval_per_patient.csv` | Per held-out patient: overlap of the tumour and stroma retrieval sets, within-patient mutual information, context-probe AUC |
| `jackknife.csv` | Full minus pathology-only difference with each patient left out in turn (main analysis and group D reference runs) |
| `per_patient_difference.csv` | Per-patient difference in balanced accuracy, full minus pathology-only (main analysis) |
