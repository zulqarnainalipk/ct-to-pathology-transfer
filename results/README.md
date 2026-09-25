# Saved outputs of the reported run

Everything in the paper is computed from the files in this folder. The run used `main.py --run full` on two NVIDIA T4 GPUs (see [`../docs/REPRODUCING.md`](../docs/REPRODUCING.md)); `aggregate/` was produced by `main.py --aggregate results --agg_out results/aggregate`, and the figures in `../docs/figures/` by `scripts/make_figures.py`.

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
```

Fold `f` holds out patient `Case{f+1}`, and the next patient in the rotation is the validation patient. Seeds are 0, 1 and 2; the SSL sweeps and channel-drop conditions use seed 0 only. `trivial_channel_stats` is the hand-crafted logistic-regression baseline, which is deterministic and has no training history.

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

## `result.json` schema

| Key | Content |
|---|---|
| `condition`, `seed`, `fold`, `n_folds` | Run identity |
| `test_patient`, `val_patient` | Held-out and validation patient |
| `sizes`, `class_counts` | Tiles and class counts in train / validation / test |
| `overrides`, `config` | Condition overrides and the full effective configuration |
| `test_metrics` | Metrics on the held-out patient (accuracy, balanced accuracy, macro-F1, AUC, kappa, MCC, sensitivity, specificity, PPV, NPV, ECE, Brier, confusion matrix) |
| `records` | One entry per held-out tile: `name`, `patient`, true label `y`, tumour probability `prob`, retrieved bank indices `topk`, attention weights `attn`, mean transfer gate `gate_mean`, and the weighted logit margins `margin_cls` and `margin_glob` |
| `history` | Per-epoch training loss and accuracy, validation accuracy, balanced accuracy, AUC and selection score |
| `best_epoch`, `best_val_score`, `epochs_run` | Checkpoint selection |
| `episodes_per_epoch`, `k_shot_used`, `n_query_used` | Episode set-up actually used |
| `bank_swap_probs` | Main model only: held-out tumour probabilities when the bank is swapped at test time |
| `time_s` | Wall-clock time of the run |

`features.npz` contains the arrays `path` (pathology features before fusion), `aligned` (after gated fusion), `context` (retrieved CT context), `y` and `names` for the held-out tiles.
