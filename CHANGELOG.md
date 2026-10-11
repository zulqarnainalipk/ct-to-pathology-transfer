# Changelog

## 3.0.0 (2026-10-11): additional experiments and analyses

Version 3.0 accompanies the second revision of the manuscript. It adds a new run group (D) and a post-hoc analysis script. The code paths of groups A to C, their saved outputs and every number derived from them are unchanged.

### Added

- **Group D in `main.py`** (696 training runs, 3 SSL pretraining runs), run as `--run D` or as the halves `D1` and `D2`, with its own reference full and pathology-only models (`ref_main`, `ref_path_only`):
  - harder tasks: PanCK removed from the input (full model, pathology-only, Gaussian-noise bank) and tumour core versus tumour margin (new `label_mode='roi_t_vs_m'`);
  - positive controls with PanCK removed: each tile's own PanCK intensity statistics, or a label-encoding oracle vector, fed through the fusion pathway in place of the retrieved CT context (new `context_source` option);
  - annotation budgets of 2, 4, 8 and 16 labelled training tiles per class (new `train_budget_per_class`);
  - checkpoint-selection variants: previous patient as validation patient (`val_rotation='prev'`) and last-epoch selection without early stopping (`selection='last_epoch'`);
  - channel drop with three seeds, and preprocessing without the DAPI-derived region mask;
  - extended test-time bank swap for `ref_main`: logit margins, fused-feature cosine and L2 change, and decision flips per tile (`bank_swap_diagnostics` in `result.json`);
  - channel-statistics baselines for tumour core versus margin and without the PanCK features;
  - the number of unique labelled training tiles per run (`unique_labelled_train_tiles`, `sizes.train_pool`).
- `--run ABC` runs groups A to C only; `--run full` now runs groups A to D.
- `scripts/analysis.py`: paired differences with patient-level bootstrap CIs, extended bank swap, retrieval null with replacement-bank models, per-patient retrieval diagnostics, leave-one-patient-out jackknife, per-patient differences and labelled-tile counts, written to `results/analysis/`.
- `scripts/make_figures.py`: new `harder_tasks` figure (harder tasks, positive controls, annotation budgets).
- Saved outputs of group D in `results/group_d/` (per-run `result.json`, `ref_main` features and CT banks, SSL histories, worker logs, aggregate tables) and the analyses in `results/analysis/`.
- `docs/DATA.md`: tumour-versus-margin labels and an ethics section.

### Changed

- The forest plot `forest_controls` shows the controls and ablations only; the channel drop is now reported with three seeds from group D.
- `README.md` and `docs/REPRODUCING.md` describe group D, the analysis step and the new results.
- Comment and docstring clean-up in `main.py`; no change in behaviour for groups A to C.

### Result

The CT context gives no detectable benefit in any of the new settings: without PanCK (75.8% versus 76.6% balanced accuracy for pathology-only), on tumour core versus margin (76.5% versus 77.9%) or at any annotation budget. The PanCK-statistics positive control raises balanced accuracy from 76.6% to 95.5%, so the fusion pathway can use an informative context.

## 2.0.0 (2026-09-25): revised study

The code, results and paper were revised after peer review. An audit of the data, labels and code found errors that invalidate the results of version 1.0. Those results are withdrawn; the files of version 1.0 remain available in the git history (commit `dd45c1d`).

### Corrections

- **Cohort.** The pathology collection contains 8 patients, not 42 as stated in the version 1.0 documentation. The analysis had grouped tiles by these 8 patients (`CaseN`), but the reported cohort size and class counts (131/137 instead of 127/141) were wrong.
- **Label.** Version 1.0 labelled a tile as tumour when more than 5% of the pixels of its `Segmentation/` image were foreground. These images are nuclear segmentation masks, not tumour masks, so the label measured nuclear density and could be predicted from the DAPI input channel (AUC 0.899). Version 2.0 uses the pathologist-selected region types: tumour core versus stroma, margin excluded.
- **CT preprocessing.** CT values were read without the DICOM rescale intercept, and the first 16 files of each series (in file-name order) were used. Version 2.0 converts to Hounsfield units, sorts slices by z-position and samples 16 slices uniformly over the series.
- **Optimization.** The domain discriminator, transfer gate and retrieval adapter were missing from the Phase-2 optimizer, and the CT projection head from the Phase-1 optimizer. All trainable modules are now optimized.
- **Protocol.** The two support sizes were trained sequentially on one model instance, and a full-dataset inference pass on mostly training data was reported. Version 2.0 trains every setting independently and reports only out-of-fold predictions.

### Changes

- Leave-one-patient-out cross-validation with three seeds, pooled out-of-fold predictions and patient-level cluster-bootstrap confidence intervals; all conditions share folds, seeds and initializations.
- New comparators and controls: pathology-only model, four replacement CT banks, random retrieval, test-time bank swap, InfoNCE with query-independent positives, free gate, component ablations, SSL epoch and temperature sweeps, channel drop, and a hand-crafted channel-statistics baseline.
- Retrieval and alignment diagnostics (bank usage, label mutual information, context probe, retrieval stability, domain separability, MMD, CKA, attention entropy, gate values, logit margins).
- SSL pretraining extended to 50 epochs with label-free convergence metrics.
- The modular code of version 1.0 was replaced by a single pipeline script (`main.py`) with a resumable multi-GPU launcher and an aggregation step, and a figure script (`scripts/make_figures.py`).
- Result: the complete pipeline is feasible and accurate, but the CT context contributes no detectable information; see `README.md`.

## 1.0.0: original submission

Code and results accompanying the originally submitted manuscript. Superseded by 2.0.0; its results are withdrawn.
