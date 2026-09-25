# Changelog

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
