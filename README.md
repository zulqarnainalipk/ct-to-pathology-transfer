# Does unpaired 3D CT context help few-shot multiplex pathology classification?

Code, results and paper for a controlled feasibility study of **unpaired cross-dimensional transfer**: a bank of self-supervised 3D CT descriptors is retrieved and fused into an episodic classifier for 2D multiplex immunofluorescence (mIF) pathology tiles, and a set of controls tests whether the model actually uses that context.

**Main finding.** The complete pipeline trains stably and generalizes to held-out patients (balanced accuracy 94.0%, AUC 0.989), but an identically trained pathology-only model performs comparably, and the predictions barely change when the CT bank is replaced by noise, a constant vector or random retrieval. In this setting the CT context contributes no detectable information; the fusion pathway is open, but what flows through it is not used.

![Computational graph of the model](docs/architecture/architecture.png)

## Study at a glance

| | |
|---|---|
| Source domain | LungCT-Diagnosis (TCIA): 61 unlabelled thoracic CT series, one per patient |
| Target domain | HNSCC-mIF-mIHC-comparison (TCIA): 8 head and neck squamous cell carcinoma patients, 5 mIF channels (CD3, CD8, FoxP3, PanCK, DAPI) |
| Task | Tumour core versus stroma, using the pathologist-selected region types (94 / 78 tiles; margin tiles excluded) |
| Protocol | Leave-one-patient-out cross-validation, 3 seeds, pooled out-of-fold predictions, patient-level cluster bootstrap CIs |
| Scale | 464 training runs and 7 SSL pretraining runs, about 3 h on 2 x NVIDIA T4 |

## Results

Leave-one-patient-out results on 172 tiles (mean ± SD over three seeds; 95% CI from a patient-level cluster bootstrap):

| Model | Balanced accuracy (%) | 95% CI | AUC | 95% CI |
|---|---:|---:|---:|---:|
| Full model (CT context), K = 2 | 94.0 ± 1.2 | 90.3 to 97.4 | 0.989 ± 0.001 | 0.975 to 0.998 |
| Pathology-only, K = 2 | 94.8 ± 0.6 | 90.7 to 98.4 | 0.979 ± 0.006 | 0.952 to 0.998 |
| Full model, K = 8 | 95.2 ± 2.5 | 92.9 to 97.8 | 0.987 ± 0.010 | 0.971 to 0.998 |
| Pathology-only, K = 8 | 95.1 ± 1.1 | 91.2 to 98.9 | 0.984 ± 0.011 | 0.968 to 0.999 |
| Channel statistics + logistic regression | 98.1 | 95.1 to 100.0 | 0.996 | 0.983 to 1.000 |

Controls and ablations, as paired differences from the full model. Only removing the PanCK channel has a clear effect; replacing or removing the CT context does not:

![Paired differences from the full model](docs/figures/forest_controls.png)

Further results, all generated from the saved outputs in [`results/`](results/):

| What | Table | Figure |
|---|---|---|
| Main results, per-patient results, pooled confusion counts | `main_results`, `per_patient`, `confusion_pooled` | `roc_pooled`, `reliability_main` |
| CT-bank controls and ablations | `controls_ablations` | `forest_controls` |
| Test-time bank swap | `bank_swap` | |
| Retrieval and alignment diagnostics | `retrieval_diagnostics` | `bank_usage_main`, `attention_gate_hist`, `tsne_main` |
| SSL convergence and sensitivity | `ssl_sensitivity` | `ssl_convergence` |
| Channel drop and label audit | `channel_drop`, `label_audit`, `single_channel_auc` | |

Tables are in [`results/aggregate/tables/`](results/aggregate/tables/) (CSV and LaTeX); figures are in [`docs/figures/`](docs/figures/) (PDF and PNG). The paper is currently under review and will be linked here once published.

## Repository structure

```text
main.py                 Experiment pipeline: data loading, SSL pretraining, episodic training,
                        all conditions and controls, multi-GPU launcher, aggregation
scripts/
  make_figures.py       Regenerates every data figure in docs/figures/ from results/
  fonts/                Latin Modern Roman fonts used by the figures
docs/
  DATA.md               Datasets, labels and the label audit
  REPRODUCING.md        Step-by-step reproduction (local and Kaggle)
  architecture/         Mermaid source and rendered architecture figure
  figures/              Result figures (PDF and PNG)
results/
  README.md             Layout of the saved outputs and the result.json schema
  runs/                 One result.json per condition, seed and fold (out-of-fold predictions)
  ssl/                  SSL training histories
  banks/                CT feature banks of the main model
  logs/                 Worker logs of the full run
  aggregate/            Tables (CSV, LaTeX) and summary.json
CHANGELOG.md            Version history, including the corrections in v2.0
CITATION.cff            Citation metadata
```

## Quick start

```bash
git clone https://github.com/zulqarnainalipk/ct-to-pathology-transfer.git
cd ct-to-pathology-transfer
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --list                                   # conditions and run counts
python main.py --run smoke --ct_dir PATH/TO/LungCT-Diagnosis --path_dir "PATH/TO/PKG - HNSCC-mIF-mIHC"
```

The smoke test runs every condition on a tiny subset on the CPU in a few minutes. The datasets are not included; see [`docs/DATA.md`](docs/DATA.md) for download and folder layout, and [`docs/REPRODUCING.md`](docs/REPRODUCING.md) for the full run, the multi-GPU launcher and the aggregation.

To rebuild the tables and figures from the saved outputs without training anything:

```bash
python main.py --aggregate results --agg_out results/aggregate
python scripts/make_figures.py
```

## Version history

Version 2.0 is the current, revised study. Version 1.0 (commit `dd45c1d`) accompanied the originally submitted manuscript; its results are withdrawn because of errors in the cohort description, the label definition, the CT preprocessing and the optimizer set-up. [`CHANGELOG.md`](CHANGELOG.md) lists every correction.

## Citation

If you use this code or the control protocol, please cite the paper (manuscript under review). Citation metadata is in [`CITATION.cff`](CITATION.cff).

```text
Ali, Z. Does Unpaired 3D CT Context Help Few-Shot Multiplex Pathology Classification?
A Controlled Feasibility Study. Manuscript under review, 2026.
```

The datasets should be cited as described on their TCIA collection pages (see [`docs/DATA.md`](docs/DATA.md)).

## Author

**Zulqarnain Ali**  
Department of Data Science, The Islamia University of Bahawalpur  
Email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)  
Website: [zulqarnainali.com](https://zulqarnainali.com)  
GitHub: [github.com/zulqarnainalipk](https://github.com/zulqarnainalipk)  
LinkedIn: [linkedin.com/in/zulqarnainalipk](https://linkedin.com/in/zulqarnainalipk)  
ORCID: [0009-0009-0281-882X](https://orcid.org/0009-0009-0281-882X)

## License

The code is released under the MIT License. The datasets are distributed by TCIA under their own data-use terms and are not part of this repository.
