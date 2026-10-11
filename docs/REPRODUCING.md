# Reproducing the results

All results in the paper come from two sets of runs of `main.py`: groups A to C (the main comparison, controls, ablations and sweeps) and group D (harder tasks, positive controls, annotation budgets, robustness checks and the extended bank swap). Each set was followed by the aggregation step; `scripts/analysis.py` and `scripts/make_figures.py` then produce the post-hoc analyses and the figures. The saved outputs of both sets are in `results/`, so the tables, analyses and figures can be rebuilt without retraining (step 4).

## 1. Environment

```bash
pip install -r requirements.txt
```

Python 3.9 to 3.12 and PyTorch 2.x work. A GPU is needed for the full runs; the smoke test runs on a CPU. The reported runs used two NVIDIA Tesla T4 GPUs on Kaggle, mixed precision (fp16) in the pathology encoder, and non-deterministic cuDNN kernels for speed, so reruns with the same seeds reproduce the results up to floating-point non-determinism.

## 2. Data

Download the two TCIA collections and arrange them as described in [DATA.md](DATA.md). Pass the folders with `--ct_dir` and `--path_dir`, or set `ct_data_dir` and `pathology_data_dir` in `ResearchConfig` at the top of `main.py` (defaults: `./data/LungCT-Diagnosis/LungCT-Diagnosis` and `./data/PKG - HNSCC-mIF-mIHC`).

## 3. Training

### Smoke test (CPU)

```bash
python main.py --run smoke --ct_dir DATA/LungCT-Diagnosis/LungCT-Diagnosis --path_dir "DATA/PKG - HNSCC-mIF-mIHC" --out smoke_run
```

Runs every condition of groups A to D on 3 patients, 3 tiles per class, 4 CT volumes, 2 epochs and 2 folds, then aggregates. It checks that the whole pipeline works, not the results.

### Groups and launcher

```bash
python main.py --list          # shows the conditions and run counts of each group
```

| Group | Conditions | Runs (8 folds each) |
|---|---|---:|
| A | full model, pathology-only, both at K = 8, InfoNCE with random positives, channel-statistics baseline | 120 |
| B | CT-bank controls (untrained encoder, Gaussian, dimension-shuffled, constant), random retrieval, free gate | 144 |
| C | ablations (fusion, adversarial, InfoNCE, prototypes), SSL epoch and temperature sweeps, channel drop (seed 0) | 200 |
| D1 | reference full and pathology-only models with extended bank-swap logging, PanCK removed (full, pathology-only, Gaussian bank), tumour core vs margin, positive controls, checkpoint-selection variants | 312 |
| D2 | annotation budgets of 2, 4, 8 and 16 tiles per class, channel drop with three seeds, no DAPI mask | 384 |

The run counts exclude the deterministic channel-statistics baselines, which take seconds: `trivial_channel_stats` runs with group A, and `tm_trivial_channel_stats` and `nopanck_trivial_channel_stats` with group D.

`--run` accepts `ABC` (groups A to C), `D` (D1 and D2), a single group (`A`, `B`, `C`, `D1`, `D2`), `full` (all groups) or `smoke`. The launcher detects the GPUs and starts `WORKERS_PER_GPU` worker processes per GPU (default 2). The workers first share out the SSL pretraining runs that the selected conditions need and then the training runs, and the launcher shows progress with an ETA. When all workers have finished it aggregates into `<out>_aggregate/` and zips the outputs into `<out>_<run>_outputs.zip`. Every finished run is saved as `runs/<condition>/seed<s>/fold<f>/result.json` and skipped when the command is repeated, so an interrupted session can be resumed by running the same command with the same `--out`.

### Groups A to C

```bash
python main.py --run ABC --ct_dir ... --path_dir ... --out results_run
```

464 training runs and 7 SSL pretraining runs; about 3 hours on 2 x T4. The groups can also be run one at a time (`--run A`, `--run B`, `--run C`) with the same `--out`.

### Group D

```bash
python main.py --run D --ct_dir ... --path_dir ... --out results_run_d
```

696 training runs and 3 SSL pretraining runs (seeds 0 to 2 at temperature 0.2 and 50 epochs); about 3.7 hours on 2 x T4. If a session is too short, run the two halves separately with the same `--out`; they are independent:

```bash
python main.py --run D1 --ct_dir ... --path_dir ... --out results_run_d    # 312 runs
python main.py --run D2 --ct_dir ... --path_dir ... --out results_run_d    # 384 runs
```

Group D was run in its own output folder, so it has its own SSL encoders and reference models (`ref_main`, `ref_path_only`), and all group D comparisons are made within group D. Its `drop_*` conditions have the same configuration as the seed-0 channel drop of group C, so if both groups share one output folder, the seed-0 runs of group C are reused.

### On Kaggle

1. Add the two datasets as notebook inputs and select the GPU T4 x2 accelerator.
2. Write `main.py` into the notebook (`%%writefile main.py` followed by the file contents) or add the repository as a dataset.
3. Run it in a cell. Running inside the notebook kernel gives live progress bars per GPU:
   ```python
   %run main.py --run ABC --ct_dir /kaggle/input/<ct-folder> --path_dir "/kaggle/input/<pathology-folder>" --out /kaggle/working/results_run
   %run main.py --run D   --ct_dir /kaggle/input/<ct-folder> --path_dir "/kaggle/input/<pathology-folder>" --out /kaggle/working/results_run_d
   ```
4. Check the ETA after about 15 minutes. If it does not fit the session limit, run the groups (or D1 and D2) in separate sessions and merge them afterwards (step 4). For an unattended run, use *Save Version, Save & Run All (Commit)*.
5. Download `results_run_ABC_outputs.zip` or `results_run_d_D_outputs.zip` from the Output tab.

## 4. Tables, analyses and figures

These steps use only the saved outputs and run on a CPU.

```bash
# tables (CSV + LaTeX) and summary.json; several output roots can be merged
python main.py --aggregate results --agg_out results/aggregate                  # groups A to C
python main.py --aggregate results/group_d --agg_out results/group_d/aggregate  # group D
python main.py --aggregate run_A run_B run_C --agg_out merged_aggregate         # merging example

# post-hoc analyses in results/analysis/ (reads results/runs and results/group_d/runs)
python scripts/analysis.py

# figures (PDF + PNG) in docs/figures/
python scripts/make_figures.py                 # all figures
python scripts/make_figures.py forest harder   # selected figures
```

On a laptop CPU, each aggregation takes about 5 to 7 minutes and `scripts/analysis.py` about 2 to 3 minutes. Only complete sets of folds enter the tables; incomplete condition-seed pairs are listed in `summary.json` under `incomplete_condition_seeds`. The statistics are computed on pooled out-of-fold predictions; confidence intervals and paired differences come from a patient-level cluster bootstrap with 2000 resamples (seed 0), so rerunning the aggregation and the analysis reproduces them exactly (up to floating-point rounding in the last digits of a few diagnostics). In the group D aggregate, paired differences are taken against `ref_main`, and the tumour-versus-margin conditions are not compared with the tumour-versus-stroma reference; their comparisons are in `results/analysis/paired_differences.csv`.

The `harder` figure reads `results/group_d/aggregate/` and `results/analysis/`, so run the group D aggregation and `scripts/analysis.py` first if you regenerate them from new runs.

The architecture figure is drawn with Mermaid:

```bash
npm install -g @mermaid-js/mermaid-cli
cd docs/architecture
mmdc -i architecture.mmd -o architecture.pdf -c mermaid-config.json --pdfFit
mmdc -i architecture.mmd -o architecture.png -c mermaid-config.json -s 3 -b white
```

## 5. Configuration

All hyperparameters are fields of `ResearchConfig` in `main.py`, and every condition is a set of overrides in `CONDITIONS`. The group D conditions use one new label mode, `label_mode='roi_t_vs_m'` (tumour core vs margin), and four new fields: `train_budget_per_class`, `context_source` (`ct`, `panck_stats` or `oracle`), `val_rotation` (`next` or `prev`) and `selection` (`early_stop` or `last_epoch`). The configurations of the reported runs are saved in `results/base_config.json` and `results/group_d/base_config.json`, and the effective configuration of each run is stored inside its `result.json`.
