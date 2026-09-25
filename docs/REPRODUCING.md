# Reproducing the results

All results in the paper come from one run of `main.py` over three independent groups of conditions, followed by the aggregation step and `scripts/make_figures.py`. The saved outputs of that run are in `results/`, so the tables and figures can be rebuilt without retraining (step 4).

## 1. Environment

```bash
pip install -r requirements.txt
```

Python 3.9 to 3.12 and PyTorch 2.x work. A GPU is needed for the full run; the smoke test runs on a CPU. The reported run used two NVIDIA Tesla T4 GPUs on Kaggle, mixed precision (fp16) in the pathology encoder, and non-deterministic cuDNN kernels for speed, so reruns with the same seeds reproduce the results up to floating-point non-determinism.

## 2. Data

Download the two TCIA collections and arrange them as described in [DATA.md](DATA.md). Pass the folders with `--ct_dir` and `--path_dir`, or set `ct_data_dir` and `pathology_data_dir` in `ResearchConfig` at the top of `main.py`.

## 3. Training

### Smoke test (CPU, a few minutes)

```bash
python main.py --run smoke --ct_dir DATA/LungCT-Diagnosis/LungCT-Diagnosis --path_dir "DATA/PKG - HNSCC-mIF-mIHC" --out smoke_run
```

Runs every condition on 3 patients, 3 tiles per class, 4 CT volumes, 2 epochs and 2 folds, then aggregates. It checks that the whole pipeline works, not the results.

### Full run

```bash
python main.py --list          # shows the conditions in each group
python main.py --run full --ct_dir ... --path_dir ... --out results_run
```

`--run full` detects the GPUs and starts `WORKERS_PER_GPU` worker processes per GPU (default 2). The workers first share out the seven SSL pretraining runs and then the training runs, and the launcher shows progress with an ETA. When all workers have finished it aggregates into `results_run_aggregate/` and zips the outputs. The complete run (464 training runs and 7 SSL pretraining runs) took about 3 hours on 2 x T4.

The runs are split into three independent groups:

| Group | Conditions | Runs (8 folds each) |
|---|---|---:|
| A | full model, pathology-only, both at K = 8, InfoNCE with random positives, channel-statistics baseline | 120 |
| B | CT-bank controls (untrained encoder, Gaussian, dimension-shuffled, constant), random retrieval, free gate | 144 |
| C | ablations (fusion, adversarial, InfoNCE, prototypes), SSL epoch and temperature sweeps, channel drop | 200 |

Use `--run A` (or `B`, `C`) to run one group, for example in separate sessions. Every finished run is saved as `runs/<condition>/seed<s>/fold<f>/result.json` and skipped when the command is repeated, so an interrupted session can be resumed by running the same command with the same `--out`.

### On Kaggle

1. Add the two datasets as notebook inputs and select the GPU T4 x2 accelerator.
2. Write `main.py` into the notebook (`%%writefile main.py`) or add the repository as a dataset.
3. Run it in a cell. Running inside the notebook kernel gives live progress bars per GPU:
   ```python
   %run main.py --run full --ct_dir /kaggle/input/<ct-folder> --path_dir "/kaggle/input/<pathology-folder>" --out /kaggle/working/results_run
   ```
4. Check the ETA after about 15 minutes. If it does not fit the session limit, run the groups in separate sessions and merge them afterwards (step 4). For an unattended run, use *Save Version, Save & Run All (Commit)*.
5. Download `results_run_full_outputs.zip` from the Output tab.

## 4. Tables and figures

```bash
# tables (CSV + LaTeX) and summary.json; several output roots can be merged
python main.py --aggregate results --agg_out results/aggregate
python main.py --aggregate run_A run_B run_C --agg_out merged_aggregate

# figures (PDF + PNG) in docs/figures/, from results/
python scripts/make_figures.py
```

Only complete sets of folds enter the tables; incomplete condition-seed pairs are listed in `summary.json` under `incomplete_condition_seeds`. The statistics are computed on pooled out-of-fold predictions; confidence intervals and paired differences come from a patient-level cluster bootstrap with 2000 resamples (seed 0), so rerunning the aggregation reproduces them exactly.

The architecture figure is drawn with Mermaid:

```bash
npm install -g @mermaid-js/mermaid-cli
cd docs/architecture
mmdc -i architecture.mmd -o architecture.pdf -c mermaid-config.json --pdfFit
mmdc -i architecture.mmd -o architecture.png -c mermaid-config.json -s 3 -b white
```

## 5. Configuration

All hyperparameters are fields of `ResearchConfig` in `main.py`, and every condition is a set of overrides in `CONDITIONS`. The configuration of the reported run is saved in `results/base_config.json`, and the effective configuration of each run is stored inside its `result.json`.
