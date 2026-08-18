# Cross-Dimensional Episodic Transfer: CT to Few-Shot Pathology Classification

This repository contains the code and saved outputs for **“Can a CT Scan Teach a Pathology Classifier?”** The project studies whether representations learned from 3D lung CT volumes can help a few-shot classifier work with 2D HNSCC histopathology images.

The work is deliberately organized as a research repository rather than a single notebook. The training, evaluation, analysis, and visualization steps are separated into modules, while the saved outputs make it possible to inspect the reported results without rerunning the full experiment immediately.

## Research question

The central question is simple: can an unlabeled bank of CT-derived features provide useful cross-dimensional context for a pathology classifier when the target pathology dataset is small?

The pipeline pretrains a 3D CT encoder with contrastive self-supervision, freezes the resulting encoder, and builds a feature bank. For each pathology query, the 2D branch retrieves similar CT features and combines them through a cross-dimensional attention bridge (CDAM). The model also uses domain-adversarial alignment and a prototypical few-shot classification head.

![Research architecture for cross-dimensional transfer](docs/figures/architecture.png)

*Figure 1. Research architecture for the CT-to-pathology transfer experiment.*

## Reported results

The reported grouped three-fold cross-validation used 268 pathology samples from 42 patients.

| Metric | Mean | 95% confidence interval |
|---|---:|---:|
| Accuracy | 88.66% | 87.69–89.52% |
| Balanced accuracy | 88.26% | 87.92–88.78% |
| Macro F1 | 0.884 | — |
| AUC-ROC | 0.970 | 0.967–0.973 |
| Cohen’s kappa | 0.769 | 0.755–0.778 |
| MCC | 0.777 | — |

These numbers should be read together with the limitations. The saved reliability analysis indicates overconfidence on held-out data, and the study uses a small target dataset, weak threshold-derived labels, and a single-seed primary estimate. The project is therefore a research prototype, not a clinical tool.

## Repository map

```text
main.py             Full experiment entry point
config.py           ResearchConfig and experiment settings
datasets.py         CT and pathology datasets and augmentations
models.py           Encoders, CDAM, discriminator, and prototype head
training.py         Self-supervision and episodic training loops
evaluation.py       Fold evaluation and inference
analysis.py          Metrics, ablations, calibration, and reports
visualization.py     Figures and visual summaries
utils.py             Reproducibility, retrieval, splitting, and logging

docs/
  paper.pdf          Research write-up, methods, results, and limitations
  architecture.mmd   Source for the system architecture diagram
  architecture.png   Rendered architecture diagram

outputs/
  final_results.json Saved aggregate results
  research_report.* Generated report files
```

## Quick start

```bash
git clone https://github.com/zulqarnainalipk/ct-to-pathology-transfer.git
cd ct-to-pathology-transfer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The original CT and pathology collections are not included. Configure their local paths in `config.py` before running. A full run includes CT self-supervised pretraining, grouped cross-validation, evaluation, analysis, and inference; a GPU is strongly recommended.

## Reproducibility notes

The repository keeps fold manifests and saved outputs so that the reported split structure can be inspected. Large trained checkpoints are intentionally not committed. This keeps the repository practical to clone and makes the distinction between source code, generated artifacts, and external data clear.

## Citation

```text
Ali, Z. “Can a CT Scan Teach a Pathology Classifier? Cross-Dimensional
Episodic Transfer from 3D Radiology to Few-Shot 2D Histopathology.” 2026.
```

## Author

**Zulqarnain Ali**  
Machine Learning Researcher  
Email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)  
Website: [zulqarnainali.com](https://zulqarnainali.com)  
GitHub: [github.com/zulqarnainalipk](https://github.com/zulqarnainalipk)  
LinkedIn: [linkedin.com/in/zulqarnainalipk](https://linkedin.com/in/zulqarnainalipk)  
ORCID: [0009-0009-0281-882X](https://orcid.org/0009-0009-0281-882X)

## License

The code is released under the MIT License. The external datasets have their own access conditions and data-use terms.
