# Notebook and experiment guide

This repository is primarily organized as Python modules rather than a single notebook. The main experiment is launched with `python main.py`, while the saved outputs under `outputs/` provide the most direct record of the reported run.

If you add exploratory notebooks, keep them in this directory and give them a short name that explains the question being investigated. A useful convention is:

```text
notebooks/
├── 01_data_quality_checks.ipynb
├── 02_embedding_analysis.ipynb
├── 03_ablation_review.ipynb
└── README.md
```

Keep exploratory work separate from the reproducible training entry point. Record the dataset version, random seed, fold manifest, and output location in the first cells of any new notebook. This makes the notebook useful to another researcher without turning it into an undocumented second implementation.

## Author

Zulqarnain Ali — [zulqararnainali.com](https://zulqarnainali.com) · [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)
