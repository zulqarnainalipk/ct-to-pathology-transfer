# Data

Both datasets are public collections of The Cancer Imaging Archive (TCIA). They are not redistributed here; download them from TCIA under the collections' data-use terms and cite them as requested on the collection pages.

| Role | Collection | DOI |
|---|---|---|
| Source (CT) | [LungCT-Diagnosis](https://www.cancerimagingarchive.net/collection/lungct-diagnosis/) | [10.7937/K9/TCIA.2015.A6V7JIWX](https://doi.org/10.7937/K9/TCIA.2015.A6V7JIWX) |
| Target (pathology) | [HNSCC-mIF-mIHC-comparison](https://www.cancerimagingarchive.net/collection/hnscc-mif-mihc-comparison/) | [10.7937/TCIA.2020.T90F-WB82](https://doi.org/10.7937/TCIA.2020.T90F-WB82) |

## Expected folder layout

`main.py` reads the two folders given by `--ct_dir` and `--path_dir` (defaults `./data/LungCT-Diagnosis/LungCT-Diagnosis` and `./data/PKG - HNSCC-mIF-mIHC`):

```text
LungCT-Diagnosis/LungCT-Diagnosis/
  R_004/<study date>/<series>/*.dcm
  R_006/...
  ...                                   61 patient folders

PKG - HNSCC-mIF-mIHC/
  mIF_Data/Case1_T1_0_0_CD3.png ...     multiplex immunofluorescence channels (used)
  mIHC_Data/...                         multiplex immunohistochemistry (not used)
  Segmentation/Case1_T1_0_0.png ...     nuclear segmentation images (not used, see below)
```

## Source domain: LungCT-Diagnosis

61 diagnostic thoracic CT series, one per patient, originally collected to study CT descriptors of lung adenocarcinoma. The series have 24 to 150 axial slices (mean 76.8) and slice thicknesses of 2.5 to 6 mm. Diagnostic labels are not used; the CT data serve only for self-supervised pretraining and the feature bank.

Preprocessing: stored values are converted to Hounsfield units with the DICOM rescale slope and intercept, slices are ordered by z-position, 16 slices are sampled uniformly over the whole series, values are windowed to [-1000, 400] HU and scaled to [0, 1], and each slice is resized to 128 x 128.

## Target domain: HNSCC-mIF-mIHC-comparison

Eight HNSCC patients (`Case1` to `Case8`). The same tumour sections were stained with mIF, restained with mIHC and co-registered. For each patient an experienced pathologist selected nine regions of interest on both the mIF and mIHC images: three in the tumour core (`T1`-`T3`), three at the tumour margin (`M1`-`M3`) and three in the adjacent stroma (`S1`-`S3`). Each region is split into up to four 512 x 512 patches, giving 268 tiles in the release (of at most 288). File names encode `Case[patient]_[T/M/S][1-3]_[patch]_[marker]`, and the `CaseN` prefix defines the patient groups for cross-validation.

Reference for the collection: Ghahremani et al., "An AI-Ready Multiplex Staining Dataset for Reproducible and Accurate Characterization of Tumor Immune Microenvironment", MICCAI 2023, LNCS 14225, pp. 704-713, doi:10.1007/978-3-031-43987-2_68.

### Labels used in this study

Tumour core (T) is the positive class and stroma (S) the negative class; margin tiles (M) contain both tissue types and are excluded. This gives 172 tiles (94 tumour, 78 stroma), and every patient contributes both classes:

| Patient | T | M | S | Used (T / S) |
|---|---:|---:|---:|---:|
| Case1 | 12 | 12 | 5 | 12 / 5 |
| Case2 | 12 | 12 | 12 | 12 / 12 |
| Case3 | 11 | 12 | 4 | 11 / 4 |
| Case4 | 12 | 12 | 9 | 12 / 9 |
| Case5 | 12 | 12 | 12 | 12 / 12 |
| Case6 | 12 | 12 | 12 | 12 / 12 |
| Case7 | 11 | 12 | 12 | 11 / 12 |
| Case8 | 12 | 12 | 12 | 12 / 12 |
| Total | 94 | 96 | 78 | 94 / 78 |

The five mIF channels (CD3, CD8, FoxP3, PanCK, DAPI) are resized to 224 x 224 and masked with a label-blind region derived from the 75th percentile of the DAPI channel.

### Note on the segmentation images

The `Segmentation/` images are nuclear instance masks (filled nuclei and their one-pixel boundaries) produced by the data contributors with an interactive whole-cell/nuclear segmentation tool. They are **not tumour masks**. A label obtained by thresholding their pixel fraction (for example at 5%) measures nuclear density: it disagrees with the region type for 55 of 94 tumour-core and 18 of 78 stroma tiles, and it can be predicted from the mean DAPI intensity alone (AUC 0.899 on the 172 T/S tiles). Version 1.0 of this repository used such a label; the current version does not use the segmentation images at all. The label audit is reproduced in `results/aggregate/tables/label_audit.csv` and `single_channel_auc.csv`, and `main.py` keeps the old definition only as `label_mode='legacy_mask'` for reference.

### Task difficulty

Tumour core and stroma differ strongly in the epithelial marker PanCK: its mean intensity alone separates the two classes with an AUC of 0.998, and a logistic regression on 15 per-channel intensity statistics reaches 98.1% balanced accuracy. Any model on this task should be compared against such a baseline.
