# BraTS 2026 Brain Metastases (METS) Segmentation - Team SAMAI

Docker submission for the [BraTS 2026 Brain Metastases Segmentation Challenge](https://www.synapse.org/#!Synapse:syn75093060), using a custom nnU-Net v2 trainer with Tversky loss and a triple-encoder, dual-branch architecture.

> **Team SAMAI**
> **Synapse project:** `syn75093060`
> **Submission image tag:** `v2.0`

---

## Overview

This repository contains the full inference pipeline submitted for the BraTS METS challenge: preprocessing, model inference, and postprocessing, packaged as a self-contained Docker container.

The core model is built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), extended with a custom trainer (`nnUNetTrainerStagedTverskyDoubleBranchFullTri`).

## Repository Structure

```
brats-mets-repo/
├── Dockerfile              # Container definition for the submission
├── requirements.txt        # Python dependencies
├── run_model.py             # Pipeline entrypoint: preprocess → predict → postprocess
├── nnunet/                  # nnU-Net workspace (results, plans, custom trainer registration)
│   └── nnUNet_results/
│       └── Dataset103_BraTS_Subsampled/
│           └── nnUNetTrainerStagedTverskyDoubleBranchFullTri__nnUNetResEncUNetMPlans__3d_fullres/
│               └── fold_0/
│                   └── checkpoint_final.pth   # tracked via Git LFS
├── src/
│   ├── preprocess.py         # Input preparation / resampling
│   ├── predict.py            # Model inference
│   ├── postprocess.py        # Label cleanup, region reconstruction
│   ├── utils.py               # Shared helper functions
│   └── custom_trainers/       # Custom nnU-Net trainer(s)
└── .gitattributes            # Git LFS tracking rules
```

## Requirements

- Docker (with GPU support via `nvidia-container-toolkit` for inference)
- Python dependencies are listed in `requirements.txt` and installed automatically during the Docker build
- Model weights (`checkpoint_final.pth`, ~400 MB) are tracked via [Git LFS](https://git-lfs.github.com/) — run `git lfs install` before cloning, or `git lfs pull` after cloning, to fetch the checkpoint
- input required to be in standard BraTS format (.nii.gz and in BraTS naming convention)

## Build

To build the image locally:

```bash
docker build -t ghcr.io/williamwhy/brats-mets-2026:v2.0 .
```

Alternatively, pull the pre-built image directly from GitHub Container Registry:

```bash
docker pull ghcr.io/williamwhy/brats-mets-2026:v2.0
```

## Usage

```bash
docker run --rm --gpus all \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  ghcr.io/williamwhy/brats-mets-2026:v2.0
```

Adjust volume mounts to match the input/output conventions expected by the Synapse evaluation harness.

The pipeline runs in three stages (see `run_model.py`):

1. **Preprocess** (`src/preprocess.py`) — resamples and prepares input MRI sequences
2. **Predict** (`src/predict.py`) — runs inference using fold 0 of the trained model
3. **Postprocess** (`src/postprocess.py`) — reconstructs final region labels from model output

## Model Details

| | |
|---|---|
| Base framework | nnU-Net v2 |
| Custom trainer | `nnUNetTrainerStagedTverskyDoubleBranchFullTri` |
| Plans | `nnUNetResEncUNetMPlans` |
| Configuration | `3d_fullres` |
| Fold | 0 |
| Loss | Tversky, CCdice and cross entropy |
| Target subregions | ET, RC, NETC, SNFH |

## Results

| Metric | ET | RC | NETC | SNFH |
|---|---|---|---|---|
| DSC | 0.60 | 0.40 | 0.61 | 0.54 |
| NSD | 0.66 | 0.31 | 0.67 | 0.57 |
| Large F1 | 0.69 | 0.07 | 0.69 | 0.68 |
| Small F1 | 0.45 | 0.00 | 0.46 | 0.35 |

*[Fill in validation/leaderboard results here]*

## Notes

- Custom trainers must be registered inside nnU-Net's package directory prior to inference (see `Dockerfile` for setup steps).
- Container built and tested with Docker Desktop + WSL2 on Windows.

## License

[Apache 2.0]

<!--## Citation

If you use this work, please cite:

```
[Add citation / BibTeX here once available]
```
-->
## Acknowledgments

This work was part of the Sprint AI Training for African Medical Imaging Knowledge Translation (SPARK) Academy 2026 summer school on deep learning in medical imaging. The authors would like to thank the instructors of the summer for providing insightful background knowledge on brain tumours that informed the research presented here, most notably: Maruf Adewole, Mohannad Barakat, Craig Jones, Noha Magdy, Tinashe Mutsvangwa, MacLean Nasrallah, Nicepho-rus Boniface Rutabasibwa, Charles Delahunt, Celia Cintas, Evan Calabrese, and Amal Saleh. The authors acknowledge the funding support provided to SPARK through the Lacuna Fund for Health and Equity(PI: Udunna Anazodo), the RSNAR & E Foundation  (PI: Farouk Dako), the University of Washington Population Health Initiative Tier2 Grant (PI: Mehmet Kurt), McGill University Healthy Brain and Healthy Lives (HBHL, Anazodo), and the National Science and Engineering Research Council of Canada (NSERC) Discovery Launch Supplement (PI:UdunnaAnazodo,DGECR-2022-00136).
