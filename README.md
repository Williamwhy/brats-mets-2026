# BraTS 2026 Brain Metastases (METS) Segmentation

Docker submission for the [BraTS 2026 Brain Metastases Segmentation Challenge](https://www.synapse.org/#!Synapse:syn75093060), using a custom nnU-Net v2 trainer with Tversky loss and a triple-encoder, dual-branch architecture.

> **Team / Author:** [Your name or team name here]
> **Synapse project:** `syn75093060`
> **Submission image tag:** `v2.0`

---

## Overview

This repository contains the full inference pipeline submitted for the BraTS METS challenge: preprocessing, model inference, and postprocessing, packaged as a self-contained Docker container.

The core model is built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), extended with a custom trainer (`nnUNetTrainerStagedTverskyDoubleBranchFullTri`) that combines:

- **Tversky loss** to better handle the severe class imbalance between tumor subregions and background
- **A triple-encoder, dual-branch architecture** to separate learned representations across tumor subregions
- **Region-based segmentation** across four target subregions: enhancing tumor (ET), resection cavity (RC), non-enhancing tumor core (NETC), and surrounding non-enhancing FLAIR hyperintensity (SNFH)

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
| Loss | Tversky |
| Target subregions | ET, RC, NETC, SNFH |

## Results

| Metric | ET | RC | NETC | SNFH |
|---|---|---|---|---|
| DSC | — | — | — | — |
| NSD | — | — | — | — |

*[Fill in validation/leaderboard results here]*

## Notes

- Custom trainers must be registered inside nnU-Net's package directory prior to inference (see `Dockerfile` for setup steps).
- Container built and tested with Docker Desktop + WSL2 on Windows.

## License

[Add license here, e.g. MIT, Apache 2.0]

## Citation

If you use this work, please cite:

```
[Add citation / BibTeX here once available]
```

## Acknowledgments

Developed as part of doctoral research on adaptive radiotherapy and AI-driven clinical tools under resource-constrained compute environments.
