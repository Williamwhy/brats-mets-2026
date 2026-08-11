"""
postprocess.py
==============
Converts nnU-Net predictions back to BraTS 2026 label convention and
writes one output file per patient to the final output directory.

BraTS 2026 METS label map:
    0 = background
    1 = non-enhancing tumor core (NETC)
    2 = surrounding non-enhancing FLAIR hyperintensity (SNFH)
    3 = enhancing tumor (ET)

If your nnU-Net training used a different internal label scheme, adjust
NNUNET_TO_BRATS below accordingly.
"""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

# Internal nnU-Net label -> BraTS 2026 METS output label
# Perfect 1:1 match — no remapping needed
NNUNET_TO_BRATS = {
    0: 0,   # background -> background
    1: 1,   # NET        -> NETC
    2: 2,   # ED         -> SNFH
    3: 3,   # ET         -> ET
    4: 4,   # RC         -> RC (resection cavity)
}


def convert_labels(seg_array: np.ndarray) -> np.ndarray:
    """Remap nnU-Net internal labels to BraTS 2026 convention."""
    out = np.zeros_like(seg_array)
    for src, dst in NNUNET_TO_BRATS.items():
        out[seg_array == src] = dst
    return out


def postprocess_predictions(
    raw_pred_dir: Path,
    output_dir: Path,
    patient_id_map: dict,
):
    """
    For each patient in patient_id_map:
      1. Load the raw nnU-Net prediction.
      2. Remap labels to BraTS 2026 convention.
      3. Save as <patient_id>.nii.gz in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0

    for case_id, patient_id in patient_id_map.items():
        # nnU-Net output filename matches the case ID used in imagesTs
        raw_pred_path = raw_pred_dir / f"{case_id}.nii.gz"

        if not raw_pred_path.exists():
            logger.error("Missing prediction for %s at %s", case_id, raw_pred_path)
            continue

        img = nib.load(str(raw_pred_path))
        seg_array = np.asarray(img.dataobj, dtype=np.uint8)

        # Label conversion
        converted = convert_labels(seg_array)

        # Save with original affine/header preserved
        out_img = nib.Nifti1Image(converted, affine=img.affine, header=img.header)
        out_img.header.set_data_dtype(np.uint8)

        out_path = output_dir / f"{patient_id}.nii.gz"
        nib.save(out_img, str(out_path))

        logger.info("  Saved: %s  (unique labels: %s)", out_path.name, np.unique(converted).tolist())
        processed += 1

    logger.info("Postprocessing complete: %d/%d patients.", processed, len(patient_id_map))