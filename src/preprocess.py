"""
preprocess.py
=============
Converts BraTS 2026 METS input layout into the nnU-Net imagesTs format.

BraTS 2026 input (per patient folder):
    BraTS-MET-00001-t1n.nii.gz   -> channel 0000
    BraTS-MET-00001-t1c.nii.gz   -> channel 0001
    BraTS-MET-00001-t2w.nii.gz   -> channel 0002
    BraTS-MET-00001-t2f.nii.gz   -> channel 0003

nnU-Net imagesTs output:
    BraTS-MET-00001_0000.nii.gz
    BraTS-MET-00001_0001.nii.gz
    BraTS-MET-00001_0002.nii.gz
    BraTS-MET-00001_0003.nii.gz
"""

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Ordered modality suffixes → nnU-Net channel indices
MODALITY_CHANNEL_MAP = {
    "t1n": "0000",   # T1 native
    "t1c": "0001",   # T1 contrast-enhanced
    "t2w": "0002",   # T2-weighted
    "t2f": "0003",   # T2-FLAIR
}


def prepare_nnunet_input(
    input_dir: Path,
    nnunet_input_dir: Path,
) -> dict:
    """
    Scans input_dir for patient subdirectories, symlinks/copies NIfTI files
    into nnunet_input_dir using the nnU-Net _XXXX channel naming convention.

    Returns a dict mapping nnU-Net case ID -> original patient ID, e.g.:
        {"BraTS-MET-00001": "BraTS-MET-00001"}
    """
    nnunet_input_dir.mkdir(parents=True, exist_ok=True)
    patient_id_map = {}

    patient_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not patient_dirs:
        # Flat layout fallback: images directly in input_dir
        patient_dirs = [input_dir]

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        found_modalities = {}
        for suffix, channel in MODALITY_CHANNEL_MAP.items():
            # Support both BraTS-MET-XXXXX-<mod>.nii.gz and <mod>.nii.gz
            candidates = list(patient_dir.glob(f"*{suffix}.nii.gz"))
            if not candidates:
                logger.warning(
                    "Patient %s: missing modality '%s', skipping.", patient_id, suffix
                )
                break
            src = candidates[0]
            dest = nnunet_input_dir / f"{patient_id}_{channel}.nii.gz"
            shutil.copy2(src, dest)
            found_modalities[suffix] = dest

        if len(found_modalities) == len(MODALITY_CHANNEL_MAP):
            patient_id_map[patient_id] = patient_id
            logger.info("  Prepared patient: %s", patient_id)
        else:
            logger.error(
                "Patient %s skipped — only found: %s",
                patient_id,
                list(found_modalities.keys()),
            )

    return patient_id_map
