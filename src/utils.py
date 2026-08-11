"""
utils.py
========
Shared utility functions for the BraTS METS 2026 inference pipeline.
"""

import hashlib
import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def verify_nifti(path: Path) -> bool:
    """Check that a NIfTI file is readable and non-empty."""
    try:
        img = nib.load(str(path))
        arr = np.asarray(img.dataobj)
        return arr.size > 0
    except Exception as e:
        logger.error("NIfTI verification failed for %s: %s", path, e)
        return False


def md5_file(path: Path) -> str:
    """Return MD5 hex digest of a file (useful for weight integrity checks)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def count_patients(input_dir: Path) -> int:
    """Count patient subdirectories in input_dir."""
    return sum(1 for p in input_dir.iterdir() if p.is_dir())
