#!/usr/bin/env python3
"""
BraTS METS 2026 - Main Inference Entrypoint
============================================
Synapse evaluation system calls this script with:
    --input-dir  /input   (read-only, contains patient subdirs)
    --output-dir /output  (writable, segmentation masks go here)

Input structure (per patient):
    /input/
      BraTS-MET-00001/
        BraTS-MET-00001-t1n.nii.gz
        BraTS-MET-00001-t1c.nii.gz
        BraTS-MET-00001-t2w.nii.gz
        BraTS-MET-00001-t2f.nii.gz

Output (one file per patient):
    /output/
      BraTS-MET-00001.nii.gz   <- segmentation mask
"""

import argparse
import logging
import sys
from pathlib import Path

from src.preprocess import prepare_nnunet_input
from src.predict import run_nnunet_inference
from src.postprocess import postprocess_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BraTS METS 2026 nnU-Net inference pipeline"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/input"),
        help="Directory containing patient subdirectories with NIfTI images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/output"),
        help="Directory where segmentation masks will be written",
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path("/tmp/brats_mets_inference"),
        help="Temporary working directory for nnU-Net I/O",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0"],
        help="nnU-Net folds to use (default: 0)",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=103,
        help="nnU-Net dataset ID your model was trained on",
    )
    parser.add_argument(
        "--configuration",
        type=str,
        default="3d_fullres",
        help="nnU-Net configuration (default: 3d_fullres)",
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainerStagedTverskyDoubleBranchFullTri",
        help="nnU-Net trainer class name",
    )
    parser.add_argument(
        "--plans",
        type=str,
        default="nnUNetResEncUNetMPlans",
        help="nnU-Net plans identifier",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("BraTS METS 2026 Inference Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input dir  : {args.input_dir}")
    logger.info(f"Output dir : {args.output_dir}")
    logger.info(f"Tmp dir    : {args.tmp_dir}")
    logger.info(f"Dataset ID : {args.dataset_id}")
    logger.info(f"Config     : {args.configuration}")
    logger.info(f"Trainer    : {args.trainer}")
    logger.info(f"Folds      : {args.folds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Reformat input into nnU-Net imagesTs layout
    # -----------------------------------------------------------------------
    logger.info("Step 1/3 — Preprocessing: reformatting input for nnU-Net ...")
    nnunet_input_dir = args.tmp_dir / "imagesTs"
    patient_id_map = prepare_nnunet_input(
        input_dir=args.input_dir,
        nnunet_input_dir=nnunet_input_dir,
    )
    logger.info(f"  Found {len(patient_id_map)} patient(s).")

    # -----------------------------------------------------------------------
    # Step 2: Run nnU-Net prediction
    # -----------------------------------------------------------------------
    logger.info("Step 2/3 — Inference: running nnU-Net predict ...")
    nnunet_output_dir = args.tmp_dir / "predictions_raw"
    run_nnunet_inference(
        input_dir=nnunet_input_dir,
        output_dir=nnunet_output_dir,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        trainer=args.trainer,
        plans=args.plans,
        folds=args.folds,
    )

    # -----------------------------------------------------------------------
    # Step 3: Post-process and write final segmentation masks
    # -----------------------------------------------------------------------
    logger.info("Step 3/3 — Postprocessing: converting labels to BraTS convention ...")
    postprocess_predictions(
        raw_pred_dir=nnunet_output_dir,
        output_dir=args.output_dir,
        patient_id_map=patient_id_map,
    )

    logger.info("=" * 60)
    logger.info("Inference complete. Results written to: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
