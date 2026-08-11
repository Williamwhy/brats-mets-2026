"""
predict.py
==========
Runs nnU-Net v2 inference programmatically (no subprocess shell calls).
Uses predict_from_raw_data for full control over fold ensemble, TTA, etc.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def run_nnunet_inference(
    input_dir: Path,
    output_dir: Path,
    dataset_id: int,
    configuration: str,
    trainer: str,
    plans: str,
    folds: list,
    step_size: float = 0.5,
    use_tta: bool = True,
    save_probabilities: bool = False,
    num_processes_preprocessing: int = 2,
    num_processes_segmentation_export: int = 2,
):
    """
    Runs nnU-Net v2 inference using the Python API.

    Weights are expected at:
        $nnUNet_results/Dataset{dataset_id:03d}_*/
            {configuration}/
                {trainer}__{plans}/
                    fold_{n}/
                        checkpoint_final.pth
    """
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  Using device: %s", device)

    # Resolve model folder from nnUNet_results env variable
    results_folder = Path(os.environ["nnUNet_results"])
    dataset_folder = f"Dataset{dataset_id:03d}_BraTS_Subsampled"
    model_folder = (
        results_folder
        / dataset_folder
        / f"{trainer}__{plans}__{configuration}"
    )

    if not model_folder.exists():
        raise FileNotFoundError(
            f"Model weights not found at: {model_folder}\n"
            f"Make sure model_weights/ is correctly copied into the container."
        )

    logger.info("  Model folder: %s", model_folder)
    logger.info("  Folds: %s", folds)
    logger.info("  TTA: %s", use_tta)

    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_folder),
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )

    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(input_dir),
        output_folder_or_list_of_truncated_output_files=str(output_dir),
        save_probabilities=save_probabilities,
        overwrite=True,
        num_processes_preprocessing=num_processes_preprocessing,
        num_processes_segmentation_export=num_processes_segmentation_export,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )

    logger.info("  nnU-Net prediction complete. Raw outputs at: %s", output_dir)