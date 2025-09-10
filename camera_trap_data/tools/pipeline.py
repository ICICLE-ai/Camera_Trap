from __future__ import annotations

import yaml
from typing import List, Optional

from datasets.dataset import Dataset
from tools.data_prep.ckp_prep import prepare_checkpoints
from tools.analysis.analysis import Analysis
from tools.data_prep.image_prep import crop_and_save_images
from tools.data_prep.utils.loader_utils import get_settings_loader


def _parse_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run(
    config_path: str,
    checkpoint_days_list: Optional[List[int]] = None,
    run_analysis: bool = False,
    prepare_data: bool = False,
    crop_images_once: bool = False,
    datasets: Optional[List[str]] = None,
    wandb_enabled: bool = False,
):
    """Execute the camera-trap pipeline.

    Args:
        config_path: Path to the YAML configuration file.
        checkpoint_days_list: List of day intervals to test; default [30].
        run_analysis: Whether to run analysis modules.
        prepare_data: Whether to write JSON files for each checkpoint interval.
        crop_images_once: If True, crop images once per camera before JSON prep.
        datasets: Optional dataset names overriding the config file.
        wandb_enabled: Enable W&B logging.
    """
    try:
        import wandb  # lazy import
    except Exception:
        wandb = None

    if checkpoint_days_list is None:
        checkpoint_days_list = [30]

    config = _parse_config(config_path)

    if datasets is not None:
        config["datasets"] = datasets
        print(f"Overriding datasets with: {datasets}")

    if wandb_enabled and wandb is not None:
        run_name = f"{datasets if datasets is not None else config.get('datasets', [])}"
        wandb.init(
            project="ICICLE Benchmark Preparation NZ",
            name=run_name,
            config={
                "datasets": datasets if datasets is not None else config.get("datasets", []),
                "checkpoint_days_list": checkpoint_days_list,
                "prepare_data": prepare_data,
                "crop_images_once": crop_images_once,
                "config_path": config_path,
                "global_config": config,
            },
        )

    # Get datetime formats from settings
    settings = get_settings_loader()
    datetime_formats = settings.get_datetime_formats()

    print("\n======================================== Multi-Day Analysis =========================================")
    print(f"Testing checkpoint intervals: {checkpoint_days_list} days")
    print(f"Prepare data: {prepare_data}")
    print(f"Crop images: {crop_images_once}")
    print(f"Datetime formats loaded: {len(datetime_formats)} formats")
    print("=" * 100)

    # Optional one-time cropping
    if crop_images_once:
        print("\n🖼️  Cropping images (one-time operation)...")
        temp_dataset = Dataset(config_path, config=config)
        prepare_checkpoints(temp_dataset, datetime_formats=datetime_formats, checkpoint_days=checkpoint_days_list[0])

        output_path = "/fs/scratch/PAS2099/camera-trap-benchmark"
        crop_and_save_images(temp_dataset, output_path, save_images=True, checkpoint_days=checkpoint_days_list[0])
        print("✅ Image cropping completed!")

    for checkpoint_days in checkpoint_days_list:
        print(f"\n🔄 Processing with {checkpoint_days}-day intervals...")
        dataset = Dataset(config_path, config=config)
        prepare_checkpoints(dataset, datetime_formats=datetime_formats, checkpoint_days=checkpoint_days)

        if run_analysis:
            Analysis(config_path, dataset, datetime_formats=datetime_formats, checkpoint_days=checkpoint_days, config=config)
            print(f"✅ Completed analysis for {checkpoint_days}-day intervals")

        if prepare_data:
            print(f"Preparing JSON files for {checkpoint_days}-day intervals...")
            output_path = "/fs/scratch/PAS2099/camera-trap-benchmark"
            crop_and_save_images(dataset, output_path, save_images=False, checkpoint_days=checkpoint_days)

    print("\n🎉 All runs completed! Check the analysis_results folder for CSV files.")
    if wandb_enabled and wandb is not None:
        wandb.finish()
