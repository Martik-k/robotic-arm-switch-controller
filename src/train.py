#!/usr/bin/env python3
"""
Model Training & Fine-Tuning Script for Industrial Switch State Detection.

European Rover Challenge (ERC) - Maintenance Task (Autonomous 6-DOF Robotic Arm)
Perception Module

Target Classes:
    - main_off (Class 0): Red lever oriented vertically (pointing downwards / "0" state).
    - main_on  (Class 1): Red lever turned horizontally (pointing rightwards / "1" state).

Domain-Specific Constraints:
    - fliplr=0.0: MUST be disabled. Horizontal flipping mirrors 'main_on' (pointing right)
                  into pointing left (an invalid/undefined state in the ERC rulebook).
    - flipud=0.0: MUST be disabled. Vertical flipping mirrors 'main_off' (pointing down)
                  into pointing up.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO model for ERC Industrial Panel Switch Detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/switches.yaml",
        help="Path to YOLO dataset YAML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained checkpoint to fine-tune (e.g. yolov8n.pt, yolo11n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without validation mAP improvement).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Training batch size (8 or 16 recommended for mobile GPUs like RTX 4050 6GB).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (square dimension).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: '0', '0,1', 'cpu', or None for auto-detection.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Destination directory for training experiments.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="switches_baseline",
        help="Experiment name / run subdirectory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for training reproducibility.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader worker processes (set 0 if debugging on Windows).",
    )
    return parser.parse_args()


def resolve_device(requested_device: str | None) -> str:
    """Resolve and validate the execution device (CUDA GPU vs CPU).

    Args:
        requested_device: User-specified device string or None.

    Returns:
        Validated device string for Ultralytics YOLO ('0', 'cpu', etc.).
    """
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if requested_device is not None:
        clean = requested_device.strip().lower()
        if clean != "cpu" and not cuda_available:
            logger.warning(
                f"Requested device '{requested_device}', but CUDA is not available. Falling back to CPU."
            )
            return "cpu"
        return requested_device

    if cuda_available:
        import torch

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"CUDA hardware detected: {gpu_name} ({vram_gb:.1f} GB VRAM). Selecting device '0'.")
        return "0"

    logger.info("No CUDA hardware detected. Selecting CPU.")
    return "cpu"


def ensure_dataset_ready(data_config_path: Path) -> None:
    """Verify that dataset configuration and image splits exist before training."""
    if not data_config_path.exists():
        raise FileNotFoundError(
            f"Dataset configuration file not found: {data_config_path}. "
            "Please ensure 'configs/switches.yaml' exists."
        )

    # Check project-relative data split directories
    project_root = data_config_path.resolve().parent.parent
    train_dir = project_root / "data" / "images" / "train"
    val_dir = project_root / "data" / "images" / "val"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        logger.warning(
            f"Training image directory '{train_dir}' is missing or empty. "
            "Checking whether 'scripts/prepare_dataset.py' needs to be run..."
        )
        prep_script = project_root / "scripts" / "prepare_dataset.py"
        if prep_script.exists():
            logger.info("Executing 'scripts/prepare_dataset.py' to generate 'data/' splits...")
            import subprocess

            result = subprocess.run([sys.executable, str(prep_script)], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Dataset preparation failed: {result.stderr}")
            else:
                logger.info("Dataset prepared successfully.")


def deploy_checkpoint(
    source_weights_dir: Path,
    deployment_dir: Path,
    model_name: str,
    training_args: dict[str, Any],
) -> Path:
    """Copy the best checkpoint to the production deployment directory.

    Args:
        source_weights_dir: Directory containing 'best.pt' (e.g. runs/detect/.../weights).
        deployment_dir: Target directory (e.g., 'models').
        model_name: Name of model weights file (default: 'best.pt').
        training_args: Training metadata dictionary for auditing.

    Returns:
        Path to the deployed model checkpoint.
    """
    deployment_dir.mkdir(parents=True, exist_ok=True)
    best_source = source_weights_dir / "best.pt"
    deployed_target = deployment_dir / model_name

    if not best_source.exists():
        logger.warning(f"Could not find '{best_source}'. Checking for 'last.pt'...")
        last_source = source_weights_dir / "last.pt"
        if last_source.exists():
            shutil.copy2(last_source, deployed_target)
            logger.info(f"Deployed fallback 'last.pt' -> {deployed_target}")
            return deployed_target
        raise FileNotFoundError(f"Neither best.pt nor last.pt found in {source_weights_dir}")

    shutil.copy2(best_source, deployed_target)
    logger.info(f"Production model successfully deployed: {best_source} -> {deployed_target}")

    # Write deployment metadata JSON
    metadata_path = deployment_dir / "model_metadata.json"
    metadata = {
        "model_file": model_name,
        "deployed_at": datetime.now().isoformat(),
        "source_checkpoint": str(best_source),
        "target_classes": {0: "main_off", 1: "main_on"},
        "training_parameters": training_args,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Deployment metadata saved to {metadata_path}")
    return deployed_target


def train() -> None:
    """Main training execution workflow."""
    args = parse_args()
    project_root = Path.cwd()

    data_config = Path(args.data)
    if not data_config.is_absolute():
        data_config = project_root / data_config

    logger.info("=" * 70)
    logger.info("ERC ROBOTIC ARM SWITCH CONTROLLER - PERCEPTION TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Dataset config   : {data_config}")
    logger.info(f"Base model       : {args.model}")
    logger.info(f"Target epochs    : {args.epochs} (patience={args.patience})")
    logger.info(f"Batch size       : {args.batch}")
    logger.info(f"Image resolution : {args.imgsz}x{args.imgsz}")

    # Verify dataset readiness
    ensure_dataset_ready(data_config)

    # Determine execution device
    device = resolve_device(args.device)

    # Lazy-import Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError as e:
        logger.error(
            "Ultralytics is not installed. Please install dependencies via: "
            "pip install -r requirements.txt"
        )
        raise e

    # Initialize model from pretrained checkpoint
    logger.info(f"Loading pretrained model: '{args.model}'...")
    model = YOLO(args.model)

    # Hyperparameter & Augmentation settings
    # CRITICAL: fliplr=0.0 and flipud=0.0 are strictly enforced because horizontal/vertical
    # switch orientations encode the physical operational state ('main_on' vs 'main_off').
    training_kwargs: dict[str, Any] = {
        "data": str(data_config),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "seed": args.seed,
        "workers": args.workers,
        # Directional augmentation constraints
        "fliplr": 0.0,  # MUST BE 0.0: Prevents mirroring horizontal ON into invalid direction
        "flipud": 0.0,  # MUST BE 0.0: Prevents mirroring vertical OFF into pointing upwards
        # Photometric variations to simulate rover lighting/shadows
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.4,
        # Subtle geometric variations
        "degrees": 12.0,       # Slight camera / end-effector roll tilt
        "translate": 0.08,     # Small spatial translation
        "scale": 0.15,         # Natural scale change due to distance
        "perspective": 0.0005, # Mild 3D perspective distortion
        "mosaic": 0.5,         # Composite background context
        "mixup": 0.0,          # Avoid switch ghosting
        # Optimization & Checkpointing
        "optimizer": "auto",
        "save": True,
        "save_period": 10,
        "val": True,
        "plots": True,
        "verbose": True,
    }

    logger.info("Starting model fine-tuning with domain-specific augmentation constraints...")
    results = model.train(**training_kwargs)

    # Artifact management: deploy best.pt into models/
    run_save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else Path(args.project) / args.name
    weights_dir = run_save_dir / "weights"
    models_dir = project_root / "models"

    deployed_path = deploy_checkpoint(
        source_weights_dir=weights_dir,
        deployment_dir=models_dir,
        model_name="best.pt",
        training_args={
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": device,
            "model_type": args.model,
            "fliplr": 0.0,
            "flipud": 0.0,
        },
    )

    logger.info("=" * 70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info(f"Run outputs      : {run_save_dir}")
    logger.info(f"Production model : {deployed_path}")
    logger.info("=" * 70)


def main() -> None:
    try:
        train()
    except KeyboardInterrupt:
        logger.warning("\nTraining aborted by user (KeyboardInterrupt). Exiting cleanly.")
        sys.exit(130)
    except Exception as exc:
        logger.error(f"Fatal error during training: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
