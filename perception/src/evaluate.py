#!/usr/bin/env python3
"""
Model Evaluation and Metric Auditing Script for Industrial Switch State Detection.

European Rover Challenge (ERC) - Maintenance Task (Autonomous 6-DOF Robotic Arm)
Perception Module

Target Classes:
    - main_off (Class 0): Red lever oriented vertically (pointing downwards / "0" state).
    - main_on  (Class 1): Red lever turned horizontally (pointing rightwards / "1" state).

Functionality:
    - Evaluates fine-tuned YOLO checkpoints against the validation split (images/val).
    - Extracts and formats overall mAP@0.5, mAP@0.5:0.95, Precision, and Recall.
    - Generates per-class breakdown for 'main_off' and 'main_on'.
    - Audits hardware latency (preprocess, inference, loss, postprocess in ms & FPS).
    - Reports paths to generated evaluation artifacts (confusion matrices, PR curves, visual predictions).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Benchmark and audit YOLO model performance on ERC switch validation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to model weights checkpoint (.pt). Defaults to 'models/best.pt' or baseline run.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/switches.yaml",
        help="Path to YOLO dataset YAML configuration file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate against.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (square dimension).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for metric evaluation.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.60,
        help="NMS IoU threshold for metric evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: '0', 'cpu', or None for auto-detection.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="val_evaluation",
        help="Subdirectory name for this evaluation run.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save metrics summary to JSON format inside the run directory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed Ultralytics logging output.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str | None) -> str:
    """Resolve and validate the compute device.

    Args:
        requested_device: User-specified device string or None.

    Returns:
        Validated device string ('0', 'cpu', etc.).
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
        logger.info(f"CUDA hardware detected: {gpu_name}. Selecting device '0'.")
        return "0"

    logger.info("No CUDA hardware detected. Selecting CPU.")
    return "cpu"


def find_default_weights(requested_weights: str | None, project_root: Path) -> Path:
    """Locate the most appropriate weights checkpoint if none is explicitly specified.

    Args:
        requested_weights: CLI-specified weights path or None.
        project_root: Workspace root directory.

    Returns:
        Path to validated weights file.

    Raises:
        FileNotFoundError: If no valid weights checkpoint can be found.
    """
    if requested_weights:
        weights_path = Path(requested_weights)
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Specified weights file not found: {weights_path}\n"
                "Please verify the path or train a model first using 'python src/train.py'."
            )
        return weights_path

    # Check candidates in order of priority
    candidates = [
        project_root / "models" / "best.pt",
        project_root / "runs" / "detect" / "switches_baseline" / "weights" / "best.pt",
        project_root / "runs" / "detect" / "switches_baseline" / "weights" / "last.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Auto-selected checkpoint: {candidate}")
            return candidate

    # Provide actionable error message
    searched_list = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "No fine-tuned model checkpoint found. Searched candidate locations:\n"
        f"{searched_list}\n\n"
        "Please run training first:\n"
        "  python src/train.py\n"
        "Or provide an explicit weights file using:\n"
        "  python src/evaluate.py --weights <path/to/checkpoint.pt>"
    )


def format_metrics_table(
    overall_metrics: dict[str, float],
    class_metrics: list[dict[str, Any]],
    speed_metrics: dict[str, float],
) -> str:
    """Format evaluation metrics into an ASCII report.

    Args:
        overall_metrics: Overall P, R, mAP50, mAP50-95.
        class_metrics: Per-class breakdown metrics.
        speed_metrics: Latency in ms for preprocess, inference, loss, postprocess.

    Returns:
        Formatted ASCII text table.
    """
    lines = [
        "=" * 78,
        "          ERC PERCEPTION MODULE - SWITCH DETECTION EVALUATION REPORT          ",
        "=" * 78,
        "",
        "--- [1. OVERALL DETECTION METRICS] ---",
        f"  Precision (P)       : {overall_metrics['precision']:.4f}  ({overall_metrics['precision']*100:.1f}%)",
        f"  Recall (R)          : {overall_metrics['recall']:.4f}  ({overall_metrics['recall']*100:.1f}%)",
        f"  mAP @ 0.50          : {overall_metrics['map50']:.4f}  ({overall_metrics['map50']*100:.1f}%)",
        f"  mAP @ 0.50:0.95     : {overall_metrics['map']:.4f}  ({overall_metrics['map']*100:.1f}%)",
        "",
        "--- [2. PER-CLASS METRIC BREAKDOWN] ---",
        f"{'Class Name':<16} {'Class ID':<10} {'Precision':<12} {'Recall':<12} {'mAP@0.5':<12} {'mAP@0.5:0.95':<12}",
        "-" * 74,
    ]

    for cm in class_metrics:
        lines.append(
            f"{cm['name']:<16} {cm['id']:<10} {cm['precision']:<12.4f} {cm['recall']:<12.4f} "
            f"{cm['map50']:<12.4f} {cm['map']:<12.4f}"
        )

    lines.append("-" * 74)
    lines.append("")
    lines.append("--- [3. LATENCY & THROUGHPUT AUDIT] ---")

    prep = speed_metrics.get("preprocess", 0.0)
    inf = speed_metrics.get("inference", 0.0)
    loss = speed_metrics.get("loss", 0.0)
    post = speed_metrics.get("postprocess", 0.0)
    total_ms = prep + inf + post
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0

    lines.extend([
        f"  Pre-processing      : {prep:6.2f} ms",
        f"  Inference           : {inf:6.2f} ms",
        f"  Loss calculation    : {loss:6.2f} ms",
        f"  Post-processing     : {post:6.2f} ms",
        f"  Total Latency/Frame : {total_ms:6.2f} ms",
        f"  Estimated Throughput: {fps:6.1f} FPS (real-time robotics requirement: >= 15 FPS)",
        "=" * 78,
    ])
    return "\n".join(lines)


def evaluate() -> None:
    """Main model validation & auditing entrypoint."""
    args = parse_args()
    project_root = Path.cwd()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config YAML not found: {data_path}")

    # Resolve weights checkpoint
    weights_path = find_default_weights(args.weights, project_root)
    device = resolve_device(args.device)

    logger.info("=" * 70)
    logger.info("STARTING VALIDATION AUDIT")
    logger.info(f"Target checkpoint : {weights_path}")
    logger.info(f"Dataset config    : {data_path}")
    logger.info(f"Split             : {args.split}")
    logger.info(f"Image resolution  : {args.imgsz}x{args.imgsz}")
    logger.info(f"Confidence thresh : {args.conf}")
    logger.info(f"IoU threshold     : {args.iou}")
    logger.info(f"Compute device    : {device}")
    logger.info("=" * 70)

    # Lazy-import Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        logger.error("Ultralytics package is missing. Install via: pip install -r requirements.txt")
        raise exc

    # Load model
    model = YOLO(str(weights_path))

    # Run validation
    validation_args: dict[str, Any] = {
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": args.conf,
        "iou": args.iou,
        "device": device,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "plots": True,
        "verbose": args.verbose,
    }

    logger.info("Executing Ultralytics model.val()...")
    metrics = model.val(**validation_args)

    # Extract overall metrics
    box_metrics = metrics.box
    overall = {
        "precision": float(box_metrics.mp),
        "recall": float(box_metrics.mr),
        "map50": float(box_metrics.map50),
        "map": float(box_metrics.map),
    }

    # Extract per-class metrics
    class_names = metrics.names
    class_metrics_list = []

    # box_metrics.p, box_metrics.r, box_metrics.ap50, box_metrics.ap are arrays indexed by class
    precisions = box_metrics.p if hasattr(box_metrics, "p") and len(box_metrics.p) > 0 else []
    recalls = box_metrics.r if hasattr(box_metrics, "r") and len(box_metrics.r) > 0 else []
    ap50s = box_metrics.ap50 if hasattr(box_metrics, "ap50") and len(box_metrics.ap50) > 0 else []
    maps = box_metrics.ap if hasattr(box_metrics, "ap") and len(box_metrics.ap) > 0 else []

    for cls_id, name in sorted(class_names.items()):
        cls_p = float(precisions[cls_id]) if cls_id < len(precisions) else 0.0
        cls_r = float(recalls[cls_id]) if cls_id < len(recalls) else 0.0
        cls_ap50 = float(ap50s[cls_id]) if cls_id < len(ap50s) else 0.0
        cls_map = float(maps[cls_id]) if cls_id < len(maps) else 0.0

        class_metrics_list.append({
            "id": cls_id,
            "name": name,
            "precision": cls_p,
            "recall": cls_r,
            "map50": cls_ap50,
            "map": cls_map,
        })

    # Speed metrics
    speed = metrics.speed if hasattr(metrics, "speed") else {}

    # Print formatted report to stdout
    report = format_metrics_table(overall, class_metrics_list, speed)
    print("\n" + report + "\n")

    # Discover and report validation artifacts
    save_dir = (
        Path(metrics.save_dir).resolve()
        if hasattr(metrics, "save_dir")
        else (project_root / args.project / args.name).resolve()
    )
    logger.info("--- [4. GENERATED VALIDATION ARTIFACTS] ---")
    logger.info(f"Artifact directory: {save_dir}")

    common_artifacts = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "val_batch0_pred.jpg",
        "val_batch0_labels.jpg",
    ]

    found_artifacts = []
    for art_name in common_artifacts:
        art_path = save_dir / art_name
        if art_path.exists():
            logger.info(f"  [+] {art_name:<30} -> {art_path}")
            found_artifacts.append(str(art_path))
        else:
            logger.debug(f"  [-] {art_name} not generated")

    # Save structured summary to JSON
    summary_data = {
        "weights": str(weights_path),
        "dataset_config": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "overall_metrics": overall,
        "per_class_metrics": class_metrics_list,
        "speed_ms": speed,
        "artifact_dir": str(save_dir),
        "artifacts": found_artifacts,
    }

    json_path = save_dir / "evaluation_summary.json"
    json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    logger.info(f"Evaluation summary saved to: {json_path}")


def main() -> None:
    try:
        evaluate()
    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user (KeyboardInterrupt).")
        sys.exit(130)
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
