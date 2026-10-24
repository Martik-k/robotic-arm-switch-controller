#!/usr/bin/env python3
"""
Runtime Detector & Robotics Telemetry Pipeline for Industrial Switch State Detection.

European Rover Challenge (ERC) - Maintenance Task (Autonomous 6-DOF Robotic Arm)
Perception Module

Target Classes:
    - main_off (Class 0): Red lever oriented vertically (pointing downwards / "0" state).
    - main_on  (Class 1): Red lever turned horizontally (pointing rightwards / "1" state).

Features:
    - Versatile input source support: static image, directory of images, video file (.mp4), or live webcam feed.
    - Dual output modes:
        1. Visual Output: Professional HUD with class styling, center crosshairs, and FPS overlay.
        2. Robotics Telemetry Payload: Structured dict formatted for direct consumption by ROS 2 nodes
           or RL (Gymnasium / MuJoCo) policy environments.
    - Clean resource management and graceful shutdown on Ctrl+C or 'q'.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("inference")

# Class styling palette (BGR format for OpenCV)
CLASS_COLORS = {
    0: (40, 60, 240),    # main_off: High-visibility Red/Crimson
    1: (60, 215, 60),    # main_on : High-visibility Emerald Green
}
CLASS_NAMES = {
    0: "main_off",
    1: "main_on",
}


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Run switch detection inference on images, video files, or live webcam feeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input modality: path to image, directory of images, MP4 video file, or integer webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to YOLO model checkpoint. Defaults to 'models/best.pt' or baseline run.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (square dimension).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: '0', 'cpu', or None for auto-detection.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="runs/predict",
        help="Destination directory to save annotated outputs.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live OpenCV window during inference (press 'q' to exit).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save structured robotics telemetry payload to JSON.",
    )
    return parser.parse_args()


def resolve_device(requested_device: str | None) -> str:
    """Resolve compute hardware device."""
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
        return "0"
    return "cpu"


def find_default_weights(requested_weights: str | None, project_root: Path) -> Path:
    """Locate model checkpoint file."""
    if requested_weights:
        weights_path = Path(requested_weights)
        if not weights_path.is_absolute():
            weights_path = project_root / weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights file not found: {weights_path}")
        return weights_path

    candidates = [
        project_root / "models" / "best.pt",
        project_root / "runs" / "detect" / "switches_baseline" / "weights" / "best.pt",
        project_root / "runs" / "detect" / "switches_baseline" / "weights" / "last.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched_list = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        "No fine-tuned model checkpoint found. Searched candidate locations:\n"
        f"{searched_list}\n\n"
        "Please train a model first using: python src/train.py"
    )


class SwitchDetector:
    """Production-grade detector and telemetry extractor for industrial panel switches.

    Designed for modular integration into autonomous robotic manipulation pipelines
    (e.g., ROS 2 perception nodes, Gymnasium / MuJoCo RL environments).
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        """Initialize the SwitchDetector.

        Args:
            weights_path: Path to YOLO model checkpoint (.pt).
            conf_threshold: Minimum confidence score for valid detections.
            iou_threshold: Non-Maximum Suppression (NMS) IoU overlap threshold.
            imgsz: Resolution for model inference.
            device: Compute device ('0', 'cpu', etc.).
        """
        from ultralytics import YOLO

        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = resolve_device(device)

        logger.info(f"Initializing SwitchDetector with checkpoint: {self.weights_path}")
        logger.info(f"Execution device: {self.device} | conf={self.conf_threshold} | iou={self.iou_threshold}")
        self.model = YOLO(str(self.weights_path))

    def detect_frame(
        self,
        frame: Any,  # np.ndarray
        render_visuals: bool = True,
        fps_val: float | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Process a single BGR frame, extract robotics telemetry, and render annotations.

        Args:
            frame: OpenCV BGR image as a numpy ndarray (H, W, 3).
            render_visuals: Whether to draw bounding boxes and HUD on the output frame.
            fps_val: Optional FPS value to overlay on the rendered HUD.

        Returns:
            Tuple of:
                1. annotated_frame (np.ndarray): Image with visual bounding boxes, crosshairs, and badges.
                2. telemetry_payload (list[dict]): Structured detection list ready for robotics controller.
        """
        import cv2
        import numpy as np

        h_img, w_img = frame.shape[:2]

        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        annotated_frame = frame.copy() if render_visuals else frame

        if not results or len(results) == 0:
            return annotated_frame, detections

        res = results[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            return annotated_frame, detections

        for box in boxes:
            # Extract coordinates and class info
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            model_names = getattr(self.model, "names", CLASS_NAMES)
            cls_name = model_names.get(cls_id, CLASS_NAMES.get(cls_id, f"class_{cls_id}"))

            # Compute bounding box center
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Compute normalized YOLO coordinates [x_center, y_center, width, height]
            x_center_norm = cx / w_img
            y_center_norm = cy / h_img
            w_norm = (x2 - x1) / w_img
            h_norm = (y2 - y1) / h_img

            # Build robotics telemetry payload matching ERC project specification
            telemetry_entry = {
                "class_name": cls_name,
                "class_id": cls_id,
                "confidence": round(conf, 4),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_center": (round(cx, 2), round(cy, 2)),
                "normalized_bbox": [
                    round(x_center_norm, 4),
                    round(y_center_norm, 4),
                    round(w_norm, 4),
                    round(h_norm, 4),
                ],
            }
            detections.append(telemetry_entry)

            if render_visuals:
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))

                # Draw main bounding box
                cv2.rectangle(annotated_frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

                # Draw switch center crosshair (useful for robotic arm end-effector alignment)
                c_pt = (int(cx), int(cy))
                arm_len = 8
                cv2.line(
                    annotated_frame,
                    (c_pt[0] - arm_len, c_pt[1]),
                    (c_pt[0] + arm_len, c_pt[1]),
                    color,
                    thickness=2,
                )
                cv2.line(
                    annotated_frame,
                    (c_pt[0], c_pt[1] - arm_len),
                    (c_pt[0], c_pt[1] + arm_len),
                    color,
                    thickness=2,
                )

                # Draw state label badge
                state_tag = "ON" if cls_id == 1 else "OFF"
                label_text = f"{cls_name} [{state_tag}] {conf:.2f}"
                (txt_w, txt_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.55, thickness=1
                )
                badge_top = max(0, int(y1) - txt_h - 8)
                badge_bottom = max(txt_h + 8, int(y1))
                badge_right = min(w_img, int(x1) + txt_w + 10)

                # Filled background rectangle for text contrast
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), badge_top),
                    (badge_right, badge_bottom),
                    color,
                    thickness=-1,
                )
                # Text label in crisp white
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (int(x1) + 5, badge_bottom - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.55,
                    color=(255, 255, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        # Draw HUD overlay (FPS & detection counts)
        if render_visuals:
            hud_y = 28
            on_count = sum(1 for d in detections if d["class_id"] == 1)
            off_count = sum(1 for d in detections if d["class_id"] == 0)

            hud_text = f"Switches: {len(detections)} (ON: {on_count} | OFF: {off_count})"
            if fps_val is not None:
                hud_text = f"FPS: {fps_val:4.1f} | " + hud_text

            (h_w, h_h), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(
                annotated_frame,
                (10, 8),
                (16 + h_w, 14 + h_h),
                (20, 20, 20),
                thickness=-1,
            )
            cv2.putText(
                annotated_frame,
                hud_text,
                (14, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        return annotated_frame, detections


def run_image_inference(
    detector: SwitchDetector,
    image_path: Path,
    output_dir: Path,
    show: bool,
    save_json: bool,
) -> None:
    """Run detection on a single image file."""
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error(f"Failed to read image: {image_path}")
        return

    annotated, telemetry = detector.detect_frame(frame, render_visuals=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_img_path = output_dir / image_path.name
    cv2.imwrite(str(out_img_path), annotated)
    logger.info(f"Saved annotated image: {out_img_path} ({len(telemetry)} detections)")

    if save_json:
        json_path = output_dir / f"{image_path.stem}_telemetry.json"
        json_path.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
        logger.info(f"Saved telemetry JSON: {json_path}")

    # Log detections summary
    for idx, d in enumerate(telemetry, 1):
        logger.info(
            f"  Det #{idx}: {d['class_name']} (conf={d['confidence']:.2f}) at center={d['bbox_center']}"
        )

    if show:
        cv2.imshow("Switch Detector (Press 'q' or any key to close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_directory_inference(
    detector: SwitchDetector,
    directory_path: Path,
    output_dir: Path,
    show: bool,
    save_json: bool,
) -> None:
    """Run detection on all images in a folder."""
    import cv2

    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".PNG", ".JPG"}
    images = [p for p in directory_path.iterdir() if p.suffix in supported_exts]

    if not images:
        logger.warning(f"No image files found in directory: {directory_path}")
        return

    logger.info(f"Processing {len(images)} images from '{directory_path}'...")
    all_telemetry: dict[str, list[dict[str, Any]]] = {}

    for img_path in sorted(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        annotated, telemetry = detector.detect_frame(frame, render_visuals=True)
        all_telemetry[img_path.name] = telemetry

        out_img_path = output_dir / img_path.name
        cv2.imwrite(str(out_img_path), annotated)

        if show:
            cv2.imshow("Batch Inference (Press 'q' to stop, any key for next)", annotated)
            key = cv2.waitKey(100) & 0xFF
            if key == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()

    if save_json:
        json_path = output_dir / "batch_telemetry.json"
        json_path.write_text(json.dumps(all_telemetry, indent=2), encoding="utf-8")
        logger.info(f"Saved batch telemetry to {json_path}")

    logger.info(f"Batch processing completed. Results saved to {output_dir}")


def run_video_stream(
    detector: SwitchDetector,
    source: str | int,
    output_dir: Path,
    show: bool,
    save_json: bool,
) -> None:
    """Run detection on a video file or live camera stream with FPS calculation and cleanup."""
    import cv2

    is_webcam = isinstance(source, int)
    source_label = f"Camera #{source}" if is_webcam else str(source)
    logger.info(f"Opening video source: {source_label}...")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source_label}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch stream properties
    fps_stream = cap.get(cv2.CAP_PROP_FPS)
    if fps_stream <= 0 or fps_stream > 120 or is_webcam:
        fps_stream = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1

    # Setup VideoWriter
    writer = None
    if not is_webcam:
        out_video_path = output_dir / f"annotated_{Path(str(source)).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_stream, (width, height))
        logger.info(f"Recording annotated video to: {out_video_path} ({width}x{height} @ {fps_stream:.1f} FPS)")

    stream_telemetry: list[dict[str, Any]] = []
    frame_idx = 0
    t_prev = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not is_webcam:
                    logger.info("Reached end of video stream.")
                break

            frame_idx += 1
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            fps_live = 1.0 / dt if dt > 0 else 30.0

            # Run detection
            annotated, telemetry = detector.detect_frame(
                frame, render_visuals=True, fps_val=fps_live
            )

            if save_json:
                stream_telemetry.append({
                    "frame_index": frame_idx,
                    "timestamp": round(t_now, 4),
                    "detections": telemetry,
                })

            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.imshow("ERC Switch Detector (Press 'q' to exit)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Stream interrupted by user ('q' key).")
                    break

            if frame_idx % 60 == 0:
                progress_str = f"Frame {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame {frame_idx}"
                logger.info(f"{progress_str} | Live FPS: {fps_live:4.1f} | Detections: {len(telemetry)}")

    finally:
        # Proper resource cleanup
        logger.info("Releasing video stream resources...")
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    if save_json and stream_telemetry:
        json_filename = "stream_telemetry.json" if is_webcam else f"{Path(str(source)).stem}_telemetry.json"
        json_path = output_dir / json_filename
        json_path.write_text(json.dumps(stream_telemetry, indent=2), encoding="utf-8")
        logger.info(f"Video telemetry saved to: {json_path}")

    logger.info(f"Completed processing {frame_idx} frames.")


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()

    weights_path = find_default_weights(args.weights, project_root)
    output_dir = Path(args.save_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    detector = SwitchDetector(
        weights_path=weights_path,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )

    # Disambiguate source input
    source_str = args.source.strip()

    # Check for webcam index
    if source_str.isdigit():
        webcam_idx = int(source_str)
        run_video_stream(
            detector=detector,
            source=webcam_idx,
            output_dir=output_dir,
            show=args.show,
            save_json=args.save_json,
        )
        return

    source_path = Path(source_str)
    if not source_path.is_absolute():
        source_path = project_root / source_path

    if not source_path.exists():
        raise FileNotFoundError(f"Specified source path does not exist: {source_path}")

    if source_path.is_dir():
        run_directory_inference(
            detector=detector,
            directory_path=source_path,
            output_dir=output_dir,
            show=args.show,
            save_json=args.save_json,
        )
    elif source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]:
        run_video_stream(
            detector=detector,
            source=str(source_path),
            output_dir=output_dir,
            show=args.show,
            save_json=args.save_json,
        )
    elif source_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        run_image_inference(
            detector=detector,
            image_path=source_path,
            output_dir=output_dir,
            show=args.show,
            save_json=args.save_json,
        )
    else:
        raise ValueError(
            f"Unsupported file type '{source_path.suffix}'. "
            "Supported modalities: image (.png/.jpg), video (.mp4/.avi), directory, or camera index."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nInference pipeline terminated by user.")
        sys.exit(130)
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        sys.exit(1)
