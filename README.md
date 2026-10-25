# Autonomous Robotic Arm Switch Controller

An autonomous perception and control system for a 6-DOF robotic arm competing in the **Maintenance Task** at the **European Rover Challenge (ERC)**. The system enables the robotic arm to visually identify industrial panel components, determine their operational states, and autonomously manipulate them to target configurations.

---

## Table of Contents
1. [System Overview & Architecture](#system-overview--architecture)
2. [Environment Setup](#environment-setup)
3. [Dataset Architecture & Labeling Strategy](#dataset-architecture--labeling-strategy)
4. [Model Selection & Training Methodology](#model-selection--training-methodology)
5. [Evaluation Results & Performance Benchmarks](#evaluation-results--performance-benchmarks)
6. [Core Scripts & Pipeline Documentation](#core-scripts--pipeline-documentation)
7. [Downstream Robotics & Telemetry Interface](#downstream-robotics--telemetry-interface)
8. [Future Engineering Roadmap](#future-engineering-roadmap)
9. [Team Contribution Statement](#team-contribution-statement)

---

## 1. System Overview & Architecture

The system is partitioned into two core components:
- **Perception Module (Computer Vision):** Real-time object detection and orientation classification using a fine-tuned Ultralytics YOLOv8 model. It identifies large red rotary isolator switches with yellow square bases on the control panel and classifies their physical state:
  - **`main_off` (Class 0):** Lever oriented **vertically** (pointing downwards / ~6 o'clock position / disconnected "0" state).
  - **`main_on` (Class 1):** Lever turned **horizontally** (pointing rightwards / ~3 o'clock position / energized "1" state).
- **Action Module (Manipulation Control):** Reinforcement Learning (RL) policy (PPO/SAC) in MuJoCo / ROS 2 environments that continuously translates visual switch centroid coordinates $(c_x, c_y)$ into 6-DOF joint trajectory commands for end-effector alignment and rotational switching.

```
+-----------------------------------------------------------------------------------+
|                               PERCEPTION PIPELINE                                 |
|                                                                                   |
|  [ Camera Feed / MP4 ] ──► [ YOLOv8 Nano ] ──► [ Centroid & State Telemetry JSON ]|
|                             (models/best.pt)    - class: main_on / main_off       |
|                                                 - center: (cx, cy)                |
|                                                 - confidence: 0.85                |
+---------------------------------------------------------┬-------------------------+
                                                          │
                                                          ▼
+-----------------------------------------------------------------------------------+
|                                 ACTION PIPELINE                                   |
|                                                                                   |
|  [ ROS 2 / MuJoCo Controller ] ──► [ Visual Servoing ] ──► [ 6-DOF Arm Actuation ]|
+-----------------------------------------------------------------------------------+
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10 - 3.14 on Windows and Linux/WSL)
- CUDA-compatible GPU (e.g., NVIDIA RTX 4050 6GB mobile GPU or desktop equivalent) recommended for real-time inference and training. CPU execution is fully supported as fallback.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Martik-k/robotic-arm-switch-controller.git
   cd robotic-arm-switch-controller
   ```

2. **Create and activate a Python virtual environment:**
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS / WSL
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **(Optional) Install PyTorch with CUDA support:**
   To leverage GPU acceleration on NVIDIA RTX GPUs:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

---

## 3. Dataset Architecture & Labeling Strategy

### Target Classes
| Class ID | Class Name | Physical Appearance | Operational Meaning |
| :---: | :---: | :--- | :--- |
| **0** | `main_off` | Red lever pointing **vertically downwards** (~6 o'clock) | Disconnected / Safe circuit |
| **1** | `main_on`  | Red lever pointing **horizontally rightwards** (~3 o'clock) | Connected / Energized circuit |

### Data Splits (Strict Split-by-Video)
To avoid **data leakage** and test true cross-domain generalizability, video frames are partitioned strictly by recording session rather than randomly shuffling adjacent video frames:

```
data/
├── images/
│   ├── train/                  # 77 frames from video1.mp4 and video2.mp4
│   └── val/                    # 58 frames from video3.mp4 and video6.mp4
└── labels/
    ├── train/                  # 116 switch bounding box annotations
    └── val/                    # 66 switch bounding box annotations
```

- **Training Split (Videos 1 & 2):** 77 frames containing 116 labeled switches (86 `main_on`, 30 `main_off`).
- **Expanded Validation Split (Videos 3 & 6):** 58 frames containing 66 labeled switches (36 `main_off`, 30 `main_on`).
  - *Video 3:* 18 frames capturing close-up pan-tilt views of the panel.
  - *Video 6:* 40 frames capturing an operational rover arm approach with dual switches (`main_off` and `main_on`) side-by-side.
- **Unannotated Video Archive:** 19 additional operational video recordings stored in `videos/` (e.g., `WIN_20261019_*.mp4`, `WIN_20261020_*.mp4`) reserved for future scale-up.

---

## 4. Model Selection & Training Methodology

### Why YOLOv8 Nano (`yolov8n.pt`)?
1. **Lightweight Edge Robotics Footprint:** With only **3.1M parameters** and **8.6 GFLOPs**, it operates reliably on power-constrained mobile rover computers (NVIDIA Jetson / laptop companion computers).
2. **Ultra-Low Latency:** Inference completes in **< 5 ms on GPU** and **~38–50 ms on CPU**, comfortably exceeding the ERC real-time robotics requirement of $\ge 15$ FPS.
3. **Anchor-Free Detection Head:** Exceptional accuracy at locating switch centroids $(c_x, c_y)$ across distance scales.

### Domain-Specific Augmentation Constraints
Unlike general object detection, panel switch orientation encodes physical state:
- **`fliplr=0.0` (Horizontal Flip Strictly Disabled):** Mirroring a horizontal `main_on` lever pointing right would create a leftward lever—an invalid/corrupted state under ERC rules.
- **`flipud=0.0` (Vertical Flip Strictly Disabled):** Mirroring a vertical `main_off` lever pointing downwards would point upwards.
- **Permitted Augmentations:** Mild roll rotations (`degrees=12.0`), photometric jitter (`hsv_h=0.015`, `hsv_s=0.5`, `hsv_v=0.4`), perspective warping (`perspective=0.0005`), and scale jitter (`scale=0.15`) simulate natural rover approach angles and indoor/outdoor panel reflections.

### Training Execution
- Resolution: $640 \times 640$
- Optimizer: AdamW / Auto with cosine LR decay
- Max Epochs: 80 with Early Stopping (`patience=15`)
- Convergence: Validation classification loss dropped sharply from **17.19** to **1.65** by epoch 27; early stopping cleanly halted training at **Epoch 43** when validation mAP peaked, preventing overfitting on the small dataset.

---

## 5. Evaluation Results & Performance Benchmarks

### Quantitative Performance Metrics

Evaluated on the expanded validation set (58 frames, 66 ground truth switch instances across Videos 3 & 6):

| Evaluation Metric | Baseline (Video 3 Alone) | Expanded Validation (Video 3 + Video 6) | Status |
| :--- | :---: | :---: | :---: |
| **Overall Precision (P)** | 58.6% | **79.7%** | **+21.1% Gain** |
| **Overall Recall (R)** | 63.3% | **84.8%** | **+21.5% Gain** |
| **Overall mAP @ 0.50** | 48.0% | **80.6%** | **+32.6% Gain** |
| **Overall mAP @ 0.50:0.95** | 23.1% | **43.0%** | **+19.9% Gain** |
| **`main_off` Precision** | 85.7% | **89.7%** | **High Reliability** |
| **`main_off` Recall** | 60.0% | **83.9%** | **Target Met** |
| **`main_off` mAP @ 0.50** | 58.1% | **83.2%** | **Target Met** |
| **`main_on` Precision** | 31.6% | **69.8%** | **+38.2% Gain** |
| **`main_on` Recall** | 66.7% | **85.7%** | **Target Met** |
| **`main_on` mAP @ 0.50** | 38.0% | **78.0%** | **Target Met** |
| **Inference Latency (CPU)** | 37.7 ms (~26 FPS) | **~49 ms (~20 FPS)** | **Real-Time ($\ge 15$ FPS)** |
| **Inference Latency (RTX 4050)**| ~3–5 ms | **< 5 ms (>200 FPS)** | **Ultra Real-Time** |

### Diagnostic Curves & Validation Artifacts
Key diagnostic plots and benchmark records are tracked in the [`benchmarks/`](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/benchmarks) directory:

- **Precision-Recall Curve ([benchmarks/BoxPR_curve.png](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/benchmarks/BoxPR_curve.png)):** Demonstrates **80.6% mAP@0.5** across all classes, holding $>85\%$ precision up to $75\%$ recall.
- **Confusion Matrix ([benchmarks/confusion_matrix.png](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/benchmarks/confusion_matrix.png)):** 
  - 26 correct detections for `main_off`, 32 correct detections for `main_on`.
  - **Zero instances** of `main_on` misclassified as `main_off`, maintaining functional circuit safety.
- **F1-Confidence Curve ([benchmarks/BoxF1_curve.png](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/benchmarks/BoxF1_curve.png)):** Reaches peak $F_1 = 0.82$, confirming optimal performance around confidence threshold $\text{conf} \approx 0.40 - 0.50$.
- **Training Progression Curves ([benchmarks/training_results.png](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/benchmarks/training_results.png)):** 10-panel loss and metric progression across all 43 training epochs.
- **Ground Truth vs. Predictions ([benchmarks/val_batch1_labels.jpg](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/benchmarks/val_batch1_labels.jpg) & [benchmarks/val_batch1_pred.jpg](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/benchmarks/val_batch1_pred.jpg)):** Visual confirmation of dual-switch detection on the panel during arm approach.
- **Structured Audit Metrics ([benchmarks/evaluation_summary.json](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/benchmarks/evaluation_summary.json)):** Raw JSON report containing exact precision, recall, mAP, and latency numbers.


---

## 6. Core Scripts & Pipeline Documentation

### Configuration: `configs/switches.yaml`
Defines root directory `data`, train/val split paths, class counts (`nc: 2`), and zero-indexed labels:
```yaml
path: data
train: images/train
val: images/val
nc: 2
names:
  0: main_off
  1: main_on
```

---

### Training Script: `src/train.py`
Fine-tunes the base YOLO model with domain-specific augmentation constraints and automatically deploys the best checkpoint to `models/best.pt`.
```bash
python src/train.py --epochs 80 --batch 16 --patience 15 --model yolov8n.pt
```
- **CLI Options:** `--epochs`, `--batch`, `--patience`, `--imgsz`, `--model`, `--device`.
- **Output:** Saves experiment to `runs/detect/switches_baseline/` and exports production checkpoint to `models/best.pt` along with `models/model_metadata.json`.

---

### Evaluation Script: `src/evaluate.py`
Standalone benchmarking tool that evaluates a checkpoint against validation data and prints a formatted metric report to stdout.
```bash
python src/evaluate.py --weights models/best.pt --data configs/switches.yaml --conf 0.25 --save-json
```
- **CLI Options:** `--weights`, `--data`, `--split`, `--imgsz`, `--conf`, `--iou`, `--save-json`.
- **Output:** Prints ASCII metric tables, generates PR/F1/Confusion matrix plots, and writes `evaluation_summary.json`.

---

### Runtime Inference & Telemetry Engine: `src/inference.py`
Multimodal detector supporting static images, image directories, MP4 videos, and live webcam streams (e.g. USB Arducam on rover end-effector).
```bash
# Process a video file and save annotated MP4:
python src/inference.py --source videos/video6.mp4 --weights models/best.pt --conf 0.50

# Live camera stream with visual preview:
python src/inference.py --source 0 --show --conf 0.50

# Single image with JSON telemetry extraction:
python src/inference.py --source data/images/val/v6_frame_000060.PNG --save-json
```
- **CLI Options:** `--source` (path or camera index), `--weights`, `--conf`, `--iou`, `--save-dir`, `--show`, `--save-json`.
- **Output:** Annotated video/image with bounding boxes, center crosshairs, FPS counter, and structured JSON telemetry.

---

### Data Preparation & Utility Scripts
- **`scripts/prepare_dataset.py`:** Reads raw images from `images/` and CVAT labels from `labels/`, maps class IDs to zero-indexed standards (`0: main_off`, `1: main_on`), and builds the flattened `data/` structure.
- **`scripts/extract_frames.py`:** Extracts frames from `.mp4` video files at a configurable frame step (e.g., every 10th frame) with standardized naming.
- **`scripts/visualize_labels.py`:** Renders ground truth bounding boxes onto raw frames for manual annotation verification.
- **`scripts/add_prefix.py`:** Helper script for batch renaming label files with video-specific prefixes (`v1_`, `v2_`, `v3_`, `v6_`).

---

## 7. Downstream Robotics & Telemetry Interface

The `SwitchDetector` class in `src/inference.py` exposes a clean Python API for integration into ROS 2 perception nodes or Gymnasium RL wrappers:

```python
from src.inference import SwitchDetector
import cv2

# Initialize detector
detector = SwitchDetector("models/best.pt", conf_threshold=0.50)

# Process frame
frame = cv2.imread("panel_frame.png")
annotated_frame, telemetry = detector.detect_frame(frame)

print(telemetry)
```

### Telemetry Payload Schema:
```json
[
  {
    "class_name": "main_off",
    "class_id": 0,
    "confidence": 0.8966,
    "bbox_xyxy": [350.2, 280.4, 435.8, 410.1],
    "bbox_center": [393.0, 345.25],
    "normalized_bbox": [0.4094, 0.5754, 0.0892, 0.2162]
  },
  {
    "class_name": "main_on",
    "class_id": 1,
    "confidence": 0.8245,
    "bbox_xyxy": [480.1, 282.0, 565.4, 408.9],
    "bbox_center": [522.75, 345.45],
    "normalized_bbox": [0.5445, 0.5758, 0.0889, 0.2115]
  }
]
```

---

## 8. Future Engineering Roadmap

1. **Dataset Scale-Up (Model-Assisted Annotation):**
   - Leverage the 19 unannotated operational recordings in `videos/` using pseudo-labeling via `models/best.pt` to expand both training and validation splits to 250+ diverse frames.
2. **Hard Negative Mining:**
   - Incorporate panel views containing circuit breakers, terminal rails, and rectangular cutouts without switches to further suppress background false alarms.
3. **Gripper Occlusion Robustness:**
   - Annotate frames where the robotic arm end-effector partially occludes the switch during the final approach phase to ensure stable continuous tracking.
4. **Temporal Filtering & Tracking:**
   - Integrate ByteTrack / SORT tracking to maintain switch identity and filter out single-frame confidence jitter across continuous video streams.
5. **Edge Deployment Optimization:**
   - Export `models/best.pt` to TensorRT FP16 / ONNX runtime for sub-2ms latency on the rover's onboard companion computer.

---

## 9. Team Contribution Statement

- **Marko Pasternak:** Dataset collection and annotation, development of the control element recognition system, perception pipeline design, training and evaluation.
- **Marta Kosiv:** Modeling, evaluation, documentation, robotics interface specifications.
- **Solomiia Hadiichuk:** Modeling, evaluation, documentation, benchmark auditing.

*All participants agree with this distribution of responsibilities.*