# Autonomous Robotic Arm Switch Controller

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8n-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.3+-orange.svg)](https://mujoco.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.9+-green.svg)](https://github.com/DLR-RM/stable-baselines3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An autonomous perception and control system for a 6-DOF robotic arm competing in the **Maintenance Task** at the **European Rover Challenge (ERC 2026)**. The system couples real-time computer vision (**Ultralytics YOLO**) for panel switch detection and state estimation with continuous reinforcement learning (**PPO / SAC** in **MuJoCo**) for trajectory planning and switch manipulation.

> **Ukrainian Catholic University (UCU)**  
> Faculty of Applied Sciences — Department of Computer Sciences  
> Course: *Artificial Intelligence* (Semester Project, Fall 2026)  
> Team: **Marta Kosiv**, **Solomiia Hadiichuk**, **Marko Pasternak**  
> Repository: [github.com/Martik-k/robotic-arm-switch-controller](https://github.com/Martik-k/robotic-arm-switch-controller)

---

## Table of Contents
1. [Motivation & Problem Statement](#1-motivation--problem-statement)
2. [Literature Review & Related Work](#2-literature-review--related-work)
3. [System Architecture](#3-system-architecture)
4. [Repository Structure](#4-repository-structure)
5. [Environment Setup & Installation](#5-environment-setup--installation)
6. [Perception Module (Computer Vision)](#6-perception-module-computer-vision)
7. [Action Module (Reinforcement Learning)](#7-action-module-reinforcement-learning)
8. [Evaluation Metrics & Performance Benchmarks](#8-evaluation-metrics--performance-benchmarks)
9. [Project Roadmap & Risk Management](#9-project-roadmap--risk-management)
10. [Team Contribution Statement](#10-team-contribution-statement)
11. [References](#11-references)

---

## 1. Motivation & Problem Statement

The European Rover Challenge (ERC) Maintenance Task requires an autonomous rover to approach an electrical control panel and manipulate small switches, buttons, and circuit breakers using an onboard 6-DOF robotic arm.

### Why Classical Inverse Kinematics (IK) Fails
Conventional robotic manipulation relies on classical analytic or numerical inverse kinematics. In real-world rover operations, this pipeline is notoriously fragile:
- **Mounting & Manufacturing Tolerances:** Slight mechanical flex or assembly deviations between the arm base and chassis invalidate rigid kinematic chain assumptions.
- **Camera Calibration Drift:** Vibration and temperature changes cause extrinsics/intrinsics drift, inducing sub-centimeter end-effector positioning errors that cause the gripper to miss or jam against toggle levers.
- **Occlusions & Dynamic Lighting:** Reflections, outdoor glare, and shadows on industrial panel enclosures degrade traditional edge-based feature detectors.

### Our Approach: Perception + Reinforcement Learning
To build a fault-tolerant system, we decouple the task into two complementary learning systems:
1. **Perception:** Fine-tuned lightweight object detector (YOLOv8) that processes monocular camera frames to locate switches and classify their operational orientation in real time.
2. **Action:** Model-free reinforcement learning policy (PPO/SAC) trained in the MuJoCo physics engine that learns to drive the 6-DOF end-effector to the target switch, robust to camera noise and spatial jitter.

---

## 2. Literature Review & Related Work

- **Marker-Based Panel Manipulation:** The Kielce ERC team (ERC 2021–2022) achieved panel task execution using fiducial ARTag markers [[1]](#1). However, fiducials require rigid panel instrumentation, cannot achieve millimeter precision on tiny levers, and fail catastrophically when occluded by the robotic arm.
- **RL for Robotic Switching:** Contrastive pre-training and RL have been demonstrated for switch flipping on physical arms [[2]](#2), and hindsight experience replay (HER) has been successfully applied to lever manipulation with sim-to-real transfer [[3]](#3).
- **Vision-Grounded RL:** End-to-end YOLO detection paired with RL policies has been validated for conveyor belt grasping [[4]](#4).
- **Our Contribution:** To our knowledge, no existing work has explicitly combined fine-grained rotary switch state classification with continuous deep RL for the ERC maintenance panel. We bridge this gap using MuJoCo [[5]](#5) and Stable-Baselines3 [[8]](#8).

---

## 3. System Architecture

```
+───────────────────────────────────────────────────────────────────────────────────+
│                                PERCEPTION PIPELINE                                │
│                                                                                   │
│  [ Camera Feed / MP4 ] ──► [ YOLOv8 Nano ] ──► [ Centroid & State Telemetry JSON ]│
│                             (models/best.pt)    - class: main_on / main_off       │
│                                                 - center: (cx, cy)                │
│                                                 - confidence: 0.85                │
+─────────────────────────────────────────┬─────────────────────────────────────────+
                                          │  Observation Vector
                                          ▼
+───────────────────────────────────────────────────────────────────────────────────+
│                                  ACTION PIPELINE                                  │
│                                                                                   │
│  [ Observation Space ] ──► [ PPO / SAC Policy ] ──► [ 6-DOF Joint Velocities ]    │
│  - Joint angles & vel      (Stable-Baselines3)       (MuJoCo / Gymnasium)         │
│  - End-effector (x, y, z)                                                         │
│  - Switch target & state                                                          │
+───────────────────────────────────────────────────────────────────────────────────+
```

### Module Responsibilities:
- **Perception (`perception/`):** Receives camera frames, performs real-time bounding box detection and state classification, and outputs switch coordinates and states.
- **Control (`rl/`):** Ingests arm kinematics and switch telemetry to output continuous joint velocity deltas $\Delta \theta \in [-1, 1]^6$ that guide the end-effector to the switch and rotate it to the target state.

---

## 4. Repository Structure

```
robotic-arm-switch-controller/
├── .gitignore                      # Production gitignore for CV and RL workflows
├── README.md                       # Master repository documentation (this file)
├── requirements.txt                # Unified dependency entry point
│
├── requirements/                   # Modular requirement manifests
│   ├── perception.txt              # PyTorch, Ultralytics YOLO, OpenCV, Albumentations
│   └── rl.txt                      # MuJoCo, Gymnasium, Meta-World, Stable-Baselines3
│
├── perception/                     # Computer Vision module
│   ├── README.md                   # In-depth perception documentation & benchmarks
│   ├── benchmarks/                 # Version-controlled evaluation curves & JSON metrics
│   │   ├── BoxPR_curve.png         # Precision-Recall curve (80.6% mAP@0.5)
│   │   ├── BoxF1_curve.png         # F1-Confidence curve (peak 0.82)
│   │   ├── confusion_matrix.png    # Validation confusion matrix
│   │   ├── training_results.png    # 10-panel training loss & metric progression
│   │   └── evaluation_summary.json # Machine-readable validation audit metrics
│   ├── configs/
│   │   └── switches.yaml           # YOLO dataset specification & class names
│   ├── models/
│   │   ├── best.pt                 # Production fine-tuned YOLOv8n weights (~6.2 MB)
│   │   └── model_metadata.json     # Checkpoint metadata & hyperparameters
│   ├── scripts/
│   │   ├── add_prefix.py           # Prefix filenames to prevent multi-video collisions
│   │   ├── extract_frames.py       # Frame extractor with configurable step
│   │   ├── prepare_dataset.py      # Flattens raw images/labels into data/
│   │   ├── rotate_video.py         # Utility for 180-degree video orientation fixes
│   │   └── visualize_labels.py     # Ground truth overlay renderer
│   └── src/
│       ├── train.py                # YOLO fine-tuning engine with early stopping
│       ├── evaluate.py             # Standalone model benchmark & validation runner
│       └── inference.py            # Real-time multi-source inference engine
│
├── rl/                             # Reinforcement Learning & Simulation module
│   ├── README.md                   # RL setup, Meta-World environment & usage guide
│   ├── configs/
│   │   ├── ppo.yaml                # PPO hyperparameters & architecture config
│   │   └── sac.yaml                # SAC hyperparameters & architecture config
│   ├── scripts/
│   │   ├── check_wrapper.py        # Validates observation and action spaces
│   │   ├── evaluate.py             # Policy evaluation across multiple seeds
│   │   ├── explore_env.py          # Interactive environment visualizer
│   │   ├── run_noise_sweep.sh      # Noise robustness evaluation runner
│   │   ├── run_seeds.sh            # Multi-seed training batch script
│   │   └── train.py                # Main RL policy training script
│   └── src/
│       ├── envs.py                 # Vectorized Gymnasium environment factories
│       └── wrappers.py             # Domain randomization & observation wrappers
│
├── data/                           # Canonical dataset directory
│   ├── images/                     # Flattened image splits (train / val)
│   └── labels/                     # Flattened YOLO annotations (train / val)
│
└── videos/                         # Raw operational panel video recordings (19 files)
```

---

## 5. Environment Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.10 – 3.14 on Linux/WSL and Windows)
- NVIDIA GPU with CUDA support recommended (e.g. RTX 4050 / RTX 4070); CPU execution fully supported as fallback.

### 1. Clone the Repository
```bash
git clone https://github.com/Martik-k/robotic-arm-switch-controller.git
cd robotic-arm-switch-controller
```

### 2. Virtual Environment Setup
```bash
# On Linux / macOS / WSL:
python3 -m venv .venv
source .venv/bin/activate

# On Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
You can install the entire pipeline or individual modules:

```bash
# Install everything (Perception + RL):
pip install --upgrade pip
pip install -r requirements.txt

# Or install module-specifically:
pip install -r requirements/perception.txt   # Computer Vision only
pip install -r requirements/rl.txt           # Reinforcement Learning only
```

---

## 6. Perception Module (Computer Vision)

The perception subsystem detects industrial rotary switches and classifies their operational state directly from monocular RGB frames.

### Target Classes
| Class ID | Class Name | Visual Indicator | Physical State |
| :---: | :---: | :--- | :--- |
| **0** | `main_off` | Red lever pointed **vertically downwards** (~6 o'clock) | Disconnected circuit ("0") |
| **1** | `main_on`  | Red lever turned **horizontally rightwards** (~3 o'clock) | Energized circuit ("1") |

### Key Design Decisions
1. **Lightweight Edge Backbone:** Ultralytics YOLOv8 Nano (`yolov8n.pt`, 3.1M parameters, 8.6 GFLOPs) provides sub-5 ms latency on mobile rover GPUs and ~45 ms on CPU.
2. **State-Preserving Augmentations:** Horizontal and vertical flips are strictly disabled (`fliplr=0.0`, `flipud=0.0`). Inverting a horizontal lever transforms a valid "1" state into an undefined leftward position, corrupting label semantics.
3. **Strict Split-by-Video:** Video 1 & Video 2 form the training set (77 frames, 116 switch instances); Video 3 & Video 6 form the validation set (58 frames, 66 switch instances). This prevents temporal correlation leakage between adjacent frames.

### Running Perception Scripts

```bash
# 1. Evaluate the production model against the validation set:
python perception/src/evaluate.py --weights perception/models/best.pt --data perception/configs/switches.yaml --save-json

# 2. Run real-time inference on a video file:
python perception/src/inference.py --source videos/video6.mp4 --weights perception/models/best.pt --conf 0.50

# 3. Run live inference on camera stream (webcam or rover end-effector USB camera):
python perception/src/inference.py --source 0 --show --conf 0.50

# 4. Fine-tune model on updated dataset:
python perception/src/train.py --epochs 80 --batch 16 --patience 15 --model yolov8n.pt
```

For complete perception documentation, see [perception/README.md](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/perception/README.md).

---

## 7. Action Module (Reinforcement Learning)

The control module learns a closed-loop policy to position the 6-DOF manipulator and actuate the switch in MuJoCo simulation via Gymnasium and Meta-World.

### Action & State Space Formulation
- **Action Space:** Continuous 6-DOF joint velocity increments $a \in [-1, 1]^6$.
- **State Space:**
  - Manipulator joint angles $\theta \in \mathbb{R}^6$ and joint velocities $\dot{\theta} \in \mathbb{R}^6$.
  - End-effector Cartesian pose $(x, y, z)$ and orientation quaternion $(q_w, q_x, q_y, q_z)$.
  - Switch centroid coordinates $(s_x, s_y, s_z)$ and current operational state (provided by the perception module).
  - Relative displacement vector $\Delta \mathbf{p} = \mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{switch}}$.
  - Target switch state $s^* \in \{0, 1\}$.

### Reward Function
$$R_t = R_{\text{dense}} + R_{\text{sparse}}$$
- **Dense Guidance:** $R_{\text{dense}} = -\alpha \|\mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{switch}}\|_2 - \beta \|\boldsymbol{\omega}_{\text{align}}\|_2$, encouraging rapid approach and gripper alignment.
- **Sparse Completion:** $R_{\text{sparse}} = +10.0$ awarded upon successfully transitioning the switch lever into the target configuration, plus a bonus for maintaining contact stability.

### Training & Robustness (Noise Injection)
To make the policy robust to real-world camera inaccuracies, we wrap the environment in `src.wrappers` to inject Gaussian jitter $\mathcal{N}(0, \sigma^2)$ into switch coordinates during training, simulating YOLO centroid estimation noise.

```bash
# 1. Verify RL environment and wrappers:
python -m rl.scripts.check_wrapper

# 2. Train baseline PPO policy:
python -m rl.scripts.train --algo ppo --env MetaWorld-Switch-v2 --total-steps 1000000

# 3. Train SAC policy with target coordinate jitter (simulating YOLO noise):
python -m rl.scripts.train --algo sac --noise-jitter 0.01 --total-steps 1000000

# 4. Evaluate trained policy across multiple random seeds:
python -m rl.scripts.evaluate --algo ppo --episodes 100
```

For complete RL documentation, see [rl/README.md](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/rl/README.md).

---

## 8. Evaluation Metrics & Performance Benchmarks

### 1. Success Criteria
1. **Success Rate (%):** Proportion of episodes in which the switch is successfully actuated to the target orientation within the step limit across varied starting arm poses.
2. **Sample Efficiency:** Total environment interaction steps required for the policy to achieve $\ge 90\%$ success rate.
3. **End-to-End Latency:** Combined runtime of the perception detection and RL action inference ($\le 66\text{ ms}$, ensuring $\ge 15\text{ FPS}$ real-time control).
4. **Detection Quality:** $\text{mAP@0.5} \ge 80\%$ with zero false negatives on safety-critical isolator states.

### 2. Perception Quantitative Results

Evaluated on the expanded multi-video validation set (58 frames, 66 ground-truth switch instances across Videos 3 & 6):

| Evaluation Metric | Baseline (Video 3 Alone) | Expanded Validation (Video 3 + Video 6) | Status / Target |
| :--- | :---: | :---: | :---: |
| **Overall Precision (P)** | 58.6% | **79.7%** | **+21.1% Gain** |
| **Overall Recall (R)** | 63.3% | **84.8%** | **+21.5% Gain** |
| **Overall mAP @ 0.50** | 48.0% | **80.6%** | **Target Met ($\ge 80\%$)** |
| **Overall mAP @ 0.50:0.95** | 23.1% | **43.0%** | **+19.9% Gain** |
| **`main_off` Precision** | 85.7% | **89.7%** | **High Reliability** |
| **`main_off` Recall** | 60.0% | **83.9%** | **Target Met** |
| **`main_off` mAP @ 0.50** | 58.1% | **83.2%** | **Target Met** |
| **`main_on` Precision** | 31.6% | **69.8%** | **+38.2% Gain** |
| **`main_on` Recall** | 66.7% | **85.7%** | **Target Met** |
| **`main_on` mAP @ 0.50** | 38.0% | **78.0%** | **Target Met** |
| **Inference Latency (RTX 4050)** | ~3–5 ms | **< 5 ms (>200 FPS)** | **Ultra Real-Time** |
| **Inference Latency (CPU)** | 37.7 ms | **~45 ms (~22 FPS)** | **Real-Time ($\ge 15$ FPS)** |

### 3. Visual Diagnostic Benchmarks

The version-controlled artifacts are located in [`perception/benchmarks/`](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/perception/benchmarks):

- **Precision-Recall Curve ([`perception/benchmarks/BoxPR_curve.png`](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/perception/benchmarks/BoxPR_curve.png)):** Demonstrates robust 80.6% mAP@0.5 across both classes.
- **Confusion Matrix ([`perception/benchmarks/confusion_matrix.png`](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/perception/benchmarks/confusion_matrix.png)):** 26 correct detections for `main_off`, 32 correct detections for `main_on`. Zero instances of `main_on` misclassified as `main_off`.
- **F1-Confidence Curve ([`perception/benchmarks/BoxF1_curve.png`](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/perception/benchmarks/BoxF1_curve.png)):** Peak $F_1 = 0.82$ around confidence threshold $\text{conf} \approx 0.45$.
- **Training Progression ([`perception/benchmarks/training_results.png`](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/perception/benchmarks/training_results.png)):** Convergence curves across 43 epochs with early stopping.
- **Audit Metrics JSON ([`perception/benchmarks/evaluation_summary.json`](file:///c:/Users/Marko/personal/UCU/third_year/ai/project/robotic-arm-switch-controller/perception/benchmarks/evaluation_summary.json)):** Exact numeric log of validation metrics.

---

## 9. Project Roadmap & Risk Management

### Milestones Timeline
- **Phase 1: Pre-Defense (Completed — October 26, 2026):**
  - [x] Physical panel data acquisition and 6-digit CVAT track annotation.
  - [x] YOLOv8n fine-tuning with orientation-preserving augmentations.
  - [x] Expanded validation benchmark reaching 80.6% mAP@0.5.
  - [x] MuJoCo simulation environment setup, Gymnasium wrappers, and action space definition.
  - [x] Baseline PPO and SAC training scripts with Gaussian noise wrappers.
  - [x] Structured JSON telemetry interface bridging perception and downstream robotics.

- **Phase 2: Final Defense (Target — December 2026):**
  - [ ] Curriculum learning integration for RL (progressive approach $\to$ contact $\to$ rotation).
  - [ ] Fine-tuning reward function weights to eliminate local minima.
  - [ ] Synthetic dataset generation via BlenderProc / domain randomization if real dataset coverage limits generalization.
  - [ ] End-to-end closed-loop benchmark (Perception feeding noisy centroids directly into RL policy).
  - [ ] Final project poster and defense presentation.

### Risk Management & Mitigation (Plan B)
| Identified Risk | Impact | Contingency Plan (Plan B) |
| :--- | :--- | :--- |
| **Small real dataset for YOLO** | Overfitting under novel lighting | Generate thousands of photorealistic synthetic frames with domain randomization using BlenderProc [[12]](#12) from panel CAD models. |
| **RL policy fails to converge** | Agent gets stuck in local minima | Apply Curriculum Learning (decompose into reaching, grasping, and rotating) and reward shaping. |
| **YOLO coordinate noise destabilizes RL** | End-effector oscillation | Add zero-mean Gaussian noise $\mathcal{N}(0, \sigma^2)$ to target coordinates during simulation training to force robustness. |

---

## 10. Team Contribution Statement

All participants contributed actively to the conception, implementation, and evaluation of this project according to the following agreed distribution:

- **Marko Pasternak (34%):** Dataset collection and annotation, development of the control element recognition system, perception pipeline design, training and evaluation.
- **Marta Kosiv (33%):** Setup of manipulator dynamics in MuJoCo (MJCF, collision modeling), Gymnasium environment integration, joint development and training of PPO/SAC algorithms.
- **Solomiia Hadiichuk (33%):** Reward function and state/action space design, joint development and evaluation of PPO/SAC algorithms, architecture documentation.

*All participants agree with this distribution of responsibilities.*

---

## 11. References

<a id="1">[1]</a> A. Annusewicz-Mistal, D. S. Pietrala, P. A. Laski, J. Zwierzchowski, K. Borkowski, G. Bracha, K. Borycki, S. Kostecki, and D. Wlodarczyk, "Autonomous manipulator of a mobile robot based on a vision system," *Applied Sciences*, vol. 13, no. 1, p. 439, 2023. doi: [10.3390/app13010439](https://doi.org/10.3390/app13010439).

<a id="2">[2]</a> A. Zhan, R. Zhao, L. Pinto, P. Abbeel, and M. Laskin, "Learning visual robotic control efficiently with contrastive pre-training and data augmentation," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2022, pp. 4040–4047.

<a id="3">[3]</a> S. B. Remman and A. M. Lekkas, "Robotic lever manipulation using hindsight experience replay and shapley additive explanations," 2021. arXiv: [2110.03292](https://arxiv.org/abs/2110.03292).

<a id="4">[4]</a> Y. Cao, X. Xu, and Y. Zhang, "Vision-based reinforcement learning for robotic grasping of moving objects on a conveyor," *Machines*, vol. 13, no. 10, p. 973, 2025. doi: [10.3390/machines13100973](https://doi.org/10.3390/machines13100973).

<a id="5">[5]</a> E. Todorov, T. Erez, and Y. Tassa, "Mujoco: A physics engine for model-based control," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2012, pp. 5026–5033.

<a id="6">[6]</a> J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," 2017. arXiv: [1707.06347](https://arxiv.org/abs/1707.06347).

<a id="7">[7]</a> T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," 2018. arXiv: [1801.01290](https://arxiv.org/abs/1801.01290).

<a id="8">[8]</a> A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-baselines3: Reliable reinforcement learning implementations," *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.

<a id="9">[9]</a> B. Sekachev, N. Manovich, M. Zhiltsov, A. Zhavoronkov, D. Kalinin, B. Hoff, et al., "Computer Vision Annotation Tool (CVAT)," 2020. doi: [10.5281/zenodo.4009388](https://doi.org/10.5281/zenodo.4009388).

<a id="10">[10]</a> A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Garushyants, I. Rakhlin, and A. A. Kalinin, "Albumentations: Fast and flexible image augmentations," 2018. arXiv: [1809.06839](https://arxiv.org/abs/1809.06839).

<a id="11">[11]</a> Y. Chen, X. Yuan, J. Wang, R. Wu, X. Li, Q. Hou, and M.-M. Cheng, "YOLO-MS: Rethinking multi-scale representation learning for real-time object detection," 2023. arXiv: [2308.05480](https://arxiv.org/abs/2308.05480).

<a id="12">[12]</a> M. Denninger, M. Sundermeyer, D. Winkelbauer, Y. Zidan, D. Olefir, M. Elbadrawy, A. Lodhi, and H. Katam, "BlenderProc: Reducing the reality gap with photorealistic rendering," 2019. arXiv: [1911.01911](https://arxiv.org/abs/1911.01911).
