# Robotic Arm Switch Controller

An autonomous control system for a 6-DOF robotic arm designed for the Maintenance Task (ERC 2026). It combines computer vision (YOLO) for object localization and reinforcement learning (RL) for precise switch manipulation on a control panel.

## Approach
* **Perception:** YOLO family models (YOLO-MS) for bounding box detection and switch state classification.
* **Action:** RL agent (PPO/SAC) in the MuJoCo environment for continuous control of joint angles.

## Setup
Requirements: Linux, Python 3.12, an NVIDIA GPU with a driver supporting CUDA 13 (for SAC; PPO runs on CPU).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All versions in `requirements.txt` are pinned (verified combination: metaworld 3.1.1, gymnasium 1.4.0, mujoco 3.3.0, stable-baselines3 2.9.0, torch 2.14.1 with CUDA 13).

Verify the installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"   # expected: True
python -m scripts.check_wrapper                              # expected: "no warnings raised"
```

Scripts are run from the repository root as modules (`python -m scripts.<name>`). Scripts that render images or video use offscreen EGL rendering (`MUJOCO_GL=egl`).

## Usage
(To be added later)

## AI Usage
(Describe the AI tools used for code generation, refactoring, literature search, text writing, or data generation, and how you verified the results).

## Contribution Statement
* **Marko Pasternak:** dataset collection and annotation, development of the control element recognition system.
* **Marta Kosiv:** modeling, evaluation, documentation.
* **Solomiia Hadiichuk:** modeling, evaluation, documentation.

All participants agree with this distribution of responsibilities.