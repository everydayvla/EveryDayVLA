# EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation

This repository contains the code, robot control scripts, evaluation utilities, and voice interface used for the EveryDayVLA paper:

[EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation](https://arxiv.org/abs/2511.05397)

EveryDayVLA combines a low-cost 6-DOF manipulator with a vision-language-action model for affordable robotic manipulation. The repo includes model training and inference code, Arduino/IK control scripts, dataset collection utilities, LIBERO/ALOHA evaluation code, plotting utilities, and an optional speech interface for robot commands.

## Repository Layout

- `vla/` - OpenVLA/Prismatic model code, training scripts, inference server, and benchmark evaluation utilities.
- `experiments/inference/` - Local robot inference clients that call a VLA action server and send actions to the arm.
- `experiments/inference/tts/` - Speech-to-text and text-to-speech interface for spoken robot commands.
- `experiments/ik/` - Inverse-kinematics experiments and collected 6-DOF trajectories.
- `software/` - Dataset collection, IK control, webcam, and Arduino-facing scripts.
- `arduino/` - Arduino sketches for 6-DOF servo control.

## Installation

Use Python 3.10 for the VLA stack. A conda environment is recommended, especially for GPU training:

```bash
conda create -n everydayvla python=3.10 -y
conda activate everydayvla
```

Install the dependency group for the workflow you need:

```bash
# Core VLA inference/server dependencies
pip install -r requirements-core.txt

# Training and fine-tuning dependencies
pip install -r requirements-training.txt

# Local robot control, IK, Arduino serial, and camera collection
pip install -r requirements-robot.txt

# Voice command interface
pip install -r requirements-tts.txt

# LIBERO or ALOHA benchmark dependencies
pip install -r requirements-libero.txt
pip install -r requirements-aloha.txt

# Formatting, linting, notebooks, and debugging
pip install -r requirements-dev.txt
```

For PyTorch, install the build that matches your CUDA or CPU setup before installing the training stack when possible. See the official PyTorch selector for the correct command.

FlashAttention is recommended for training but is CUDA/platform-specific. After PyTorch is installed and CUDA is configured, install it manually if your machine supports it:

```bash
pip install "flash-attn==2.5.5" --no-build-isolation
```

## System Dependencies

The robot and audio paths require hardware/system packages in addition to pip dependencies:

- Arduino IDE or CLI for sketches in `arduino/`.
- A connected Arduino/servo controller with the correct serial port configured in the scripts.
- A webcam or IP camera for dataset collection and inference.
- FFmpeg for `pydub` audio playback in the TTS interface.
- PortAudio development libraries for `pyaudio`.
- ROS/Interbotix packages for the real ALOHA path: `rospy`, `cv_bridge`, `sensor_msgs`, `interbotix_xs_modules`, and `interbotix_xs_msgs`.

On Ubuntu, the audio dependencies can usually be installed with:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg portaudio19-dev python3-pyaudio
```

## Common Workflows

The model-side EveryDayVLA demos live in `vla/README.md`, including an action-chunk smoke test, LIBERO AdaHorizon evaluation, real-robot action-server launch, and fine-tuning command.

Start an OpenVLA action server from the `vla/` code, then use the local inference clients in `experiments/inference/` to send camera observations and language instructions to the server. The local client serializes the model action and sends it to the Arduino-controlled arm.

Dataset collection and IK experiments live in `software/` and `experiments/ik/`. These scripts expect the robot URDF files, an available camera, and a configured Arduino serial port.

The optional voice interface lives in `experiments/inference/tts/` and uses OpenAI speech APIs. Set your API key in the environment before running it:

```bash
export OPENAI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

## Development

The `vla/Makefile` exposes basic formatting and linting commands:

```bash
cd vla
make check
make autoformat
```

## Citation

```bibtex
@article{chopra2025everydayvla,
  title={EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation},
  author={Chopra, Samarth and McMoil, Alex and Carnovale, Ben and Sokolson, Evan and Kubendran, Rajkumar and Dickerson, Samuel},
  journal={arXiv preprint arXiv:2511.05397},
  year={2025}
}
```
