# Experiments

This folder contains laptop-side experiment code for the EveryDayVLA robot. The model server lives in `../vla/`; these scripts capture camera frames, send observations and language instructions to the server, convert returned end-effector actions into IK joint angles, and stream commands to the Arduino-controlled arm.

## Layout

- `inference/deploy.py` - main real-robot client for text instructions.
- `inference/openvla_inf.py` - compatibility wrapper for `inference/deploy.py`.
- `inference/deploy_tts.py` - voice-enabled robot client using speech-to-text and text-to-speech.
- `inference/vla_test_tts.py` - older voice-control test client kept for reference.
- `inference/tts/` - reusable speech-to-text/text-to-speech modules.
- `ik/dataset_collection_ik.py` - IK trajectory execution and dataset collection.
- `ik/user_input_ik.py` - manual IK control from typed target poses.
- `ik/datasets/` - collected trajectory examples.
- `ik/seal.urdf` and `inference/seal.urdf` - robot descriptions used by IKPy.

## Dependencies

From the repository root, install the relevant groups:

```bash
pip install -r requirements-robot.txt
pip install -r requirements-tts.txt  # only needed for voice control
```

Hardware and system requirements:

- Arduino running the matching 6-DOF control sketch.
- USB serial connection to the Arduino.
- Webcam, video file, or IP camera stream.
- A running EveryDayVLA action server from `../vla/`.
- FFmpeg and PortAudio for voice control.

For voice control, set the OpenAI API key in the shell:

```bash
export OPENAI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

## Start the Model Server

Run this from `../vla/` before using the laptop clients:

```bash
python vla-scripts/deploy.py \
  --pretrained_checkpoint /path/to/everydayvla_checkpoint \
  --unnorm_key consolidated_dataset \
  --use_l1_regression True \
  --use_diffusion False \
  --num_open_loop_steps 8 \
  --host 0.0.0.0 \
  --port 9000
```

## Text-Instructed Robot Client

Run from `experiments/inference/`:

```bash
python deploy.py \
  --instruction "pick up the block and move it away from the robot" \
  --unnorm-key consolidated_dataset \
  --arduino-port COM3 \
  --baud-rate 115200 \
  --video 0 \
  --server-url http://localhost:9000 \
  --urdf seal.urdf
```

Use `--video` for either a camera index, a video file, or an IP camera URL such as a DroidCam stream. Press `q` in the OpenCV window to stop.

## Voice-Controlled Robot Client

Run from `experiments/inference/`:

```bash
python deploy_tts.py \
  --port COM3 \
  --baud 115200 \
  --video 0 \
  --dataset consolidated_dataset
```

The voice client listens for an instruction, sends frames to the model server, speaks status updates, and accepts keyboard controls:

- `space` - listen for a new instruction.
- `e` - end the current task.
- `q` - quit.

`vla_test_tts.py` is an older test client with similar behavior. Prefer `deploy_tts.py` for demos.

## IK Dataset Collection

`ik/dataset_collection_ik.py` executes scripted end-effector trajectories, records `end_effector_angles.txt`, writes `language_instruction.txt`, and optionally records videos. The active trajectory list is the `datasets` variable in that file; each entry contains:

- `Coordinates` - a sequence of 7-DOF end-effector poses.
- `Instruction` - the natural-language task instruction.
- `Folder` - output folder name for the collected trajectory.

Run from `experiments/ik/`:

```bash
python dataset_collection_ik.py
```

Before collecting new data, check:

- Arduino serial port and baud rate near the top of the script.
- Camera source if recording video.
- `dataset_dir` output path.
- `seal.urdf` path.
- The active trajectory in the `datasets` list.

The notebook `ik/ik_test.ipynb` is useful for visual IK debugging. Install `ipympl` if the interactive plots do not render.

## Safety Notes

- Keep the robot powered off while changing hardware or repositioning objects.
- Restart Arduino power if the arm stops responding or the gripper stalls.
- Use conservative gripper values for hard objects; an over-tight gripper can stall and delay release.
- Keep the server, camera, and Arduino logs visible during experiments.
