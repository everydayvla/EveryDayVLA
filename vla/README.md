# EveryDayVLA Code

This folder contains the model-side contributions for [EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation](https://arxiv.org/abs/2511.05397).

The code extends OpenVLA/OFT-style training and deployment with:

- Joint continuous L1-regression and discrete autoregressive action prediction.
- Chunked action prediction for 7-DOF end-effector controls.
- AdaHorizon, an adaptive-horizon controller that uses continuous/discrete action disagreement as an uncertainty signal.
- LIBERO evaluation scripts for continuous, discrete, and AdaHorizon policies.
- A FastAPI action server for real-robot deployment.

The original OpenVLA-OFT setup notes are still useful for environment details; see [SETUP.md](SETUP.md), [LIBERO.md](LIBERO.md), and [ALOHA.md](ALOHA.md).

## Main Contribution Files

- `prismatic/models/action_heads.py` - continuous action heads, including the L1 regression head used by EveryDayVLA.
- `prismatic/extern/hf/modeling_prismatic_new.py` - hybrid prediction path that returns continuous and discrete action chunks.
- `experiments/robot/mad_comp.py` - AdaHorizon mean-absolute-difference horizon selection.
- `experiments/robot/openvla_utils_hybrid.py` - hybrid model loading and action generation helpers.
- `experiments/robot/robot_utils_hybrid.py` - wrappers for hybrid, ensembled, and AdaHorizon policy queries.
- `experiments/robot/libero/run_libero_eval_test.py` - LIBERO evaluation path using AdaHorizon.
- `vla-scripts/finetune_custom.py` - LoRA fine-tuning with continuous and autoregressive action objectives.
- `vla-scripts/deploy.py` - action server for real-world robot clients.

## Environment

From the repository root:

```bash
conda create -n everydayvla python=3.10 -y
conda activate everydayvla

pip install -r requirements-training.txt
pip install -r requirements-libero.txt
```

For GPU training, install a PyTorch build matching your CUDA version before the requirements above. FlashAttention is optional but recommended for large training runs:

```bash
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Demo 1: Generate an Action Chunk

This smoke test loads a checkpoint, runs the policy on the included LIBERO sample observation, and prints the predicted action chunk. Replace `PRETRAINED_CHECKPOINT` with a local checkpoint path or Hugging Face model id that includes the matching `action_head` weights.

Run from this `vla/` directory:

```python
import pickle

from experiments.robot.libero.run_libero_eval_test import GenerateConfig
from experiments.robot.openvla_utils_hybrid import get_action_head, get_processor, get_vla
from experiments.robot.robot_utils_hybrid import get_ada_action
from experiments.robot.mad_comp import AdaptiveHorizon
from prismatic.vla.action_tokenizer import ActionTokenizer

cfg = GenerateConfig(
    pretrained_checkpoint="PRETRAINED_CHECKPOINT",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=1,
    use_proprio=False,
    center_crop=True,
    num_open_loop_steps=8,
    task_suite_name="libero_spatial",
    load_in_8bit=False,
    load_in_4bit=False,
)

model = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, model.llm_dim)
action_tokenizer = ActionTokenizer(processor.tokenizer)
ada_horizon = AdaptiveHorizon(min_actions=4, threshold=0.03, replan_threshold=0.06)

with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as f:
    observation = pickle.load(f)

actions = get_ada_action(
    cfg=cfg,
    model=model,
    obs=observation,
    task_label=observation["task_description"],
    processor=processor,
    action_head=action_head,
    action_tokenizer=action_tokenizer,
    ensembler=ada_horizon,
)

print(actions)
```

Save it as `demo_everydayvla_action.py` or run it in a Python shell from `vla/`.

## Demo 2: Run LIBERO Evaluation with AdaHorizon

Run from `vla/` after installing the LIBERO dependencies and preparing the LIBERO datasets:

```bash
python experiments/robot/libero/run_libero_eval_test.py \
  --pretrained_checkpoint /path/to/everydayvla_checkpoint \
  --task_suite_name libero_spatial \
  --num_trials_per_task 10 \
  --use_l1_regression True \
  --use_diffusion False \
  --num_open_loop_steps 8 \
  --center_crop True
```

Use `libero_object`, `libero_goal`, or `libero_10` for the other suites.

## Demo 3: Start a Real-Robot Action Server

The deployment server exposes `/act` on port `9000`. A client can send an RGB image plus language instruction and receive a predicted action chunk.

Run from `vla/`:

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

The local robot clients outside this folder, such as `../experiments/inference/deploy.py`, post camera observations and instructions to this server, then send the returned actions to the Arduino-controlled arm.

## Fine-Tuning

Use `vla-scripts/finetune_custom.py` for the EveryDayVLA training recipe:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_custom.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /path/to/tensorflow_datasets \
  --dataset_name consolidated_dataset \
  --run_root_dir runs_everydayvla \
  --use_l1_regression True \
  --use_auto_regression True \
  --use_diffusion False \
  --batch_size 8 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --lora_rank 32 \
  --max_steps 50000
```

Important configuration knobs:

- `use_l1_regression=True` trains the continuous action head.
- `use_auto_regression=True` keeps the discrete action-token objective active.
- `use_diffusion=False` matches the L1 objective described in the EveryDayVLA paper.
- `num_open_loop_steps=8` matches the chunk length used by AdaHorizon.
- `unnorm_key` must match the dataset statistics stored in the checkpoint.

## Notes

Some files in this folder are inherited from OpenVLA/OFT and remain for baseline comparison or experiments. The EveryDayVLA path is the hybrid model plus AdaHorizon flow listed above.

If a local checkpoint is moved between machines, `experiments/robot/openvla_utils_hybrid.py` may update its Hugging Face `auto_map` configuration so the checkpoint uses the local EveryDayVLA model classes.

## Citation

```bibtex
@article{chopra2025everydayvla,
  title={EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation},
  author={Chopra, Samarth and McMoil, Alex and Carnovale, Ben and Sokolson, Evan and Kubendran, Rajkumar and Dickerson, Samuel},
  journal={arXiv preprint arXiv:2511.05397},
  year={2025}
}
```
