---
base_model: lerobot/pi05_base
datasets:
- lerobot/robotwin_unified
library_name: lerobot
license: apache-2.0
model_name: pi05_robotwin
pipeline_tag: robotics
tags:
- lerobot
- robotics
- vision-language-action
- pi05
- robotwin
- aloha
- bimanual-manipulation
---

# π₀.₅ for RoboTwin 2.0

This is a [LeRobot](https://github.com/huggingface/lerobot) π₀.₅
Vision-Language-Action policy fine-tuned from
[`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) on the
[`lerobot/robotwin_unified`](https://huggingface.co/datasets/lerobot/robotwin_unified)
dataset. It predicts joint-space action chunks for the 14-DoF Aloha-AgileX
bimanual robot used by [RoboTwin 2.0](https://robotwin-platform.github.io/).

The checkpoint in this repository is the final checkpoint from training step
50,000. The model consumes language instructions, one 14-dimensional robot
state, and three RGB camera views.

## Model details

| Property | Value |
| --- | --- |
| Policy | π₀.₅ (`pi05`) |
| Fine-tuned from | `lerobot/pi05_base` |
| Framework | LeRobot / PyTorch |
| Robot | Aloha-AgileX bimanual, 14 DoF |
| Action representation | Absolute joint-space actions |
| Action chunk size | 50 |
| Actions executed per prediction | 50 |
| Inference steps | 10 |
| Internal image resolution | 224 × 224 |
| Model dtype | bfloat16 |
| License | Apache-2.0 |

## Inputs and outputs

The feature names below are part of the checkpoint configuration. Camera names
must match these keys, or be mapped to them with `--rename_map`.

### Inputs

| Feature | Type | Shape |
| --- | --- | --- |
| `observation.state` | State | `(14,)` |
| `observation.images.cam_high` | RGB image | `(480, 640, 3)` |
| `observation.images.cam_left_wrist` | RGB image | `(480, 640, 3)` |
| `observation.images.cam_right_wrist` | RGB image | `(480, 640, 3)` |
| `task` | Natural-language instruction | string |

### Output

| Feature | Type | Shape |
| --- | --- | --- |
| `action` | Absolute joint-space action | `(14,)` per step, 50-step chunks |

The 14 state/action dimensions are ordered as follows:

```text
left_waist, left_shoulder, left_elbow, left_forearm_roll,
left_wrist_angle, left_wrist_rotate, left_gripper,
right_waist, right_shoulder, right_elbow, right_forearm_roll,
right_wrist_angle, right_wrist_rotate, right_gripper
```

## Training data

The policy was trained on `lerobot/robotwin_unified` in LeRobot v3.0 format.
The dataset metadata available for this training run reports:

- 27,500 episodes
- 6,075,103 frames
- 30 FPS
- three RGB views: high, left wrist, and right wrist
- 14-dimensional Aloha joint state and action

RoboTwin 2.0 covers 50 bimanual manipulation tasks with varied objects,
layouts, lighting, backgrounds, and language instructions.

## Training configuration

| Setting | Value |
| --- | --- |
| Training steps | 50,000 |
| Batch size | 16 |
| Optimizer | AdamW |
| Peak learning rate | 2.5e-5 |
| Weight decay | 0.01 |
| LR schedule | Cosine decay with 1,000 warmup steps |
| Decay steps / final LR | 30,000 / 2.5e-6 |
| Seed | 1000 |
| Gradient checkpointing | Enabled |
| `torch.compile` | Enabled (`max-autotune`) |
| Vision encoder frozen | No |
| Expert-only training | No |
| Image augmentation | Disabled |
| State/action normalization | Mean and standard deviation |

The saved preprocessor and postprocessor files contain the normalization state
needed for inference; upload them together with `model.safetensors` and
`config.json`.

## Installation

Install LeRobot with the π policy dependencies:

```bash
pip install "lerobot[pi]"
```

RoboTwin evaluation additionally requires Linux, an NVIDIA GPU, and the
RoboTwin SAPIEN/CuRobo environment. See the
[LeRobot RoboTwin guide](https://huggingface.co/docs/lerobot/main/en/robotwin)
for simulator setup.

## Loading the policy

The checkpoint includes serialized pre- and postprocessing pipelines. Load all
three components from the same Hub repository:

```python
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy

model_id = "SidneyXie/pi05_robotwin"
device = "cuda"

policy = PI05Policy.from_pretrained(model_id)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=model_id,
    preprocessor_overrides={"device_processor": {"device": device}},
)
```

At inference time, pass a batch containing the three image features,
`observation.state`, and a natural-language `task`. Use
`policy.select_action(preprocessor(batch))`, then apply `postprocessor` to the
result before sending it to the robot or simulator.

## RoboTwin evaluation

RoboTwin's environment camera keys differ from the names stored in this
checkpoint, so the rename map is required. A quick five-episode evaluation on
one task is:

```bash
lerobot-eval \
  --policy.path=SidneyXie/pi05_robotwin \
  --env.type=robotwin \
  --env.task=beat_block_hammer \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --rename_map='{"observation.images.head_camera":"observation.images.cam_high","observation.images.left_camera":"observation.images.cam_left_wrist","observation.images.right_camera":"observation.images.cam_right_wrist"}' \
  --output_dir=outputs/eval/pi05_robotwin/beat_block_hammer
```

For an official-style result, evaluate 100 episodes per task and report Easy
(`demo_clean`) and Hard (`demo_randomized`) settings separately. Consult the
[RoboTwin leaderboard](https://robotwin-platform.github.io/leaderboard) for the
current submission protocol.

## Evaluation results

| Setting | Task | Episodes | Success rate |
| --- | --- | ---: | ---: |
| Easy | `adjust_bottle` | 100 | 100% |
| Easy | `beat_block_hammer` | 100 | 93% |
| Easy | `click_alarmclock` | 100 | 90% |
| Easy | `click_bell` | 100 | 86% |
| Easy | `dump_bin_bigbin` | 100 | 96% |
| Easy | `grab_roller` | 100 | 99% |
| Easy | `handover_mic` | 100 | 23% |
| Easy | `lift_pot` | 100 | 57% |
| Easy | `move_can_pot` | 100 | 48% |
| Easy | `move_pillbottle_pad` | 100 | 74% |
| Easy | `move_playingcard_away` | 100 | 96% |
| Easy | `move_stapler_pad` | 100 | 14% |
| Easy | `pick_diverse_bottles` | 100 | 59% |
| Easy | `pick_dual_bottles` | 100 | 69% |
| Easy | `place_a2b_left` | 100 | 49% |
| Easy | `place_a2b_right` | 100 | 41% |
| Easy | `place_bread_skillet` | 100 | 60% |
| Easy | `place_burger_fries` | 100 | 96% |
| Easy | `place_container_plate` | 100 | 86% |
| Easy | `place_dual_shoes` | 100 | 72% |
| Easy | `place_empty_cup` | 100 | 90% |
| Easy | `place_fan` | 100 | 50% |
| Easy | `place_mouse_pad` | 100 | 45% |
| Easy | `place_object_scale` | 100 | 46% |
| Easy | `place_object_stand` | 100 | 90% |
| Easy | `place_phone_stand` | 100 | 66% |
| Easy | `place_shoe` | 100 | 92% |
| Easy | `press_stapler` | 100 | 97% |
| Easy | `rotate_qrcode` | 100 | 71% |
| Easy | `scan_object` | 100 | 18% |
| Easy | `stamp_seal` | 100 | 45% |
| Easy | `turn_switch` | 100 | 55% |
| **Easy** | **Overall (32 tasks)** | **3200** | **67.91%** |

## Intended use and limitations

This checkpoint is intended for research on RoboTwin 2.0 and compatible
14-DoF Aloha-style bimanual setups. It expects the same joint ordering, camera
semantics, observation preprocessing, and action convention used during
training.

- It has not been validated for direct deployment on a physical robot.
- Distribution shifts in camera placement, calibration, control frequency,
  joint scaling, objects, or scene appearance can substantially reduce
  performance.
- The policy can produce unsafe or infeasible actions. Use workspace limits,
  collision checking, emergency stops, and human supervision on real hardware.
- This is a learned policy and does not provide correctness or safety
  guarantees.

## References

- [π₀.₅: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)
- [OpenPI](https://github.com/Physical-Intelligence/openpi)
- [LeRobot](https://github.com/huggingface/lerobot)
- [RoboTwin 2.0](https://arxiv.org/abs/2506.18088)

```bibtex
@article{intelligence2025pi05,
  title   = {Pi 0.5: a Vision-Language-Action Model with Open-World Generalization},
  author  = {Physical Intelligence and Kevin Black and Noah Brown and others},
  journal = {arXiv preprint arXiv:2504.16054},
  year    = {2025}
}

@misc{cadene2024lerobot,
  title        = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in PyTorch},
  author       = {Cadene, Remi and Alibert, Simon and Soare, Alexander and others},
  year         = {2024},
  howpublished = {\url{https://github.com/huggingface/lerobot}}
}
```
