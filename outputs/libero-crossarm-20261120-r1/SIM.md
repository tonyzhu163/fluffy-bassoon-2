# LIBERO-Pro2 physical-arm interface

Panda and Sawyer are the source bodies. UR5e is available as a distinct six-joint MuJoCo model and is reserved for evaluation. All three use their own arm geometry, mass properties, joint limits and actuators with the same physical two-finger PandaGripper. The coordinator's allocated Linux CPU check at `2620e725` restored one shared Goal 4 scene, rendered both cameras and executed three controls on each arm. The public-observation motion path in `703c0b9d` and bit-for-bit Panda trajectory parity have separate verification requirements below. [Execution record](EXECUTION.md)

## Python API

Run this code inside the pinned LIBERO interpreter. The experiment runner owns the OpenPI process and simulator transport.

```python
import numpy as np
from arm_env import ArmEnv

with ArmEnv(arm="Sawyer", suite="libero_goal", task_id=4) as env:
    first = env.reset(init_state_id=0, seed=20261520)
    raw = first["observation"]
    prompt = first["language_instruction"]
    identity = first["model_identity"]
    action = np.array([0.4, 0, 0, 0, 0, 0, -0.4], dtype=np.float32)
    transition = env.probe(action)  # Same live, counted operation as env.step(action).
    motion = transition["sensor_motion"]
```

| Call | Returned fields and semantics |
| --- | --- |
| `reset(init_state_id, seed)` | `observation`, `language_instruction`, `model_identity`, `environment_reproducibility`, `tool_observation`, manifest `max_steps`, `worker_step_index=0`. Reusing a handle for reset builds a fresh simulation. |
| `step(action7)`, `probe(action7)` | `observation`, `language_instruction`, `issued_action`, `sensor_motion`, `tool_observation`, native scalar `reward`, booleans `success`, `terminated`, `truncated`, and `worker_step_index`. Each call advances one 20 Hz control. A separate reset creates an independent support episode. |
| `env.model_identity` | Available after reset. Includes arm family, resolved class, mount, arm XML SHA-256, mesh names on compiled arm bodies, joint names and link order, model `nq/nv`, gripper identity, controller site, camera parent and OSC scale. |
| `close()` | Releases the compiled environment. The context manager closes it on exit. |

**Motor input.** The released public fields are `agentview_image`, `robot0_eye_in_hand_image` (256×256×3 uint8), `robot0_joint_pos`, `robot0_joint_vel` (7/7/6 float32), `robot0_eef_pos` (3), `robot0_eef_quat_xyzw` (4), and `robot0_gripper_qpos` (2). The prior run's `to_openpi_observation(raw, prompt)` rotates both views by 180°, resizes with padding to 224×224 and constructs the 8D state from world grip-site position, hand-link axis-angle and both signed finger slides. The runner issues each decoded and bounded π0.5 command through `step`. [Motor preprocessing](../libero-xemb-sysid-20261119-r1/motor_iface.py) · [Runner transport](runner_native.py)

**Command and reward.** Input is a finite normalized 7-vector in [-1, 1]. The fixed-impedance `OSC_POSE` controller interprets the first six components as world-relative translation and axis-angle increments, with per-axis maxima of 0.05 m and 0.5 rad. The last component uses its sign to increment the finger actuator target. Negative opens, positive closes and zero holds that target. A reward is the value returned by the released task's `step`. The BDDL success predicate ends the episode. The manifest horizon truncates a still-active episode. The study uses BASE tasks to isolate the arm change. [Pinned reward and step](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/bddl_base_domain.py) · [Released adapter](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/libero_remote/environment/libero.py)

**Sensor frames.** `sensor_motion` is a float32 7-vector computed from the previous and returned public observations. Its components are world grip-site displacement in metres, the rotation vector of `R_after @ R_before.T` in radians from the hand-link quaternions, and change in `finger1 - finger2` in metres (positive opening). The installed gripper is rigidly mounted, so its constant hand-to-tool rotation cancels in a world-frame relative rotation. The public samples follow robosuite's observable clock. The audit-only `tool_observation` independently computes current forward kinematics on scratch MuJoCo data. It exposes `tcp_xyz`, `tcp_matrix`, `hand_xyz`, `hand_quat_xyzw`, `link_xyz`, `link_quat_xyzw` and `jaw_opening`. Link arrays follow `model_identity["joint_links"]`. Keep those diagnostics, joint arrays, robot names, scene fingerprints and IK solutions outside the learned context. [Arm interface](arm_env.py) · [Kinematic readback](sim_reset.py)

## Source and reset contract

The local bundle and its checkout identify LIBERO-Pro2.0 revision `e5500c690c1498eca1b7050350654964190cdf70`. The existing Linux setup uses robosuite 1.4.0 and MuJoCo 3.2.3. The robosuite wheel SHA-256 is `aba065e7b36745738cede259457b2cb349427f3608728d867ef3a2034cb62994`. Its model metaclass registers robot XML classes. Its separate `ROBOT_CLASS_MAPPING` selects the `SingleArm` runtime. Stock Sawyer defaults to RethinkGripper, and stock UR5e defaults to Robotiq85Gripper. Explicitly selecting PandaGripper gives all three the same two slide joints, finger geometry and action sign. [Runtime pins and asset hydration](../libero-meta-learning-20261117-r1/bootstrap.py)

| Physical model | Arm joint and actuator count | Meshes declared / referenced by arm geoms | Wrist camera |
| --- | ---: | ---: | --- |
| Panda | 7 | 58 / 57 | `right_hand/eye_in_hand`, 75° vertical |
| Sawyer | 7 | 50 / 50 | `right_hand/eye_in_hand`, 75° vertical |
| UR5e | 6 | 27 / 27 | `right_hand/eye_in_hand`, 75° vertical |

The runtime attaches `robot0_` to arm joints, links and the hand camera, and `gripper0_` to finger joints and the controller's `grip_site`. Each camera has hand-relative position (0.05, 0, 0) m and the same local quaternion. The other image comes from the task arena's `agentview`. The camera and position sensor follow different physical frames. Robosuite's `SingleArm.eef_quat` sensor reads the hand body, while its `eef_pos` sensor reads the grip site 0.097 m along the common tool's local Z. The tool root is rotated −90° about hand-local Z. The implementation preserves both frames. [Pinned observation normalization](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/libero_remote/environment/observations.py)

The domain constructors prepend `Mounted` in tabletop, kitchen and study scenes, and `OnTheGround` in floor, coffee and living-room scenes. The release registers both Panda variants. For Sawyer and UR5e the integration creates the corresponding stock-arm subclasses and registers them with both model factories in the simulator process. It reuses the released arena offset and mount properties, including RethinkMount or the ground variant's NullMount. This preserves the stock alternate arm MJCF and its rest posture. [Released arena classes](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/tree/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/problems) · [Released Panda placement classes](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/tree/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/robots)

1. On Panda, load the published row through native `set_init_state`. On an alternate arm, load it first into a same-task, same-seed, headless Panda donor. Check the complete row length against `1 + nq + nv`. With the common gripper Sawyer has the same length as Panda despite different kinematics. UR5e has one fewer position and velocity. The qualified wheel's flattened state contains time, qpos and qvel.
2. Transfer time, the common fingers and every task joint by MuJoCo joint name, parent body and type. Copy seeded fixture roots from `model.body_pos/body_quat`, which live outside the flat row. Capture the target's updated LIBERO model baseline before native state regeneration. Require exact pre-settle scene readback. Use `environment_reproducibility["scene_fingerprint"]` for cross-body pairing.
3. Use the alternate arm's own stock rest pose as the seed for joint-limited IK. Match the released Panda grip-site world pose by moving only the alternate arm's joints. Set alternate arm velocity to zero and reset the controller's nullspace and orientation goals at the solution. An IK admission failure identifies an unmatched start.
4. Apply the ten native open-gripper settling controls and restart the released rollout clock. Compare every named task-object and fixture-link world pose against the independently settled donor. Admission allows 1 mm and 0.01 rad, each 2% of a full normalized single-axis command. The reset record stores the worst errors, the affected bodies, and the settled tool-pose difference. `initial_state_fingerprint` hashes the complete settled state for within-body pairing. Task success at reset rejects that start. [Named transfer and admission checks](sim_reset.py) · [Native reset implementation](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/env_wrapper.py)

## Qualification evidence

The macOS source check compiled the four sim Python files and inspected the pinned wheel without importing robosuite or MuJoCo. It checked the stock single-arm registry, OSC bounds, arm joint-to-actuator order, gripper slides and motor ranges, wrist-camera parents/extrinsics and every file referenced by the four MJCF assets. It verified all 40 BASE BDDL hashes. Static archive inspection found 50 released float64 rows per pilot task, with width 79 on Goal 1/4/5 and 92 on Spatial 1. The following command reproduces the check from this directory. [Static check](sim_source_check.py)

```sh
python3 sim_source_check.py \
  /tmp/libero-meta-wheelhouse-20261117/robosuite-1.4.0-py3-none-any.whl \
  /tmp/libero-pro2-review-20261117-r1
```

**Allocated CPU evidence from the coordinator.** On Goal 4, initialization 0, seed 20261520, the first +0.4 X command at `2620e725` moved the physical grip site +2.157 mm on Panda, +2.143 mm on Sawyer and +2.351 mm on UR5e. The three models shared the pre-settle scene fingerprint. Both 256×256 images were present. This measures one translation at one start. General reachability, other rotation axes, image parity, and native task completion need their own evidence. The training revision derives reader motion from public samples. The earlier physical readback and public samples can have different timing. [Coordinator's Linux record](EXECUTION.md) · [Three-body qualification driver](qualify.py)

**Panda reference check.** In the qualified Linux CPU simulator environment with the hydrated assets, software EGL and its device 0, run `"$CROSSARM_SIM_PYTHON" -Bu sim_panda_parity.py`. It compares the released adapter and the physical Panda wrapper on the same task/start, instruction, full settled-state fingerprint, all seven public array hashes, rewards and termination flags for three fixed controls. This worker did not execute that comparison. No model or simulator ran on the Mac. [Panda parity check](sim_panda_parity.py)

**Fallback decision.** UR5e is the selected holdout. If a future installed runtime lacks its stock MJCF or a required start fails admission, record the failed model/start and its cause before scoring. A separately declared fallback could use the stock seven-joint IIWA, which has a single-arm registry entry and the same hand-camera contract in this wheel. That alternative requires its own model identity, reset and reachability qualification in a separate pilot.
