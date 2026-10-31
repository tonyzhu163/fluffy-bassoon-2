# Native state and calibration interpretation

The pinned native RoboTwin state vector reports commanded arm drive targets.
It must not be described as measured arm joint position feedback. This is a
static source finding and a limitation on interpreting the proposed controlled
calibration experiment; it does not admit source collection, change the
condition, or alter the current capacity probe.

## Source chain and evidence boundary

- RoboTwin `dca9ec682688821c42944eacd7f5dd9bfb15396f`,
  `envs/robot/robot.py` SHA
  `1f4d0284eaf47fe5de2c0ba98a173b78beb83f2f2cc43409d37072a4774df4c8`:
  the retained source audit locates `get_left/right_arm_jointState` at528–539,
  reading each joint's `get_drive_target()[0]` and appending the normalized
  gripper value. The separate real-state method at542–549 reads articulation
  qpos; it is not the method used by `get_obs` for this vector.
- `envs/_base_task.py` SHA
  `6b3f74b038509ef989913b3f6f37c74e1d0b7156d201a1bf0100f9ccb6b0d46b`,
  actual cached source lines512–519: calls those left/right arm methods and
  concatenates their values into `joint_action.vector`.
- `robotwin/envs/vector_env.py` SHA
  `3a068625f08e006162a5583ddf5175e42ad9ab7941d1016de89bbb7fe63ebc0d`,
  cached line56: `state = observation["joint_action"]["vector"]`.
- This repository's `_RoboTwinBackend._encode` sends the raw observation to
  the pinned Motus/RLinf observation converter, then appends
  `observation.state[:, :14]` to the2048 masked-mean VLM prefix. Student2076
  additionally appends the previous actual valid pre-calibration request14.
- RLinf `c605b9d55ed93a180119b099e6f836abff293b08`,
  `rlinf/envs/sim/robotwin/robotwin_env.py` SHA
  `9d468358e2286b55d807f8fe9569419f1ef4e343488b2f0bfd4eabd521fad210`,
  lines172–215: `_extract_obs_image` appends each `obs["state"]`, stacks it,
  and returns that tensor as raw `states`; no measured-qpos substitute occurs.
- `rlinf/models/embodiment/openpi/env_io.py` SHA
  `ba36371be49982fc921cb4bd1a1ec1e7a32f2c3f5eef084be59385f0155b86dc`,
  lines61–65 and128–137: the configured state selector returns the supplied
  state when no subset is configured; the environment dictionary goes through
  configured repack/input transforms. Deterministic model-side normalization
  does not turn drive targets into measured joint positions.

The actual cached evidence files were rehashed locally without loading models,
arrays or pickles:

| Retained evidence | SHA-256 |
| --- | --- |
| `phone-data-preparation/source-conventions.json` | `351dd8765aaae75366bb3485730f824b47f39cba308e4ad276bd07d6858ff127` |
| `phone-stage0-cpu-artifacts/author-source-excerpts.json` | `509916516706537a41250e24d6f1fbb268eb11da9d0064d864fb699e08f30861` |
| `PHONE-STAGE0-CPU-CHECK.json` | `ef8f6357fc876e609fd0aea507a5ece730880cd6cdb5badea921983f6f66c783` |
| `PUBLIC-STATE-SOURCE.json` | `4d0f77bac7edd7c8d20855c10c87ba9f7382f4ec8e826205231167d66c2b2bcc` |

The first two are published by the execution owner at commit
`2436eec1574d917dae6e2bda876575113e73f3e0` under
`outputs/frozen-pi05-endtoend-20261031-r1/`. The wrapper receipt is retained in
the shared `vla-source40-integration-1m` worktree. It embeds the exact source
excerpts, not a fresh runtime observation. The robot-method distinction is
supported by the existing hash-bound source-convention audit; no physical
equivalence claim was inferred from numeric arrays here.
The final missing RLinf wrapper/EnvIO snippets were supplied in the existing
source owner's commit`65354eb14a5385714e9e562067de7c64f8bf20c6`, then read and
rehash-checked from the shared filesystem. A byte-preserved copy of that compact
JSON is retained beside this audit. No new Git/SSH query, numerical-array load,
model, test, simulator, scheduler query or allocation was needed for the chain.

## Consequence for the controlled task

Following a chunk, the agent has both the previous requested commands and
current public commanded-target state. Away from clipping and partial/interpolated
execution, their difference can reveal the calibration offset after one chunk.
Normalization, TOPP interpolation and early termination can affect the simple
equality; this audit does not claim a universally exact one-step estimator.

Both signals already enter the trained static student's2076D input. Therefore
an A2 contrast against mismatched and one fixed conditional teacher establishes
condition dependence, not history necessity over a trained static policy.
Memory may still help at the first query decision: a fresh query reset zeros
its previous-command field while full history can retain support information.
That claim needs actual paired first-query behavior and complete controls.

Retain trained-static, same-weight whole-history reset, reward-blind and
consistent-reward-shuffle comparisons, with identical actual reset/command
inputs and frozen query weights. Report the first query decision separately
from later query adaptation. If rewards are unnecessary, describe any result
as memory-assisted dynamics/system identification rather than evidence of
reward-causal in-context reinforcement learning. No observation channel,
condition, task, scope or resource allocation is changed by this finding.

## Actual raw gripper compatibility finding

The authorized phone-stage0 CPU job inspected17,591 actual valid native rows
(17,760 issued slots): raw gripper6/13 ranges were
`[-0.0101426,1.0090805]` and `[-0.0069030,1.0086253]`, with2,362/5,913 values
outside[0,1], all finite. Original evidence
`phone-stage0-cpu-artifacts/a1-gripper-statistics.json` hashes to
`1e126be345a4641df36ec14fd902cf9de9db2cb4572e78a7f4fcb01d5fd35536`.

The integration applies only the demonstrated compatibility correction:
calibration headroom and limit handling cover arm channels; raw gripper
requests remain byte-identical and are passed to the unchanged native actuator
clamp. Arm limits, physical correction magnitude, labels and raw corpora are
unchanged. Four new NumPy/Torch/sign regressions join the same composite CPU
gate; no extra robot episode or qualification job is requested.
