# Public-state command-target chain — static handoff

The pinned deployed source makes the public arm state a drive-target channel:

| Source | Exact lines | Mapping |
| --- | --- | --- |
| RoboTwin `envs/_base_task.py` |462–519|`get_obs()` captures images and `joint_action/vector` from `get_left/right_arm_jointState()`|
| RoboTwin `envs/robot/robot.py` |528–549|The public getters read `joint.get_drive_target()[0]`; the separately named `get_left_arm_real_jointState()` reads actual articulation qpos|
| RoboTwin `robotwin/envs/vector_env.py` |48–63|`joint_action/vector` becomes returned observation `state`|
| RLinf `rlinf/envs/sim/robotwin/robotwin_env.py` |198–212|Per-environment `obs['state']` is stacked into raw `states`|
| RLinf `rlinf/models/embodiment/openpi/env_io.py` |128–137|That environment dictionary passes through the configured repack and input-transform pipeline|

Full file hashes and the added exact snippets are in `PUBLIC-STATE-SOURCE.json`; the first two source files are already hash-bound in `phone-data-preparation/source-conventions.json` and the allocated stage0 author excerpts. RoboTwin pin is `dca9ec682688821c42944eacd7f5dd9bfb15396f`, RLinf pin `c605b9d55ed93a180119b099e6f836abff293b08`.

The14 public channels therefore contain transformed commanded arm drive targets plus the gripper value, without an independent measured arm-qpos channel on this path. Previously verified paired hashes establish equal public inputs; they are not a separate privileged physical-qpos measurement. This is a task/interface limitation to describe accurately, not a source-feature change or a new gate. Only standard-library file inspection was used; no VLA/Torch/simulator imports, numerical arrays, simulator RPC, tests, scheduler reads or allocations were used for this handoff.
