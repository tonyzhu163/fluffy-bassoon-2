# Native fork interfaces

The runner calls `teacher.make_teacher(arm, obs)` once at the handoff and once on an independent reset. `arm` is the already-running native `ArmEnv` (the camera-free session's `env.arm`). `obs` is the current camera-free public observation with `state` and `prompt`. The returned callable takes each new public observation and returns one finite, bounded 7-D action. The teacher reads the existing simulator through `arm._env`. The runner owns native reset and stepping. The fixed teacher computes endpoint geometry in scratch data.

Use the prior CPU launcher environment and invoke the following from the qualified simulator interpreter.

```sh
python -Bu "$here/fork.py" --checkout "$orig" --panel "$panel" \
  --prior "$r/outputs/embodiment-sawyer-source-contact-20261120-r1/native01/replay" \
  --out "$run/fork" --workers 6
```

`--out` must be fresh. The standard CLI runs all three branches on the first registered seed, 740000, as its Linux admission. Once the replay and capture checks pass, that case counts toward the 50 and the remaining 49 run in parallel. Both expert outcomes are admissible. The runner writes `cases.json` with the 50 ordered layouts, `cases/<seed>.json` with branch receipts or diagnostic errors and `captures/<seed>.npz` with numeric trajectories and fork states. A nonzero exit indicates incomplete or failed validation.

For a first contact in control `k`, the handoff capture is the observation and native state after that entire control. The first replacement action is control `k+1`. Each NPZ uses `raw__`, `contact_expert__` and `reset_expert__` prefixes. `actions[i]` is the issued float32 action for native control `i+1`. `action_source[i]` is 0 for the archived Huber action or 1 for the fixed teacher. Public states and geometry endpoints begin at reset index 0. Contact masks and substep distances follow the [prior trace format](../embodiment-sawyer-source-contact-20261120-r1/geometry.py).

The `opening__` and `handoff__` fields retain the actual MuJoCo integration state, OSC controller attributes, gripper command, native counters, clocks and public state used for each comparison. The JSON gives their digests, teacher phase and alignment, action counts, available and consumed controls, native rewards and termination. A fresh raw replay must reproduce every archived issued action, motion, reward, done flag, success and stored replan state, plus every value of the prior geometry trace. The paired branch must reproduce the raw prefix and handoff byte for byte.
