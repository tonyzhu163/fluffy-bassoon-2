# State lane result

**No native reward was measured, so nothing here is task success.** Archived Kinova3 records show one state difference from the source convention: the hand quaternion is reported on the opposite sign branch, which moves the packed state by 2π. Four adapters, seven CPU-fixture tests and a launch script are ready for one Baige RTX 4090 job. [Plan](PLAN.md)

## Measurement (archived records, CPU)

`uv run --no-sync python measure.py` → [measurement.json](measurement.json).

| Channel | Kinova3 | Source arm | Evidence |
| --- | --- | --- | --- |
| Rotation vector X, as packed | −3.29 to −3.09 on all 10 archived public frames (layouts 14 and 15, controls 0–90) | Positive in every source training example (checkpoint minimum +0.353 rad, SmolVLA) | [source-view frames](../frozen-vla-source-view-20261123-r1/evidence/cpu01), [state](../frozen-vla-factory-20261123-r1/state/RESULT.md) |
| Same frames after the sign flip | +2.97 to +3.15; state moves by 6.2832 in L2 | Panda on these layouts: +3.06 to +3.17 | [pose replays](../frozen-transfer-pose-20261124-r1/evidence/replay) |
| Grip-site position at reset | Kinova3 − Panda on layout 14: (−2.2, −0.8, −3.2) mm; layout 15: (3.3, 7.4, 5.0) mm | | pose replays |
| Grip-site rotation at reset | 3.35° and 2.31° from Panda on layouts 14 and 15 | | pose replays |
| Hand frame against grip site | Fixed 90° about Z, constant to 0.09° over the 10 frames | Same gripper | frames and replays |
| Finger qpos | 0.0333–0.0402 and −0.0400 to −0.0387 | Not archived as numbers | source-view frames |
| Released reset, fixed kinematics | Same grip-site target and 0.0775 m jaw opening as IIWA and UR5e; jaw 0.0437–0.0771 m over the 15 controls on all three | | [stage-2c cells](../xemb-stage2c-cpu-20261121-r1/evidence) |

- The reset position differences change sign between layouts, so they give no fixed offset to remove. All three arms carry `PandaGripper`.
- The pose replays hold rotation matrices. The Kinova3 sign along a whole rollout is known only at the 10 frames.
- No record in the repository holds a source arm's reported quaternion or finger qpos as numbers. The source branch rests on the SmolVLA checkpoint statistic and the UR5e finding in FINDINGS. `state-dropped` checks the π₀.₅ checkpoint's own mean for a positive X at launch.
- Every archived native Kinova3 run packed state with the raw sign.

## Adapters ([adapter.py](adapter.py))

All four act in the `observation` hook on `robot0_eef_pos`, the quaternion and `robot0_gripper_qpos`. Images, prompt, noise, choice and commands are untouched.

| Name | State given to the policy |
| --- | --- |
| `state-mapped` | Quaternion negated when its X is negative. Position and fingers unchanged, because the measurement supports no fixed offset. Uses fixed kinematics only; the calibration frames are counted, not fitted. |
| `state-dropped` | The source mean at every replan, read from the checkpoint's `norm_stats.json`. Zero would place the grip site about 1 m below its workspace with no rotation, which is an out-of-range state and not an absent one. On CPU without the checkpoint it falls back to the archived Panda mean on this task ([source_reference.json](source_reference.json)). |
| `state-shuffled` | A whole real state of this body from another time, raw sign: drawn with a fixed seed from the four calibration frames and earlier replans of the episode, never the current one. |
| `state-mapped-shuffled` | The same draw on the source branch. Added after the plan: it separates the branch from the time alignment, which `state-shuffled` alone confounds. |

Change from the plan: the shuffle draws with replacement from the past, since an online adapter cannot permute states it has not seen.

## CPU fixture

`uv run --no-sync pytest -q -W error outputs/frozen-shift-state-20261124-r1`: 7 passed. The fixture's rewards are synthetic. The tests cover panel discovery and a full paired start without lane faults, that only `observation/state` changes, rotation equivalence of the map, the constant dropped state and its source, and that shuffled states are whole earlier states.

## Not done

- No native rollout and no π₀.₅ forward pass. Whether the sign branch changes proposals or reward on Kinova3 is unmeasured.
- `run.sh` was syntax-checked only. It has not run on Baige.
- The checkpoint's state mean was not read here; the launch script requires it by checksum.

## Launch

In a Baige Distributed Training allocation with one RTX 4090, on a clean checkout of this branch:

```
outputs/frozen-shift-state-20261124-r1/run.sh "$(git rev-parse HEAD)" state-r1 --starts development
```

It runs raw, repeat and the four adapters on the 11 development starts (66 episodes) through the shared panel and writes to `outputs/frozen-vla-proposal-shift-20261124-r1/state-r1` under the Baige root. Read `state-mapped` against `raw`, `repeat` and `state-dropped` in `summary.json`.
