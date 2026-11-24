# Why streaming SymMeta scored 401/1,400 on ReGround's released 90° rotation test

Exploratory. Seven arms on the 280 distinct task–start pairs (seed label 0), 1,960 scored episodes plus 280 sixteen-control supports. Estimated 12 GPU-hours from the streaming run's 8.5 GPU-hour cap for 1,400 episodes; the launcher reserves at most 14.

## Question

The [reference run](../reground-rotation-reference-20261123-r1/RESULT.md) scored 0 / 401 / 1,271 of 1,400 for the frozen policy, streaming SymMeta and fitted ReGround. Three properties of the SymMeta arm differ from an exact inverse of the released shift, apart from the estimate itself:

1. **Late start.** The first correction can act on control 10.
2. **Wrist commands.** The released shift rotates the translation and the wrist-rotation command triplets ([RotZ](https://github.com/umersiddique94/ReGround/blob/16a98486f560676b9036d1a6d067b9dd51a5917a/reground/envs/shifts.py#L70-L87)). The adapter passed wrist commands through, and its validation asserts that.
3. **Policy EEF input.** The released shift leaves observations untouched ("The VLA and every controller see identical observations; only the body differs"). The adapter rotated the policy's EEF position about its start by the estimated yaw ([policy_observation](../reground-rotation-reference-20261123-r1/symmeta_adapter.py)), which is correct for this paper's shift family, where the robot reports position in its own frame, and wrong here.

Which of these, or the estimate, accounts for the gap? And what does the paper's own protocol, a calibrated retry from 15 uncorrected controls, score on this test?

## Arms

| Arm | Correction | First control | Wrist rotated | Policy EEF rotated |
| --- | --- | ---: | --- | --- |
| `exact_asrun_0` | exact 90° | 0 | no | yes |
| `exact_asrun_10` | exact 90° | 10 | no | yes |
| `exact_world_0` | exact 90° | 0 | no | no |
| `exact_full_0` | exact 90° | 0 | yes | no |
| `exact_full_10` | exact 90° | 10 | yes | no |
| `stream_full` | seed-41 reader, streaming | 10 | yes | no |
| `retry_full` | seed-41 reader on a 15-control support | 0 | yes | no |

Readings: `exact_asrun_10` against the recorded streaming arm isolates estimate error. `exact_asrun_0` against `exact_world_0` isolates the EEF input, `exact_world_0` against `exact_full_0` the wrist, and `exact_full_10` against `exact_full_0` the ten uncorrected controls. `exact_full_0` is the analytic inverse and should approach the nominal policy. `stream_full` and `retry_full` are the two protocols with the correction family matched to the shift.

`retry_full` runs 16 uncorrected controls on a calibration start of the same task (`(start − 15) mod 15`, so 0–14), reads the estimate after the third five-control chunk, then scores the test start with that estimate fixed from the first control. The sixteenth control exists only so the third chunk is read before the support ends. A support that ends early keeps the nominal interface.

## What would count against the hypotheses

If `exact_asrun_0` is already near `exact_full_0`, the wrist and EEF input were not the cause and the 401 reflects the estimate and the late start. If `exact_full_10` is far below `exact_full_0`, ten uncorrected controls are costly on this test whatever the estimator. If `retry_full` stays near the streaming result, the second-attempt protocol does not transfer to this test.

## Limits

- That the shift rotates wrist commands and leaves observations untouched is supplied to the adapter here. The reader does not estimate either property.
- One reader seed (41), one seed label, 35 starts per task. The exact arms are deterministic given the start and policy noise.
- `retry_full` uses a different start for support, as in the paper, and costs one extra reset and 16 unscored controls per query. ReGround's fit uses 10,000 target transitions shared across queries.
- Supports and scored episodes draw different policy noise, each from the released rule for its start.
- A simulator worker's earlier resets of a task move that task's fixtures ([prefix audit](../reground-rotation-reference-20261123-r1/prefix-audit/RESULT.md)). Every shard here starts from new workers, and supports reset a separate set of workers, so all seven arms score the same scenes; the summary lists any start whose reset image differs between arms. The recorded streaming run kept one worker across a shard boundary, so six of its drawer scenes differ from these.

## Run

```bash
bash outputs/reground-rotation-decompose-20261123-r1/run.sh <pin> <attempt> <2|4|5 GPUs> <wall seconds> [--arms ...] [--shards ...]
```

Each arm's shards go to `decompose-native/<arm>/shards` under the reference runtime root; `<attempt>/decompose-summary.json` holds successes, unsafe counts and per-task counts by arm, for the arms and shards the attempt selected.

[arms.py](arms.py) subclasses the reference run's `MatchedBatch` and reuses its lane (GPU checks, shard locks, staging), its reader process and the released episode loop. `arms.sh` and `run.sh` are `symmeta_adapter.sh` and `stream.sh` with the entry point, store and budget changed, and with the current source ahead of the archived source on `PYTHONPATH` so the wrist rotation in `retarget_action_from_motor` is found. The archived and current `embodiment.py` are otherwise identical.

**Not run anywhere.** The Devin VM has no simulator, policy weights or route to Baige. Checked here: the wrist rotation inverts the released `RotZ(90)` to 3e-17 on the pinned upstream source, [its test](../../tests/test_xemb_wrist_retarget.py) passes, and the three files parse. Start with one shard of one arm (`--arms exact_full_0 --shards 0`) before the full run.
