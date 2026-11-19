# What the released LIBERO-Pro2.0 demonstrations can teach

The released demonstrations supply examples of successful actions under normal timing and S20 delay. A reader calibrated on two fitting tasks identified the timing condition on **40/40 archived frozen-policy attempts**. In a separate gain comparison, the native policy and half translation gain each scored **0/8** delayed placements. On four additional starts, executing ten controls followed by a charged 14-control wait gave **8/8** delayed placements, versus **0/8** for default execution on those same starts. Always pacing is therefore a fixed control for evaluating an adaptive selector. [Transfer readout](transfer.json) · [Native branch outcomes](../libero-bowl-decisions-20261118-r1/RESULT.md)

## Training examples available

The pinned release contains 40,470 stored demonstrations across the 32 official training tasks. Its declared-copy filter retains 28,361 records in our 24 fitting tasks and 9,035 in our eight development tasks. This filter removes the copies identified by the publisher. None of the official test task IDs occurs in the manifest. [Pinned dataset](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro2.0-Train/tree/6c4c5524676815a88880a7025badd17fb724f7f4) · [Coverage by task](pairs/coverage.json)

| Partition | Tasks | Tasks with two reset IDs common to BASE and S20 | Matched blocks | Support/query examples |
| --- | ---: | ---: | ---: | ---: |
| Fitting | 24 | 10 | 122 | 244 |
| Development | 8 | 2 | 15 | 30 |
| Existing eight-task fitting pilot (subset of first row) | 8 | 5 | 49 | 98 |

Each block contains one BASE example and one S20 example for the same task. Support and query have different recorded reset IDs. Both arms use the same IDs in the same roles. For each eligible query ID, we pick the preceding common ID cyclically for support, then the first eligible demo by numeric group order. This makes the index reproducible without consulting outcomes. Reuse across blocks creates dependencies. The 122 fitting blocks use six Goal, two Object and two Spatial tasks. No LIBERO-10 task supplies two common IDs. The pilot's coverage gaps are LIBERO-10 tasks 2 and 9 and Object task 6. [Fitting pairs](pairs/train.jsonl) · [Development pairs](pairs/development.jsonl) · [Builder](build_pairs.py)

A recorded reset ID is an index into the start bank. In a follow-up check on the two bowl tasks, **0/24** matched query IDs had identical first recorded observations across BASE and S20. We compared row zero of each indexed query's main and wrist images, end-effector position and quaternion, and gripper positions with exact array equality. The first action also differed in all 24. A later replay established that row zero follows the first action. It measures the effect of the first commands as well as any differences at reset. For action supervision, pair observation `t` with the next command. To measure alternative actions at an identical query state, save and fork that state in the simulator. [Per-reset comparisons](first-observations.json) · [Replay alignment](../libero-s20-motor-20261119-r1/RESULT.md)

## A timing reference on measured robot motion

We verified the release hashes for all four bowl HDF5 files and their collection metadata. Each contained 30 retained successful oracle demonstrations with terminal reward 1. For S20 the cabinet generator recorded 30 successes in 152 attempts and the plate generator recorded 30 in 108. The 200 failures are counts in the metadata. The four demonstration files contain the successful trajectories. Both BASE generators recorded 30 successes in 30 attempts. [Raw-data summaries and per-demo readouts](bowls.json)

Using the first 80 controls, the command offset with the highest correlation to subsequent end-effector displacement was 0 on all 60 BASE demonstrations and 13 on all 60 S20 demonstrations. Using Goal 4's medians to classify Goal 8 gave 60/60 correct. Reversing the tasks also gave 60/60. Each task used the same recorded instruction across the two timings. The offset measures alignment in the stored arrays and includes the observation and command paths. [CPU readout](read_bowls.py)

We then applied those medians to the raw public commands and post-step positions from two previously completed frozen OpenPI panels. The estimates ranged from 0 to 4 under BASE and 8 to 14 under S20. The reference classified 8/8 nominal-screen and 32/32 gain-comparison tapes correctly. This retrospective check uses six starting states and two bowl destinations. The rule sees no rewards, task IDs or perturbation labels at prediction time. It measures transfer of timing recognition from the oracle to the VLA. Success of a history-conditioned policy on held-out tasks is still unmeasured by this experiment. [Transfer code](check_transfer.py) · [Per-attempt evidence](transfer.json)

## Use in the benchmark

Use the indexed demonstrations for a demo-conditioned baseline or for action supervision along successful trajectories. For decision pretraining from the robot's own experience, collect its failed and successful supports under the same condition as the query. Supervise the query on a continuation that the frozen motor can actually execute. A privileged teacher queried at a failed-policy state or repeated native continuations from that state can supply the missing label.

The CPU run used Baige's allocated development instance, CUDA hidden, and zero new GPU-hours or simulator episodes. The index and oracle readout ran at `e84e886e`, and the retrospective transfer at `37628ee3`. The output indices refer to HDF5 groups in the pinned dataset. Supply public observations and causally available feedback to the reader. Keep filenames, suite/task IDs, condition names, seeds, source metadata and query targets in the data loader.

To regenerate the index with Python 3.9 or later, download the pinned release's `manifests/episodes.jsonl` and run the following from the repository root with a new output directory.

```sh
python3 outputs/libero-pro2-demonstration-pairs-20261118-r1/build_pairs.py episodes.jsonl pairs-out
```
