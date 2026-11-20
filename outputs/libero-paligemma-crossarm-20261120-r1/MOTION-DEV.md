# Source-dynamics joint reader on native UR5e Goal 5

Does a frozen reader trained on Panda and Sawyer dynamics use a new arm’s practice to help a frozen Panda-only PaliGemma motor push the plate to the front of the stove? The outcome is native LIBERO Goal 5 task success. A paired loss or tie against cleared and wrong histories counts against a benefit from identifying the arm on that start. The registered two-arm successful-demonstration experiment remains stopped. IIWA stays sealed for every outcome.

## Linux invocation and input evidence

Use the deployed run directory and the qualified motor and simulator interpreters. Supply immutable, completed inputs (including the trainer’s `motor.lock` in the motor directory) and a fresh output directory. The parent launcher supplies the deployed revision and runtime environment, including `CROSSARM_SIM_PYTHON` for a separate simulator interpreter. [Evaluator](motion_dev.py) · [Shared rollout code](adapt_eval.py)

```bash
python motion_dev.py --motor-output "$MOTOR_DIR" --weights "$VLM_WEIGHTS" \
  --fit "$MOTION_JOINT_OUT" --out "$MOTION_DEV_OUT" --revision "$CODE_REVISION"
```

The fit directory supplies `reader-adapter.pt` in the version-1 `adapt_eval.load_reader` format and the producer’s `report.json`. The producer must name the bodies that supplied reader training examples. Simulated UR5e is excluded from policy training and selection. The following fields bind the completed fit to the exact reader and motor served here.

| Fit report field | Value |
| --- | --- |
| `stage` | `source_motion_joint_diagnostic` |
| `status` | `complete` |
| `config.bodies` | Exactly `["Panda", "Sawyer"]`, in either order |
| `checkpoint_sha256` | SHA-256 of the final serialized `$MOTION_JOINT_OUT/reader-adapter.pt` |
| `motor_sha256` | SHA-256 of the actual frozen `$MOTOR_DIR/checkpoint.pt` used during joint training |

The evaluator also checks the task ID, proposal receipts and every update’s per-arm motion-loss keys against the producer’s declared sources. The producer exports `motor_sha256` from the audited motor checkpoint digest and checks the same file again after joint training.

Before constructing any simulator session, the evaluator verifies both file hashes and audits the motor’s saved training and validation episode records, validation anchors, pose split and instruction. The motor loader verifies the original PaliGemma snapshot and restores the validation-selected update. Both modules must have gradients disabled and be in evaluation mode. Target practice supplies transient context to this fixed policy. Target pulse responses supply only the privileged comparator.

## Paired native episodes

| UR5e task pose | UR5e practice | Command-matched wrong practice | Reset seed for task |
| --- | --- | --- | --- |
| 37 | 35 | Panda 35 | 20261157 |
| 38 | 36 | Sawyer 36 | 20261158 |

Each practice supplies exactly 16 transitions from a separate reset. For the privileged comparison, the evaluator applies signed Cartesian pulses on 32 and 33 to Panda and UR5e, checks each measured response on 34, and uses the existing damped Panda-to-UR5e correction. On each task pose, raw executes the unchanged motor proposal, correct uses UR5e practice, cleared uses a zero embedding, wrong uses the donor’s motion, and privileged applies the analytic calibration.

All five conditions reset independently with the same pose and seed. Before issuing the first action, the shared rollout compares the raw opening observation, instruction, simulator fingerprints, horizon and entire raw action chunk with the raw baseline for that pose. The first issued action may differ once the condition applies its correction. An opening mismatch saves the discrepant openings and fails the evaluation.

## Evidence and interpretation

A complete `report.json` contains ten task episodes. `native_successes` gives counts out of two, and `terminal_rewards` lists the final native rewards in pose order 37, 38. `versus_raw` lists the poses won and lost by each alternative. `correct_versus` compares correct practice with cleared and wrong practice on the same poses. The report records each opening, first issued action, native return, success, step count and tape path. The NPZ files preserve practice transitions and closed-loop proposals, actions, measured motion, rewards and termination flags. A partial report retains finished episodes. Only `status=complete` denotes a scored panel, including all-zero panels.

A correct-only rescue supports a practice-dependent benefit on that pose. A rescue shared with cleared history also admits an unconditional motor correction. A privileged-only rescue shows an opportunity for response calibration that the learned policy missed. Two starts provide a diagnostic comparison with limited precision.

## Implementation evidence

Verification on macOS was limited to static checks of syntax, shared call signatures, panel IDs, command-line arguments and the documented shell invocation. The native task metrics require execution on allocated Linux hardware. This implementation produced no rollout scores.
