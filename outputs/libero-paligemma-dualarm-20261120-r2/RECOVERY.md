# Goal 5 closed-loop recovery collection

Can an approved native teacher still finish Goal 5 after a frozen PaliGemma motor has moved the arm and plate? Each source rollout supplies branches at 5, 10, 20, 30 and 45 native controls by default. A successful branch contributes its **teacher suffix only**, starting with both camera frames and the hand state at the handoff. Early learner successes, exhausted horizons, failed teachers and replay mismatches contribute zero training labels. The experiment measures recovery yield on training layouts. An autonomous policy gain would need a separate fit and evaluation.

## Linux CLI

From the deployed repository root, set `PIN` to its revision and `PANDA_BANK` and `SAWYER_BANK` to approved native bank directories on the shared volume. Each directory must contain its original `index.json` and the referenced NPZs. The two stages below use fresh launcher attempts and the existing original Panda-trained 3B MIX LoRA checkpoint. The launcher supplies the verified base model, the separate Python environments, EGL and the released Goal 5 HDF5. [Launcher](run.sh) · [Existing checkpoint provenance](../libero-paligemma-crossarm-20261120-r1/RESULT.md)

```bash
set -eu
run=outputs/libero-paligemma-dualarm-20261120-r2
shared=/mowenjie/fluffy-bassoon-2
previous="$shared/outputs/libero-paligemma-crossarm-20261120-r1"
campaign="$shared/$run"

bash "$run/run.sh" "${PIN:?}" recovercap01 vlm recover.py capture \
  --checkpoint "$previous/panda01/motor/checkpoint.pt" \
  --checkpoint-sha256 30355c806c4fa91a1beffae6190b9b62becda1c1e220d46040453332e521921f \
  --bank "${PANDA_BANK:?}" "${SAWYER_BANK:?}" --count 4 --out prefixes

capture="$campaign/recovercap01/prefixes/capture.json"
cap_sha=$(awk '$1 == "capture_sha256" {print $2}' "$campaign/recovercap01/run.log")
bash "$run/run.sh" "$PIN" recoverteach01 sim recover.py collect \
  --capture "$capture" --capture-sha256 "${cap_sha:?}" --out bank
```

For a dual-arm candidate, pass the snapshot file and its published digest. The following shell fragment selects update 400 from the completed fit's candidate index. Substitute the resulting two values in the capture command with a fresh attempt ID. The loader restores the stored LoRA, action head and normalization from `best.model`, verifies the original MIX base files and freezes the entire model. The source-only checkpoint contract may include offline validation on 28–31. Recovery resets and teacher donors are restricted to 0–27. [Candidate writer](motor_train.py)

```bash
step=400
candidate=$(jq -er --argjson s "$step" '.snapshots[] | select(.step == $s) | .file' "${FIT:?}/candidates.json")
candidate_sha=$(jq -er --argjson s "$step" '.snapshots[] | select(.step == $s) | .sha256' "$FIT/candidates.json")
checkpoint="$FIT/$candidate"
```

`--count` accepts 3–5 independent body/layout starts. Automatic selection alternates available arms in ascending layout order and uses the first eligible accepted receipt in the supplied bank order. For a fixed panel across checkpoints, replace `--count 4` with, for example, `--rolls Panda:0 Sawyer:0 Panda:1 Sawyer:1` after checking the bank covers those pairs. Each roll uses its accepted receipt's seed. Multiple handoffs from one roll are paired observations. Set `--prefixes 5 10 25` to shorten a panel. Supported lengths are 5, 10, and any integer from 20 through 45. Each branch shares the original native episode horizon. Both output directories must be fresh.

## What runs at the handoff

1. Capture runs the frozen motor in the policy interpreter. A subprocess under `CROSSARM_SIM_PYTHON` owns the camera-enabled native session. Every five controls the motor receives the current public observations over a pipe. The child verifies the motor's input hashes against the frame it sent and records the full proposal plus the actual float32 controls returned by `ArmEnv.step`.
2. Collection runs in the simulator interpreter. It checks the capture manifest digest, source files, configuration, bank indices, selected positive receipts and their NPZ digests. Every handoff starts with a fresh released reset on the captured body, layout and seed. Reset release and native initial fingerprints must match the approved bank. Replay checks the reset, the opening, and every recorded prefix transition, including the raw released-wrapper state, MuJoCo integration state, public hand state, both images, instruction, issued control, measured motion, reward and terminal flags. Only a fully matching prefix reaches the teacher.
3. A Paddle teacher uses the approved row's exact `params` for that body and layout, including a rendered-command success produced by the local Paddle search. At handoff the controller compares the current low gripper surface, lateral alignment and orientation against the route's approach and push targets. It enters approach, descent or push with fresh timers and integral, then recomputes its command from live geometry each control. Scratch probes used to measure alignment never step the simulator. [Paddle](paddle.py)
4. For Panda rows without a Paddle recipe, the teacher uses a successful official demonstration of the **same training layout**. It checks the pinned archive and donor metadata before the donor reset, then qualifies the official native endpoint in a separate session. The tracker matches the live hand and plate to a pre-action waypoint using position, orientation, jaw opening and plate displacement. It preserves the reset plate offset and advances using native feedback. A missing matched demonstration or a failed native donor replay yields no labels. The target's physics evolves through the issued prefix and teacher commands. [Official tracker](../libero-paligemma-crossarm-20261120-r1/oracle_fallback.py)

## Reading the evidence

| Artifact | What it establishes |
| --- | --- |
| `prefixes/config.json`, `capture.json` | Exact checkpoint, base identity, code and bank hashes, revision, CLI, prefixes and hash of each independent source roll. Capture prints the manifest digest used by collection. |
| `prefixes/learner-<body>-<pose>.json` | Frozen proposals, policy input hashes, issued learner actions, native responses and hash of every requested reachable prefix. |
| `bank/receipts/<body>-<pose>-prefix-<n>.json` | One requested handoff, its source receipt, outcome, and each issued teacher action and resulting native transition. Successful replay records the prefix digest and handoff observation hashes. Teacher decisions include live phase or reference index. |
| `bank/<body>/*.npz` | Positive native terminal suffix in the existing motor tape schema. `steps` counts teacher controls and `total_steps` includes the prefix. Before indexing a tape the collector reads it back and matches each pre-action camera pair and state, actuator command, measured motion, reward and terminal hand state to the native trace. |
| `bank/index.json` | All requested branches, counts of accepted recoveries and usable labels, tape and receipt SHA-256s. A complete zero-yield panel is a result. A native replay mismatch records the first differing step or reset field and produces exit code 2 after the panel. An interrupted collection retains its completed attempt records and running status. |

```bash
bank="$campaign/recoverteach01/bank"
jq -r '.attempts[] | [.body,.pose,.prefix_steps_requested,.outcome,.steps,.terminal_reward] | @tsv' "$bank/index.json"
jq '{status, accepted, labels}' "$bank/index.json"
```

At each reachable handoff the receipt reports mean absolute learner action deviation from the accepted source path. It also reports same-time wrist position error when the source tape has that frame. Use these with the recovery counts by prefix length to see whether positives cover states away from the demonstration. Zero positives despite exact replay counts against using that teacher at those handoffs. Add a completed positive recovery bank to the original approved banks via the existing trainer's `--demos` argument. The original banks supply offline validation. [Motor bank reader](motor_bank.py)

Static verification on macOS covered Python compilation, Ruff, local import names and documentation links. All three shell examples passed `bash -n`. Native replay parity, recovery yield and downstream motor performance remain unmeasured.
