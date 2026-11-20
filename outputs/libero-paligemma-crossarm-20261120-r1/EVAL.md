# Native Goal 5 evaluation

**Implementation status.** The source-only learner and native evaluation driver are implemented. Static checks passed on the submitted driver. Native motor success, reader training loss, UR5e calibration wins and IIWA success are all unmeasured. No cluster run was launched for this implementation. [Run question](PLAN.md) · [Driver](adapt_eval.py)

## Pilot and decision rules

The task is BASE `libero_goal/5`, push the plate to the front of the stove. Each number below is a released pose ID. Each simulator reset uses seed `20261120 + pose`. Source fit and model selection must stay below pose 30. The driver reads both splits and the arm names from the motor checkpoint before source qualification.

| Stage | Bodies and poses | Evidence and gate |
| --- | --- | --- |
| `qualify` | Raw Panda and Sawyer, each on 30 and 31 | Four independent native episodes. Each arm must succeed at least once on its own two starts. A zero on either arm stops the pipeline before reader fitting. |
| `fit` | Panda and Sawyer only. Reader support 20, 21. Motion prediction queries 22, 23. Joint demonstration queries use successful common poses from 24–27. | Save both training curves, cached motor proposals with demonstrated labels, and the frozen reader/adapter checkpoint. |
| `dev` | UR5e. Analytic calibration 32, 33, independent response check 34. Practice 35 and 36 precede fresh queries 37 and 38 respectively. | Run five conditions on each query. Open the reserved arm only if privileged calibration succeeds where raw fails on **both** distinct starts. |
| `reserved` | IIWA. Analytic calibration 39, 40, independent response check 41. Practice 42 and 43 precede fresh queries 44 and 45 respectively. | Verify the source checkpoint and the passing UR5e report before constructing any IIWA session. Run the same five conditions and report the held-out paired outcomes. |

The repeatability rule is two native rescues on different poses. A tie on a raw success, a raw ceiling, or a single rescue keeps IIWA sealed. The complete UR5e comparison survives that stop, including any learned-controller losses. The motor gate establishes feasibility on both sources. A 1/2 result remains visible and gives a weak estimate of source reliability.

## Linux commands

Use Bash from the repository root in the qualified motor/simulator environment. Supply the actual absolute paths in `MOTOR_PYTHON` (the motor environment's Python), `MOTOR_DIR` (containing the finished source `checkpoint.pt`), `VLM_WEIGHTS` (the original PaliGemma snapshot), `DEMO_DIR` (the successful source demonstration root), `PILOT_DIR` (a fresh evidence location) and `CODE_REVISION` (the deployed revision). When the simulator uses a different interpreter, export its path in `CROSSARM_SIM_PYTHON` and use the simulator environment from the launcher. These commands use the saved motor's selected weights and successful common demonstration poses from 24–27.

```bash
set -euo pipefail
driver=outputs/libero-paligemma-crossarm-20261120-r1/adapt_eval.py
common=(--motor-output "${MOTOR_DIR:?}" --weights "${VLM_WEIGHTS:?}" --revision "${CODE_REVISION:?}")
"${MOTOR_PYTHON:?}" -Bu "$driver" qualify "${common[@]}" --out "${PILOT_DIR:?}/qualify"
"$MOTOR_PYTHON" -Bu "$driver" fit "${common[@]}" --qualification "$PILOT_DIR/qualify" \
  --demos "${DEMO_DIR:?}" --demo-poses 24,25,26,27 --pretrain-steps 100 --joint-steps 150 \
  --batch-size 32 --out "$PILOT_DIR/fit"
"$MOTOR_PYTHON" -Bu "$driver" dev "${common[@]}" --fit "$PILOT_DIR/fit" --out "$PILOT_DIR/dev"
"$MOTOR_PYTHON" -Bu "$driver" reserved "${common[@]}" --fit "$PILOT_DIR/fit" \
  --dev "$PILOT_DIR/dev" --out "$PILOT_DIR/reserved"
```

Each stage creates a fresh directory. Exit 0 means the stage completed and passed its applicable gate. Exit 2 records a scientific gate stop. Exit 1 records an execution or evidence error. A stage writes `report.json` with its status and reason. Bash stops the sequence on either nonzero exit. The CLI independently enforces gate ordering when invoked one command at a time. A running report with partial episode tapes is incomplete evidence.

To replay the UR5e panel, use the same environment, frozen motor and fit directory. This reloads the reader and regenerates practice and closed-loop queries on their original seeds. A reserved replay also requires the passing development report.

```bash
"$MOTOR_PYTHON" -Bu "$driver" dev "${common[@]}" --fit "$PILOT_DIR/fit" --out "$PILOT_DIR/dev-replay"
```

The stage identities bind the motor checkpoint, base configuration and implementation files by SHA-256. The motor verifies the original VLM files when it loads. Qualification records the original model and revision, selected source motor update, train/validation pose IDs and source successes. Later stages record the parent report and reader checkpoint digests, exact command, deployed revision and library versions. Immutable input artifacts and fresh output directories allow replays to be compared without overwriting the first run.

## Source fitting

Each support contains 16 issued seven-dimensional commands and their measured six-axis Cartesian motion and jaw displacement. Probe commands depend only on the pose seed. A 16-dimensional reader pools features of commands, source-normalized motions and all signed command/motion products. For pretraining it sees one support episode and predicts the response to a new command on a **different pose and reset** of the same source arm. Motion scales use source supports only. Training samples a source arm uniformly, then a support, a query episode and a transition. The two stages use seed 41, Adam with learning rate 0.003 and gradient clipping at norm 5.

Joint fitting reads only `Panda/<pose>.npz` and `Sawyer/<pose>.npz`. The supplied demonstration must have a positive final native reward, zero earlier rewards, finite bounded controls and aligned observations. The driver checks body, pose and native success when stored and requires the qualified instruction. The motor sees the demonstration's pre-action external image, wrist image, eight-dimensional state and instruction. It proposes an action chunk once per execution prefix. At most five consecutive raw actions are aligned with the demonstrated actions from the same time indices, including the final partial prefix. Those fixed pairs are cached before any joint update. The motor remains frozen.

The reader and a zero-initialized, context-conditioned affine residual learn all seven demonstrated command coordinates by squared error. On 1/8 of sampled joint examples the context is cleared, so the cleared controller can learn an unconditional correction. Source training uses unsaturated predictions to keep gradients available. Native execution clips the adapted command to the simulator bounds. The checkpoint includes both heads, the reader and source motion scales. No learned parameter, optimizer or normalization statistic uses development or reserved-arm observations.

The launcher collects fresh successful source demonstrations on poses 24–27 for joint fitting. Motor training and validation use only poses 0–15. Independent evidence for the combined policy comes from the fresh native poses in the table.

## Native conditions and privileged diagnostic

Each target query has its own 16-transition practice and a command-matched wrong-arm practice on the same layout. The first donor is Panda and the second is Sawyer. Practice and query use separate sessions, poses and seeds. The query never appends transitions to practice. Every condition starts with its own native reset, uses the same frozen motor and replans from its own resulting observations.

| Condition | Command executed |
| --- | --- |
| Raw | Frozen motor's proposal on the query arm. |
| Correct | Residual conditioned on that query arm's separate practice. |
| Cleared | Same learned adapter with zero reader context. |
| Wrong | Same adapter with the donor's measured responses to the identical probe commands. |
| Privileged | Analytic correction from independently measured local responses of raw Panda and the query arm. The proposed gripper command passes through. |

For the privileged control, both arms receive the same signed single-axis pulses at amplitude 0.4 and four zero Cartesian commands per calibration reset. The jaw stays open. Symmetric response differences give each row of the six-by-six local gain. The mean zero-command motion gives drift. For a raw proposal, the desired motion is the independently measured **Panda response to that unmodified proposal**. A damped linear inverse on the query arm converts the motion discrepancy to a Cartesian action correction, scaled per output coordinate by the Panda response norm. Damping is fixed at 0.01. These closed-form measurements belong only to the privileged diagnostic. No query reward chooses coefficients, poses or damping. A separate probe compares per-axis prediction RMSE with constant drift and records the measured rank. Only paired native success opens the reserved gate.

Motor planning uses the same seed `900000 + 1000 * pose + elapsed_controls` for Python, NumPy and PyTorch in each condition. Before issuing the first command, the driver checks both image hashes, public state, prompt, native horizon, available simulator fingerprints and the full opening raw proposal against the raw episode. A mismatch saves the discrepancy and invalidates that panel. The complete evaluation runs under PyTorch inference mode with frozen weights. Each rollout ends at native success, termination or the released horizon (300 by default). The native success flag must agree with a positive native reward.

## Reading outcomes

`report.json` lists each condition's native successes, paired wins and losses against raw, and correct-history wins and losses against cleared and wrong history. Per-episode records include opening proposals and issued actions, native return, terminal reward and control count. The referenced NPZ tapes retain each executed proposal, action, measured motion, native reward, success and done flag. Practice tapes retain the actual 16 transitions. The fit report links each accepted demonstration hash to its cached proposal/label bank and gives the losses on a fixed sampled training batch before and after each stage.

A correct-history gain over raw with comparable gains from cleared history is compatible with generic source imitation. Gains over raw, cleared and wrong on the same poses support a contribution from measured practice. A privileged win establishes an opportunity for local response correction. It does not establish that the reader learned that correction. The two-pose panel cannot estimate population-level success precisely. Local free-space response and a first-order inverse may describe contact motion poorly. Joint labels come from expert-state trajectories, while live states depend on previously issued commands.

## Static verification

From the run directory, the following checks parse and compile the driver in memory. They do not import the motor, simulator or learner.

```bash
python3 -I - <<'PY'
import ast
from pathlib import Path
p = Path("adapt_eval.py")
source = p.read_text()
ast.parse(source, filename=str(p))
compile(source, str(p), "exec")
print("AST and bytecode compilation passed")
PY
```

Static inspection also verified the driver against the implemented `Motor` and `Session` method signatures, the disjoint hard-coded pose groups, all five control names and the inference-mode decorators. The documented Bash blocks passed `bash -n` and their local links resolved.
