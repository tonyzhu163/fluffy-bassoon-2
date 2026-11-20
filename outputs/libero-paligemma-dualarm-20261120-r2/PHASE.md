# Source Paddle skill distillation

Can a PaliGemma policy reproduce a particular successful push when it learns where the plate is, which part of the maneuver it has reached and what wrist error still needs correcting? The first experiment uses the exact primary route on Sawyer, which has the most qualifying camera receipts. A second fit can use a single straight-flank route shared by Panda and Sawyer. Each checkpoint trains on one route. The policy receives neither route nor arm identity at deployment.

Fit on released Goal 5 BASE layouts **0–23**. Reserve **24–27** for offline checks and native rollouts. A positive needs a camera tape, native terminal success and positive terminal reward. The loader checks released reset provenance, arm, seed, instruction, bounded actions, quaternion norms and final reward, stages cameras on disk and deduplicates trajectories. Exact labels and the ordered public observations share a tape digest for dense capture. Generated layouts and official Panda replay variants are outside this experiment. [Data reader](phase_data.py)

## Available inputs

These counts come from accepted camera receipts in the four local summaries, before deduplication. An entry is **episodes / distinct layouts**. The route filter uses the full parameter tuple. Both routes use rim contact, tilt 28°, yaw 180° and bottom height 0.006 m.

| Route | Arm | Fit 0–23 | Holdout 24–27 | Issued controls, fit / holdout |
|---|---|---:|---:|---:|
| Primary, flank −35° | Sawyer | 20 / 19 | 2 / 1 | 2,033 / 165 |
| Primary, flank −35° | Panda | 0 / 0 | 0 / 0 | 0 / 0 |
| Straight, flank 0° | Sawyer | 12 / 9 | 0 / 0 | 1,363 / 0 |
| Straight, flank 0° | Panda | 3 / 3 | 3 / 2 | 232 / 213 |

The primary Sawyer holdout receipts both come from layout 26. The twelve straight Sawyer fit receipts combine seven from sawyer01 and five from sawyer02. Panda01 supplied released demonstrations. The headless search hit contributes zero camera tapes. The local run tree contains **zero corresponding NPZs and zero dense captures**. Actual unique inputs, phase coverage and positive yields on another capture are measured by the inventory CLI. [Panda01](evidence/panda01/summary.json) · [Panda02](evidence/panda02/summary.json) · [Sawyer01](evidence/sawyer01/summary.json) · [Sawyer02](evidence/sawyer02/summary.json)

## What the policy learns

The two RGB views and instruction pass through the verified general PaliGemma-3B MIX 224 backbone. Source gradients update rank-eight query/value LoRA in Gemma layers 16 and 17, the visual plate localizer, the three phase readouts and the action decoder. The fit stops if either the adapter group or the decoder has no finite nonzero first-step gradient. It also verifies that the zero-initialized adapter up-projections and action-readout weights changed after the optimizer step. The checkpoint contains the actual learned delta, normalization, optimizer and RNG. [Model](phase_policy.py) · [Fit](phase_fit.py)

The head predicts plate world XY from spatial image tokens. It combines the plate relative to the observed hand, image features, normalized public XYZ/XYZW/jaw and elapsed executed controls. Five learned slots score approach, descent and push. Each phase predicts plate-relative target XY and world Z, a desired hand orientation, a translational force correction and a jaw command. Translation uses target minus observed wrist, learned per-phase gain and the force correction. Rotation uses the shortest world rotation vector from the observed **hand-link** quaternion to the predicted hand orientation. Native OSC limits bound the seven outputs. Slot zero predicts the current error. Later slots approximate the future corrective errors as seen from this anchor, using supervision from subsequent ticks. Stride one uses a fresh camera pair for every action.

Each phase starts with the median demonstrated target offset. If legacy sampling missed the descent geometry, known-phase wrist positions and issued translations initialize it. Dense camera capture supplies pre-action plate XY, Paddle's 0.1 mm-resolution position error, pre-switch phase and actual switch at every tick. The desired hand quaternion uses the teacher's orientation and the measured rigid grip-site/hand transform. Sparse banks supply plate/error only at recorded ticks. Equal neighboring phase observations certify the intervening phase through ordering. A changed phase leaves the boundary interval partially unknown. For legacy orientation supervision the observed wrist at the smallest logged angular error approximates the route orientation. Saved headless commands can differ from the live shadow teacher on a replay. The fit uses the actual issued commands for imitation. A known switch tick combines old-phase position error with new-phase gains in Paddle, so it supplies geometry and phase supervision only. [Teacher](paddle.py) · [Dense capture](phase_capture.py) · [Input alignment](phase_data.py)

The sampler balances arms, layouts and tapes, then oversamples physical labels and dense switch neighborhoods. No future observations, trace, masks, geometry or demonstration retrieval enter the deployed decoder. All metric targets and initializers come from fit episodes. The fixed update count prevents selecting weights on the four evaluation layouts.

## CLI and dependencies

Run from this directory with the existing Linux VLM and simulator environments. Set `VLM_PY` to the qualified PyTorch interpreter, `SIM_PY` to the native LIBERO interpreter, `WEIGHTS` to the local verified MIX snapshot, `BANK` to the **original** sawyer02 bank containing index.json and tapes, and `WORK` to writable project NVMe. The simulator also uses the existing release/config, EGL device and library environment. The [existing launcher](run.sh) documents the qualified interpreter, environment paths and GPU probe. The commands below call the new files directly.

```sh
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export CROSSARM_SIM_PYTHON="$SIM_PY"
mkdir -p "$WORK/scratch"
"$VLM_PY" -B phase_data.py --banks "$BANK" --route-scope primary --scratch "$WORK/scratch"

# Optional online teacher capture in the simulator interpreter.
"$SIM_PY" -B phase_capture.py --route primary --bodies Sawyer --poses 0-27 \
    --out "$WORK/dense-primary"

"$VLM_PY" -Bu phase_fit.py fit --route primary --bodies Sawyer \
    --banks "$BANK" "$WORK/dense-primary" --weights "$WEIGHTS" \
    --output "$WORK/fit" --scratch "$WORK/scratch" --steps 1600 --microbatch 4

"$VLM_PY" -Bu phase_fit.py check --banks "$BANK" "$WORK/dense-primary" \
    --weights "$WEIGHTS" --checkpoint "$WORK/fit/checkpoint.pt" \
    --output "$WORK/check" --scratch "$WORK/scratch"

"$VLM_PY" -Bu phase_fit.py native --weights "$WEIGHTS" \
    --checkpoint "$WORK/fit/checkpoint.pt" --stride 1 --output "$WORK/native"
```

For a legacy-only fit and check, omit the dense bank from both commands. For the two-arm straight fit use `--route straight --bodies Panda Sawyer`, the original panda02, sawyer01 and sawyer02 banks, and a separate straight dense capture. Capture all four holdout layouts regardless of their native teacher outcomes. A completed capture reports each arm's positive count, including zeros. The fit requires positives for every requested training arm. A thin or absent positive holdout for an arm limits the offline check. Every trained arm still gets all four starts in the native command.

The fit needs one exclusive BF16-capable Pro6K, Python, NumPy, SciPy, PyTorch, the existing Transformers PaliGemma implementation, tokenizer and safetensors, plus sibling R1 motor modules. MIX revision is `d1d8734c9c3ad0ccfeea4afc270faa356c2ba515`. Capture/native additionally need the pinned LIBERO-Pro2 release `e5500c69`, robosuite 1.4.0, MuJoCo 3.2.3 and NVIDIA EGL. Allow 0.375 MiB scratch per loaded control for both uint8 images. The global batch is eight, with microbatch 1, 2, 4 or 8. The fit logs update time, allocated bytes, weighted losses and first-step gradients. It replaces the checkpoint atomically. Use the same fit command with `--resume` to continue. Checkpoints bind base identity, source hashes, input indices and actual episode/label digests.

## Source controls and decision

The offline report compares matched RGB with both cameras replaced by a different fitting layout of the same arm at the same elapsed tick. Each condition advances its own causal phase memory. Reports include plate and wrist-target errors in metres, phase accuracy on identifiable ticks, predicted entry times against teacher intervals, and normalized translation/rotation/jaw errors. A forced wrong-phase readout, constant plate predictor, training-mean action and timed donor replay help identify whether visual localization, phase selection or a nearly fixed action sequence explains the scores. Averages give each represented layout equal weight within arm. [Controls and native runner](phase_check.py)

Prespecify the last update and stride one. **Three or four native successes on the four held layouts per claimed arm** motivates a larger independent source experiment. **Two or fewer counts against robust skill reproduction on this screen.** Comparable matched and swapped-camera localization/action errors or comparable correct and wrong-phase translations count against the proposed mechanism even with successful rollouts. Good offline metrics followed by native failures implicate on-policy state drift, timing or contact force. A small positive-only offline cohort and four native starts cannot establish a population success rate. Earlier mixed SFT used these layouts for training, so a numerical improvement claim requires a comparator refit on the same 0–23 partition.

## Source large_panel integration

Construct `PhaseMotor.load(checkpoint_path, weights)` in the model process. Create one `Memory()` per simulator **after reset**. Pass unchanged `{rgb, wrist, state, prompt}` dictionaries and the matching memories to `model.plan(observations, memories)`. It returns bounded `[B,5,7]` commands and `[B,5]` proposed phases. Once a prefix executes, call `memory.advance(phases[i].tolist(), len(trace))` for that simulator. Discard the memory on termination. Keep memory keyed by simulator, since active batch positions change. For fixed batch padding pad observations and memories together, then discard padded proposals. Planning never mutates memory. The ordered decoder advances at most one phase per command and asks adjacent prediction slots to support an advance. Direct `model(observations)` provides a trained clock-free variant. The native CLI exercises the stateful integration on the permitted released layouts. The existing large_panel registration and acceptance protocol would need a separate architecture entry before that driver could score this checkpoint.

The implementation started from local revision `e2f794c41c9141a7aadaf67339da919336362f22` with concurrent working-tree edits. The fit records its own source hashes. Local verification covered the executable CLI parsers, source receipt counts, source-interface review and in-memory compilation. No training, online capture or native evaluation ran for this implementation. Native success and fitted controller quality remain unmeasured.
