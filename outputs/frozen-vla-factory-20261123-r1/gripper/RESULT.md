# The shared PandaGripper accumulates command sign. Measured motion predicts the policy tapes best.

**Keep gripper polarity as the command adapter for this family.** A bounded actuator state matters on deliberate reversals. On the recorded π₀.₅ manipulation tapes, a single fitted coefficient multiplying the last public finger displacement gave the smallest mean forward error. These are jaw-width forecasts, with zero motor updates and zero new native task-success evaluations.

## Held-out policy trajectories

Each of eight folds held out an entire LIBERO development task, including both friction settings and both frozen π₀.₅ tapes (default and paced, reset 1). There were 32 native replay episodes and 7,322 control transitions. Each fit used 14 ordinary-material episodes from the other seven tasks (2,862–3,368 controls). At test time the model receives the issued gripper sequence and public `q_left - q_right`. A five-control forecast starts with the current width and previous displacement. It receives no finger observations inside that block. The command accumulator also retains signs since reset. No task or arm ID, wire label, contact flag or reward enters a fit. [Source tapes and provenance](../../libero-xemb-friction-dev-20261119-r1/RESULT.md) · [Plan](PLAN.md)

| Width forecast | One-control MAE (mm) | Five-control endpoint MAE (mm) | First control after reversal (mm) |
|---|---:|---:|---:|
| Persist current width | 0.424 | 2.074 | 0.571 |
| Extrapolate last observed displacement, command-free | **0.158** | **1.614** | 0.560 |
| Signed gain times command | 0.690 | 3.372 | **0.311** |
| Direction-specific linear response to current width | 0.649 | 2.946 | 1.359 |
| Same map scaled by analog magnitude | 0.649 | 2.946 | 1.358 |
| Separate opening and closing dead zones | 0.649 | 2.946 | 1.358 |
| Direction, width and last displacement | 0.245 | 2.582 | 0.461 |
| Add a bounded sign accumulator | 0.183 | 1.646 | 1.412 |

The accumulator integrates sign into a desired width and uses the gap to observed width as one additional predictor. All eight training folds selected a step of 1/8 of source width span per control. Against the sign-and-width map it reduced one-control error by 0.467 mm (95% paired task interval 0.312–0.610). At five controls the improvement over persistence was 0.428 mm (−0.201 to 1.111). The command-free velocity control, added as an ablation, beat the accumulator by 0.025 mm (0.006–0.044) at one control and 0.032 mm (−0.294 to 0.364) at five. At the first control after a switch the accumulator did worse than sign plus last displacement by 0.951 mm (0.717–1.194). There were 54 sign switches across paired branches and seven tasks with any switch. A reversal can occur while the jaw is nearly stationary. Transition errors above use equal episode means within task and equal task means. The percentile intervals resample paired task means 20,000 times, seed 20261123, conditional on the fitted folds. [Task differences, support and cases](metrics.json)

## Command magnitude and memory

The archive contains no issued magnitude below 0.990. Asymmetric thresholds on this support change five-control error by less than 0.001 mm versus the sign-and-width map. The direct component intervention held the sign schedule and starting state fixed, then issued close/open/close pulses at magnitudes 0.0001, 0.01, 0.1, 0.25, 0.5 and 1. Every nonzero magnitude produced exactly the same width at every control. A zero command from identical midrange positions and zero velocities moved +2.758 mm after an opening drive state and −2.758 mm after a closing drive state. The fork changes only the stored controller drive. Robosuite 1.4.0's `PandaGripper.format_action` adds a signed step to a clipped actuator target at each 2 ms substep. This identifies path dependence in the command handler. The experiment does not identify mechanical backlash in another jaw. [Component receipt](component.json)

## Transfer and failure modes

- **Free-space reversal.** On independent 15-control BASE probes (Goal 1, layout 13) for UR5e, IIWA and Kinova3, the held-out learned accumulator had 2.04 mm one-control error but 20.09–20.11 mm five-control endpoint error. The sign-and-width map gave 4.52–4.53 and 10.31–10.32 mm respectively. A white-box component prior with the released gripper XML, actuator limits and sign formatter gave 0.78 and 2.33–2.34 mm. The three arms share the jaw model, task, layout and pulse sequence. They supply one matched scenario. [BASE receipts](../../xemb-stage2c-cpu-20261121-r1/RESULT.md)
- **Fixture and contact.** Compiling the hand straight against world generated spurious hand/finger contacts. After inspecting those contacts, we used a separate hand body welded to a stationary mount and retained both results. The free-space prior's five-control error was 10.97 mm on the policy tapes, rising to 28.83 and 29.51 mm on the two object tasks. The fixture has no grasped object. Its favorable pulse result is an exploratory mechanism check with known hardware. [Fixture contacts and per-task errors](component.json)
- **Frozen π₀ and π₀.₅.** Four real policy prefixes pair Panda and UR5e on the same Goal 1 start. After one observed five-control block, the next four five-control endpoints per cell had 0.053–0.090 mm error with memory and 0.250–0.279 mm with persistence. All 100 issued controls open an already open jaw. This comparison cannot rank closure models. [Matched motor prefixes](../../xemb-pi0-source-smoke-20261122-r1/evidence/result.json)
- **Polarity.** On three native Panda replay tapes with a synthetic command transform, two opposite five-control support pulses predicted the direction of the third pulse on 3/3, including the inverted wire. They share one reset. Independently, archived SymMeta grip predictions scored 96/96 on each of the four frozen-policy arm panels when joined to the injected signs. Those compressed case records provide a polarity reference only. [Native probe](../../libero-xemb-joint-sysid-20261119-r1/RESULT.md) · [Four-panel count](metrics.json)

For a frozen-VLA adapter, use a short bidirectional probe to establish polarity on this jaw. Retain a recent measured finger displacement for passive tracking. A bounded actuator target is a candidate for active pulse prediction when the mechanism is known. The next discriminating transfer test would pair equivalent reversals and contact loading on a second gripper mechanism. One-step prediction on a common jaw cannot establish improved closed-loop grasp success.

## CPU job receipt and reproduction

Local Linux CPU, CPython 3.10, NumPy 1.26.4, MuJoCo 3.2.3, robosuite 1.4.0, 0 GPUh. Both scripts exited 0. Input revision `757a44cd4374833cad790852df8b87ddf412e638`. The source π₀.₅ replays used `c45aa965` frozen tapes in LIBERO-Pro2.0 `e5500c69`. The paired π₀/π₀.₅ prefixes came from `job-ev9ie3w3ki3q`, wrapper `565af610` and dependency `437da7f6`. The wheel SHA-256 matched the cross-arm qualification pin, `aba065e7b36745738cede259457b2cb349427f3608728d867ef3a2034cb62994`. The component script checks the gripper XML digest.

From the repository root with this lane applied, run

```sh
uv venv --python 3.10 /tmp/gripper-env
uv pip install --python /tmp/gripper-env/bin/python numpy==1.26.4 mujoco==3.2.3 robosuite==1.4.0 termcolor==2.5.0
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
lane=outputs/frozen-vla-factory-20261123-r1/gripper
/tmp/gripper-env/bin/python "$lane/screen.py"
/tmp/gripper-env/bin/python "$lane/component.py"
```
