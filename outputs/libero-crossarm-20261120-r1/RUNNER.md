# Native cross-arm pilot

This pilot asks whether 16 measured control transitions from a simulated UR5e improve the next native LIBERO-Pro2.0 task attempt. The reader and residual action adapter train on simulated Panda and Sawyer arms with a common PandaGripper. The frozen JAX OpenPI π0.5 policy supplies every task-directed proposal. The simulator owns the arm kinematics, cameras, object initialization, reward and termination. The learner sees only the issued seven-dimensional command, the measured seven-dimensional motion and the current frozen proposal. [Simulator contract](SIM.md) · [Learner contract](LEARNER.md)

## Fixed panel

The table groups measurements by purpose. Both source training stages finish and the runner saves and reloads the source checkpoint before collecting any UR5e calibration or native admission outcomes. The native admission outcomes constrain interpretation and never select runs or starts. [Completed run](RESULT.md)

| Stage | Bodies and tasks | Released BASE reset IDs | Evidence |
| --- | --- | --- | --- |
| Action calibration | All three bodies on Goal 4 and Goal 5 | 0, 1 | 72 measured control intervals per episode. Two independently permuted signed sweeps excite all six Cartesian components. Each pulse lasts three controls. Gripper excitation is independent. UR5e calibration is privileged oracle data. |
| Calibration check | Each body on both calibration tasks | 9 | A fresh 16-control sequence mixes axes. Each body's forward response must predict observed XYZ and rotation better than constant fitted drift, measured by summed squared error within each group across the two episodes. |
| Native source controls | Panda and Sawyer on Goal 4 | 10 | Unadapted and calibrated π0.5 on identical starts and flow noise. Unmodified Panda and both calibrated controllers must reach native success. The Panda teacher uses the reference arm on both sides of the inverse. |
| Target reachability | UR5e on Goal 1 and Spatial 1 | 10 | The target action oracle must reach native success on at least one admission task. Both tasks enter the fixed exploratory panel. |
| Reader pretraining | Panda and Sawyer on Goal 4 and Goal 5 | 2, 3, 4 | Independent 16-control episodes with mixed commands. Predict a fresh command's physical response using support from a different episode on the same body. |
| Joint fit | Same source bodies and tasks | Practice 5, 6. Frozen proposals 7, 8. | Collect the two practice sets and eight unadapted native rollouts. Both source teachers label every recorded proposal bank. The reader and residual head jointly fit these supervised actions for 400 updates following 300 pretraining updates, seed 41. |
| Held-out comparison | UR5e on Goal 1 and Spatial 1 | Practice/query 11/13 and 12/14 on each task | Four paired query cases × five conditions. Practice uses 16 controls on a fresh scene. The wrong-body donor alternates between Panda and Sawyer using the same task, pose, seed and commanded sequence. |

The reader predicts world TCP translation in metres, world relative rotation in radians and change in common jaw aperture in metres per 20 Hz control. The teacher fits an affine forward Cartesian response to command/observed-motion pairs on separate calibration trajectories, then uses its damped inverse to match the reference Panda response. The teacher passes the common gripper command through. The joint objective is supervised inverse calibration. There are no expert on-body demonstration labels in this run. Task IDs come from the registered train and development pools. The official test tasks stay sealed. [Registered split](../libero-meta-learning-20261117-r1/SPLITS.json)

## Native comparison

The five conditions are unadapted π0.5, the fitted adapter with UR5e support, the same adapter with cleared support, the same adapter with a wrong-body support, and the target's independently calibrated action oracle. Each condition receives a fresh UR5e reset. The runner checks the full same-body reset identity, every public observation hash, instruction, renderer, complete flow-noise bank and opening proposal before comparing actions. Later policy proposals follow each controller's own live observations. Support is collected once per case, held fixed and never consumes query controls. Each target calibration is available only to the oracle and feasibility checks.

The motor uses the prior xemb run's JAX loader and exact observation conversion. It executes five of ten decoded actions per replan, with noise from the episode seed plus 817117. The seed for a case is 20261120 + 10000 × suite index + 100 × task ID + reset ID, with Goal indexed 0 and Spatial indexed 1. The two training tasks share pick-and-place and manipulation skills with the development tasks. This is a four-start feasibility panel for one held-out arm. A failed action oracle bounds what this local calibration and checkpoint achieved on those starts. It does not establish that the task is mechanically impossible.

## Launch and evidence

The coordinator should first qualify the three registered bodies on the pinned Linux simulator, including the named scene placements. [Linux qualification](qualify.py) Use the integrated Git revision in a clean detached worktree. Inside a single exclusive RTX 4090 or RTX PRO 6000D Baige **分布式训练** job, run the following from the worktree root with a fresh attempt name.

```bash
pin=$(git rev-parse HEAD)
bash outputs/libero-crossarm-20261120-r1/run.sh "$pin" pilot pilot01
```

The wrapper checks the deployed project pin, the LIBERO and OpenPI pins, the GPU identity and the installed NVIDIA EGL libraries. It runs the motor in the qualified OpenPI interpreter and launches a persistent simulator subprocess in the LIBERO interpreter. A fresh attempt directory under `/mowenjie/fluffy-bassoon-2/outputs/libero-crossarm-20261120-r1/` receives the logs. The GPU probe runs before the driver. The driver has a 90-minute wall limit. The coordinator owns scheduling and cumulative accounting. [Pilot](pilot.py) · [Transport](runner_native.py) · [Launcher](run.sh)

Read `pilot/status.json` with `exit.txt`. The calibrated-control flag records whether motion prediction and native positive controls passed. The limited fit and four paired evaluations still run if those controls score poorly. Nonzero exits identify runtime, data, or pairing failures. A reset or first-proposal mismatch writes `mismatch.json` before that controller executes. Process termination by the wall limit may leave partial episode files. Every completed episode has `reset.json`, `episode.json` and `tape.npz` with issued controls, measured motion and every native scalar reward. Motor tapes also retain the executed prefix of bounded frozen proposals. The fit records source bank provenance, pairings, teachers, optimization diagnostics and the frozen PyTorch checkpoint. A complete evaluation has four groups in `pairs.json` and paired wins and losses against the unadapted and learned conditions in `status.json`.

## Worker outcome

Implementation began on `af1dd6e00dfd22f7bccafbdda8868ffddd998d80`. On macOS, both runner modules compiled in memory and passed a global-name check. Bash syntax, the simulator reset/step signatures and return fields, learner fit keywords and source-only arguments, the four task memberships, the manifest digest, and source whitespace checks passed. This worker used zero GPU-hours and made no cluster changes or commit. Native calibration, source success, target reachability and held-out scores require an integrated Linux run. [Execution record](EXECUTION.md)
