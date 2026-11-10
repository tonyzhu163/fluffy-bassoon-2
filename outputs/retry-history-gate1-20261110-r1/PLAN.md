# Gate 1: do earlier attempts carry usable information?

Prepared 2026-11-10. Not launched. This is the first gate of the [in-context learner plan](../retry-feedback-plan-20261110-r1/PLAN.md); no training runs until it passes.

## What is collected

Frozen native Human300 π0.5 with the unchanged closed loop (predict 50, execute 5, ten denoising steps, 600 controls), `pretrain` split.

- **8 tasks × 16 scenes × 8 attempts = 1,024 episodes.**
- Scenes are seeds 986000 to 986015. A text search found no earlier use of this range.
- Every attempt on a scene starts from one exact simulator and public-observation start, selected as in the [Nov 10 persistence check](../icrl-idea-checks-20261110-r1/PLAN.md). Only the motor-noise seed differs: scene + 700000 + 1000 × attempt.
- Each episode saves the three camera inputs and state at every inference step, every predicted chunk, every executed action and the native outcome.

| Task | Atomic baseline, successes of 50 |
| --- | ---: |
| OpenDrawer | 34 |
| OpenCabinet | 22 |
| CloseFridge | 27 |
| OpenStandMixerHead | 32 |
| TurnOnSinkFaucet | 28 |
| SlideDishwasherRack | 40 |
| PickPlaceCounterToStove | 29 |
| PickPlaceDrawerToCounter | 24 |

Tasks were chosen from the [atomic reproduction](../robocasa-sft-reproduction-20261109-r1/ATOMIC-RESULT.md) for mid-range success and spread across doors, drawers, appliances and pick-and-place. That baseline used the released evaluator's horizons, so rates at 600 controls will differ. Only OpenDrawer and CloseFridge are confirmed to resolve in the pinned environment; the script checks the task name before loading the policy.

## Readout

**Table ([readout.py](readout.py)).** Per task: single-attempt success, best-of-8 success, scenes that always fail, always succeed or are mixed, and scene repeatability.

**Predictor (CPU, written once the first archives exist).** Within a task, predict whether an attempt succeeds from its first action chunk and the starting state, with and without the other attempts on that scene and their outcomes. Fit on some scenes, score on others.

- The comparison that matters is ranking attempts **within a scene**. Knowing that a scene is hard predicts failure but does not say which action to take.
- Only mixed scenes can show this, so the count of mixed scenes per task is reported first.

## Pass and fail

- **Pass:** on at least two tasks, earlier attempts improve within-scene ranking of held-out attempts on held-out scenes, with a bootstrap interval over scenes that excludes zero.
- **Fail:** no task shows it. Earlier attempts then say how hard a scene is but not what to do differently, and meta-training has nothing to read. Stop before gate 2.
- **Too few mixed scenes** (fewer than about 5 per task on most tasks): the gate is undecided; report that and decide whether more scenes are worth their cost.

With 16 scenes per task this is a coarse screen. A pass justifies gate 2; it is not evidence for the paper.

## Cost

- The Nov 10 persistence check measured 1.90 GPU-hours for 128 episodes. At that rate 1,024 episodes cost about 15 GPU-hours; worst case about 18 if most attempts run the full horizon.
- Storage: about 54 MB of images per full-length episode before compression; plan for 20 to 30 GB. The script stops if less than 60 GiB is free.
- One job per task, one GPU each, 3.5-hour limit per job.

## Launch

On an allocated Baige GPU container, from a clean checkout of the commit containing this folder:

```sh
bash outputs/retry-history-gate1-20261110-r1/run.sh <commit> <fresh-id> OpenDrawer
```

Run OpenDrawer first as the qualification: confirm the GPU probe, matched starts, advancing episodes and a readable first archive, and record the measured seconds and bytes per episode. Then launch the other seven. A job that stops early can be continued under a new id with a later starting scene:

```sh
bash outputs/retry-history-gate1-20261110-r1/run.sh <commit> <fresh-id> OpenCabinet 986008 8 8
```

Record job ids, GPU usage and checks in this folder, and the live row in [coordination.md](../coordination.md).

## Not verified before launch

- The script has only been syntax-checked on macOS; it has not run against the simulator.
- Six of the eight task names are untested in the pinned environment.
- Compressed archive size is an estimate.
