# Per-dimension drift diagnostic for the 10M Flow-SDE sources

**Status 2026-11-04: written, statically checked, not launched.** Needs a Baige
owner. Reads only retained artifacts of
[`robocasa-flow-sde-10m-per-task-20261103-r1`](../robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md).
No simulator controls, sampling or SGD.

## Question

The fixed-B16 likelihood replay
([table](../robocasa-flow-sde-10m-per-task-20261103-r1/result-tables-r1/initial-policy-log-ratios.csv))
shows that PickPlaceDrawerToCounter moves away from the initial policy far faster
than the other four tasks. Its TRAIN first-success rate stays flat at ~40%.

| Task | step 1 | step 10 | step 130 |
| --- | ---: | ---: | ---: |
| OpenDrawer | +0.04 | +0.06 | −0.34 |
| CloseToasterOvenDoor | +0.08 | −0.01 | −2.58 |
| PickPlaceDrawerToCounter | **−2.73** | **−10.5** | **−137.5** |
| CoffeeSetupMug | +0.02 | +0.01 | −0.77 |
| TurnOffStove | +0.00 | −0.11 | −1.45 |

The task's iteration-1 approx KL was 0.274, against 0.11–0.20 for the others.
Gradient norms (35–65) always exceed `clip_grad` 1.0, so every task takes
equal-norm steps. A 100× larger drift therefore means a persistently aligned
gradient. Where does that drift live, and does it change behaviour?

## Hypotheses

- **H1, binarized dimensions.** The env binarizes gripper (index 6) and base
  mode (index 11). PPO still scores the continuous pre-binarization values. Means
  can move far out without changing executed actions. Predicts the drift
  concentrates in dims 6/11.
- **H2, arm dimensions.** Real motion change that does not raise success.
  Predicts drift spread over dims 0–5 (or the base/torso dims).
- **H3, cohort artefact.** One saved cohort happens to be sensitive. Weak, because
  all 16 samples drift (max −55 at step 130). H1/H2 split this further.

CoffeeSetupMug is the key control: it is also pick-and-place and uses the
gripper, but did not drift. If H1 holds for Drawer→Counter, the per-task
difference still needs explaining. Compare its dim-6 share and the cohort's
dim-6 values (`cohort.action`, `cohort.model_action`).

## What runs

[`DIMENSION-DRIFT.py`](DIMENSION-DRIFT.py) imports the source run's
helpers unchanged and does three things.

1. `metrics` (CPU, any Linux): joins every retained
   `<task>/*/updates/iteration-*-rank-*.json` into `metrics/<task>.csv`. Each row
   has approx KL, clip fraction, ratio, grad norm, explained variance and value
   loss for all 130 updates. Only iteration 1 is in Git today.
2. `stage` (dev CPU with the BOS mount): the same one-private-model-at-a-time
   bridge as R3, walking tasks in `ORDER`.
3. `run` (one GPU, [`COMMAND.sh`](COMMAND.sh)): the same 16 checkpoints × 5
   tasks and the same fixed R2 fixtures as R3. It saves the full
   `[16, 50, 12]` log-ratio tensor per checkpoint, the per-dimension,
   per-timestep and per-sample sums, the binarized-dimension share, and the
   cohort's action statistics. Each row checks that it reproduces the R3 mean
   within 1%. Order: Drawer→Counter, CoffeeSetupMug, TurnOffStove, OpenDrawer,
   Toaster.

Budget: R3 used 4,285 GPU-s for the same 80 forwards. Ceiling one GPU ×
180 minutes (3 GPU-h), no automatic retry. This is under the 8 GPU-hour ask
threshold.

## Decision rules

- **dims 6+11 ≥ 70% of the step-130 drift on Drawer→Counter:** H1. Propose a
  recipe change: mask the binarized dims in the PPO ratio, or score them after
  binarization. Requalify on Drawer→Counter before any rerun. Do not swap the
  task.
- **arm dims ≥ 70%:** H2. Inspect the per-update clip fraction and KL for a
  regime change, and look at rollouts from late checkpoints. Swapping the task
  becomes reasonable if the drift is task-specific (CoffeeSetupMug stays small).
- **Neither:** report the split as-is and do not change the recipe.

TurnOffStove needs no GPU here. Its drift is in the normal range. The
`metrics` join will show whether its clip fraction/KL ever separate from the
improving tasks.

## Evidence to record here

`metrics/METRICS.json` and the CSVs; `gpu/<task>/RESULT.json`;
`gpu/gpu-probe.json`; `gpu/clock.json`; and the job ID and scheduler times. Keep
`gpu/*/deltas/` on Baige (≈80 KB each) unless asked.
