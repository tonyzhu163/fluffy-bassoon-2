# Per-dimension drift diagnostic for the 10M Flow-SDE sources

**Status 2026-11-04: RUNNING on Baige `job-aesultap3qa8`.** Owner
chat `01a1a4a2`; user requested pull and dimension-drift check. Reads only retained artifacts of
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
Iteration-1 gradient norms exceed `clip_grad` 1.0. Gradient clipping does
not imply equal parameter-update norms under Adam, and likelihood drift does
not establish persistent gradient alignment. Where does the observed drift
live, and is a behavioural diagnostic warranted?

## Hypotheses

- **H1, binarized dimensions.** The env binarizes gripper (index 6) and base
  mode (index 11). PPO still scores the continuous pre-binarization values. Means
  can move on the same side of the decoded 0.5 threshold without changing the corresponding executed
  binary action, but crossing that threshold changes it. Predicts concentration
  in dims 6/11; saved transition likelihoods alone do not test sign changes.
- **H2, arm dimensions.** Real motion change that does not raise success.
  Predicts drift spread over dims 0–5 (or the base/torso dims).
- **H3, cohort artefact.** One saved cohort happens to be sensitive. Weak, because
  all 16 samples drift (max −55 at step 130). These samples share one cohort;
  this does not establish generalization across states. H1/H2 split it further.

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
   loss for all 130 updates. The join requires both actor ranks for every
   update and verifies the 24-step schedule. Only iteration 1 is in Git today.
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

- **dims 6+11 ≥ 70% of the step-130 signed drift on Drawer→Counter:**
  evidence of concentration consistent with H1. Also report absolute and
  negative dimension shares so cancellation cannot masquerade as concentration.
  This does not establish unchanged binary actions or justify dropping their
  likelihood terms. Before a recipe change, inspect binary-action probabilities
  and qualify a mathematically valid objective. No recipe change or task swap
  is authorized by this diagnostic.
- **arm dims ≥ 70%:** evidence consistent with H2. Inspect the per-update clip fraction and KL for a
  regime change, and look at rollouts from late checkpoints. Swapping the task
  would require a separate scientific decision; CoffeeSetupMug staying small
  only establishes a difference on these cohorts.
- **Neither:** report the split as-is and do not change the recipe.

TurnOffStove remains in the five-task GPU panel as an exact R3 reproduction
control. Its drift is in the normal range. The `metrics` join will show whether
its clip fraction/KL ever separate from the improving tasks.

## Evidence to record here

`metrics/METRICS.json` and the CSVs; `gpu/<task>/RESULT.json`;
`gpu/gpu-probe.json`; `gpu/clock.json`; and the job ID and scheduler times. Keep
`gpu/*/deltas/` on Baige (≈80 KB each) unless asked.

## Review before launch

The original staging assertion required an output inside the source campaign,
but COMMAND.sh puts it in this sibling diagnostic run. Staging is now bound
to this run's exact staging/gpu paths. Consumption receipts bind task, step,
model hash and READY hash; R3 reproduction is required before completion.
Initial fixture hashes and source completion are checked. Linux CPU checks
passed: all 1,300 update/rank rows, optimizer counts, finite scalars, and two
reference decomposition cases. [Receipt](CPU-QUALIFICATION.json). The deployed
clean commit is `0dbdafa4`; the allocated-device probe and initial replay pass
exactly. [Execution](EXECUTION.json), [first check](MONITOR-FIRST.json).

The GPU artifact preserves the proposed flattened `action`/`model_action`
statistics. These are indexed by flattened position, not physical dimension.
A separate CPU readout will reshape them to `[16,50,12]` / `[16,50,32]` for
per-dimension statistics; no active source file or GPU execution is changed.

## Fixed-cohort comparison caveat verified on Linux CPU

[The native-sampler readout](COHORT-READOUT.json) finds one denoising index
per original cohort: OpenDrawer 0, Toaster 4, Drawer→Counter 9, Coffee 1 and
Stove 4. At noise level 0.3 their transition standard deviations are 0.3000,
0.1162, 0.03162, 0.2846 and 0.1162. The Drawer cohort thus has 90× the inverse
variance of OpenDrawer and 81× that of Coffee. For a fixed Gaussian variance,
log-density changes scale with squared residual changes divided by that
variance. The raw cross-task drift ratio therefore cannot establish a
corresponding difference in parameter movement, gradient alignment or behavior.
Within-cohort checkpoint trends and dimension splits remain valid conditional
readouts. This diagnostic does not resample or modify the active GPU inputs.
