# Dimension drift readout

**Completed: all 80 forwards and independent tensor audits pass; the Baige job succeeded and its allocation is released.**

Drawer→Counter’s update-130 signed conditional log ratio is **−137.5202** on the original fixed B16 saved-chain cohort. All 16 retained checkpoints reproduce the earlier R3 report exactly, including every per-sample chunk log ratio. An independent Linux CPU NumPy reduction of all 16 saved float64 tensors matches every dimension sum exactly.

| Action group | Indices | Share of final negative log ratio |
| --- | --- | ---: |
| Arm | 0–5 | 61.43% |
| Gripper | 6 | 16.86% |
| Base motion | 7–9 | 11.27% |
| Torso | 10 | 2.87% |
| Base mode | 11 | 7.58% |

Gripper plus base mode contribute **24.44%**; arm dimensions contribute **61.43%**. All twelve final contributions are negative, so signed and absolute dimension shares agree. Neither predeclared 70% concentration rule fires. This result does **not** support changing the PPO objective or swapping the task.

## The cross-task noise-scale confound

The five original cohorts use different denoising indices. The native sampler and an independent formula agree on their fixed transition scales:

| Task | Denoising index | Transition standard deviation |
| --- | ---: | ---: |
| OpenDrawer | 0 | 0.30000 |
| CloseToasterOvenDoor | 4 | 0.11619 |
| PickPlaceDrawerToCounter | 9 | 0.03162 |
| CoffeeSetupMug | 1 | 0.28460 |
| TurnOffStove | 4 | 0.11619 |

Drawer→Counter therefore has **90× OpenDrawer’s and 81× Coffee’s inverse transition variance**. Gaussian log-density changes depend on residual changes divided by that variance. The raw “~100× faster drift” comparison does not control for denoising time or states and cannot establish comparable parameter movement, gradient alignment or behavioral change. Matching denoising indices and using additional cohorts would be needed for that comparison; this run keeps the original inputs unchanged.

These are conditional transition likelihoods on one original cohort per task. They are not full-action marginal KL or a test of later binary-action probabilities. The original decoded gripper values in the Drawer cohort all lie below the 0.5 threshold, but no later-policy execution claim follows.

## Five-task reproduction controls

All original R3 per-sample results match exactly. Shares below use the absolute final per-dimension totals to avoid signed cancellation in the control tasks. They are conditional fixed-cohort readouts at the different denoising indices listed above.

| Task | Final signed log ratio | Arm absolute share | Gripper/base-mode absolute share |
| --- | ---: | ---: | ---: |
| OpenDrawer | -0.3403 | 50.76% | 23.07% |
| CloseToasterOvenDoor | -2.5751 | 67.63% | 8.12% |
| PickPlaceDrawerToCounter | -137.5202 | 61.43% | 24.44% |
| CoffeeSetupMug | -0.7713 | 77.52% | 5.07% |
| TurnOffStove | -1.4524 | 65.13% | 31.36% |

The variance-multiplied log-ratio column in the CSV is descriptive arithmetic; it does not create a matched-denoising comparison or a marginal action KL.

## Per-update metrics

All 1,300 task/update/rank records are present, with finite scalars and the expected 24 optimizer steps per logical update. Drawer’s approximate KL averages 0.1189 over updates 1–10 and 0.1019 over 121–130; clip fraction is 0.2240→0.2309. Its largest logged approximate KL is the first update (0.2739). These data show no late per-update KL surge matching the accumulated fixed-cohort drift.

## Completed replay, audit and cost

Baige Pro6K job `job-aesultap3qa8` completed successfully. At the 13:12 Shanghai closeout, all five tasks have 16/16 checkpoint forwards; the independent CPU audit checks all 80 float64 tensors, decomposition and consumption hashes, and reproduces every original R3 sample exactly. The worker ended at 13:04:54 and the scheduler marked success at 13:05:17; the single allocation is released. The staging worker finished naturally with all 80 receipts. Process cost is 4,330 GPU-seconds (1.203 GPU-hours); conservative creation-to-scheduler-completion cost is 4,373 GPU-seconds (1.215 GPU-hours), within the 3 GPU-hour ceiling. This run performed zero new controls, sampling or SGD. Monitoring ends after publication of this closeout.

- [Full independent audit](FINAL-AUDIT.json), [five-task summary](dimension-summary.csv), [terminal scheduler/cost receipt](SCHEDULER-CLOSEOUT.json), and [GPU clock](gpu/clock.json).
- [Independent target audit](TARGET-AUDIT.json) and [complete target readout](gpu/PickPlaceDrawerToCounter/RESULT.json).
- [Noise-scale and correctly reshaped action statistics](COHORT-READOUT.json), [CPU qualification](CPU-QUALIFICATION.json), and [all update metrics](metrics/METRICS.json).
- [Execution and budget](EXECUTION.json), [protocol](PLAN.md), and [full-run audit implementation](FINAL-AUDIT.py).
