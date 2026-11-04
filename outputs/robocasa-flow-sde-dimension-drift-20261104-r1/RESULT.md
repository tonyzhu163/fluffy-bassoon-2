# Dimension drift readout

**Target verified; four-task control replay and whole-run closeout are still in progress.**

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

## Per-update metrics

All 1,300 task/update/rank records are present, with finite scalars and the expected 24 optimizer steps per logical update. Drawer’s approximate KL averages 0.1189 over updates 1–10 and 0.1019 over 121–130; clip fraction is 0.2240→0.2309. Its largest logged approximate KL is the first update (0.2739). These data show no late per-update KL surge matching the accumulated fixed-cohort drift.

## Running controls and evidence

Baige Pro6K job `job-aesultap3qa8` continues under the original one-GPU / 180-minute ceiling. At the 12:48 Shanghai check, 61/80 forwards are complete: Drawer, Coffee and Stove each 16/16; OpenDrawer 13/16 through checkpoint 100; Toaster pending. Every recorded row reproduces R3; the GPU clock is still open and the verified staging worker is healthy. Elapsed time since the device probe is 3,306 seconds (a lower bound, not final scheduler cost). No new controls, sampling or SGD are performed. [Latest monitor](MONITOR-20261104-1248.json). The next check is due by 13:08; the heartbeat will finish the independent 80-tensor audit, scheduler release and cost closeout, publish the final comparison, and stop itself.

- [Independent target audit](TARGET-AUDIT.json) and [complete target readout](gpu/PickPlaceDrawerToCounter/RESULT.json).
- [Noise-scale and correctly reshaped action statistics](COHORT-READOUT.json), [CPU qualification](CPU-QUALIFICATION.json), and [all update metrics](metrics/METRICS.json).
- [Execution and budget](EXECUTION.json), [protocol](PLAN.md), and [full-run audit implementation](FINAL-AUDIT.py).
