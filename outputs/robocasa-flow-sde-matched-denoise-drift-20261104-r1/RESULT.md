# Matched-denoising drift result

**Complete: all 80 checkpoint loads / 800 forwards passed the independent Linux
CPU audit; Baige job `job-pjgeuietm5of` succeeded and released its GPU.**

Drawer→Counter does not show a uniformly larger conditional likelihood shift
once denoising index is controlled. Its variance-normalized average is close
to OpenDrawer's. The original comparison of different noise levels cannot be
used as evidence of a roughly100-fold Drawer-specific PPO problem. A remaining
excess at the final denoising step is descriptive evidence worth retaining;
it does not establish a cause of robot-performance regression.

## Final policy130 versus policy0

These are mean signed log ratios on fixed policy-zero saved transitions,
summed over H50×12 coordinates. Normalization multiplies each index's result
by σ_k²/σ_0² before averaging the ten index means equally. It does not turn the
result into full-action KL or a behavioral metric.

| Task | Mean variance-normalized log ratio over indices0–9 |
| --- | ---: |
| Drawer→Counter | -0.369302 |
| OpenDrawer | -0.348370 |
| Coffee | -0.257153 |
| Toaster | -0.338000 |
| Stove | -0.474643 |

Drawer's magnitude is about1.06× OpenDrawer's on this aggregate, versus the
much larger original comparison made at different indices. Stove has the
largest magnitude of the five aggregate means. Neither observation establishes
a relationship to learning quality.

The raw per-index means make the residual index dependence explicit:

| Denoising index | Drawer→Counter | OpenDrawer | Coffee | Toaster | Stove |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | -0.180 | -0.024 | -0.259 | -0.109 | -0.798 |
| 1 | -0.138 | -0.141 | -0.436 | -0.336 | -0.080 |
| 2 | -0.703 | -1.349 | -0.020 | -0.734 | -0.840 |
| 3 | -1.062 | -1.767 | -1.123 | -0.747 | -2.221 |
| 4 | -1.049 | -1.806 | -1.063 | -2.126 | -2.338 |
| 5 | -3.184 | -2.875 | -2.060 | -3.758 | -2.125 |
| 6 | -3.124 | -4.396 | -4.140 | -3.160 | -4.412 |
| 7 | -7.105 | -3.732 | -5.836 | -4.502 | -7.172 |
| 8 | -20.381 | -18.111 | -9.092 | -27.884 | -34.418 |
| 9 | -122.565 | -82.541 | -47.794 | -63.438 | -89.728 |

Drawer is not the most negative task at nine of ten matched indices. At index9,
it remains more negative: −122.565 versus OpenDrawer −82.541, Coffee −47.794,
Toaster −63.438 and Stove −89.728. Across the16 saved samples, Drawer has sample
SD49.917 and OpenDrawer SD26.908 at that index. These are descriptive sample
spreads; no new cross-task significance test or policy-selection rule is added.
The [50-row summary](matched-final-summary.csv) supplies sample SD, minimum,
maximum and normalized mean for every task/index; the five result reports
retain each sample and every checkpoint curve.

The predeclared criterion for a persistent Drawer-specific excess at every
matched index is not met. This closes the noise-scale diagnostic; it does not
justify changing PPO, discarding Drawer, claiming the critic bug caused its
curve, or making a teacher-admission decision. Actual paired robot evaluation
and the separate critic validation answer different questions.

## Audit and cost

The [independent audit](FINAL-AUDIT.json) checks all50 selected B16 fixtures
against the original10 restored Q1 actor records, all800 float64 index tensors,
variance-normalized and per-dimension reductions,80 checkpoint marker / READY /
CONSUMED joins, and complete staging cleanup. Initial likelihood replay is
exact at all ten indices for all five tasks. Maximum independent reduction
error is1.5543e-15. Every temporary private checkpoint copy was reclaimed by
its own stager; shared originals are untouched. Audit source is `d1850311`,
separately Git-deployed, while active execution remained pinned to `a28f8405`.

The worker finished at14:20:18 Shanghai; the scheduler and its sole worker are
terminal Succeeded, zero restarts, with scheduler completion at14:21:51.
Process cost is4370GPU-seconds. Conservative creation-to-release cost is
4472GPU-seconds (1.2422GPU-hours), inside the14400GPU-second ceiling.
There were zero new simulator controls, sampling or SGD.
[Scheduler closeout](SCHEDULER-CLOSEOUT.json), [clock](gpu/clock.json),
[execution](EXECUTION.json), [protocol](PLAN.md).
