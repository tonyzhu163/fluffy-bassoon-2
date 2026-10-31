# LIBERO collection execution evaluation

Verified30October2026 on Baige, job `job-1l5gxk7pglgo`, executable source
`c6f46e8cd80c7cca0877983b659c76ef26b1edbd`. Driver exited0 with status
`passed`; scheduler-active GPU duration131seconds. See the
[terminal audit](../../outputs/libero-collector-speed-20261030-r4a/TERMINAL-AUDIT.json)
and [full report](../../outputs/libero-collector-speed-20261030-r4a/terminal-report.json).

## Same-process result

One frozen official OpenPI pi0.5 motor served independent LIBERO simulator
processes. Every model call retained scalarB1 kernels. Task Spatial00/state0/
seed27101 used zero steering coordinates, native H10 prediction and at most
five executed actions per decision. Each complete episode executed77actions
in16decisions. Model weights/controllers were not trained.

All16current observations of a native-executed episode were checked against
cached-prefix features/actions with the identical observation and noise.
Maximum feature and action errors were both0. The second native episode, cached
scalar episode and both parallel lanes reproduced actual full initial state,
raw initial observations and complete per-chunk traces exactly.

| Pure execution cell | Task actions | Complete episode wall time | Aggregate actions/s |
| --- | ---: | ---: | ---: |
| Canonical native scalar |77|9.81247s|7.84716|
| Cached-prefix scalar |77|9.07304s|8.48668|
| Cached-prefix, two simulator processes |154|10.72814s|14.35478|

These are descriptive measurements of one matched case on one GPU/process.
Cached scalar achieved1.0815×the native aggregate rate; parallelB2 achieved
1.6914×cached scalar and1.8293×native. Reset/process startup dominates much of
these short episodes. Warm interleaved model-path median was98.9597ms native
features+actions versus58.8941ms cached features+actions, approximately1.6803×.
The shadow episode computes both paths and is excluded as a throughput baseline.

Baige's probe retains `timing_comparison_eligible:false`, including when called
with training purpose. Preserve that limitation. These rates do not establish
an all-task/hardware speedup, source learning, adaptation or convergence. Earlier
cross-GPU R2/R3 trajectory differences are not causal execution comparisons.

## Reset diagnosis

Four resets in one owned process, each with a fresh environment, used
Spatial00/state19/seed1916071045. The92-coordinate float64 actual simulator
state, public proprioception and main RGB were exact across all four resets.
Relative to repeat0, repeats1–3 differed in only wrist channels[128,52,0] and
[128,52,1], each+1uint8 level; wrist RMS difference0.00318944. Repeats1–3 raw
NPZ artifacts are byte-identical. No reset acceptance profile was applied.

Full actual state/proprio/rawRGB NPZ artifacts remain protected with hashes in
[the audit](../../outputs/libero-collector-speed-20261030-r4a/TERMINAL-AUDIT.json)
and on Baige under this run's `benchmark/` directory. Compact reports are tracked;
raw NPZ copies stay untracked locally and on Baige.

## Accounting and integration boundary

This attempt consumed485physical actions:385task actions+100settling, zero
unresolved RPC ceilings. All three attempts total759actions and411scheduler
GPU-seconds, inside the existing2880-action/1800-second systems envelope. The
unused2121actions/1389seconds remain sealed, unreleased and unauthorized for
another attempt. The qualification receives zero source-training credit.

Production sourcec4cadc8 and its PPO/SAC drivers/recoveries were not migrated.
Eligible numerical execution is cached-prefix scalarB1, with two independent
simulators qualified for this case. General source40 integration still requires
owner binding of the exact API/reset/feature contracts plus controller-memory,
old-policy replay and unchanged global-loss-weighting checks. TrueB2/B4 model
kernels remain rejected for exceeding the original numerical tolerances.
