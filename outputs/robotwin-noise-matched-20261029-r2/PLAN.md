# Actual experiment retry after allocation-visibility fix

**Superseded by [r4](../robotwin-noise-matched-20261029-r4/RESULTS.md), the completed run.**

2026-10-29 Asia/Shanghai. Scientific contract is unchanged from
[r1](../robotwin-noise-matched-20261029-r1/PLAN.md): six-task source training,
GTrXL/PPO versus source-pretrained feature-conditioned noise SAC and trained
episode-reset GTrXL, exact equal10.24M source primitive-action ceilings,
K4 training and K8 evaluation at1/2/4/8. This is the first training attempt;
r1 stopped before tests or model computation.

The launcher preserves CUDA visibility if supplied and otherwise selects the
single device exposed inside the exclusive Baige container. Run the full
allocation/CUDA probe and all focused tests before native parity and source
collection. New immutable worktree and output identity; no r1 files changed.

Resource envelope: one GPU,16CPUs,128GiB RAM,8GiB shared memory,43-minute
scheduler cap. Source driver wall budget35minutes with a3-minute boundary
reserve, after a maximum5-minute test allowance. Together with r1's2-minute
reserve, both submissions remain bounded by45GPU-minutes. Save resumable
coordinator state and timing before considering larger continuation compute.
