# Can the released S20 demonstrations support a stronger motor?

Replay the first two HDF5 groups per condition on each of the two cached
fitting bowl tasks (Goal 4 and 8). Use their recorded seeds, reset IDs,
commands, camera resolution and horizons, the pinned LIBERO-Pro2.0 simulator
`e5500c69`, and the training archive at `6c4c5524`. Match the first recorded
observation at its measured timing, record divergence in wrist motion, and
score native success at the benchmark endpoint. Where the GPU is occupied,
first check the same physics, proprioception and rewards with the camera
sensors disabled. Inspect command activity across all 120 cached
demonstrations for evidence that the successful S20 executions rely on
waiting. This is a fitting-side replay check of recorded actions.

For the first learned motor, train a small timing-conditioned action adapter
on the 24 fitting tasks' released BASE and S20 demo commands. Deduplicate
the declared copies, report effective reset coverage, and hold π0.5 weights
fixed. Use the public image and state at the time the command was issued.
On the audited archives, observation row `t` is the result of action `t`.
Pair it with `actions[t + 1]` and subsequent commands when training.
Provide the true timing setting as an adapter input during this diagnostic.
Use learned action-chunk targets that respect the issued-command convention.
An analytic predictive comparator would need to account for both stale
observations and pending commands. The 10-action proposal cannot reach a
14-control offset directly. [Pinned timing implementation](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/nonphysical_perturbations.py)

On fresh paired development resets and matched noise, compare the informed
adapter with the original and paced controls, the strongest source-selected
fixed adapter setting, and the same adapter conditioned by the public lag
rule. Repeat paired draws to estimate run-to-run variation. Report BASE and
S20 successes, changes in feasible actions by timing, and paired gains and
losses. An advantage over fixed conditioning qualifies motor coverage. A
residual advantage over the lag rule on the same adapter would measure room
for a more capable reader. If the lag rule captures the S20 gain, use S20
as a diagnostic and seek history-specific supervision on another released
mechanism. Keep the official test tasks reserved. [Prior reader pilot](../libero-decision-pretraining-20261119-r1/RESULT.md)
