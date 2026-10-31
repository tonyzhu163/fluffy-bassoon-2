# Meta-World CTM pilot, 28 October 2026

**Superseded by [r2](../metaworld-ctm-pilot-baige-20261028-r2/PLAN.md), a retry after this run failed.**

User authorized testing state-based meta-RL before expanding VLA integration,
then explicitly selected Baige. Use a fresh exclusive one-GPU allocation,
16 CPUs, 64 GiB RAM, 8 GiB shared memory, low queue priority and a 180-minute
scheduler limit. Existing jobs and their immutable worktrees remain untouched.

Implement and test the contract in `docs/metaworld-ctm.md`, based on latest
available main. Record the final source SHA and actual validation before the
training gate opens. The development machine is for code and dependency setup;
all simulator/model tests and learning run in the scheduler allocation.

Pin Meta-World 3.1.1, MuJoCo 3.3.0 and record the complete installed environment.
Use a new environment with the existing read-only torch runtime available by
path; do not modify packages in the environment used by active VLA jobs.
Check storage capacity, quota if available, and a write probe first.

Admission requires the allocation-local one-GPU training probe; fake-environment
boundary and visibility tests; exact next-update checkpoint recovery for all
three cores; real two-episode Meta-World trials on ML1 Push and ML10; relevant
existing PPO/checkpoint tests; and behavior likelihood/value replay in each
learner's actual first full trial.

Pilot matrix: ML1 Push and ML10 × GRU T1, persistent CTM T4 and Transformer-XL
T1 × seed 1000. Trainable policy cap 250k, five 500-step episodes per trial,
ten training lanes, one PPO epoch, minibatches of two complete trials. Target
20 updates with a 900-second soft cap per cell. Evaluate on ten balanced lanes
and five episodes at initialization, every two updates and the final boundary.
Final diagnostics add memory reset and reward blinding on paired trial seeds.
These are throughput and learning-admission pilots; a single seed, shortened
evaluation trial and reduced PPO epochs cannot establish benchmark superiority.

The six soft caps total 1.5 GPU-hours; setup, full-update overshoot and final
diagnostics are additional. The scheduler limit bounds the entire job at three
GPU-hours. Save complete recovery after every update. A scheduler kill may
interrupt an update; only the preceding atomic checkpoint is resumable.
Do not retry a failure under the same output identity. Diagnose and record it.

Save raw evaluations, timing, parameter/state sizes, package freeze, probe,
checkpoints and success-versus-GPU-hour plots. Do not overlay paper wall times
as a measured hardware-matched baseline: METTA used an RTX 3090 Ti and a
different optimizer/runtime. After first advancing updates, monitor every
20 minutes and report cluster/job, progress, last check and next action.

Once measured capacity and update times are available, choose a feasible
configuration for three screening seeds. Any materially larger resource budget
needs user approval; the initial capped pilot does not authorize an open-ended
benchmark sweep. No scientific conclusions or VLA admission decisions should
be based on this execution pilot alone.
