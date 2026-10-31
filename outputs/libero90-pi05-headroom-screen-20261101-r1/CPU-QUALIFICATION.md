# Stage A driver: CPU qualification

2026-11-01. Code commit `5e90a2949a86b5fc27594ad2368df0bc3987a19a`, a child
of main `4e42258`. It adds `meta_rl.vla_headroom_screen`, the launcher
`scripts/diagnostics/vla_headroom_screen.sh` and
`tests/test_vla_headroom_screen.py`. No GPU, model, simulator or robot
action was used.

- **Where it ran:** thu2 Slurm job 49931 on node01, a 4-CPU/16G step with
  no GPU GRES, COMPLETED 0:0. The worktree was
  `worktrees/libero90-pi05-headroom-screen-cpu-20261101-r1`, pinned and clean.
  The env was `envs/rlinf-pi05-libero-e1a82eb-20261029-r1`.
- **Result:** 43 passed, 0 failed. That is 23 new screen tests plus the
  unchanged onset (5), task57 (11) and noise-headroom (4) tests. The log and
  JUnit file are in `outputs/libero90-pi05-headroom-screen-cpu-20261101-r1/`
  on thu2.
- **Covered:** the closed 12-task × 8-state × 2-arm × 2-repeat plan (384)
  and its seeds; refusal of unlisted tasks and of states 0, 9 and 40–49;
  grant scope binding; the cumulative clock; the selection rule (closest to
  20%, tie-breaks, inclusive band edges, tied-noise requirement, stop when
  nothing qualifies, refusal of incomplete or duplicate records); and the
  arm noise shapes (native independent per chunk, tied repeated 32D).
  Also: journal action charging, stopping before an unfunded episode,
  resuming with the same seeded noise, aborting when the nominal state
  hash changes, and the rates/selection report.

The native arm uses `run_episode(native=True)`, which draws an independent
H10×32 Gaussian per chunk, the same distribution as the native panels of
`frozen_query`. The tied arm draws one fresh unbounded 32D Gaussian per chunk,
the `OnlineSourceSimulator` warmup rule, with no learner.

**Not yet done before Stage A starts:** a grant JSON, deploying this commit
to Baige (where the motor/simulator runtime and task57 paths live), the GPU
probe, the motor parameter signature, same-noise replay and the first real
episode, all of which the driver does inside the allocation.
