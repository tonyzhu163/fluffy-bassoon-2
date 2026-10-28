# Initial runtime qualification, 2026-10-28

Baige `job-06qgbfj648bf`, source `fe5b8fc36f0fc55b6723f5c0c4847246cca03eca`, completed with exit 0. Scheduler pod ran 23:42:03–23:48:30 Shanghai (387 seconds, about 0.108 allocated GPU-hours; task status ended 23:48:40).

**169 tests passed, 3 skipped.** All eight cells completed exactly one update; source, parameter budgets, finite metrics, interaction/optimizer counts, frozen replay and all eight full checkpoint hashes were independently audited.

| Task/core | Real steps | Collection seconds | PPO seconds | Optimizer steps |
| --- | ---: | ---: | ---: | ---: |
| halfcheetah-vel-cmm | 25000 | 2.70 | 60.10 | 100 |
| halfcheetah-vel-gru | 25000 | 1.69 | 3.27 | 100 |
| halfcheetah-vel-persistent_ctm | 25000 | 3.08 | 73.75 | 100 |
| halfcheetah-vel-transformer_xl | 25000 | 2.41 | 37.75 | 100 |
| pointmaze-cmm | 125000 | 21.05 | 15.78 | 25 |
| pointmaze-gru | 125000 | 11.24 | 1.38 | 25 |
| pointmaze-persistent_ctm | 125000 | 23.89 | 18.70 | 25 |
| pointmaze-transformer_xl | 125000 | 17.66 | 9.49 | 25 |

All cores were within 85–100% of the 250k total policy parameter cap. PointMaze used 25 lanes/K10/H500/five epochs and minibatches of five trials; HalfCheetah used 50 lanes/K5/H100/ten epochs and minibatches of five trials. Cached replay length was 100. These measure different task workloads, not equal optimizer schedules across tasks.

These are execution/cost observations, not trained performance. All initial PointMaze rollout successes were zero. The first PointMaze adapter retained the installed one-substep (0.01s) action duration. It is superseded for training by an explicitly declared 0.1s adapter; do not transfer its absolute throughput or scores to that new protocol.

The follow-up `metta-bench-qualification-20261028-r2` / Baige `job-ehmq2vza51zm`, source `c6196b8`, passed 172 tests/3 skips but stopped at its oracle headroom gate (exit 1). Oracle and white-noise success were zero on both splits; persistent-noise success was 0/48 fresh and 1/48 held-out episodes. Its GPU probe was saved at 23:50:52 and the scheduler was terminal at 23:52:06 (displayed duration 89 seconds). No long training ran. The controller saturated the XML gear-100 actuators; the next source declares action duration and bounds PD force.

Prepared worktrees named 20261028-r3/r4 were not launched. The next actual allocation uses the new date/run identity `metta-bench-qualification-20261029-r1`, source `47a37b7`. It also tests padding-length invariance and enables real-transition loss normalization only for the new PointMaze protocol, preserving existing fixed-trial PPO behavior.
