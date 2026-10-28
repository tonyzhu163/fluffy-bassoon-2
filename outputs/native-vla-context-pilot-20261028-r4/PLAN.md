# Bounded native VLA context pilot

User requested "run it on baige". The runtime source is
`0d6dee3d2a792dbede86fe754425bdd361e3cdde`; latest main `1a21c86` was
incorporated before this retry. The driver and scientific scope are the
[original bounded pilot](../native-vla-context-pilot-20261028-r1/PLAN.md).
Differences from r3: return CPU-prepared commands to the policy device,
expose preprocessing/preparation hooks for integration tests, and run a
two-attempt CPU-command round-trip test on the allocated CUDA device.
The original nonfinite-gradient guard remains enabled.

Baige **`job-lwpvhzys2vtc`**, name `mwj_1028_native_context_g1_0d6dee3`,
submitted 2026-10-28 19:27:55 Shanghai. One NVIDIA RPBZZZ6, 16 CPU,
64 GiB RAM, 8 GiB shared memory, 60-minute scheduler cap. Automatic retries
and deletion disabled; low priority, existing PFS mounted at `/mowenjie`.
The driver has its original 2,700-second boundary-checked budget. Earlier
failed admissions consumed about four driver minutes in total.

Immutable source and output directories:

- `/mowenjie/fluffy-bassoon-2/worktrees/native-vla-context-pilot-20261028-r4`
- `/mowenjie/fluffy-bassoon-2/outputs/native-vla-context-pilot-20261028-r4`

Allocated CPU qualification: **16 passed, 1 CUDA skip in 24.40 seconds**.
Inside the actual GPU allocation: **8 passed, 1 intentionally hidden-CUDA
skip in 3.10 seconds**, then the CPU and real CUDA collector regressions
**both passed in 1.85 seconds**. GPU health/count admission also passed.

Plan: repeat real native parity/gradient/fitting/replay/frozen-weight gate;
collect three K4 training histories at -45/0/+45 and two validation histories
at held-out -30/+30 with disjoint overlays; qualify paired oracle execution;
perform 20 supervised warm-start updates, one full-trial PPO step, and paired
normal/wiped/reward-blind held-out smoke trials. Save complete recovery
state and raw outcomes. A failed competence gate ends the pilot without
scaling. Small one-seed results cannot establish adaptation/generalization.

Monitor only this job every 20 minutes once progress is verified. Do not
change active source, extend the cap, start a sweep, or alter other jobs.
