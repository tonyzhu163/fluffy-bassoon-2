# LIBERO collector speed qualification handoff

Terminal: Baige job-1l5gxk7pglgo passed, exit0. R4a485actions/131schedulerGPU-seconds; all attempts759/411. Native/cached/parallel traces exact in the matched case. See TERMINAL-AUDIT.json and docs/libero-collection-speed/EVALUATION.md. Unused envelope remains sealed; no retry or production migration authorized.

Executable source: `c6f46e8cd80c7cca0877983b659c76ef26b1edbd`.
The clean allocated worktree is
`/mowenjie/fluffy-bassoon-2/worktrees/libero-collector-speed-20261030-r4a`.
No production PPO/SAC driver was migrated. The source40 implementation remains
based on owner source33827ab; only23ad427 reset metadata lines were copied.

## Admission and interpretation

- Execution microbatch remains1. Native B2/B4 model kernels failed strict
  numerical parity; grouping independent B1 calls preserves the scalar path.
- On every current observation of the first episode, the motor executes native
  actions and shadows cached features/actions with identical explicit noise.
  Require feature error≤2e-4 and action error≤1e-5 on every decision. Initial
  observation parity alone is insufficient. Shadow work restores native RNG.
- The shadow episode is a qualification cell, not a throughput baseline. Pure
  native, pure cached scalar and parallelB2 cells provide within-process/GPU
  episode times. All compared parallel lanes repeat the scalar case and must
  match initial state/public observations and complete per-decision traces.
- Two native-executed serial episodes are compared before cached/parallel
  admission. Any reset nonrepeatability stops those cells; no pixel acceptance
  profile or model tolerance is silently relaxed.
- Four state19/seed1916071045 resets run in one owned process with fresh
  environments. Full float64 simulator state, raw proprioception and RGB are
  retained in NPZ artifacts. Exact state/proprio and RGB count/max/RMS are
  reported separately. A completed diagnostic does not apply the owner's
  separately qualified two1LSB wrist-channel profile automatically.
- Preserve Baige's `timing_comparison_eligible:false` receipt. Report measured
  within-run rates and their limits; do not turn cross-GPU R2/R3 differences
  into a causal speedup claim. No architecture, learning or convergence claim
  follows from this execution benchmark.
- Production integration still needs owner review of exact API/feature/reset
  contracts and controller-memory, old-policy replay and unchanged global
  weighting. Keep production sourcec4cadc8 and active recoveries immutable.

## Shared resource accounting

The canonical sidecar is
`outputs/libero40-source-1m-20261029-r1/terminal-capacity-aliases.json`
on Baige. Alias `libero-collector-speed-20261030-r2` consumes terminal unused
capacity **within** original SAC9500 reservations:1633Spatial03+1247Goal02.
Original sourcecaps, indexes, journals and recoveries remain unchanged.

R2 actual95actions/170scheduler GPU-seconds; R3 actual179/110.
R4a reserves at most2100actions/1200scheduler GPU-seconds. Combined
actual+reservation2374actions/1480seconds stays inside2880/1800.
The1800-second systems hold is within the existing approved32GPU-hour pool,
released from terminalSACworker0. It is not additional compute authorization.
All continuations/retries share these bounds. Do not infer another allocation
from unused reservation capacity.

## Terminal settlement workflow

1. Require the scheduler's terminal state and actual allocation-active duration
   from the parent. A driver exit/report wall time is not GPU billing duration.
2. Read R4a `exit-code.txt`, `benchmark/report.json`, probe and physical journal.
   Verify report sourcec6f46e8 and recompute known charges from returned RPCs plus
   maximum ceilings for unresolved RPCs. Count every qualification/episode/reset
   settling action. Preserve the unknown ceiling on interrupted RPCs.
3. Take the canonical sidecar's `.lock` with exclusive `fcntl.flock`. Re-read
   its matching alias/attempt. Keep original maximums and donor/index bindings.
   Replace only R4a's reserved status with terminal evidence: exact source/job,
   exit/result status, physical charge and unresolved ceiling, verified scheduler
   GPU-seconds and report/journal hashes. A repeated settlement must match.
4. Recompute `combined_actual_and_reserved_physical_actions` using actual
   charges for terminal attempts and maximums for outstanding attempts. Do the
   same for GPU-seconds. Reject totals above2880/1800 and reject R4a actuals
   beyond its2100/1200 reservation. Atomically write/fsync/replace the sidecar.
5. Mirror thin reports and raw reset NPZ artifacts, hash-verify them, and commit
   a local evidence receipt. Report residual capacity as accounting only. The
   donor streams remain sealed against future execution without explicit
   reconciliation. Parent/budget owner settles the shared compute-ledger hold;
   this child does not edit another owner's compute component automatically.

## Scientific quantity boundary

The shared next-phase plan is approximately1M executed source actions for each
of SAC, persistent PPO and separately trained reset PPO: approximately3M source
training actions per benchmark. LIBERO's future schedule still requires its
owner's frozen contract. These speed-qualification interactions receive zero
source-training credit and do not launch, reserve or alter that next phase.
