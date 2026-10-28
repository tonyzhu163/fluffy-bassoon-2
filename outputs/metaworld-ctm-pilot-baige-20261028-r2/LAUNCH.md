# Baige Meta-World pilot launch and first update receipt

- Job: `job-x2x4kdpfn011`, name `mwj_1028_metaworld_ctm_pilot_g1_r2_adf741d`.
- Submitted: 2026-10-28 19:20:18 Asia/Shanghai.
- Source: `adf741d6e955a624e9f785b9d1572dabf9ff4ab1`, based on reviewed
  main `decabec00981734d9fe2dfe8d42cd0d705238efa`.
- Allocation: one NVIDIA RPBZZZ6 (97,887 MiB reported by NVML), 16 CPUs,
  64 GiB RAM, 8 GiB shared memory; low priority, 180-minute scheduler cap,
  no automatic retry or deletion.
- Immutable worktree:
  `/mowenjie/fluffy-bassoon-2/worktrees/metaworld-ctm-pilot-baige-20261028-r2`.
- Outputs:
  `/mowenjie/fluffy-bassoon-2/outputs/metaworld-ctm-pilot-baige-20261028-r2`.
- Launcher: that worktree's `scripts/run_metaworld_pilot.sh`, with the full
  source SHA and r2 identity as explicit arguments.

The one-GPU training probe passed 19:20:45 Shanghai, including scheduler
allocation count, CUDA computation, PCI mapping and own-PID allocation checks.
The allocated test suite passed **71 tests, two skips, in 33.22 seconds**.
The two skips are existing CUDA-marker cases; new environment, all-core
checkpoint recovery and real simulator tests passed. No tests ran on macOS or
the development machine.

Python 3.11.17, torch 2.11.0+cu128, NumPy 2.4.6, Meta-World 3.1.1,
MuJoCo 3.3.0, Gymnasium 1.4.0. New environment
`/mowenjie/fluffy-bassoon-2/envs/metaworld-3.1.1-r1` uses the existing torch
environment by read-only site-packages path; no active environment was changed.
Shared storage had 1.4 TiB free and the write probe passed. A quota query did
not expose an additional limit. Complete version-only freeze is mirrored here.

At the 19:25:15 Shanghai saved-artifact check, GRU completed two updates,
50,000 training environment steps and ten optimizer steps. Width 236 gives
245,757 trainable parameters. Its first behavior replay errors were
4.77e-7 in log probability and 2.98e-8 in value, within the declared 2e-5 bound.
Updates had finite metrics and a complete `latest.pt`; no failure receipt.
Update-two deterministic success remained zero on both fresh train-family and
held-out goal trials. This is initial execution evidence, not adaptation.
The full raw snapshot is `CHECK-20261028-1925.json`.

The fixed sequence is ML1 Push then ML10, each with GRU, CTM T4 and
Transformer-XL, seed 1000, matching the dated plan. Each cell targets 20 updates
under its 900-second soft cap, plus final paired memory/reward interventions.
Measured GPU-hour plots are written after each completed cell. All learned
weights remain fixed during evaluation. The local 20-minute heartbeat
`meta-world-ctm-pilot-progress` monitors this run only, reports meaningful
changes and audits final raw outcomes/checkpoint hashes before closure.

The first failed admission, r1 `job-ikplxyvyh7p9`, remains preserved. The only
r2 source correction initializes an unset CUDA selector to container-local
ordinal zero while retaining the scheduler's NVIDIA allocation. No active run
contract or worktree was modified.
