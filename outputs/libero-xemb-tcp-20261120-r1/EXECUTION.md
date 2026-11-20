# Baige execution record (`libero-xemb-tcp-20261120-r1`)

## Code revisions and Linux qualification

All Linux CPU and GPU executions ran from clean detached Git worktrees on Baige shared storage using pinned OpenPI (`215abfb217dbac7d5f1273282331b9b1866c0479`) and LIBERO-Pro2.0 (`e5500c690c1498eca1b7050350654964190cdf70`), keeping the 8-task official test split untouched.

| Attempt | Revision | Stage / Host | Outcome | Notes |
| --- | --- | --- | --- | --- |
| `cputest01` | `af1dd6e0` | CPU `test` | Exit `0` | `3 passed in 1.21s` (`pytest` on [`tests/test_xemb_tcp.py`](../../tests/test_xemb_tcp.py)). |
| `cpugate01` | `af1dd6e0` | CPU `gate0-cpu` | Exit `1` | Failed EGL import: CPU Mesa software renderer exposes device `0` only, got `MUJOCO_EGL_DEVICE_ID=1`. |
| `cpugate02` | `825896c7` | CPU `gate0-cpu` | Exit `1` | Fixed `MUJOCO_EGL_DEVICE_ID=0` + joint pre-contact boundary; failed `max_shift_err_m` check (`1.792e-05 m`) because `sim.data.geom_xpos` lags `qpos` by one `mj_step` substep after reset settling. |
| `cputest02` | `6c3b2f11` | CPU `test` | Exit `0` | `4 passed in 1.54s`; added stale-kinematics test after switching `apply_mujoco_tcp_offset` verification to temporary `mujoco.MjData` + `mj_forward`. |
| `cpugate03` | `6c3b2f11` | CPU `gate0-cpu` (4 Mesa workers) | **Exit `0` (`291.0 s`)** | **Gate 0 PASSED** (`0` GPUh): `104` replays, `0` pre-contact leaks (`max_shift_err_m <= 2.08e-16 m`), selected `r* = 0.035 m` (`8/8` BASE, `8/32` shifted, `8/8` separable tasks). [Receipt](evidence/cpugate03/gate0-summary.json) |
| `cputest03` | `c6417128` | CPU `test` | Exit `1` | `4 passed, 1 failed` (`test_run_tcp_episode_chunk_clear_and_nom_passthrough` horizon `38` was shorter than the `40`-step TCP maneuver). |
| `cputest04` | `795f48f2` | CPU `test` | **Exit `0`** | **`5 passed in 1.47s`** after raising test horizon to `45`. [Log](evidence/cputest04/run.log) |
| `gpu01smoke` | `795f48f2` | Baige Distributed `job-gmj5rsb0xd59` (`gate1-smoke`) | **Exit `0` (`119 s`)** | `gpu.json` passed (`NVIDIA RPBZZZ6`, `97887 MiB`, exclusive container), NVIDIA EGL verified (`b'NVIDIA Corporation' b'NVIDIA RPBZZZ6/PCIe/SSE2'`), `2/2` on `libero_goal/4` across all 3 arms. [Receipt](evidence/gpu01smoke/gate1-summary.json) |
| `gpu01gate1` | `795f48f2` | Baige Distributed `job-gmj5rsb0xd59` (`gate1`) | **Exit `3` (`1469 s`)** | All `56 × 3 = 168` paired episodes completed (`0` reset/proposal mismatches); exited `3` because Gate 1 thresholds failed (`22/48` grasp-only and `20/48` grasp+release vs `23/48` nominal; `8/8` BASE). [Compact receipt](evidence/gpu01gate1/gate1-summary-compact.json) |

## Scheduler job ledger and compute budget

- **Baige Distributed Training job:** `job-gmj5rsb0xd59` (`mwj_1120_libero_xemb_tcp_gate1`, `1× NVIDIA RPBZZZ6`, queue `算法Z组/aihcq-rh0wwk9muei4`, `NCCL_P2P_DISABLE=1`, 90-minute limit, 0 retries).
- **Timestamps (Shanghai):** Created `2026-11-20 12:08:35`, Running `12:08:46`, Ended `2026-11-20 12:35:37`.
- **Scheduler GPU-hours:** `26m 50s` = **`0.447` GPU-hours** (`0.447 / 8.0` Phase 2 budget used).
- **Full evidence on shared NVMe:** `/mowenjie/fluffy-bassoon-2/outputs/libero-xemb-tcp-20261120-r1/gpu01gate1/gate1/gate1-summary.json` (`SHA-256: 3314c0d62beeaf1bd14258af107bc715549ca9bab3669a1645186e4eb6aee401`, replaced in Git by compact JSON [`evidence/gpu01gate1/gate1-summary-compact.json`](evidence/gpu01gate1/gate1-summary-compact.json) with per-step `native_rewards` lengths and nonzero counts).
