# Kinova3 plate pushing with the learned velocity residual zeroed

The registered test crossed at the first look. Removing the learned history velocity residual rescued 63 of 64 fresh native cases, with zero losses and one failure in both mixed arms. The success difference for MIXED-ZERO minus MIXED-ON was **+98.44 percentage points** (simultaneous 95% interval **+69.33 to +99.998 points**, paired two-sided exact McNemar **p = 2.1684 × 10⁻¹⁹**, incremental α = .005). We stopped at 64 without a subsequent allocation. [Registration](PLAN.md) · [Sealed look](evidence/look-64.json)

| Policy | Native successes | Descriptive exact 95% interval |
| --- | ---: | ---: |
| STOCK | 64/64 | 94.4–100% |
| MIXED-ON | 0/64 | 0–5.6% |
| MIXED-ZERO | 63/64 | 91.6–99.96% |

MIXED-ZERO succeeded on 63 placements where STOCK succeeded and fell short on case 65. Its paired difference from STOCK was −1.56 percentage points (descriptive 95% interval −9.56 to +9.32). The shared history sampling program preserved success on 63 cases when it received zeros. Issuing the fitted term therefore caused the loss on those paired cases under this sampler. At the first post-opening replan, the maximum absolute proposal difference between ON and ZERO had a median of 0.2864 across cases. The corresponding median between STOCK and ZERO was 0.001894. The small proposal difference and the one ZERO failure are consistent with a remaining sampling-path effect. The experiment isolates use of this fixed fitted term on this Kinova3 task and placement/noise distribution. [Per-case evidence](evidence/readback-64.json)

## Native and implementation checks

All 192 HDF5 trajectories passed an independent Linux readback. Each triplet started with identical recorded physics, public observations, paired noise and five genuine-stock opening controls. Readback reconstructed the four-transition public history, the two cameras, executed controls and native terminal flags. MIXED-ON executed all 300 controls in every case. Median controls were 98 for ZERO and 97.5 for STOCK. Median end-effector path length was 0.718 m for ON, 0.734 m for ZERO and 0.734 m for STOCK. The raw physics and proprioception remain with each tape. [First native triplet](evidence/first-native.json) · [Full readback](evidence/readback-64.json)

Each lane reused one actual compiled history executable for both mixed arms. The frozen ON adapter supplied a nonzero dynamic offset on all 3,776 history replans. ZERO supplied exact float32 zeros on all 1,299 history replans. STOCK invoked the published policy throughout. The archived optimized HLO files reproduce each lane's receipt digests. The four lanes matched the sealed resident and deployed stock, fitted head digests and frozen before/after records. All fixed enrollment rows 64–127 were disjoint from the evaluated old target placements and eligible fitted-source geometry. The old results were excluded from the estimate and test. [GPU and HLO receipts](evidence/lanes/) · [Enrollment audit](evidence/enrollment-01.json)

## Execution and retention

Four exclusive RTX 4090 jobs on Baige Distributed Training exited successfully. Launcher time was 4,100 GPU-seconds in total, or 1.139 GPU-hours. Source pin `0db6cec5a66e3f7ed024bc87f9c4ffff2dd3c34d`. No additional models were fitted. [Launch and storage handoff](evidence/ready-01.txt) · [Scheduler jobs](evidence/submitted-01.txt)

The 3,871,965,121 bytes of native HDF5 tapes and per-replan receipts remain under `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-plate-zero-branch-20261124-r1/gpu01s{0,1,2,3}/shard{0,1,2,3}` on Baige shared storage. Use the matching lane and case directory. The original Linux readback and sealed look are `gpu01s3/audit/readback-128.json` and `gpu01s3/audit/look-128.json`, where 128 is the exclusive manifest index. The copies in this directory use the sample size, 64, and have identical file hashes. [Tape hashes and case records](evidence/readback-64.json)
