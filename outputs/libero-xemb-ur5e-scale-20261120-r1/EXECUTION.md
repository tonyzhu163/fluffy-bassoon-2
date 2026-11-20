# Execution Record — `libero-xemb-ur5e-scale-20261120-r1`

## Cluster & Runtime
- **Compute:** Baige Distributed Training (`1 × 8 NVIDIA RPBZZZ6`, `64` CPU, `320` GiB RAM, `32` GiB SHM, queue `算法Z组 / aihcq-rh0wwk9muei4`, `NCCL_P2P_DISABLE=1`).
- **Image:** `ccr-registry.baidubce.com/aihc/nvcr.io/nvidia/pytorch:2.7.0-cu12.8.0-py3.12-ubuntu24.04` with `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.
- **Code Revision:** `d597ee0f1e99937bbeb2705132d709b514a4b5c2` (Linux unit tests `3 passed` + 8-shard plan verified before launch).

## Stage Receipts
| Stage | Job ID | GPUs | Wall-Clock | Scheduler GPUh | Exit | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 24-Task CPU Admission | `baige-dev` | `0` | `114 s` | `0.000` | `0` | `Panda 24/24`, `Sawyer 24/24` |
| Linux Unit Tests + Import Check | `baige-dev` | `0` | `14.6 s` | `0.000` | `0` | `3 passed`, `1440 = 8 × 180` |
| S1 Collect + Train (`gpu01`) | `job-6prgb2hkbe8c` | `8` | `1489 s` (`24m49s`) | `3.31` | `0` | [offline-audit-compact.json](evidence/offline-audit-compact.json) |
| S2 Live Eval (`gpu02`, `1,056` eps) | `job-5s10a9cdqh77` | `8` | `2161 s` (`36m01s`) | `4.80` | `0` | [live-eval-summary-compact.json](evidence/live-eval-summary-compact.json) |
| **Total** | | | **`60m50s`** | **`8.11` GPUh** | **All `0`** | [receipts.json](evidence/receipts.json) |

## Integrity
- Both 8-GPU jobs passed `scripts/[train]gpu_probe.py --expected-count 8` (`status: passed`, `gpu_count: 8`).
- Compact summaries preserve full-file SHA-256 hashes of the raw PFS JSONs (`offline-audit.json`, `live-eval-summary.json`).
- Zero official test split (`SPLITS.json["test"]`) tasks were accessed or evaluated.
