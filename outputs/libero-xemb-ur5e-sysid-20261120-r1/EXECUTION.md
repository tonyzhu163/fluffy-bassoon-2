# Execution Record — `libero-xemb-ur5e-sysid-20261120-r1`

## Cluster & Runtime
- **Compute:** Baige Distributed Training (`NVIDIA RPBZZZ6`, queue `算法Z组 / aihcq-rh0wwk9muei4`) + Baige CPU dev (`baige-dev`).
- **Image:** `ccr-registry.baidubce.com/aihc/nvcr.io/nvidia/pytorch:2.7.0-cu12.8.0-py3.12-ubuntu24.04` with `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`.
- **Policy:** Frozen off-the-shelf $\pi_{0.5}$ (`openpi`, zero fine-tuning).

## Stage Receipts
| Stage | Job / Host | Commit | Wall-Clock | Scheduler GPUh | Exit | Evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| CPU Admission (`cpuprobe01`) | `baige-dev` | `a5ac693d` | `148.7 s` | `0.000` | `0` | [admission.json](evidence/admission.json) |
| Linux Unit Tests (`cputest02`) | `baige-dev` | `1dc4dffe` | `0.67 s` | `0.000` | `0` | `2 passed` |
| CPU Gate 0 (`cpugate02`) | `baige-dev` | `6c16d9f8` | `443 s` | `0.000` | `0` | [gate0-summary.json](evidence/gate0-summary.json) |
| Pro6K Gate 1 (`gpu01gate1`) | `job-34nf57gggb4z` (1× Pro6K) | `6c16d9f8` | `997 s` (`19m11s` job) | `0.320` | `0` | [gate1-summary-compact.json](evidence/gate1-summary-compact.json) |
| Pro6K Stage B1 (`gpu02col` + `gpu02fit`) | `job-2aymxz7y0alu` (1× Pro6K) | `d639787a` | `1795 s` (`30m10s` job) | `0.503` | `0` | [offline-audit-compact.json](evidence/offline-audit-compact.json) |
| Pro6K Stage B2 (`gpu03eval`) | `job-omd2j5x1wj0f` (1× Pro6K) | `62854a88` | `5392 s` (`89m52s` job) | `1.498` | `0` | [live-eval-summary-compact.json](evidence/live-eval-summary-compact.json) |
| **Total** | | | | **`2.321` GPUh** | **All `0`** | [receipts.json](evidence/receipts.json) |

## Integrity
- Compact summaries preserve full-file SHA-256 hashes of the raw PFS JSONs (`gate1-summary.json`, `offline-audit.json`, `live-eval-summary.json`).
- Zero official test split (`SPLITS.json["test"]`) tasks were accessed or evaluated.
