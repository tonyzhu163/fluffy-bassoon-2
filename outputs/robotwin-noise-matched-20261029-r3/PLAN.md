# Asset-complete actual training retry

**Superseded by [r4](../robotwin-noise-matched-20261029-r4/RESULTS.md), the completed run.**

2026-10-29 Asia/Shanghai. Source `5f2d1c4fc7c1a5e93010aa530d9ab82234edd4fa`
merges the reviewed latest remote main `26d8be7` and preserves the matched
six-task experiment contract in r1. Prior r2 failed before any training
transition; this is a fresh learner run, not a checkpoint continuation.

Install the full official object archive at dataset revision
`3dc3b798668feb99ac61cc9086d84cbcc3d79186`, SHA256
`6aa56b3cf1e1064f7c809308144da36b00815f8b137fef2d7e4de856f8becf27`.
Verify byte count, archive SHA, every extracted CRC, and preserve every existing
matching shared asset. A static preflight checks every archived file in every
object family referenced by all 12 source/target tasks before motor loading.
Process-local setup/reset wrappers retain successful seed order and bound
retries at 32 attempts, exposing the final exception. External sources stay
unchanged and pinned. No source/target sampling or learning hyperparameter
was changed.

Run focused Linux tests and real motor/noise parity inside the actual GPU
allocation, then automatically collect the original source training. One GPU,
16 CPU cores, 128 GiB RAM, 8 GiB shared memory, **28-minute scheduler cap**.
Driver budget 1,200 seconds, with 180-second trial-boundary recovery reserve;
external driver timeout 1,380 seconds after up to 300 seconds of test allowance.
Original source cap remains 10.24M executed actions per learner, K4 training,
K8 evaluation at indices 1/2/4/8. A wall stop is incomplete training.

Combined ceiling: r1 two-minute reserve + r2 measured 14m49s + r3 28m =
44m49s, below the original 45 GPU-minute envelope. Do not launch a materially
larger continuation without measured throughput and the user's approval.
