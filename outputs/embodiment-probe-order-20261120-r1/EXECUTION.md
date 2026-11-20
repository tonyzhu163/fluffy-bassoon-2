# Probe-order execution on Baige

Times are Shanghai time on November 20, 2026.

| Time | Observation |
| --- | --- |
| 15:49 | The existing `mwj-dangerous-dogs` Jupyter terminal verified write/readback on the GPFS project mount under this run's `setup/` directory. The filesystem had 417 GiB available of 2.0 TiB and 1,683,518 free inodes. The container lacked both `quota` and `mmlsquota`, leaving the per-project quota unavailable. The archived analysis checkout returned `b1eb9dd72be6cff3dcedfb256ecd47e176dc25e6`. Its Git common directory is the checkout's own `.git`. |
| 15:52 | Opened a draft from the successful H1 native02 Distributed Training job on Beijing Pro6K queue `算法Z组` (`aihcq-rh0wwk9muei4`). The refreshed queue reported 2,184 free CPU cores, 23,547 GiB free RAM and zero queued tasks. The draft requests one PyTorch worker, zero accelerator cards, 4 CPUs, 16 GiB RAM, 4 GiB shared memory, and the existing writable PFS mount at `/mowenjie`. The scheduler limit is 119 minutes with automatic retry off. |
| 15:55–15:57 | Root supplied `383e291254940bab51388fcd2333f22c00a27530`. The 74,421-byte incremental Git bundle matched SHA-256 `e5f9a5722e7d575c330ffbb1204cc6913150ad48156e1f415da557be39883483` locally and on PFS. Verified the fresh detached worktree at `setup/source-383e2912` using the archived CPU analysis checkout's common directory. The checked-out launcher, collector and scorer matched their local SHA-256 digests. |
| 15:57:54 | Baige admitted `job-owxei8dm5dvv` (`mwj_1120_probe_order_native01`, owner `mowenjie`) on Pro6K `算法Z组`. Scheduler readback showed one worker, zero cards, 4 cores and 16 GiB. The native02 shell sets up the system Mesa vendor file if needed, then calls the pinned launcher under `timeout -k 10s 118m`. |
| 15:58–16:01 | Worker `job-owxei8dm5dvv-master-0` began at 15:58:03. It measured 4 cgroup CPUs, 17,179,869,184 bytes RAM and zero NVIDIA card nodes. Python 3.11.14 / NumPy 1.26.4 loaded, the Mesa manifest passed, and the renderer reported llvmpipe (LLVM 15.0.7, 256 bits). At 15:59:30 the raw file held 55 completed tapes. At 16:01:21 it held 140/384 tapes (2,240/6,144 actions). |
| 16:03–16:04 | The live file held 223 tapes at 16:03:05 and 293 at 16:04:36. |
| 16:06 | Collection completed at 16:06:35 with 384 tapes and 6,144 returned native actions. The frozen scorer's synthetic self-test passed. Scoring ran from 16:06:37 to 16:06:38 and the worker exited 0 at 16:06:38. Baige reported `Succeeded` at 16:06:55. Elapsed time from admission was 9m01s on one attempt, using zero GPU-hours. |
| 16:08 | Retrieved the 269,627-byte compact archive through the authenticated Baige session. Its SHA-256 matched the remote value `fa10c7575164756892861da3ea61ce1b1609db7e7cfaf40948b54254648750ff`. All ten payload files passed checksum verification after transfer. |

## Results

The scorer validated 128 blocked supports, 128 shuffled supports and 128 shared queries across 16 paired layout indices. It scored 1,664 query rows. Lower normalized query motion MSE is better. The primary comparison uses the frozen physical reader with matching support.

| Family | Blocked | Shuffled | Blocked minus shuffled | Fixed common control |
| --- | ---: | ---: | ---: | ---: |
| Panda | 0.720325 | 0.717644 | +0.002681 | 0.863632 |
| Sawyer | 1.017404 | 1.014453 | +0.002951 | 1.101331 |
| Pooled | 0.868865 | 0.866049 | +0.002816 | 0.982481 |

The pooled paired-layout 95% bootstrap interval for blocked minus shuffled was [−0.001598, +0.007581]. The registered decision was `prefer_blocked=false`. Both schedules predicted query motion more accurately with matching support than with the fixed common, swapped-yaw or swapped-gain controls on each arm. Pooled shuffled-support gains over these controls were 0.11643, 0.11641 and 0.03410 normalized MSE respectively. The secondary lag-ridge reader gave a pooled blocked-minus-shuffled difference of −0.04864 with a 95% interval of [−0.11418, +0.00677]. These endpoints measure motion prediction. [Frozen report with per-case scores](evidence/native01/score/report.json)

## Evidence readback

The pinned launcher ran `collect.py --out "$out/collection"`, `score.py --self-test` and `score.py --tapes "$out/collection/tapes.jsonl" --out "$out/score"` in that order. Local readback checked the archived launcher, collector and scorer hashes against the pinned Git objects, checked per-case donor assignments and reconstructed the score summary means from the returned rows. The worker saw 414 GiB of GPFS space and 1,617,346 free inodes. [Launcher](run.sh) · [Source digests](evidence/native01/source.sha256) · [Allocation](evidence/native01/allocation.txt) · [Renderer](evidence/native01/renderer.log) · [Worker log](evidence/native01/run.log) · [Self-test](evidence/native01/self-test.log) · [Exit receipt](evidence/native01/exit.txt)

The full 3,936,601-byte raw tape file remains on Baige at `/mowenjie/fluffy-bassoon-2/outputs/embodiment-probe-order-20261120-r1/native01/collection/tapes.jsonl`. Its SHA-256 is `689087d25bf1862c2eea6946b5c8f44bf8889627b22a53a33097dfd3c2a23bde`. The local receipt manifest includes one record per tape and a digest of the exact raw line, including its newline. [Collection summary](evidence/native01/collection/collection.json) · [Per-tape native receipts](evidence/native01/native-receipts.jsonl) · [Raw file checksum](evidence/native01/tapes.sha256) · [Evidence checksums](evidence/native01/SHA256SUMS)
