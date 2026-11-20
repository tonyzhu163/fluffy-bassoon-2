# Source correction gate on Baige

**The frozen Huber motor had the most native successes on both arms.** On the 100 paired development starts, the gate rescued zero raw failures and lost five raw successes. All 128 production training pairs, the residual and gate fit, and all 300 development episodes completed. Baige job `job-sc2k6rvp1v69` succeeded at 19:48:31 Shanghai on November 20, 2026. The three allocations together used **0.758 of 4 GPU-hours** on the conservative submission-to-final-status ledger. [Scheduler](evidence/scheduler.json) · [Study result](RESULT.md)

## Paired development results

Each controller ran on the same 50 registered Goal 5 BASE starts per arm, seeds 740000–740049. The success criterion was the simulator's terminal reward. The gate could switch once, after 60 issued controls. Nine Panda starts terminated before that decision. [100 raw per-case records](evidence/native03/readback/eval-cases.csv) · [Native evaluation](evidence/native03/evaluation.json)

| Arm | Frozen raw | Always correct | Gated | Switched / all starts | Switched / actionable starts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Panda | 33/50 | 30/50 | 30/50 | 5/50 (10%) | 5/41 (12.2%) |
| Sawyer | 20/50 | 0/50 | 18/50 | 11/50 (22%) | 11/50 (22%) |

| Paired comparison | Panda gains / losses / ties | Sawyer gains / losses / ties |
| --- | ---: | ---: |
| Always correct against frozen raw | 5 / 8 / 37 | 0 / 20 / 30 |
| Gated against frozen raw | 0 / 3 / 47 | 0 / 2 / 48 |
| Gated against always correct | 5 / 5 / 40 | 18 / 0 / 32 |

On Panda, both constant controllers succeeded on 25 starts and both failed on 12. The gate avoided five of eight harmful corrections and missed all five available rescues. On Sawyer, both failed on 30 starts. The gate avoided 18 of the 20 harmful corrections. A hindsight selector between the two measured continuations could score 38/50 on Panda and 20/50 on Sawyer. [Per-arm counts and switch cross-tabs](evidence/native03/readback/readback.json)

## Training and checkpoint

In the successful attempt, the eight-slot Linux pilot collected on training seeds 360000–360007 and evaluated all three controllers on separate training seeds 360008–360015. The pilot used a separate fit directory. Production then independently executed 64 raw/teacher pairs per arm over seeds 360000–360063. The 64 donor files and manifest matched the earlier independent admissions byte for byte. The fitter enumerated only the production training manifest. [Donor audit](evidence/native02/layout-comparison.json) · [Pilot training cases](evidence/native03/readback/pilot-train-cases.csv) · [Pilot evaluation cases](evidence/native03/readback/pilot-eval-cases.csv)

| Arm | Raw successes | Teacher successes | Teacher gains / losses / ties | Actionable forks | Successful teacher suffixes / controls fitted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Panda | 39/64 | 29/64 | 5 / 15 / 44 | 58 | 23 / 509 |
| Sawyer | 33/64 | 24/64 | 13 / 22 / 29 | 64 | 24 / 1,507 |

The six early Panda successes contributed to both training outcome counts. The gate used signed teacher-minus-raw outcomes on 122 actionable forks. The bounded residual used 2,016 issued suffix controls. The production checkpoint was frozen at 19:34:22 before development execution. Its SHA-256 is `7c164a76073adfe7f878232b74210923684cf427e1426bf7d46e58014b3a563a` and the bytes matched again after scoring. [128 training records](evidence/native03/readback/train-cases.csv) · [Fit and exposure](evidence/native03/fitted/fit.json) · [Checkpoint](evidence/native03/fitted/model.npz)

## Action, reward and physical replay checks

The runner used the real frozen step-6400 Huber PaliGemma motor on one GPU. A single inference process ran with eight independent simulator children for full groups, five issued actions per replan, and a 300-control horizon. The original R2 simulator and VLM interpreters, evaluator revision `fc61bf2a`, GPU EGL renderer, checkpoint `77d18c386b2c…` and original 50-layout panel were verified. The GPU probe confirmed one exclusive NVIDIA RPBZZZ6 with 101,973,950,464 bytes. Telemetry recorded the GPU inference PID and its eight native child PIDs. [GPU probe](evidence/native03/gpu.json) · [Telemetry](evidence/native03/telemetry.log) · [Input checks](evidence/native03/input-verification.txt)

Each condition reset and stepped a separate native simulation. The matched prefix used the raw motor's recorded public packets and actions. The gate committed its choice on the common public fork packet before the first raw suffix action. The gated simulator then executed its selected controller using the measured public packets and proposals of that continuation. Each branch also recorded its own rendered observations. This protocol permits a physical replay check even when the GPU renderer produces different image bytes. [Repair protocol](FORK-REPAIR.md) · [Native interface](RUN-API.md)

Independent Linux readback rehashed every production tape and checked the joins, native indices, seven-dimensional issued actions, reward and success arrays, terminal flags, episode lengths, raw proposals and correction bounds. For all 100 development starts it checked the opening and shared prefix, including per-control integration and replan controller, clocks and public state. On all 91 actionable starts it verified the saved pre-outcome gate choice and fork input. On all 100 starts it verified the full gated action, reward and physical trajectory against the selected continuation. The compact archive contains all **63,439** issued development controls, segmented by arm, layout seed and controller. [Readback program](evidence/verify.py) · [Controls, raw proposals, indices and rewards](evidence/native03/readback/eval-controls.npz) · [Transferred file hashes](evidence/native03/SHA256SUMS)

Independent development rerenders produced 21 wrist-image hash differences across six Panda starts, five in the always-on branch and 16 in the gated branch. All verified physical matches held. Sawyer produced zero rerender differences. The historical raw archive is a separate reproducibility comparison. All 100 layout IDs and donors matched. Raw outcomes reproduced on 47/50 Panda and 50/50 Sawyer starts, and complete action arrays reproduced on 46/50 and 49/50, respectively. The historical Panda score was 34/50. Outcomes changed on seeds 740002, 740019 and 740024. The paired comparisons above use the contemporaneous raw controls. [Per-case archive comparisons](evidence/native03/readback/eval-cases.csv)

## Attempts, allocation and restart artifacts

SSH through `baige-dev` matched the requested host and port and reached the expected staging host. A fresh write/readback passed on the mounted project GPFS, with 398 GiB free at admission. Staging lacked a user-storage-quota command. The existing Distributed Training console verified queue and personal GPU availability before submission. Each attempt used one isolated Pro6K, 12 CPU cores, 96 GiB RAM and 8 GiB shared memory, with a separate directory and pinned detached Git checkout. [Storage](evidence/storage.txt) · [Actual allocation](evidence/native03/allocation.txt) · [Scheduler ledger](evidence/scheduler.json)

| Attempt and Baige job | Source | Submitted → final status, Shanghai | Conservative GPUh | Outcome |
| --- | --- | --- | ---: | --- |
| native01 · `job-iqtkovukpgsj` | `0c4370dd` | 18:56:25 → 19:05:07, released | 0.1450 | Single-layout pilot passed. First eight-slot production group stopped on Panda 360001. |
| native02 · `job-oayt8gwyuk34` | `7115fad0` | 19:08:34 → 19:14:17, released | 0.0953 | Eight-slot training qualification identified the first mismatch at the wrist image after 35 controls. |
| native03 · `job-sc2k6rvp1v69` | `2813211d` | 19:17:28 → 19:48:31, succeeded | 0.5175 | Training-only qualification, production collection, fit and development scoring passed. |

On the second attempt, the first differing motor action was control 36, with maximum change 0.02326. The third attempt preserved canonical public inputs and verified physical replay. The same layout then retained one rerecorded wrist-image difference and passed all physical prefix checks. All failure evidence, source pins and intermediate artifacts remain available. [First attempt](evidence/native01/exit.txt) · [Field-specific failure](evidence/native02/mismatch-readback.json) · [Repaired layout](evidence/native03/qualification-panda-360001.json) · [Successful deployment](evidence/deploy-2813211d.txt)

The successful attempt completed the training-only pilot at 19:21:47, production collection at 19:34:21, fitting at 19:34:22 and evaluation at 19:48:27. The scheduler showed the worker Running at the 20-minute check. At 19:43 the monitor saw 50 Panda and eight Sawyer development starts complete. SSH availability subsequently interrupted polling; the resumed monitor retrieved exit 0 and the remaining completed records at 21:39. The scheduler and stage timestamps place completion at 19:48. Total conservative allocation time was 2,728 single-card seconds. The three principal worker pods accounted for 2,538 seconds. [Stage times](evidence/native03/stages.tsv) · [Monitor log](evidence/native03/progress.jsonl) · [Exit receipt](evidence/native03/exit.txt)

The full shared run is `/mowenjie/fluffy-bassoon-2/outputs/embodiment-source-correction-gate-20261120-r1`. Under `native03/`, `layouts/` holds the donors, `train/` and `eval/` hold per-case numeric tapes and receipts, `eval/decisions/` holds precommitted gate decisions, `fitted/` holds the checkpoint, and `readback/` holds the independent audit and compact control archive. The full-state fingerprints, model inputs, outcome labels and raw arrays remain on PFS. All 20 transferred scoring, qualification and readback files passed SHA-256 comparison against Linux. [Receipt manifest](evidence/native03/SHA256SUMS)
