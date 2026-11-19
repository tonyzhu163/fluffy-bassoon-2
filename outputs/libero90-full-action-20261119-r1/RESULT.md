# The full-action LIBERO-90 study stopped during development

Baige marked `job-tl0r1fybgx8y` failed at 19:41:36 Shanghai on November 19, 2026. The frozen readers were ready. The native single-draw trajectory failed an input-equivalence check against the compute-matched trajectory on development case `task40-r03`. The 30-task test never opened. The data support an execution diagnosis and leave the effect of cross-episode practice unmeasured. [Terminal inventory, traceback, and scheduler accounting](failure-evidence.json)

| Stage | Complete registered cases | Episodes in complete cases |
| --- | ---: | ---: |
| Fitting archive, 48 tasks | 960 / 960 | 2,880 |
| Live development, 12 tasks | 66 / 144 | 462 |
| Live test, 30 tasks | 0 / 720 | 0 |

The four fitting shards finished successfully. They collected 959 new cases and reused the one completed distributed qualification case. Each development case required a practice and six live retries. The monitor counted 61 published cases at its error signal. Ranks 0–3 ultimately published 17, 17, 17 and 15. Four incomplete development directories contain nine additional complete episode records. The table counts only published whole cases. On those 66 development cases, independent checks verified 858 payload hashes and 462 native reward tapes. The base and matched arms agreed on all 4,397 decision-input hashes and their issued-command, wrist-position and reward trajectories. [Stage counts and registered task IDs](failure-evidence.json)

## Failure and recovery evidence

Rank 3 (`worker-2`) returned exit code 1 at 19:38:27 Shanghai. Development task 40, repeat 3 used practice pose 8, query pose 2, policy seed 27114106 and scene seed 7. The practice and compute-matched retry finished. The matched retry saved 80 decision-input hashes for its 400 controls. The native base passed the initial query-reset check, then a motor-input hash differed at the same replan ordinal. The base stopped before saving its tape or running a learned arm. Its traceback and the matched tape establish the mismatch. The base did not persist the failing ordinal or individual channel hashes, so the evidence cannot distinguish camera, state or trajectory divergence. Baige identifies this worker's exit as the job failure. [Traceback, partial files and hashes](failure-evidence.json)

The CPU fit exited successfully after 450 updates per recipe and seed (41, 42 and 43). Independent CPU readback found both ensembles tensor-identical to the corresponding three checkpoint members. The checkpoint files also contained optimizer, sampler and Torch RNG state. Both ensemble hashes matched the live development contract. The registered manifest has 48, 12 and 30 disjoint task IDs, with 20, 12 and 24 cases per task. All 1,026 published trial headers matched their registered rows, source pin and stage design. [Frozen weights, checkpoint hashes and split readback](failure-evidence.json)

The shared recovery root is `/mowenjie/fluffy-bassoon-2/outputs/libero90-full-action-20261119-r1`. It retains `corpus01/fitting`, `fit01/weights`, the four partial directories and the worker logs. The scientific source is `4750c0971e6ba9b467faa47aac0dd8641f7d5076` and the distributed launcher is `3e4038df3645b4e7f4bab9212bf735d398c06943`. A hash-verified local copy of the weights, all three resumable checkpoints and the failed case's surviving tapes is under `recovery-copy/` outside Git. [Artifact inventory](failure-evidence.json)

## GPU cost and remaining physical work

| Allocation | Accounting basis | GPU-seconds | GPU-hours |
| --- | --- | ---: | ---: |
| Three RTX 4090 development checks | Measured command time | 626 | 0.174 |
| Initial four-instance fitting launch, `job-6g40zc00z7m9` | 4 × 7 min pod lifetime | 1,680 | 0.467 |
| Runtime qualification, `job-woc2swgr5z28` | 1 × 3 min 08 s | 188 | 0.052 |
| Runtime qualification, `job-3xwv8arajvkw` | 1 × 2 min 53 s | 173 | 0.048 |
| Full panel, `job-tl0r1fybgx8y` | 3 × 3 h 51 min 34 s + 1 × 3 h 48 min 45 s | 55,407 | 15.391 |
| **Study total** | | **58,074** | **16.132** |

The distributed jobs used one Pro6K GPU per pod. The ledger sums scheduler-reported instance lifetimes, including startup and teardown. The panel's job-level running phase lasted 3 h 48 min 28 s. These measurements leave 48.868 GPU-hours against the approved 65 on this basis. These figures estimate allocated GPU time. Billed usage remains unverified. [Scheduler and command-time evidence](failure-evidence.json)

Recovery requires a GPU replay of the practice and both baseline arms with the failed case’s registered poses and seeds. Capture the first differing ordinal and compare the two camera inputs, proprioception and issued commands. Re-establish the registered native equivalence before releasing test. A same-design resume can verify and reuse the 66 published cases and must restart the four partial cases from their practice reset. The remaining 78 development and 720 test cases require 5,586 complete physical episodes. At the observed 103.24 seconds per published development case, that is about 22.9 GPU-hours of rollouts. The estimate depends on task mix and excludes diagnosis, provisioning and synchronization. This verification used CPU only. [Registered contract](PLAN.md) · [Resume behavior](IO.md)

## Interpretation

Neither registered split summary exists. The three ordered comparisons (outcome reader versus native base, versus its cleared form, then versus intact imitation) remain untested. Absolute test success, paired task-cluster 95% intervals, imitation's history gain and the between-recipe gain interaction are unavailable. The study tests whether an earlier, completed practice helps after a pose reset. HDH uses preceding queries within the ongoing episode and supplies no estimate for this cross-episode comparison. [Registered estimands](PLAN.md) · [HDH protocol and scope](../../docs/literature/frozen-vla-offline-icrl-review.md)
