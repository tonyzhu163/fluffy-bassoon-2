# Meta-World released pi0.5 noise screen

Status checked 2026-10-29 19:42 Asia/Shanghai: Baige
`job-orgyjb0z9h6d` running. The visible-goal reach condition completed all
**544 episodes**, independently audited. Native success is 31.25%; split-half
noise-arm selection reaches 37.5%, a 6.25-point gain below the 15-point gate.
Two of four goals remain unreached by any arm. A read-only snapshot of the owned worker confirms **425/544 hidden-goal
episodes completed** (last row424), with the next rollout at step155. Its
JSONL remains buffered until the condition closes; no exit code is present. [Partial results and limits](PARTIAL-RESULTS.md); the overall screen
and final motor fingerprint remain pending. Both GPU probes and all nine
focused Linux checks previously passed, with zero skips.

The authenticated WebShell produced the closed [19:42 worker receipt](checks/worker-progress-194207.json). A separate scheduler-detail read became unavailable when the browser connection closed; this check therefore relies on actual owned-worker progress and saved artifacts.

Scientific source: `4d4f46f4cdcd8cfd7c7056e95bf85d0a90e0ed59`, reviewed from
main and deployed through verified Git bundles into a fresh detached worktree.
The Meta-World scientific implementation and focused tests are byte-identical
to their earlier eight-test CPU qualification at `428335c`. No active source
worktree was changed. The plan is
[the released-motor noise screen](../libero-noise-icrl-20261029-r1/PLAN.md).

| Item | Contract |
| --- | --- |
| Allocation | One RPBZZZ6 GPU, 16 CPUs, 96 GiB RAM, 8 GiB shared memory |
| Scheduler cap | 175 minutes; automatic retries disabled |
| Combined budget | Five minutes conservatively reserved for the failed r1; r1 plus r2 at most 180 GPU-minutes |
| Checkpoint | `RLinf/RLinf-Pi05-MetaWorld-SFT`, revision `60b2d0309af8bb087c9e9b250f8c7f9db50c0f8e` |
| Model SHA-256 | `6e877ae8e2c9c33a2b8d1e3a8a60649ef39b0268da60a3ee71bc5e33cbca7221` |
| Runtime | Existing `envs/metaworld-3.1.1-r1`, managed with uv; local Transformers 4.53.2 and the five matching OpenPI replacement files |
| Execution | Eager inference (`TORCH_COMPILE_DISABLE=1`), five flow steps; inference and MuJoCo EGL selected on allocated device 0 |
| Collection | Reach-v3 first, visible then hidden goals; push-v3 only within the same deadline |
| Per task/visibility | Four goals × four scenes × (native + 16 fixed arms) × two repeats = 544 episodes, each at most 160 task steps |
| Training | None; frozen motor and fixed noise arms |

The [r1 failure and repair](../metaworld-pi05-noise-baige-20261029-r1/RESULTS.md)
are preserved separately. Runtime changes happened after r1 terminated and
before r2 launched. No protected shared runtime files were modified. The
actual source, model and runtime receipts remain in the remote preparation
directory. [launch-receipt.json](launch-receipt.json) records the allocation
and budget; [launch.sh](launch.sh) is its exact launcher.

Remote run/output root:
`/mowenjie/fluffy-bassoon-2/{worktrees,outputs}/metaworld-pi05-noise-baige-20261029-r2`.
Inspect `exit-code.txt`, `qualification-tests.log`, `s0/contract.json`, the
per-condition outcome files and `s0/summary.json` before interpreting progress.
The final audit must check complete paired rows, repeated outcomes, mode
effects, goal coverage, G1–G4 and frozen motor fingerprints. Partial collection
does not establish passing gates or in-context learning.

The live JSONL remained empty despite advancing worker counters. Read-only
inspection found the writer's OS file position still zero: rows were buffered
in Python. The filesystem reports 4 MiB I/O blocks, and the script does not
explicitly flush each row. Its normal condition boundary closes the file.
Until then, use [worker_progress.py](worker_progress.py) only inside the exact
owned container; it verifies the process/cwd and records selected counters
through a read-only profiler. [worker-progress-185618.json](worker-progress-185618.json)
and [worker-progress-190813.json](worker-progress-190813.json) preserve the
observed counters. Full raw outcomes must be audited once the
condition files close. The active scientific worktree was not modified.

The chat heartbeat `monitor-metaworld-pi05-noise-screen` checks every 20 minutes,
reports meaningful results or failures, and stops after final audit. Other
chats' LIBERO allocations have separate ownership.
