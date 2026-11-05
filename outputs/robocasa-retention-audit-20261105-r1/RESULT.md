# RoboCasa retention audit — complete

November 5, 2026, Baige. Audited **79 existing historical run folders** from 93 selected paths (14 absent), inspected 243 compact result records, and removed **155.08 GiB of file allocation / 156,447 files**. Every reviewed deletion was independently checked absent; zero errors or skipped source folders.

| Removed material | Allocated GiB |
| --- | ---: |
| 11 extracted checkpoint copies, full hashes matched to verified cold members | 71.19 |
| Regenerable per-run vLLM/Triton/FlashInfer compiler caches | 30.36 |
| Redundant interface-screen/capacity-test videos | 24.37 |
| Closed evaluation environment chunks outside retained first banks | 19.62 |
| Repeated initial-state snapshots outside retained samples/banks | 9.54 |

The checkpoint copies came from the stopped checkpoint-selection sweep and migrated original 512 test. Their cold receipts, archive sizes, exact member identities and prior complete archive readbacks were reconciled; the private copies themselves were fully hashed before deletion. No new full-run archive was created. Other retired raw payloads were deleted under the user's cleanup instruction.

| Largest reductions by run | Allocated GiB removed |
| --- | ---: |
| `robocasa-flow-sde-checkpoint-selection-20261104-r1` | 78.10 |
| `robocasa-interface-screen-20261102-r3` | 25.70 |
| `robocasa-flow-sde-endpoint-test512-20261104-r1` | 15.80 |
| `robocasa-opendrawer-flora-resume-20261102-r1` | 3.54 |
| `robocasa-opendrawer-flora-first-20261102-r1` | 3.54 |
| `robocasa-flora-five-task-fifty-score-20261102-r1` | 3.54 |
| `robocasa-flora-throughput-three-view-20261102-r2` | 3.52 |
| `robocasa-flora-throughput-20261102-r2` | 3.49 |
| `robocasa-endpoint512-acceleration-20261104-r1` | 3.14 |
| `robocasa-interface-screen-20261102-r1` | 1.84 |

Retained **55,822 regular files / 27.97 GiB** in these reviewed folders, plus directory metadata and symlinks: original code, configs, logs, per-case results and audit receipts; first evaluation banks, canonical reset and sampler records; 1,080 representative screen videos; all Flora training/calibration videos; model/recovery checkpoints and numerical diagnostic tensors. The 27 retained source-code files passed full SHA256 checks; all 55,830 retained file/symlink identities were independently rechecked. Referenced calibration videos and qualification metadata remain available.

Live meta-learning, new source/evaluation work, held steering/critic recovery, Flow10M/FlowR4 teacher corpora, common-before calibration data, shared environments and shared model caches were preserved. No live artifact tree was refreshed, no accelerator computation/job was launched, and no other owner's worktree or job was changed.

PFS free space was **507.20 GiB** at 2026-11-05T11:22:43+0800. It increased 148.32 GiB during the deletion/verification interval while other jobs continued writing; this global free-space delta is separate from the exact allocation removed.

[Final manifest](MANIFEST.json) · [Reviewed decisions](REVIEW.json) · [Dependency review](DEPENDENCY-REVIEW.json) · [Cold recovery proofs](RECOVERY-PROOFS.json) · [Deletion summary](PRUNE-SUMMARY.json) · [Linux qualification](PRUNE-QUALIFICATION.json)

The private per-file inventories, review excerpts and 21,889 directory-group receipts remain in the Baige audit folder. They were reconciled against all 156,447 deletions and retained-record identities; bulky/raw source logs were not committed.
