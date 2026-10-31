# Source40 runtime qualification

Read-only artifact audit at 2026-10-29 22:26 Shanghai. Both Baige jobs exited
zero; their original output directories and complete recoveries remain intact.
The [receipt](receipts.json) independently verifies final paired dataset,
source-checkpoint and full-recovery SHA256 values, episode/case counts and
interaction totals. The copied indices are exact saved source records.

| Partition / Baige job | Episodes / successes | Completed cases | Query chunks | SAC updates | Primitive + settling | Worker seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train `job-f0gv7b32ers5` | 60 / 59 | 15 | 606 | 2,220 | 5,796 + 610 | 642.16 |
| Validation `job-1g94ufvin6wa` | 68 / 68 | 17 | 617 | 2,400 | 5,848 + 690 | 666.66 |

- Both use source task `libero_spatial:00`, official frozen OpenPI40-task
  `pi05_libero`, receipt `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.
- Train source `b48b3da95021d369f2cc9ede7cec8b5d07db7a83`; validation source
  `1650ae2075646c5748ced929032e5fc0df06c10b`. No active worktree was migrated.
- Train visited nominal states 0–14, fifteen distinct state-vector hashes.
  Validation visited states 40–49, ten distinct hashes and seventeen seeded
  cases. Each case has four episodes; repeated seeds are not extra nominal
  placements. The partitions have independent learners/replays and disjoint
  nominal state IDs.
- Train observed 503 Gaussian warmup chunks and 103 SAC-behavior chunks;
  validation observed 505 and 112. Optimizer updates happen after complete
  episodes; weights and behavior stage stay fixed within each episode.
- Final actor/critic losses are finite. This establishes real updates and
  exports, not an improving source policy or expert-label competence.
- Focused Linux tests: train 133 passed/2 skipped; validation 138 passed/2
  skipped. The actual OpenPI runtime separately passed typed and legacy JAX
  PRNG serialization round-trips. Skipped JAX checks belong to the separate
  probe/test environment.
- Actual allocation GPU is **NVIDIA RPBZZZ6**, approximately 102 GB. Earlier
  generic RTX6000 descriptions do not identify the measured hardware; no
  RTX6000 throughput comparison follows from these timings.
- These runs stop after reaching 600 query chunks and finishing a case. They
  do not exhaust all scheduled cases and do not collect all40 tasks. The new
  `complete_cases` route exhausts the requested case schedule.

Across both source partitions: 128 episodes, 1,223 query chunks, 11,644
primitive +1,300 settling actions, and 4,620 source optimizer updates. Source
generation is counted once for AD and DPT. Teacher histories and source
optimization are separate from the deployment support-shot budget.

This easy source task succeeded on 127/128 episodes. No frozen controller has
been evaluated under the new source40 protocol, and no adaptation, memory,
sample-efficiency or final-performance advantage is established. The next
technical check assembles the paired histories and trains matched full/static
controllers on an allocation. The substantive experiment still requires the
all40 source corpus and separate-query target evaluations.
