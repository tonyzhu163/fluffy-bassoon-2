# Independent terminal verification: DPT validation teacher binding

Verified 2026-10-31 Shanghai from read-only Baige saved records.

- Job `job-z9tjwlkovscv`; immutable producer
  `850dd96f6e8ff9b96076c7da4618f175dc473688`, verified clean.
- Run `libero40-dpt-same-train-teacher-cpu-20261031-r1` has completed index,
  exit code 0 and finished worker clock. Charge: 348 conservative CPU-worker
  seconds, zero GPU seconds. Before masking, the actual allocation attested
  zero CUDA devices, CUDA uninitialized, and cgroup limits of four CPU cores
  and 16 GiB. No simulator actions, VLA inference or optimizer updates.
- All 49 focused tests passed in 16.49 seconds on that allocation.
- Eight derived validation goals contain 8,202 rows; both TRAIN objectives
  retain 40 histories and 33,729 rows. All use 4,104 observation features and
  32 noise coordinates. The metric is explicitly
  `heldout_state_same_train_teacher`.
- The allocated final verification reports exact unchanged bytes for all
  TRAIN AD/DPT fields and validation AD fields. Every derived validation
  history reports exact unchanged non-target fields. Source-history exposure
  and the original 457,851-action collection charge remain unchanged; the
  eight TRAIN teachers are not charged again.

All eight actor-unchanged and TRAIN-label replay proofs are present. Pure actor
digests are distinct across the eight goal-specific teachers and are recorded
in full in the JSON receipt. Maximum TRAIN first-row replay difference was
`3.2782554626464844e-7`, below the declared `1e-5` CPU/backend tolerance.

| Goal | Pure actor digest prefix | TRAIN replay max absolute difference |
| --- | --- | ---: |
| libero_spatial:00 | 4c145c442f40 | 2.3841858e-7 |
| libero_spatial:01 | 2a11ea0b8767 | 1.4901161e-8 |
| libero_object:00 | b6fffd5eecc0 | 3.2782555e-7 |
| libero_object:01 | 72fbebaeddb6 | 3.7252903e-8 |
| libero_goal:00 | e2445c3a83a7 | 2.9802322e-8 |
| libero_goal:01 | 6bef36d5daf8 | 1.4901161e-8 |
| libero_10:00 | e1e9e482b980 | 4.4703484e-8 |
| libero_10:01 | 03b869d5408e | 8.9406967e-8 |

The independent verifier rehashed 34 saved files, including the complete
binding, combined datasets and sidecars, all eight derived datasets/sidecars,
original admission and terminal evidence. All 25 supplied reference hashes
matched; nine additional file hashes were captured. The assembly content hash
also recomputed exactly.

- Binding file SHA256:
  `9da01d662b45fe3bdc584cceb1dd55732527f9babe9ff0ef0db93d1cae1a1b23`.
- Assembly file SHA256:
  `1d7a93c8c0ec64a92687003058dac372bf6c19ab7c871aa1e9d218e624b4c29e`.
- Assembly content SHA256:
  `05cb614a21805e2dee390fadc36a278f05704169f1c8c764942fc107f0a4bf0b`.

This is successful data derivation and admission, not a student-training or
robot-performance result. Array and actor parity proofs came from the actual
allocated execution; this independent pass checked its saved metadata and file
bytes without constructing actors, running inference, rerunning tests or
launching an allocation. Original validation publications and labels remain
preserved with their legacy teacher-transfer interpretation. No collection
launch depends on this receipt.
