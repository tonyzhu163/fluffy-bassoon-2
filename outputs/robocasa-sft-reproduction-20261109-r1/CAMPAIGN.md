# Frozen native 50-task reproduction campaign

The native batch-one inference server and four simultaneous simulator clients
passed GPU04 qualification. Campaign01 extends that qualified path without
changing model precision, inference batch, camera/state transforms or actions.
Original checkpoint, norm stats and public source pins remain as in PLAN.md.

- Eight exclusive Baige containers, each one allocated GPU, 21 CPU, 224 GiB
  RAM and 16 GiB shared memory; six-hour scheduler maximum; retry disabled.
- 50 episodes per task, seed 7, pretrain scenes/objects, H50 prediction,
  five executed controls and ten ODE steps. No optimization or training.
- Static manifest covers all 18 atomic and 16+16 composite tasks exactly once.
  All 317 registry horizons were checked; primary allowance is the new registry
  (equivalent to the old registry times 1.5); old cutoff is also saved.
- Stage 1 is all 900 atomic episodes. Only a complete aggregate within five
  percentage points of 39.6% continues to composites. Outside that prospective
  practical band, workers close and save sampler state for a fresh 1.0.0
  comparison. This screen does not establish statistical equivalence.
- Stage 2 is all 1,600 composite episodes, four simulator processes per GPU.
  Native inference remains batch one. Eight separate sampler streams and
  interleaved requests mean this is aggregate reproduction, not exact replay
  of the original evaluator's random-number sequence.
- Each rank verifies its in-step GPU probe; all eight distinct UUIDs and matching
  source/manifest/job identities are required before model loading. Peer failure
  closes the other workers; no automatic retry. Fresh output identity required.
- Every completed case saves success time, both horizon outcomes, action digest,
  input hash and input NPZ. Selected videos and server RNG state are retained.
  Deadline is 5.7 hours after worker start with finite episode closure reserves.

Worst-case controls are 4,522,500. Measured qualification throughput projects
33.50 GPU-hours before task-mix/reset/startup overhead. Reservation is 48
GPU-hours plus a conservative 0.267-hour startup/release allowance. Closed
qualification allocations cost 908 GPU-seconds (0.2522 GPU-hours), giving a
maximum reservation charge of 48.519 GPU-hours, within this task's cumulative
50 GPU-hour limit. Monitor actual cumulative usage and advancing cases at least
every 20 minutes. Do not open a further allocation without recomputing the
remaining budget.

Manifest and launcher are committed and deployed to a fresh detached checkout
before scheduling. Immutable active source is never updated in place.
