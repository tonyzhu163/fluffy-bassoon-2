# Frozen native baseline reproduction campaign01

The native batch-one server and four simultaneous simulator clients passed
GPU04 qualification. Campaign01 uses that qualified path for the complete
18-task atomic set. All policy/checkpoint/source settings remain as in PLAN.md.

- Fresh queue snapshot: three idle NVIDIA RPBZZZ6 GPUs, no pending queue.
  Allocate three exclusive containers, each one GPU, 21 CPU, 224 GiB RAM,
  16 GiB shared memory; four-hour scheduler maximum; retry disabled.
- All 18 Atomic-Seen tasks, 50 episodes each, seed 7, pretrain scenes/objects,
  H50 prediction, five executed controls, ten ODE steps, native batch one.
  Balanced task shards run waves of at most four simulator processes.
- Primary allowance is the new registry (old registry times 1.5). Preserve
  success at the old cutoff too. Zero optimization or training.
- All 900 complete cases are required for the prospective practical screen:
  within five percentage points of 39.6% -> prepare the remaining 1,600
  composite cases with a fresh capacity/budget check; otherwise -> fresh
  detached 1.0.0 comparison. Workers close after this atomic stage either way.
  This screen does not establish statistical equivalence.
- Each rank verifies its actual in-step GPU probe. Three distinct UUIDs and
  matching source/manifest/job identities are required before model loading.
  Peer failure closes remaining workers; no automatic retry.
- Each completed case records success time, both cutoffs, action/input hashes
  and first input NPZ. Selected videos and native sampler RNG are retained.
  Worker deadline is 3.7 hours with finite episode closure reserves.
- Separate sampler streams and interleaved requests establish aggregate
  reproduction, not exact replay of the original evaluator RNG sequence.

The full 50-task manifest is preserved as FULL-MANIFEST.json. The active
manifest covers only this complete atomic stage. Qualified collection rate
is 37.50 controls/second; task mix, waves and reset costs may change throughput.
This stage reserves 12 GPU-hours plus 0.10 hour startup/release allowance.
Closed qualification cost is 908 GPU-seconds (0.2522 hours). Subsequent jobs
must be sized from actual closed usage within the cumulative 50 GPU-hour cap.

Commit and deploy through Git before scheduling; never edit active sources.
Inspect advancing episodes at least every 20 minutes through closure.
