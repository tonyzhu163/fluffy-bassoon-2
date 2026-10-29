# Source40 collection: measured proposal

2026-10-29, Baige. **Proposal only; no full 40 collection launched.**
Collect all 40 official source tasks with40 train cases and20 validation cases
per task, four real episodes per case. Request a **160 GPU-hour collection cap**
and **1.0 TiB storage reservation**, with at most eight independent single-GPU
workers (16 CPU/96 GiB RAM each). These are planning limits, not a performance
claim or permission to launch. Controller training and target evaluation need
separate budgets.

## Exact interaction and state schedule

The reviewed complete-case plan is
[`../libero40-all 40-plan-20261029-r1/schedule.json`](../libero40-all 40-plan-20261029-r1/schedule.json),
content hash `06cdf07de074da60cf8619d21013e7364f9ee80c47b1e2e757531a3aefb1bd67`.
It uses source generator/collector code `1650ae2075646c5748ced929032e5fc0df06c10b`.

- 40 tasks ×60 cases ×4 episodes = **2,400 cases /9,600 episodes**.
- Train:40 distinct nominal state IDs0–39 per task, one seeded case per ID.
  Validation:10 distinct IDs40–49 per task, two seeded cases per ID.
  The validation20 cases are **10 nominal placements**, not20 placements.
- Horizon400, native/executed chunk10: **384,000 query chunks /
  3,840,000 primitive actions maximum**, plus **96,800 settling actions**
  with one qualification reset per task/partition. Failed/interrupted physical
  actions remain charged by the execution journal.
- Fresh independent SAC/replay per task+partition:80 source learning histories;
  actor/critic/replay persist through that partition's cases. Batch256,
  UTD20, warmup>500 chunks, optimizer updates between complete episodes.
  Four episodes/case is not the source teacher's total history length.
- All requested cases must complete. No chunk-target early stop. Report actual
  initial-state hashes and case coverage; seeds do not establish new states.
- The source40 data stays separate from full LIBERO90/Plus/Pro evaluation.
  No target-suite trajectory enters source pretraining.

## Completed calibration data

Both jobs exited0 and exported paired AD/DPT prefixes plus complete recoveries.
These are one-task runtime qualifications, not full 40 collections or evidence
that a controller learns to adapt.

| Source qualification | Train | Independent validation |
| --- | ---: | ---: |
| Job | `job-f0gv7b32ers5` | `job-1g94ufvin6wa` |
| Exact code | `b48b3da95021d369f2cc9ede7cec8b5d07db7a83` | `1650ae2075646c5748ced929032e5fc0df06c10b` |
| Task | `libero_spatial:00` | `libero_spatial:00` |
| Actual cases /episodes | 15 /60 | 17 /68 |
| Distinct nominal state IDs /hashes | 15 /15 | 10 /10 |
| Distinct initial observation hashes | 15 | 17 |
| Successes | 59/60 | 68/68 |
| Query chunks | 606 | 617 |
| Primitive /settling actions | 5,796 /610 | 5,848 /690 |
| Actual SAC updates | 2,220 | 2,400 |
| Worker wall seconds | 642.16 | 666.66 |
| Probe-to-exit seconds | 670.56 | 711.07 |

The validation17 observation hashes include different seeded observations of
10 nominal state vectors. Do not count these as17 distinct placements.
Training has one failure; validation has none. A high success rate here verifies
execution and leaves little room to demonstrate recovery. The saved final
source actors are candidates, not established expert teachers.

The actual allocation probe identifies **NVIDIA RPBZZZ6,101,973,950,464 bytes**
(about102 GB decimal), with `timing_comparison_eligible=false`. These measurements
must not be called RTX6000 timings. The official OpenPI pi05_libero motor receipt
is `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.
No RLinf130 motor/data/features enter this estimate.

## Compute projection

Train journal timing separates340.17 seconds of collection windows,
172.88 seconds of postepisode gaps and129.10 seconds of startup/unattributed
work. The first SAC-update episode has a111.52 second postepisode gap; later
mixed gap excess is approximately17.82 ms/update. Validation independently
measures18.29 ms/update. These gaps include synchronization and disk work;
they are not isolated GPU optimizer timings.

The model carries cold/startup overhead for every task/partition and the larger
44.40 second measured launcher overhead. It deliberately retains old every-case
IO timing even though the full collection publishes less often.

| Profile | Easy-task observed episode length | Every episode hits400 steps |
| --- | ---: | ---: |
| Modeled queried chunks | 96,960 | 384,000 |
| Modeled primitive actions | 927,360 | 3,840,000 |
| Approximate SAC updates after500 chunk warmup | 1,139,200 | 6,880,000 |
| GPU-hours before uncertainty | 27.63 | 100.80 |
| GPU-hours with20×every-chunk update allowance and1.5uncertainty | 47.38 | 157.14 |

**160 GPU-hours is the proposed collection cap**, rounded above the conservative
projection. Eight continuously available workers would consume it in20 hours;
that is a budget conversion, not an ETA. Actual task lengths, reset cost,
checkpoint IO and resource availability determine completion time. The cap
must be metered across all attempts and resumptions. Do not infer a negative
meta-learning result if a small or incomplete corpus fails.

## Storage projection and retention

Observed train footprint is1,099,940,337 logical bytes (1,103,200,768 allocated):
719.28 MB checkpoints,202.88 MB decision journal,126.98 MB raw episode archives,
50.53 MB cumulative paired prefixes,0.27 MB other metadata.
Validation footprint is1,231,617,019 logical bytes.

The full-case plan publishes immutable full checkpoints and paired cumulative
AD/DPT prefixes every16 cases plus final: cases16/32/40 for training,16/20 for
validation. Full optimizer/replay/sampler/Python/NumPy/motor RNG recovery still
saves at every complete episode. Raw archives and write-ahead journals remain.
One latest committed paired prefix per history is selected for assembly;
overlapping prefix versions must never be concatenated.

| Category,40 train/20 validation | Easy-task length | All400 step timeouts |
| --- | ---: | ---: |
| Raw episode archives +decision journals | 52.78 GB | 209.02 GB |
| Immutable full checkpoints +latest recovery | 17.77 GB | 39.95 GB |
| Paired cumulative prefixes | 5.98 GB | 23.70 GB |
| Retained total | 76.53 GB | 272.67 GB |
| Peak with an atomic recovery copy for all 80 partitions | 81.91 GB | 285.28 GB |

These measured compression projections are not storage ceilings. The exact
schedule uses larger fixed allowances:2 MiB/decision for raw+journal,
40 KiB/row for paired prefixes,32 KiB/replay row plus256 MiB fixed/checkpoint,
immutable publications and two atomic recovery copies, then10% reserve.
The resulting **1,084,181,446,680 bytes =0.986 TiB** matches the reviewed schedule
byte-for-byte. Reserve1.0 TiB. At the audit,1.2496 TiB was free, leaving0.2635 TiB
after the exact schedule allowance. Recheck capacity/write access before launch
because unrelated outputs consume the same filesystem.

For comparison,80 train/20 validation cases require a1.633 TiB conservative
allowance and exceed this free capacity.32 train/20 validation require0.846 TiB.
The40/20proposal preserves every train state ID while fitting the current disk
allowance. No existing data needs to be deleted or moved.

## Resumability and limits

The collector already persists a complete recovery at episode boundaries and
journals execution intents before each physical RPC. An unresolved intent is
conservatively charged on resume. Preserve the exact source worktree, schedule,
output identity and complete recovery. The fresh-run `vla_source40_collect.sh`
rejects an existing output; resume must invoke the pinned collector's `--resume`
path rather than rerunning that fresh-output launcher.

The40/20episode budget cannot guarantee500 query chunks on a task solved in
very few decisions. Record whether each source teacher actually updates and
produces a changing-policy prefix; never label a raw-only prefix as AD. This
proposal does not extend a task automatically beyond its approved case/action
budget.

`derive.py --refresh` reads completed JSON/stat evidence over authenticated SSH;
`derive.py` recomputes from the compact snapshot in `measurements.json`.
No pickle loading, simulator/model execution, tests, GPU computation or target
outcomes are used by the derivation. Source JSON hashes, fitted checkpoint
sizes and episode timing windows are retained in the snapshot. Prior untracked
draft derivation files were preserved outside the repository before revision.
