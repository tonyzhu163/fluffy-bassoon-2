# Complete-case source40 collection preparation

This is a prepared collection route and budget example, not an approved large
run. It uses all source LIBERO40 tasks and leaves full LIBERO90, Plus and Pro as
targets. It performs no goal-family filtering and does not change active runs.

`chunk_target` is the bounded smoke mode: reach the queried-chunk target and
finish the current case. `complete_cases` instead finishes **every requested
case**, even if a chunk target would already have been reached. It rejects a
chunk stopping target. A task and partition share one persistent official SAC
learner, replay and optimizer across cases. Train and validation use independent
learners and disjoint initial-state banks.

The index reports requested, started and completed cases from actual episode
receipts. It separately reports initial-state IDs, nominal state-vector hashes,
state/seed pairs and repeated episodes. For example, 80 training cases below
are 40 initial-state IDs with two seeds each, not 80 distinct nominal states.
One support episode plus one query episode remains a separate target protocol;
source generation episodes are not deployment shots.

## Concrete all40 plan example

Run static schedule generation using the reviewed code; it does not load a
model or submit jobs. Omitting `--task` includes all four ten-task source suites.
Use a fresh plan path.

```sh
PYTHONPATH=src python -m fluffy_bassoon.meta_rl.vla_source40_schedule \
  --libero-root /mowenjie/fluffy-bassoon-2/deps/libero-dsrl-87edbd1-20261029-r1/repo \
  --output /mowenjie/fluffy-bassoon-2/outputs/<fresh-plan-id>/schedule.json \
  --collection-mode complete_cases --train-cases 80 --validation-cases 20 \
  --episodes-per-case 4 --publish-every-cases 16 --seed 0
```

The following launcher runs **one** selected task/partition inside an existing
allocation. It does not fan out or submit the 80 task/partition jobs. The runtime
and wall-time budget require qualification before full collection. `<SHA>` is
the exact reviewed integrated commit, and the run ID must be fresh.

```sh
bash scripts/diagnostics/vla_source40_collect.sh <SHA> <fresh-run-id> \
  /mowenjie/fluffy-bassoon-2/outputs/<fresh-plan-id>/schedule.json \
  libero_spatial 0 train
```

Change the suite/task and partition to select another source stream. The
launcher extracts its partition-specific primitive ceiling from the hash-bound
plan, uses the same official frozen OpenPI `pi05_libero` checkpoint, and runs
focused tests and GPU checks inside the allocation. It uses batch256, UTD20,
500 queried warmup chunks and episode-batched official SAC updates.

## Conservative example ceilings

Native H10 chunks and a 400-primitive episode horizon imply at most 40 queried
chunks per episode. Early success reduces actual cost.

| Quantity | Train per task | Validation per task | All40 total |
| --- | ---: | ---: | ---: |
| Requested seeded cases | 80 | 20 | 4,000 |
| Requested source episodes | 320 | 80 | 16,000 |
| Maximum queried chunks | 12,800 | 3,200 | 640,000 |
| Maximum executed primitive actions | 128,000 | 32,000 | 6,400,000 |
| Maximum settling actions, including one qualification reset per stream | 3,210 | 810 | 160,800 |
| Storage planning allowance with 10% reserve | 35,631,870,772 B | 9,260,918,375 B | 1,795,711,565,880 B |

The total storage allowance is approximately **1.80 TB / 1.63 TiB**, without
taking compression credit. Assumptions are 2 MiB per decision for combined raw
traces and decision journals, 40 KiB per row for paired AD/DPT prefix arrays,
32 KiB per saved replay row and 256 MiB of fixed state per full SAC checkpoint.
The estimate includes growing replay in retained checkpoints, all retained
prefix versions and two atomic recovery copies. Input VLA weights, exceptional
orphan files and repeated process qualifications are excluded. This is an
explicit planning allowance, not a verified filesystem or quota bound; measure
artifact sizes and throughput on the smoke before choosing a collection budget.

Full recovery still occurs after every completed physical episode. Immutable
actor snapshots and paired prefix exports occur every 16 completed cases and
at the final case, reducing repeated prefix/replay storage. All raw stages,
actual noise and rewards remain. Overlapping prefix versions must not be
concatenated into extra unique interactions.

Completing a case schedule does not guarantee sufficient learning signal or
warmup clearance: very short episodes can still produce too few queried chunks.
The index separately reports complete case coverage, actual SAC updates and
whether a changing-policy AD/DPT prefix exists. A raw-only history is retained
honestly. Interrupted actions consume the write-ahead physical budget; a run
that cannot finish within its cap fails with partial coverage rather than
claiming that every scheduled case was visited.

## Resume the same pinned source stream

The launcher accepts a final `--resume` argument. Reuse the exact source commit,
worktree, schedule and output identity; a new scheduler allocation may run it.
It retains previous attempt logs and GPU/test receipts under `attempts/`, starts
a fresh simulator process and restores the last complete episode's full SAC,
replay, optimizer, sampler and motor RNG state. The collector charges interrupted
physical actions separately. Do not increase the source ceiling while resuming;
failure to finish within the declared ceiling remains partial coverage.

```sh
bash scripts/diagnostics/vla_source40_collect.sh <same-SHA> <same-run-id> \
  /mowenjie/fluffy-bassoon-2/outputs/<same-plan-id>/schedule.json \
  libero_spatial 0 train --resume
```

An allocated numerical diagnostic in `vla_source40_training_smoke.sh` checks
that two restorations reproduce the same next real SAC update and motor PRNG
split without changing the source recovery. This does not substitute for an
interrupted full-collection continuation test or authorize a larger run.
