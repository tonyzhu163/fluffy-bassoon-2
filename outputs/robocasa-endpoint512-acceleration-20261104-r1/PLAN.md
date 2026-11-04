# Parallel completion of the existing matched 512-case endpoint test

The human requested more GPUs to accelerate the two endpoint tasks on Nov4.
This is an execution migration of the original policy0/policy130 comparison,
not a new training run or statistical comparison. Owner chat `01a1a512`.

Use one exclusive Baige Pro6K allocation with eight GPUs, 128 CPUs, 1536 GiB
RAM and 64 GiB shared memory for at most 35 scheduler minutes. First verify
queue capacity, resource limits, actual devices, storage/write access and the
actual eight-device GPU probe. No automatic retry. Eight independent workers
each run the unchanged qualified eight-lane evaluator with one allocated GPU.
Each worker starts a separate local Ray service, limited to16CPUs and2GiB
object store, to prevent attaching to another worker's service. This changes
orchestration only, not model computation, batches or sampling.

Before stopping either original job, qualify the wrapper and static bank
partition on allocated Linux CPU and stage a clean Git-pinned code copy.
Then verify both exact original job IDs, ownership, source and outputs in the
scheduler. Stop only those two evaluation jobs, preserving every raw receipt,
checkpoint and staging copy. Record scheduler release and conservative
creation-to-release charges, including the earlier41GPU-second duplicate.

After both releases, freeze a migration manifest from their completed paired
banks. Retain every complete pair unchanged. If a bank was interrupted, retain
its raw evidence and exclude it from the final valid-case table; rerun that
whole bank under its original ABBA order on one helper. Divide the remaining
banks into four disjoint contiguous groups per task. The manifest asserts
coverage, case identity, checkpoint hashes and the original manifest hash.

Preserve all512cases/task, bothpolicies, the original600controls, global bank
ABBA order, per-bank noise seeds, B8 inference, 12H50 calls, physical/public
initial-state checks and common-RNG receipts. Changing GPU allocation changes
the time schedule, not these numerical inputs. Rendering variability retains
the original qualification boundary. Strictly load the original weights and
require policy0 likelihood parity in every worker before any simulator bank.

Keep the original27600GPU-second all-attempt ceiling. Admission requires the
old conservative charge plus8×2100seconds to fit that ceiling. Reserve at most
19200 additional physical controls for two interrupted bank pairs; all2048valid
policy episodes and1228800validcontrols remain exactly the original comparison.
All abandoned attempts stay charged and separate from valid-case statistics.
No scientific scope expansion, PPO update or student fitting is included.

Combine retained complete pairs and helper outputs once every helper exits.
Reject missing/duplicate case-policy keys, wrong source hashes, changed bank
noise, unresolved helper action intents or incomplete horizons. Reuse the
original exact two-sided McNemar, Holm-2 and descriptive paired bootstrap.
Publish results in this migration directory with a pointer to the original
run; do not overwrite the original executable worktree or its result files.
Reclaim only this migration's files, after verification; original staging
processes and source checkpoints remain protected. Monitor every20minutes.

## Explicit migration after the shared-container attempt

The shared-container attempt `gpu-r3` was stopped and released Nov4 14:34:10.
Independent Ray services in one container caused ambiguous address discovery;
the RLinf node probe also advertised the whole allocation. This layout is
retired. Its completed paired banks remain usable only after original seed,
noise, state and sampler audits; retain both policies from the same attempt.

Use eight scheduler containers with one GPU,16 CPUs,224GiB RAM and8GiB shared
memory each, with a27-minute maximum and automatic retry disabled. The Baige
PyTorch master-0 maps to shard0 and worker-0 through worker-6 map to shards1–7.
`NODE.sh` rejects other identities. Each container uses its scheduler GPU
visibility, the actual one-device probe and the unchanged qualified evaluator.
The runtime and RLinf commits are unchanged. A fresh continuation manifest
retains original complete pairs and complete pairs from the stopped helper
attempt, preserving hashes and the local-to-global bank mapping. Never stitch
two incomplete banks from separate attempts into a pair.

All three acceleration attempts stay charged. At most96000 recovery controls
are reserved across interrupted original/helper banks, replacing the earlier
19200 reserve for two original banks only. Valid cases, outcomes and1228800
valid controls are unchanged; this is replay overhead, not additional cases.
The original27600GPU-second all-attempt cap remains binding. The latest stopped
receipt and27-minute reservation must fit it before another launch. Combine
on allocated Linux CPU after all eight shard results and clocks are complete.

The first separate-container attempt exposed an accounting collision: the
qualified evaluator uses `output.parent/accounting`, so sibling shard outputs
shared a ledger root and disagreed on ceilings for unequal shard lengths.
Release that attempt, retain complete pairs, then place each new model output
at `<task>/shard-XX/gpu-r1`. Its parent and action journal are now private.
Qualify the real frozen ActionJournal with all eight roots and unequal limits
on Linux CPU before launch. Reuse prior complete native import qualification;
the evaluator and all model imports are unchanged. Retain previous manifest
pairs recursively, with raw-file hashes and no stitching of interrupted banks.
The next eight-container reservation is20minutes, with a1170-second worker
timeout and30-second kill allowance. Recompute cumulative charges on release
and admit only when that reservation fits the original27600GPU-second ceiling.
