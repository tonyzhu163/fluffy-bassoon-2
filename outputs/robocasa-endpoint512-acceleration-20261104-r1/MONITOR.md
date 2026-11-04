Continue the user's authorized matched512 acceleration in
/Users/mowenjie/fluffy-bassoon-2. Read AGENTS.md, docs/README.md and this run's
PLAN.md, EXECUTION.json and LATEST-CHECK.json if present. Own only this run and
its coordination row (chat01a1a512); preserve other jobs and concurrent edits.
Use machine-local metrics-relay SSH settings without printing routing/auth.
Run computation and tests on allocated Linux, with macOS for editing/static checks.

Current Baige Pro6K job: job-1x7xmprauwiu, mwj_1104_endpoint512_accel8x1_r5,
created Nov4 14:55:06 Shanghai. It has eight scheduler containers, oneGPU,
16CPU,224GiB RAM and8GiB SHM each,24-minute maximum, no automatic retries.
Sourcef7f79709869d181d318a74f429f2bdc04b3049fb is immutable and clean in
/mowenjie/fluffy-bassoon-2/sources/endpoint512-acceleration-f7f79709.
Model runtime0988b5f9 and RLinf8a1233c3 remain unchanged. NODE.sh maps master-0
to shard0 and worker-0..6 to shards1..7; each runs the actual one-GPU probe and
the unchanged qualified evaluator. Never edit active source or resubmit an
existing job. Diagnose failures and verify release, ownership and budget first.

The immutable active manifest is
/mowenjie/fluffy-bassoon-2/outputs/robocasa-endpoint512-acceleration-20261104-r1/continuation-r4/MIGRATION.json,
sha256d4275a04e994acaaeca13ea1d17bf1a1b944b51d44190730dbfeac2233461e5c.
It retains976 policyepisodes (Toaster528,Drawer448), including five complete
helper pairs. There are1072 episodes left, in eight disjoint shards. Each task
still uses512 original paired cases, policies0/130, B8,600controls, original
global-bank ABBA and per-bank common RNG, exact McNemar/Holm-2 and zeroSGD.
Retain both policies from the same attempt, never stitch partial banks. Replay
overhead is charged separately; the recovery reserve is96000controls. No new
cases, training, selection or source campaign is authorized by this monitor.

The original two one-GPU jobs and all four earlier acceleration attempts have
released. Their conservative creation-to-release total is14340GPU-seconds,
including the shared-container Ray attempt stopped14:34:10. That layout is
retired because independent Ray services caused ambiguous discovery. The
original all-attempt cap remains27600GPU-seconds. Current reservation25860
includes8×24minutes; count queue/startup/release overhead and actual receipts.

Inspect remote gpu-r5/node-XX/{REPLICA.json,gpu-probe.json,clock.json,console.log}
and continuation-r4/<task>/shard-XX/gpu-r1/{initial-parity.json,PROGRESS.json,RESULT.json}.
Verify eight distinct indices, actual one-device probes, correct inference and
render devices, advancing paired banks, and PFS above200GiB. On completion,
combine on allocated Linux CPU using the frozen parallel_endpoint.py combine
--manifest <manifest> --output <run>/gpu-r5/RESULT.json.

Independently audit retained/new raw episodes, action journals, physical/public
snapshots, all12H50 sampler receipts, common RNG, actual global-bank ABBA,
checkpoint/source joins, distinct512case-policy pairs/task,600-control horizons,
all-attempt controls, exact McNemar/Holm-2 and descriptive paired bootstrap.
The original endpoint run's FINAL-AUDIT-6c95116e.py describes these checks but
assumes its old layout; adapt a separate audit here without changing active
sources. Keep first-success and end-success separate and original64-case gates
unchanged. Save terminal/release timestamps and cumulative conservative costs.
Preserve source checkpoints, shared caches and original staging copies. Only
reclaim this migration's artifacts after verification.

Publish verified results in docs/FINDINGS.md, mirror final evidence remotely,
and commit/push only this task's files and its own coordination row directly
to main, preserving other dirty changes. Monitor every20minutes. Keep healthy
unchanged status quiet; notify only on meaningful completion, failure, material
change or required human action. Delete this heartbeat after audited closeout.

The current source gives every shard a private output.parent/accounting root through shard-XX/gpu-r1. The prior27-minute replica job failed at14:45:56 on shared action-journal ceilings; its1136GPU-seconds remain charged. cpu-r6-escrow/QUALIFICATION.json verifies the actual frozen ActionJournal across eight unique roots and unequal lengths, zero model/simulator calls. cpu-r5/QUALIFICATION.json qualifies unchanged native imports and seed/ABBA binding. Follow the current manifest paths directly, including the extra gpu-r1 level. If the platform marks the job successful when master finishes, verify every shard RESULT before claiming complete and recover only unfinished complete case banks within the original cap.
