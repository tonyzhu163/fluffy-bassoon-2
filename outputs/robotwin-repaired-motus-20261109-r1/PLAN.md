# Repaired Motus continuation — 2026-11-09

Continue the approved single-task `place_phone_stand` experiment under the fresh
100 GPU-hour budget recorded in
`../robotwin-restoration-20261106-r1/LATEST-BUDGET.json`. Historical costs remain
archived there; this run uses fresh output identities and starts from frozen SFT.

## Stage 1: paired controller verification

Use the 16 fixed pilot DEV cases from `MOTUS-DEV-CASES.json` (also the first
16 cases of `MOTUS-SOURCE-READOUT-CASES.json`), native B2,
H32 delta-joint actions, the pinned Motus checkpoint and identical case-addressed
initial noise across three arms:

1. Legacy chunk controller, K5/H192.
2. Repaired waypoint controller, K5/H192 (isolates controller change).
3. Repaired waypoint controller, K10/H384 (proposed learning configuration).

The waypoint controller follows the native per-waypoint planning structure and
adds linear interpolation on nonstationary TOPP failure. It is therefore a
modified controller, not an exact reproduction of unmodified `take_action`.
Record actual planner failures, arm commands, full-chunk freezes, reset receipts,
physical execution, complete outcomes and frozen-weight checks. Compare fresh
paired outcomes; the historical 6/64 is context, not a matched-noise control.

Reserve at most 1.5 GPU-hours for one GPU with an 80-minute worker deadline and
90-minute scheduler deadline, automatic retries disabled. Use project shared
storage for immutable code/model inputs, job-local NVMe for scratch, and BOS for
evidence; check capacity and write/read/fsync before launch. Qualify the actual
allocated device and renderer inside the job.

Advance only after all 48 episodes and independent trace reduction pass,
repaired K10/H384 reaches at least 4/16, and the repaired arms have zero
nonstationary full-chunk arm freezes. A failed gate triggers diagnosis.

## Stage 2: one-task PPO and GRPO

Use fresh run identities and reviewed latest-main commits after Stage 1.
Start with a fresh C0–C2 prefix for each method using `train_prefix.sh`; both
launchers require the successful independent Stage 1 receipt. Each prefix has a
150-minute worker limit and needs a 155-minute scheduler limit (20.667 GPU-hours
reserved per eight-GPU job). Measure actual collection/update/checkpoint time
and audit C2 before reserving C3–C10. These are fresh repaired lineages; none of
the historical H192 checkpoints is resumed into them.
Preserve B32/accum8/world8, global batch2048, five update epochs, exact behavior
likelihood checks, GRPO reward filtering [0.1,0.9]/upper clip .28, and PPO VLM
prefix critic with actual rollout/replay value parity. H384 means 12 decisions
per episode and **30**, not 15, Adam steps per iteration; a 15-step critic warmup
is half an iteration. Validate checkpoint/optimizer/sampler/RNG restoration
before continuation. Obtain measured collection/update time before reserving
C1–C10; the earlier 35–45 GPU-hour figure is an estimate, not a qualification.

Use the complete 64-case DEV bank for training readout and method selection;
the 16-case pilot is exploratory controller qualification, not a learning claim.
Select checkpoint/method using DEV, then use the disjoint fresh 64-scene
confirmation bank once. Require >=10 percentage points and exact two-sided
McNemar p<.05 versus a paired repaired SFT baseline. No student fitting or
six-task scaling until this gate passes. Six-task results will be per-task,
with their mean reported only after matched evaluation of all six tasks.

Monitor launched jobs every 20 minutes and record scheduler allocation charges,
including failed attempts, against the same 100-hour cap.
