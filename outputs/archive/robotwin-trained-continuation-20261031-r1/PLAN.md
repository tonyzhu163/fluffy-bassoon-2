# Original100-round continuous continuation — prepared, not submitted

The initial inputs are the independently audited, terminal R5 boundaries:
PPO57rounds/546,458sourceactions/285Adamsteps and trained-reset59rounds/564,658sourceactions/295Adamsteps.
Both full checkpoints, whole-boundary manifests, canonical heads and unchanged
motor receipts are bound in `compiled-plan.json` and `owner-release.json`.
No active predecessor PPO writer is reused. SAC remains independently owned.

Use the existing clean launcher0c16c2b794f8de5b78603206917e72195083c265 at
`/mowenjie/fluffy-bassoon-2/worktrees/robotwin-trained-continuation-code-20261031-r2`.
The scientific children remain unchanged runtime57016ba08eed79043fd67bf8b085bd6d42e6d766
at the separate runtime worktree carried from R5. No runner code is duplicated.

The fresh c353-format initial identity is
`robotwin-trained-pair-initial-20261031-r1`, separate from outer identity
`robotwin-trained-continuation-20261031-r1`. Its original2700scheduler/2520driver/2160child
admission envelope is INCLUDED once in the long1GPU/32CPU/192GiB/shm16GiB,
46800scheduler/43200driver envelope. Subsequent children use at most3600seconds;
each still closes one original full N6/K4 round and update. Input/output names
are fresh; no automatic failed retry is admitted. Each arm independently
continues to100, completes its original final evaluation, then retires.

Original parent/source/total caps remain unchanged. PPO has43rounds remaining;
reset has41. Maximum additional source exposure is907,200actions, and original
final evaluation is at most48,800actions before subtracting completed units.
These956,000maximum actions fit the separate residual worker caps. They are
not new caps, and no episode/trial is cut to hit an action target.

Conservative planning is43x450seconds=19,350seconds=5.375GPU-hours of source
intervals, before final evaluation, journal-audit growth, cold loads, filesystem
and failure overhead. This is no completion guarantee or scalar speedup claim.
The13-hour scheduler slice is a metered operational upper bound, not an
invented overall budget; the human-approved original100goal and null global
compute ceiling remain explicit.

Known terminal charges76962+R5actual371=77,333GPU-seconds. Active SAC and the
old unverified reported630GPU-seconds are excluded; full cumulative charge is
therefore null. R5unused2329seconds is sealed, not borrowed. The new long
allocation is charged once; its initial2700subunit is not additive. CPU
software qualification used131schedulerCPU-seconds and0GPU-seconds.

The exact0c16 Linux CPU gate passed67production and43LIBEROCPU-fake cases,
zero failures/errors/skips, with actual0GPU admission and clean source pins.
Raw gate summary SHA55aeb39f1c3b6f534c7cb553739940ceebbf3c52f04ac9fcab5494720e9993e2
and independent JUnit/scheduler receipts are copied and bound. All long inputs
remain ready_pending_owner_review_cpu_gate_passed_not_submitted; this folder
submits/deploys nothing. Root independently reviews and deploys these input
bytes to `/mowenjie/fluffy-bassoon-2/inputs/robotwin-trained-continuation-20261031-r1`,
then uses `command.txt` in its exclusive allocation. Root owns actual GPU
probe, initial production preflight, current canonical-head admission and
terminal scheduler metering. No source controls or new tests ran here.
