# Parallel source collection and the ready robot readout

Proposed, not authorized. This updates the unanswered25h source-only request
by adding the now-qualified4.5h robot readout. Approval of only the earlier
25h request still authorizes only that source phase.

| Additional phase | GPU ceiling | Additional PFS storage |
| --- | ---: | ---: |
| Same all40 source continuation, fixed source checks, four500-update fits |25h|130GB|
| Fixed100-model all90 K0 and nine-case K1 robot readout |4.5h|20GB|
| Total |29.5h|150GB|

The original32h ledger and its1,180 unassigned seconds remain separate. No
budget is granted by these files. Source and readout run independently as
allocations become available; the existing immutable100 snapshot allows
evaluation while the remaining source collection proceeds.

Source execution is pinned toe93c742eaf1782dc6a3c024e9347321442d5ff11.
The actual CPU-produced plan file SHA is
bf868ac2b9854cf2a0a26201cddb2b4dc2c32f961b6879e18ab3254f2a5297ec.
It admits all40 goals, launches38 remaining streams, retains two filled
bridges and imports fixed8 validation streams. The cumulative source guard
is998,554 actions, including historical costs and final source checks. Four
AD/DPT full/static controllers fit at fixed25/100/500 update checkpoints.
The [25h source proposal](../libero40-source-scale-25h-20261031-r1/PROPOSAL.md)
contains exact commands and complete cost accounting; the new plan has now
actually been generated and verified without granting funding.

Readout execution is pinnede764afdd04c9ea4dcbb7ac3e3e25d245c186dd42.
Its additive profile passed48 allocated CPU tests without skips, including
the actual fixed100 matrix and all model/recovery hashes. Native and four
students run K0 on all90 tasks, state0/seed123/H400. Four full/same-weight
reset conditions additionally run K1 on the nine predeclared development
tasks0,10,...80, using actual support plus separate query episodes. Total522
physical episodes,220k actions including settling/failures/retries, six
2,700-second cumulative workers. The fixed producer matrix SHA is
dc9ebd1274170e021906c1f66623a98713405e227229bc022e03eb4c1c92a623.
The [qualified readout contract](../../docs/vla-source40-interim100-readout.md)
and [prepared grid](../libero40-interim100-target-readout-plan-20261031-r1/README.md)
contain exact commands. Target data never enters source pretraining.

Every resume retains its original cumulative time/actions, full recovery,
optimizer/sampler/RNG where applicable and unresolved RPC charges. Limits
win over completion; partial work remains preserved and reported. Storage
is an additional planning envelope; current measured free PFS exceeds1.2TB.
No original artifacts or active worktrees are removed or edited.

This is one-initialization development screening, not full Plus/Pro, complete
2/4/8-shot evaluation or a positive memory/sample-efficiency result. PPO/reset
1M scaling (44h/arm) and full shifted-suite runs remain separate and unfunded.


The optional validation-label path has now completed on an actual0GPU CPU
allocation:8 matching TRAIN actors/8,202 VAL decisions,49 focused tests,
full derived-assembly round trip,348 conservative CPU-worker seconds and
zero GPU time/actions. Return the proposed600GPU-s label reserve to
contingency: source planned88,880/90,000s, contingency1,120s. The source25h
and combined29.5h caps and the pending approval request are unchanged.
Use a preselected TRAIN actor per goal for both TRAIN and heldout-state VAL
labels in a fresh derived generation; keep behavior observations,
actions/noises/rewards and all previous archives unchanged. If its binding
has not been verified for a later snapshot, report the old teacher-transfer
metric explicitly. This affects validation interpretation, introduces no
target data or checkpoint selection, and never gates source collection.
[Completed CPU evidence](../libero40-dpt-same-train-teacher-cpu-20261031-r1-audit/RESULT.md).
