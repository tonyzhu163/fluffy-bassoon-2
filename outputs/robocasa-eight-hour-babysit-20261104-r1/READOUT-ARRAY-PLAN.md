# Matched readouts without serializing independent histories

After the real B8 six-episode qualification passed144episodes/86400controls
and432decoded videos, the full readout may distribute whole fixed cohorts.
Each task retains the same fifty final cases, case seeds, eight-case ascending
cohorts and final two-case cohort, all six actual episodes, and query-specific
native draws. A case's history never crosses a process or resets in full mode.

Two native-anchor replicas each own one of two cohort partitions. Four student
replicas own full/shard0, full/shard1, episode-reset/shard0 and episode-reset/
shard1. Cohort assignment is(global task index+cohort index)%2; rotating it
across tasks balances the expensive final small cohorts. Shared canonical
physical/public snapshots are published atomically and checked before each
actual attempt. Partitions are exhaustive/disjoint; neither GPU batch size,
row order, policy, history, sampler nor saved training contract changes.

Each worker has1GPU/16CPU/64GiB RAM/8GiB SHM, a real GPU probe and a unique
node output. The master stays alive until every worker has a complete receipt.
Initial ID ceilings are75minutes (70minute worker); adjust only from actual
same-horizon measurements and the existing96GPU-hour cumulative ledger.
Native anchors use two replicas; each student uses four. Existing full
learners remain immutable. Their inference snapshots are frozen only after
that individual learner's declared validation convergence.

Run ID first. OOD begins after that method's complete ID panel. The current
three-task baseline's OOD panel is CloseFridge/TurnOnMicrowave, retaining their
native900/450 horizons. This does not substitute for the future locked15-task,
five-source steering panel. Its prior exposure is disclosed in TRANSFER-PLAN.

READOUT-COMBINE.py requires exact50-case coverage per task/mode, all six
complete episodes, intact checkpoint references, fixed cohorts and unchanged
candidate/source/model identities. It reports each query's first/final success
and separate cumulative ever-success. Full-minus-reset differences and paired
intervals across the curve are descriptive; no multiplicity-adjusted memory or
adaptation claim is made. Independent raw command/draw/video audits remain
required after reduction and are labeled pending until completed.
