# Shared-runtime robot comparison

Baige job-7mznlehmgocq, source f5552d651eae7e10523e8f696fa09de58bf19d6b.
Submitted2026-10-30 03:02:40Shanghai, oneGPU/40min,32k action ceiling.
**Completed exit0:** all28groups,76actualepisodes,2,976chunks and30,505actions
including760settling actions.52focused Linux checks and the actual allocation
GPUprobe passed. All groups used one actual motor process; full motor
parameter SHA256 stayed
`c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9`.

Both one-goal and two-goal fixed500 AD/DPT full/static/reset candidates,
native and frozen final target-online SAC share one motor process on the
original task0/1 cases. No optimizer updates or new source training data.
76plannedepisodes; motor-parameter identity and complete raw/context/RNG
recoveries required. The [contract](../../docs/frozen-vla-libero/COMMON-RUNTIME-SOURCE-COMPARISON.md)
records controls, source pins and fresh-process resume limitations.

## Robot outcomes

Each denominator is the same two predeclared development cases, not a full
suite. Source1 uses one source goal; Source2 uses two. All students have500
optimizer steps. K1 includes one actual support followed by a separate query.

| Candidate | K0 query successes | K1 query successes |
| --- | ---: | ---: |
| Native |0/2|not rerun|
| Frozen final target-SAC, after15 training episodes per case |0/2|not rerun|
| Source1 AD full |0/2|1/2|
| Source1 AD static/reset (each) |0/2|0/2|
| Source2 AD full/reset (each) |0/2|0/2|
| Source2 AD static |1/2|1/2|
| All Source1/2 DPT full/static/reset (each) |0/2|0/2|

Task0: Source1 AD full succeeds339primitives after failed400primitive support;
its same-weight episode-reset fails. Source2 AD static succeeds103primitives
both at K0 and K1 after a failed support. Those two static query trajectories
are bitwise identical, so this benefit does not depend on support history.
Task1: every query fails. All24 support episodes fail. The Source1 discovery
has been remeasured here; do not count it as an independent new-case success
or pool it with the earlier six-state replication.

Steering can help this particular case without memory. More source goals did
not yield monotonic history-conditioned improvement. This is not reliable
adaptation, high topline performance, a full-suite result or proof that the
transformer architecture beats SAC. Students use offline source pretraining
and prefix features, while ordinary pixel-SAC had a short fresh target-only
learning schedule. Its original per-case training costs5,976/6,150 actions
remain separate from these frozen queries. Source costs are14,430actions for
Source1 and33,419 for Source2, with Source1 reused inside Source2.

## Independent artifact audit

[Audit script](audit_saved.py) constructs no model or simulator. It checked
all76trace hashes, exact chunks/actions/rewards/boundaries/observation
continuity, actual initial public arrays/nominal states/query RNG across
methods, native independent Gaussian draws, tied learned noise, all76complete
context/RNG archives and every group recovery/physical journal. All8full/reset
pairs match actual supports and empty-history queries; static K0/K1 queries
match bitwise. All8student files and both final target-SAC recoveries/contracts/
indices were rehashed. All charges reconcile with no unresolved RPCs.

Evidence: [independent audit](independent-artifact-audit.json),
[full final index](index.json). The original [launch snapshot](launch-progress.json)
remains as an explicitly intermediate receipt. Raw images/traces and full
recoveries remain on Baige. The [source-quality audit](../libero-source-bank-quality-20261030-r1/REPORT.md)
finds no corruption, but near-ceiling source behavior, no demonstrated learned
recovery and untested final DPT candidate actors. These limit interpretation
of this tiny data-scale comparison.

Full40 bulk collection has not launched.32k is funded inside1M by equally
reducing future unlaunched PPO full/reset streams to172k each; no existing
run is modified. Full40bulk and stable RoboTwin learning/comparison remain
unfinished; their additional compute requests are still pending.
