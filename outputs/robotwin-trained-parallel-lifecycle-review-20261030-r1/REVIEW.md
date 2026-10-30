# Trained parallel migration and process-lifetime review

Date: 2026-10-30, Asia/Shanghai. This review did not launch GPU work, stop
jobs, initialize a live ledger, modify an active worktree, or migrate a source
learner. Review/test work stayed in the original detached
`robotwin-process-pool/fluffy-bassoon-2` worktree and fresh Linux CPU worktrees.

Reviewed immutable sources:

- Packed result/base: `42abc1b1d8dbeeafa5b4d42df7ec42d5d2c618fc`.
- Owner lossless recovery: `e825821a7c2231952bcf1042162f3470bd8ecdea`.
- Existing supervisor: `c715e7b74218f2d54eac7837c4dfefcc2644e69a`.
- Integrated trained candidate: `1aa4d1de5b6749ebac72b8c8242656d13edeaa22`.
- Additional static review of parent binding:
  `5454c0a8f379e0306e8880587f4f85598336f0e2` (CPU validation pending here).
- Scientific lineage: `db0184089fa88a5f96c153f9b80029bb91b77def`.

The packed result alone is insufficient for trained migration. Its branch
lacked the owner's selected-trial/whole-boundary recovery path and put
parallel execution options in the scientific contract. The integrated
candidate restores the owner path and places execution changes outside the
unchanged predecessor scientific contract. No production scalar
`robotwin_noise.py` diff exists between the DB lineage and packed base.

## Required handoff and ownership

Only a hash-bound whole-trial handoff after a completed trained PPO update is
eligible for N6. Require one PPO arm, no selected trial, no retained prefix,
empty pending PPO, positive completed iterations, trials equal six times
iterations, schedule cursor equal completed trials, learner updates equal
iterations, and five optimizer steps per iteration. The candidate checks
trained Adam steps, LR and memory-reset convention, then checks the loaded
policy/optimizer/config/counters/private sampler/task sampler identities and
restores the coordinator's global RNG. Full checkpoint bytes remain bound by
the owner's manifest hash. Selected or partial trials must finish on the
owner's scalar path; diagnostic weights are ineligible.

The original DB scientific contract, method, campaign ID, header and action
caps must remain identical. Reuse the original canonical campaign with a new
unique execution run ID; do not import prior receipts into a new campaign.
The candidate validates the predecessor ledger fence before CUDA and again
after opening the exclusive live ledger. Copied ordinary recoveries retain
canonical journal checks; failed/unfinished groups remain fenced and charged.
Whole-group action accounting and replay admission remain atomic. Unknown
native outcomes never become refunded or silently retried actions.

An immutable handoff is not an ownership lease. The active scalar supervisor
can publish one and advance its next slice immediately. The owner must provide
an exclusive handoff boundary before any eventual migration; this review does
not stop the supervisors. A changed canonical hash or held lock must reject
the candidate rather than replay the protected checkpoint.

The predecessor scalar schema binds a ledger snapshot but does not contain
its canonical live root. Initial 1aa4d1d derives that root from CLI arguments;
a matching cloned history was therefore insufficient as an ownership proof.
After review, 5454c0a adds a required hash-bound original parent manifest and
checks its `worker.ledger_root`, campaign ID, caps, source goal, evaluation
schedule and native allowance before CUDA. The final launch must use the
original reviewed parent SHA, not a replacement parent. Subsequent parallel
checkpoints also embed a canonical root.

## Bounded source and evaluation lifetime

For the first trained candidate, require exactly six lanes and
`max_source_trials_per_process=6` with round-boundary recycling. The driver
adds six to its process counter only after admitting the complete physical
group. It finishes the unchanged PPO update, then the lifetime guard stops
before a due milestone or another source trial. There is no future-policy
collection or partial global-round expansion.

Each child owns one fixed source task and one K4 trial: four reset RPCs and
at most 2,400/2,000/1,600 control actions for a 600/500/400-action task.
The six-task round reserves at most 10,800 source actions. Four reset RPCs
are not four native scene setup attempts: the existing retry limit is 32
for constructor setup and separately for each reset, giving a conservative
bound of 32 + 4*32 = 160 setup attempts per child before an error. Native
actual/requested seeds remain recorded separately.

A due milestone resumes in a fresh process with no vector pool. Completed
older milestones and individual frozen evaluation units must be skipped
using the hash-bound coordinator's `completed_units`. Scalar evaluation
retains its original source-validation, target and native cases:

| Per complete milestone | Persistent PPO | Trained reset PPO |
| --- | ---: | ---: |
| Source-validation episodes | 24 | 24 |
| Target episodes | 48 | 16 |
| Native episodes | 4 | 0 |
| Scalar reset calls, including one preflight | 77 | 41 |
| Maximum milestone policy actions | 31,600 | 17,200 |
| Maximum with one following N6 source group | 42,400 | 28,000 |

Source validation costs at most 4*(600+500+4*400)=10,800 actions. Persistent
target evaluation costs 2*3*8*400=19,200 and native cases another 1,600;
reset targets cost 2*8*400=6,400. These reproduce the original campaign caps:
1.1M + 4*31,600 = 1,226,400 and 1.1M + 4*17,200 = 1,168,800. A source-only
10,800-action proposal cannot cover a process with a due evaluation.

**Remaining finding at 5454c0a:** the trained argument gate does not yet bind
`eval_trials`, a field omitted from the old scientific contract. Passing
`--eval-trials 2` can double target cases while preserving contract bytes,
invalidating these lifetime/action bounds. Require the original single
repetition before CUDA (or an equivalent trusted parent binding). This was
reported to the PPO implementation agent and root; no fix or new test was
made by this reviewer.

After the scalar stage finishes, the same fresh parent may perform its one
N6 group. Admission first closes the idle scalar scene and clears prefix/raw
cache while retaining the identical frozen motor. The vector pool remains
lazy until then. Evaluation with an already-created pool fails before new
files, RNG consumption or scalar interaction; there is no six-idle-scene
overlap. Normal/error shutdown joins or terminates only owned workers and
clears lane caches; the next process uses a fresh output/pool identity.

The original c715e7b supervisor permits cleanup exit 134/139 only after an
independently verified complete-round handoff with empty pending data and
zero unresolved ledger actions. Other errors, missing receipts, timeouts or
no progress halt. This is handoff validation, not a GIL recovery algorithm.
SAC remains on its existing scalar path and is outside the N6 candidate.

## Verification and remaining limits

Added focused tests in commit `8d0741bfd3e48d4416a414b79021115e3a5cd177`.
All eight passed in 0.60 s on Linux with CUDA hidden, against the exact
candidate plus those tests (combined remote source in
`linux-source-commit.txt`). The tests execute the actual nested driver
`guard`, `native`, `evaluate_frozen_sources` and `evaluate_stage` bodies with
CPU fakes. They cover the exhausted update boundary, earlier milestones,
77/41 reset-call bounds, completed-unit suppression, RNG restoration on
success/failure, and rejection of scalar evaluation after pool startup.
See `linux-cpu-tests.txt`. Two deployment setup errors occurred before any
test execution: worktree-local FETCH_HEAD resolution and absent remote Git
committer identity. Only the fresh unused test setup was repaired.

The successful packed fresh-controller round demonstrated twelve allocated
renderers and clean parent exits, but had recurring OIDN errors and no matched
scalar speed baseline. It did not qualify a genuinely trained handoff, long
native lifetimes or a fix for GIL failures. Process lifetime reduces exposure;
it is not proof of native stability. Scalar evaluation still has more resets
than an individual K4 source worker and retains bounded native retry risk.

Frequent parent reloads, trained optimizer/checkpoint I/O and original
milestone evaluation must be included in any cost/time proposal. The packed
run's roughly 114-second motor startup and roughly 236-second collection
round do not establish trained restart cost or sustained GPU saturation.
Concurrent old and new source consumers are prohibited even if GPU memory
appears available. Immutable trained handoffs after PPO40/reset41 and an
owner-reviewed resource/ownership transition remain prerequisites; this
review grants no automatic migration or larger budget.
