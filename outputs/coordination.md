# CTM in-context RL: coordination

Live coordination for `fluffy-bassoon-2`, updated **2026-11-01 Asia/Shanghai**.
Keep current decisions and jobs here, method contracts in `docs/`, verified
results in `EVALUATION.md`, and dated plans and evidence in
`outputs/<run-id>/`. Follow [AGENTS.md](../AGENTS.md) and the
[retention policy](../README.md#artifact-retention).

The [2026-11-01 snapshot](archive/coordination/coordination-20261101.md) holds
the full table before this trim, including terminal rows and jobs unconfirmed
since 2026-10-28. Older state is in the
[2026-10-31 snapshot](archive/coordination/coordination-20261031.md), the
[cutover placeholder](archive/coordination/coordination-pre-cutover-20261028.md)
and the [source-repository records](https://github.com/tonyzhu163/fluffy-bassoon/blob/lycoreco_rl/outputs/coordination.md).

## Question and decision

Can a persistent CTM learn from interaction history with fixed deployment
weights, and does its recurrence offer an advantage over conventional memory?
Separate architecture performance from motor competence, steering headroom,
informative history and optimization.

- If XL adapts and CTM does not, retain the qualified pipeline and use the
  stronger learner while revising the CTM claim.
- If neither adapts, diagnose the interface, histories and training signal
  before attributing failure to memory.
- Judge adaptation with paired fresh probes, trained static/reset controls
  and valid history interventions.

Contracts: [support/probe](../docs/ctm-support-probe.md),
[feature bridge](../docs/pi05-feature-bridge.md),
[PPO replay](../docs/pi05-bridge-policy.md).
Results: [meta-RL evaluation](../src/fluffy_bassoon/meta_rl/EVALUATION.md).

### Redesign (2026-10-31)

No robot comparison so far shows memory or adaptation. The governing plan is
the [frozen pi0.5 steering screen](frozen-pi05-steering-screen-20261031-r1/PLAN.md):
first show a controller can improve the frozen motor on cases it fails, then
show experience changes a useful later decision. The
[steering-headroom audit](steering-headroom-audit-20261031-r1/AUDIT.md) gives
the paired fixed100 evidence. The earlier
[headroom diagnosis](archive/icrl-headroom-diagnosis-20261031-r1/PLAN.md) is
archived as superseded, with corrections listed at its top. A
[criteria proposal](steering-interface-criteria-20261031-r1/PROPOSAL.md)
informs the newly authorized [targeted scope](frozen-pi05-endtoend-20261031-r1/SCOPE.json),
which adopts A1/A2/A3 with the [review safeguards](frozen-pi05-steering-screen-20261031-r1/CRITERIA-REVIEW.md).
The proposal is not adopted verbatim: search is explicitly bounded, improvement
must be measured rather than assumed, and the initial mechanism panel does not
replace fresh32-case confirmation before broader claims.

## Coordination rules in force

From the [2026-10-31 review](archive/coordination/coordination-20261031.md#coordination-review-2026-10-31):

- Push coordination edits to `main`, rebased, editing only your own row.
  Branch-only edits leave this file stale.
- One or two readable sentences per row: job ID, last check, progress, next
  action and a link. Put hashes, byte counts and budget ledgers in the
  linked run record.
- Budget: roughly **1M source actions per learner** (human, 2026-10-30;
  confirmed 2026-10-31), superseding "1M per benchmark, shared". Charge each
  learner's actions to its own ledger; do not borrow across learners or
  benchmarks.
- Baseline priority (human, 2026-10-31): same-motor DSRL/SAC first, native
  VLRLA PPO afterward. `frozen-vla-poc` (`finish_assembly`/`finish_ppo`) is
  the sole LIBERO launcher; RoboTwin AD GPU work stays deferred.
- Active `codex/*` branches were merged into `main` on 2026-10-31; merge
  further verified work promptly (see [AGENTS.md](../AGENTS.md)).

## Ongoing work

Rows are as last recorded by their owners; none was re-queried for this
update.

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| LIBERO task44 DSRL on frozen pi0.5 | R3 `job-2qn8dpw3yvlb` Running; sole `finish_assembly` | **12:30:**394240updates/2308TRAIN/20206replay,2269success39failure.22completepanels,last38000020/20,next400k;stablefull/newrawRPC/finitelosses pass. 367898phaseactions/22856910460B;activeGPUelapsed notfinalcharge. Next12:50. [Proof](libero90-pi05-headroom-runtime-20261101-r1/CHECK-1230-PARENT-REVIEW.json). |
| Frozen pi0.5 interface/data redesign | AssessmentR2 `job-bc2ooyztqsfh` Running on9d29; sole `finish_assembly` | Actualprobe/resource/migration pass;cursor168=161oldvalid+6newvalid+1nullabort.26722combinedactions/626new/0uncertain;firstnew162success96+10steps,0SGD. Original1340GPU-s+9420reservation inside10800;32GiBshared,oldCPU154/180 andfresh77/120 separateclosed. Failed161 neverreplayed;finalC565pendingactive. NextPRIMARY12:50,assessment12:59. Current-bank500preparation separate7200,notyetlaunched. [Proof](libero40-teacher-assessment-20261101-r1/R2-LAUNCH-PARENT-REVIEW.json). |
| LIBERO90 pi0.5 headroom screen for DSRL replication | Owned Stage A terminal Succeeded/exit0, `4ec35eb9`; owner chat `01a192cf` | **Nov1 09:54 saved-reference review:** own384 audit/task59 native3/16 vs untrained tied9/16 unchanged (3,890GPU-s/134,629actions). Separate task44 recovery repair qualified; executiond236/scienceb665 resumes preserved learner to302180 with3newEP/480updates. Fixed500k/finalnative/motor pending; no duplicate learner or interface GPU launch. [Reconciliation and gate](libero90-pi05-headroom-screen-20261101-r1/REFERENCE-RECONCILIATION.md). |
| Flora-Rewarder PPO quick test, LIBERO90 task 44 | Run `flora-ppo-comparison-20261101-r1`; owner tonyzhu163 Claude Code session; planning only | **2026-11-01:** Single-seed RLinf PPO on task 44 from the RLinf-130 SFT, comparing a simulator-success reward (S) with a Flora outcome-label reward (F); training runs full horizon so episode length cannot leak success. Nothing implemented or launched. Next: Flora smoke test and a 64-episode SFT baseline (start rate, Flora agreement) as a go/no-go; kept separate from the running task-44 DSRL job. [Plan](flora-ppo-comparison-20261101-r1/PLAN.md). |

### Needs owner refresh

- Native VLRLA-recipe PPO pilot: last recorded 2026-10-31 14:06 Shanghai as
  queued behind the task57 DSRL run, which has since finished. Owner
  `frozen-vla-poc`: confirm whether it is still queued.
  [Plan](vlrla-native-ppo-pilot-20261031-r1/PLAN.md).
- Three jobs last recorded on 2026-10-28 (RoboTwin GRPO gate
  `job-vh9lp7fvgf9n`, MetaWorld pilot r2 `job-x2x4kdpfn011`, thu2 CMM
  48983/48984) are listed in the
  [snapshot](archive/coordination/coordination-20261101.md#needs-owner-refresh).
  Re-add a row only if the job is still live.

## Paused

- **Meta-World released pi0.5 noise screen**, Baige `job-orgyjb0z9h6d`: paused
  at user request 2026-10-29; reach screens fail headroom.
  [Pause record](metaworld-pi05-noise-baige-20261029-r2/PAUSED.md).

## Recently completed work

- **Original Pi0 DSRL task57 replication**, Baige `job-psgxs9n5b08w`:
  complete and verified 2026-11-01. Final learned 10/10 vs native 4/10, one
  seed; the paper's three-seed claims are unverified. Full receipts in the
  [snapshot](archive/coordination/coordination-20261101.md#ongoing-work).
- **RoboTwin final100 AD export (R4)**, Baige `job-vth2qv6r4lxu`: complete,
  exit 0, 2026-10-31; AD GPU fit deferred.
  [Final report](robotwin-final100-ad-export-20261031-r4/RESULTS.md).
- Work completed on 2026-10-28/29 (CMM calibration, DSRL Stage 0 smoke,
  native-context pilot r3, CTM/XL PPO gates, cached screens, Baige cleanup) is
  summarized in the [2026-11-01 snapshot](archive/coordination/coordination-20261101.md#recently-completed-work).
