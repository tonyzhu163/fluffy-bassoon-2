# CTM in-context RL: coordination

Live coordination for `fluffy-bassoon-2`, updated **2026-11-01 Asia/Shanghai**.
Keep current decisions and jobs here, the five shared documents in
[`docs/`](../docs/README.md), verified results in `docs/FINDINGS.md`, and dated
plans and evidence in `outputs/<run-id>/`. **Cleanup 2026-11-01:** the old
method docs and finished runs without usable evidence were deleted (git history
keeps them; old links are permalinks). Dead runs that code still loads moved to
[`archive/`](archive/) under the same run IDs. Live runs stay at the top level.
Record new results in the run folder and `docs/FINDINGS.md`; the old
`docs/frozen-vla-libero/EVALUATION.md`, `docs/paper/DRAFT.md` and
`docs/frozen-vla-steering-summary.md` are gone (paper: `docs/PAPER.md`). Follow [AGENTS.md](../AGENTS.md) and the
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

Contracts: [support/probe](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/ctm-support-probe.md),
[feature bridge](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/pi05-feature-bridge.md),
[PPO replay](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/docs/pi05-bridge-policy.md).
Results: [meta-RL evaluation](metta-common-ppo-chart-20261029-r1/ML10-EVALUATION.md).

### Redesign (2026-10-31)

No robot comparison so far shows memory or adaptation. The governing plan is
the [frozen pi0.5 steering screen](archive/frozen-pi05-steering-screen-20261031-r1/PLAN.md):
first show a controller can improve the frozen motor on cases it fails, then
show experience changes a useful later decision. The
[steering-headroom audit](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/outputs/steering-headroom-audit-20261031-r1/AUDIT.md) gives
the paired fixed100 evidence. The earlier
[headroom diagnosis](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/outputs/archive/icrl-headroom-diagnosis-20261031-r1/PLAN.md) is
archived as superseded, with corrections listed at its top. A
[criteria proposal](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/outputs/steering-interface-criteria-20261031-r1/PROPOSAL.md)
informs the newly authorized [targeted scope](frozen-pi05-endtoend-20261031-r1/SCOPE.json),
which adopts A1/A2/A3 with the [review safeguards](archive/frozen-pi05-steering-screen-20261031-r1/CRITERIA-REVIEW.md).
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
| RoboCasa rollout batching capacity | Root; Baige job-pvcv8in7o1kb terminal exit0 on all4nodes; source3bda6a0a | Same8fullOpenDrawer750cases ×8GPU workers:64EP/48,000controls/0SGD. Mean1/2/4/8lane rates15.76/27.95/51.67/86.93controls/s/GPU; batch8 besttested5.51×, sampled18.77GiB. Closed2,732GPU-s + prior640 =3,372/14,400. Small BF16batch drift recorded; activejobs unchanged, Fcolocation not qualified. Human cap40experimentGPUs excludesdev. [Audited result](robocasa-rollout-batch-capacity-20261102-r3/PARENT-REVIEW.json). |
| LIBERO research / RoboCasa comparison priority | Owner chat `01a1965f-42f4-7dc0-82b6-3bf65118e069`; research only, no jobs | **2026-11-01 human decision: defer LIBERO and focus on RoboCasa365 for ordinary PPO vs Flora-rewarded PPO vs meta-learning trained on the resulting chronological PPO histories.** Keep the existing [RoboCasa implementation/protocol/execution owners](robocasa-flora-ppo-preparation-20261101-r1/AUTHORITY.json); compare true simulator success and interaction efficiency on held-out cases, with matched PPO arms and explicit meta-training/deployment costs. |
| LIBERO task44 DSRL on frozen pi0.5 | HUMANPAUSE: job-2qn8dpw3yvlb ManualTermination12:59:03/allocationreleased | Fullclosed406780updates/2389TRAIN/20833replay,SHA7d2d0532;2350S39F,last400120panel20/20.376339actions=staleindex376249+lastclosed90;9RPCcomplete/no nextintent. OwnerGPUupper40992;rawclockunfinished/currentexitunknown/staleR2exit1preserved. Fixed500k/finalnativeNOTcomplete;alloldlaunchesheld. [Auditedclosure](owned-jobs-pause-20261101-r1/PARENT-REVIEW.json). |
| Frozen pi0.5 interface/data redesign | HUMANPAUSE: R2naturallySucceeded12:57:11;allallocationsreleased/successorsheld |319valid+1nullabort161/160initialomissions;native78/80,tied70/79+1abort,mean78/80,sample78/80.53081actions/2643GPU-s/finalC565unchanged/fullrecovery5b60163e. No initialcurve/source-studentadmission. Cfit/a7NEVERallocated/0charge;oldheartbeatpausedafteraudit. NewpriorityRLinfPPOsimrewardvsFloraRewarder separateprotocol,notlaunchedhere. [Closure](owned-jobs-pause-20261101-r1/PARENT-REVIEW.json). |
| LIBERO90 pi0.5 headroom screen for DSRL replication | Owned Stage A terminal Succeeded/exit0, `4ec35eb9`; owner chat `01a192cf` | **Nov1 09:54 saved-reference review:** own384 audit/task59 native3/16 vs untrained tied9/16 unchanged (3,890GPU-s/134,629actions). Separate task44 recovery repair qualified; executiond236/scienceb665 resumes preserved learner to302180 with3newEP/480updates. Fixed500k/finalnative/motor pending; no duplicate learner or interface GPU launch. [Reconciliation and gate](libero90-pi05-headroom-screen-20261101-r1/REFERENCE-RECONCILIATION.md). |
| Flora benchmark order and gates | Owner chat `fa8e2bae`; planning only, no jobs | **2026-11-01 human direction:** LIBERO is a test bed only (task59 native3/16, tied9/16 on C565 if a Flora plumbing check is needed); report on RoboCasa365 with the RLinf-loadable pi0.5 checkpoint, then RoboTwin vs VLRLA, then real robots. Each benchmark passes Flora agreement/headroom, a sim- and Flora-reward positive control, then history collection and meta-learning. Meta-learner fixed to a plain causal decoder (AD/DPT/Vintix), now the `vla_history.py` default; a matched backbone ablation (decoder vs GTrXL vs CTM) may follow a main result. Steering interface: bounded action residual (human); alternatives conditioning residual, noise, direct action, velocity residual, sample-and-select ([review](../docs/literature/vla-steering-review.md)). Baseline and history learner share the interface; residual PPO is the matched baseline, full-weight PPO the reward check. [Order and gates](../docs/FLORA-PPO.md#benchmarks-in-order). |
| Residual PPO / RoboCasa365 | Owner chat `fa8e2bae`; code on `main` (`1b60857`); thu2 CPU tests passed (job 49948, 19/19), no GPU jobs | **2026-11-01:** bounded 6-D eef action residual per H50 chunk, bound 0.25×norm-stat std (0.5× only as a declared fallback), Markov PPO with sim or Flora terminal reward, history export for the decoder. Code changes listed first; reuses the full-weight owner's env, pi0.5 loading and Flora contract without touching their worktree. Residual bound, memoryless PPO, history export and deployment adapter written; RoboCasa binding waits for the full-weight owner's qualified runner and Flora scorer (theirs, not touched). Next: RoboCasa365 binding and zero-residual equivalence once the owner's runner qualifies. [Plan](robocasa-residual-ppo-20261101-r1/PLAN.md). |
| Flora-Rewarder PPO comparison / RoboCasa365 pi0.5 | Eight-hour monitor window CLOSED Nov2 07:56:01; all comparison benchmarks terminal; six-update2.5MbudgetAPPROVED; joint launch awaits separate taskgate decision | Full250 qualification/scoring complete: native29/37/23/7/17 and AUROC.966/.900/.977/.894/.923. AllAUCgates pass; Counter74% remains failed70nativegate and explicit keep/replace decisionpending. Nov2 09:39 human approvedsix-update2.5Mstage, including180k alreadyspent and2,301,020remainingproposedbound; HUMAN-ACTION-AUTHORITY saved. Taskauthoritystillnull/NOT_ADMITTED, nojointSGD. Native15,884+Flora5,972GPU-s; closedbenchmarktotal75,141,unusedsealed/0benchmarkholds. Devcontainerreachable,lastattested1GPU/lifetimechargeunknown. OldS6-update fullcheckpoint14654853B remains preserved; statonly, no modelrestore/rehash. Common/S/F entriesreviewed; source629b5ef0 fixes local6shard normal-exit race without weakening80labelbarrier; main6c03426f carries status. ActualcombinedjointGPUiteration/colocationunqualified. Ownedrecurringmonitor DELETED afteroriginal8h audit; noendedjobpoll/ACKloops. [Closeout](robocasa-eight-gpu-speed-20261102-r1/MONITOR-CLOSEOUT.json). ResearchcomparisonINCOMPLETE; no oldbudgettransfer or taskchange withoutnewhuman decision. |
| OpenDrawer Flora PPO catch-up | Owner chat `01a199ef-f9c0-75c1-8c1e-1c88c63a17f1`; Baige `job-btcpf0pdsj0c`, source `13b88083` | **2026-11-02T11:41:34+08:00: 4 complete Flora PPO updates/256 TRAIN episodes, all64-label barriers and full boundary checkpoints verified; no errors.** Latest update at198000controls, checkpointfd6ddc55; no performance-improvement claim. Up to6 updates, final50 ID+50 OOD, max500k actions/359min/no retry;20-minute heartbeat active. [Advancement](robocasa-opendrawer-flora-first-20261102-r1/OPTIMIZER-ADVANCEMENT-REVIEW.json). |
| Saved simulator-PPO checkpoint eval50 | Owner chat `01a199ef-f9c0-75c1-8c1e-1c88c63a17f1`; Baige `job-7gjull5v6epq`, source `7666d294` | **2026-11-02T11:41:34+08:00: all8 GPU checks and strict saved S6 checkpoint restore passed; native ID episodes advancing, 82350 controls observed.**50 OpenDrawer ID+50 CloseFridge OOD,0new SGD; max100k newactions/119min/24GiB, original SHA84b4a25f unchanged.20-minute heartbeat active; next complete both panels. [Startup](robocasa-simulator-ppo-eval50-20261102-r1/STARTUP-REVIEW.json). |
| OpenDrawer PPO-history AD/DPT convergence | Owner chat `01a199fb-3586-7171-bd20-52fa70e5378c`; Baige CPU preparation complete, GPU launch pending | **Nov2:** paired336 TRAIN/48 VAL episodes exported;28 core checks pass and operational resume is bitwise identical to uninterrupted training. Human authorized validation-plateau fitting and comparison against initial/final PPO on50 identical fresh cases; full/static controls included. Next publish qualified source and submit a separate two-GPU job. [Plan](opendrawer-ppo-ad-dpt-20261102-r1/PLAN.md). |

| FLORA camera coverage / RoboCasa365 | Owner chat `01a19839-7acf-7301-afab-88ab56612045`; source bd1bce60 CPU-qualified | Human selected left + right + wrist with one shared official terminal prompt and native instruction, no task-specific rubrics. New synchronized left/wrist/right recordings and all-view checks default to a distinct three-view protocol; archived one-view results/calibration stay separate. Baige allocated CPU core checks15/15 passed; native three-view episode, throughput/memory and agreement remain for experiment owner01a18677 under existing budgets. Active run worktrees were not changed. [Validation](robocasa-flora-three-view-20261102-r1/CPU-VALIDATION.json). |

### Needs owner refresh

- Native VLRLA-recipe PPO pilot: last recorded 2026-10-31 14:06 Shanghai as
  queued behind the task57 DSRL run, which has since finished. Owner
  `frozen-vla-poc`: confirm whether it is still queued.
  [Plan](archive/vlrla-native-ppo-pilot-20261031-r1/PLAN.md).
- Three jobs last recorded on 2026-10-28 (RoboTwin GRPO gate
  `job-vh9lp7fvgf9n`, MetaWorld pilot r2 `job-x2x4kdpfn011`, thu2 CMM
  48983/48984) are listed in the
  [snapshot](archive/coordination/coordination-20261101.md#needs-owner-refresh).
  Re-add a row only if the job is still live.

## Paused

- **Meta-World released pi0.5 noise screen**, Baige `job-orgyjb0z9h6d`: paused
  at user request 2026-10-29; reach screens fail headroom.
  [Pause record](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/741064b4d8df00555f92a7a8449055c0edb50707/outputs/metaworld-pi05-noise-baige-20261029-r2/PAUSED.md).


## Recently completed work

- **Original Pi0 DSRL task57 replication**, Baige `job-psgxs9n5b08w`:
  complete and verified 2026-11-01. Final learned 10/10 vs native 4/10, one
  seed; the paper's three-seed claims are unverified. Full receipts in the
  [snapshot](archive/coordination/coordination-20261101.md#ongoing-work).
- **RoboTwin final100 AD export (R4)**, Baige `job-vth2qv6r4lxu`: complete,
  exit 0, 2026-10-31; AD GPU fit deferred.
  [Final report](archive/robotwin-final100-ad-export-20261031-r4/RESULTS.md).
- Work completed on 2026-10-28/29 (CMM calibration, DSRL Stage 0 smoke,
  native-context pilot r3, CTM/XL PPO gates, cached screens, Baige cleanup) is
  summarized in the [2026-11-01 snapshot](archive/coordination/coordination-20261101.md#recently-completed-work).
