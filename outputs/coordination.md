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
| Native pi0.5 first-success timing / RoboCasa365 | Owner chat `01a19a71`; Baige `job-kpdi47srqc2m` manually stopped; source `c5d271b4` | **Human selected600 controls (12 H50 chunks); 126 complete EP, observed per-task first-success maxima279/538/538/317/524, no full population-p99 claim.** Both GPUs released13:00 Shanghai;90,600 actions/0pending/0SGD, cumulative GPU upper2,260/9,600s. One DrawerToCounter end-success lost at600; future600-step comparison needs matching Flora calibration. [Decision and evidence](robocasa-pi05-success-timing-20261102-r3/RESULT.md). |
| RoboCasa rollout batching capacity | Root; capacity benchmark terminal; continuous eight-lane production core qualified on allocated Linux | Eight lanes/GPU measured5.51× singleton speed with one shared motor. Future matched S/F templates use continuous refill,64CPU/256GiB per two-GPU node and80episodes/update. Twelve affected core cases pass (11carried,1fixture repair); noise/order/nextAdam/recovery protected. Require27,408MiB free after Flora warmup; colocation still unqualified. Active runs unchanged;40experimentGPUs excludesdev. [Integration](robocasa-rollout-batch-integration-20261102-r1/QUALIFICATION.json). |
| LIBERO research / RoboCasa comparison priority | Owner chat `01a1965f-42f4-7dc0-82b6-3bf65118e069`; research only, no jobs | **2026-11-01 human decision: defer LIBERO and focus on RoboCasa365 for ordinary PPO vs Flora-rewarded PPO vs meta-learning trained on the resulting chronological PPO histories.** Keep the existing [RoboCasa implementation/protocol/execution owners](robocasa-flora-ppo-preparation-20261101-r1/AUTHORITY.json); compare true simulator success and interaction efficiency on held-out cases, with matched PPO arms and explicit meta-training/deployment costs. |
| LIBERO task44 DSRL on frozen pi0.5 | HUMANPAUSE: job-2qn8dpw3yvlb ManualTermination12:59:03/allocationreleased | Fullclosed406780updates/2389TRAIN/20833replay,SHA7d2d0532;2350S39F,last400120panel20/20.376339actions=staleindex376249+lastclosed90;9RPCcomplete/no nextintent. OwnerGPUupper40992;rawclockunfinished/currentexitunknown/staleR2exit1preserved. Fixed500k/finalnativeNOTcomplete;alloldlaunchesheld. [Auditedclosure](owned-jobs-pause-20261101-r1/PARENT-REVIEW.json). |
| Frozen pi0.5 interface/data redesign | HUMANPAUSE: R2naturallySucceeded12:57:11;allallocationsreleased/successorsheld |319valid+1nullabort161/160initialomissions;native78/80,tied70/79+1abort,mean78/80,sample78/80.53081actions/2643GPU-s/finalC565unchanged/fullrecovery5b60163e. No initialcurve/source-studentadmission. Cfit/a7NEVERallocated/0charge;oldheartbeatpausedafteraudit. NewpriorityRLinfPPOsimrewardvsFloraRewarder separateprotocol,notlaunchedhere. [Closure](owned-jobs-pause-20261101-r1/PARENT-REVIEW.json). |
| LIBERO90 pi0.5 headroom screen for DSRL replication | Owned Stage A terminal Succeeded/exit0, `4ec35eb9`; owner chat `01a192cf` | **Nov1 09:54 saved-reference review:** own384 audit/task59 native3/16 vs untrained tied9/16 unchanged (3,890GPU-s/134,629actions). Separate task44 recovery repair qualified; executiond236/scienceb665 resumes preserved learner to302180 with3newEP/480updates. Fixed500k/finalnative/motor pending; no duplicate learner or interface GPU launch. [Reconciliation and gate](libero90-pi05-headroom-screen-20261101-r1/REFERENCE-RECONCILIATION.md). |
| Flora benchmark order and gates | Owner chat `fa8e2bae`; planning only, no jobs | **2026-11-01 human direction:** LIBERO is a test bed only (task59 native3/16, tied9/16 on C565 if a Flora plumbing check is needed); report on RoboCasa365 with the RLinf-loadable pi0.5 checkpoint, then RoboTwin vs VLRLA, then real robots. Each benchmark passes Flora agreement/headroom, a sim- and Flora-reward positive control, then history collection and meta-learning. Meta-learner fixed to a plain causal decoder (AD/DPT/Vintix), now the `vla_history.py` default; a matched backbone ablation (decoder vs GTrXL vs CTM) may follow a main result. Steering interface: bounded action residual (human); alternatives conditioning residual, noise, direct action, velocity residual, sample-and-select ([review](../docs/literature/vla-steering-review.md)). Baseline and history learner share the interface; residual PPO is the matched baseline, full-weight PPO the reward check. [Order and gates](../docs/FLORA-PPO.md#benchmarks-in-order). |
| Residual PPO / RoboCasa365 | Owner chat `fa8e2bae`; code on `main` (`1b60857`); thu2 CPU tests passed (job 49948, 19/19), no GPU jobs | **2026-11-01:** bounded 6-D eef action residual per H50 chunk, bound 0.25×norm-stat std (0.5× only as a declared fallback), Markov PPO with sim or Flora terminal reward, history export for the decoder. Code changes listed first; reuses the full-weight owner's env, pi0.5 loading and Flora contract without touching their worktree. Residual bound, memoryless PPO, history export and deployment adapter written; RoboCasa binding waits for the full-weight owner's qualified runner and Flora scorer (theirs, not touched). Next: RoboCasa365 binding and zero-residual equivalence once the owner's runner qualifies. [Plan](robocasa-residual-ppo-20261101-r1/PLAN.md). |
| Flora-Rewarder PPO comparison / RoboCasa365 pi0.5 | Eight-hour monitor window CLOSED Nov2 07:56:01; all comparison benchmarks terminal; six-update2.5MbudgetAPPROVED; joint launch awaits separate taskgate decision | Full250 qualification/scoring complete: native29/37/23/7/17 and AUROC.966/.900/.977/.894/.923. AllAUCgates pass; Counter74% remains failed70nativegate and explicit keep/replace decisionpending. Nov2 09:39 human approvedsix-update2.5Mstage, including180k alreadyspent and2,301,020remainingproposedbound; HUMAN-ACTION-AUTHORITY saved. Taskauthoritystillnull/NOT_ADMITTED, nojointSGD. Native15,884+Flora5,972GPU-s; closedbenchmarktotal75,141,unusedsealed/0benchmarkholds. Devcontainerreachable,lastattested1GPU/lifetimechargeunknown. OldS6-update fullcheckpoint14654853B remains preserved; statonly, no modelrestore/rehash. Common/S/F entriesreviewed; source629b5ef0 fixes local6shard normal-exit race without weakening80labelbarrier; main6c03426f carries status. ActualcombinedjointGPUiteration/colocationunqualified. Ownedrecurringmonitor DELETED afteroriginal8h audit; noendedjobpoll/ACKloops. [Closeout](robocasa-eight-gpu-speed-20261102-r1/MONITOR-CLOSEOUT.json). ResearchcomparisonINCOMPLETE; no oldbudgettransfer or taskchange withoutnewhuman decision. |
| OpenDrawer Flora PPO catch-up | Owner chat `01a199ef-f9c0-75c1-8c1e-1c88c63a17f1`; Baige recovery `job-9fa6axdtbhlw` SUCCEEDED12:58:31, source `72ae2202` | **Terminal verified:** six updates/384 TRAIN episodes; final31/50 OpenDrawer ID and28/50 CloseFridge OOD. All384 official labels/48 shards audited, all100 final sidecars/canonical cases verified; full5760-decision history equals retained4-update prefix plus continuation. All4 exit0; original+resume430,528 controls (5 old unresolved charged)/46,384GPU-s; original21600s clocks closed, output11.37GiB within112GiB. FinalSHA1e4184af retained. Original full matching baseline now human-requested; no before/after or ICRL claim. Flora terminal monitoring complete; shared heartbeat continues AD/DPT plus original baseline. [Audit](robocasa-opendrawer-flora-resume-20261102-r1/TERMINAL-REVIEW.json). |
| Saved simulator-PPO checkpoint eval50 | Owner chat `01a199ef-f9c0-75c1-8c1e-1c88c63a17f1`; Baige `job-7gjull5v6epq` SUCCEEDED, source `7666d294` | **Complete:** saved S6 policy scored34/50 OpenDrawer ID and32/50 CloseFridge OOD. All100 independent sidecars audited;82,500 controls,0unresolved/0SGD,7700GPU-s,4exit0 and original checkpoint hash unchanged. Evaluation monitor deleted. These are after-training panels; do not substitute the different calibration bank for matched before data. [Terminal audit](robocasa-simulator-ppo-eval50-20261102-r1/TERMINAL-REVIEW.json). |
| Original pi0.5 matching eval50 | Owner chat `01a199ef-f9c0-75c1-8c1e-1c88c63a17f1`; Baige `job-028lwmw7ikwc` SUCCEEDED13:36:35, source `0be8cc27` | **Terminal verified13:48:** untouched original28/50 ID and28/50 OOD; trained S34/32, F31/28. All100 sidecars, actual case/noise schedules for all3 policies and1650 zero-residual/base-identical chunks verified; checkpoint unchanged/0SGD. All4 exit0,82,500 controls/0unresolved/7,332GPU-s. Initial public proprio matches; cross-run frozen outputs remain nonbitwise. No net OOD drop for F versus original; one seed/small-panel limits. Original/S/F execution complete; now investigating reward alignment and PPO dynamics on retained data, shared monitor continues AD/DPT. [Final comparison](robocasa-original-pi05-eval50-20261102-r1/TERMINAL-REVIEW.json). |
| OpenDrawer PPO-history AD/DPT convergence | Owner chat `01a199fb`; Baige training `job-8kb761dxfvtm`, immutable core `68f2c5aa`; independent driver `e8c76416` | **Nov2 13:40 Shanghai:** AD/full+static converged1900, DPT/full23800; DPT/static continues. Per-model evaluation barrier removed and AD dispatched separately, but `job-gv4po7gq8aaf` failed exact initial-feature/base pairing; DPT/full `job-t4xwnjnjuxp9` stopped to hold the same issue. Preserve/charge all records; diagnose repeated native reset/render identity before new attempts, no paired result admitted. [Independent dispatch](opendrawer-ad-first-id-ood-20261102-r1/PLAN.md), [diagnosis](opendrawer-reset-pairing-20261102-r1/PLAN.md). |

| Joint residual PPO: uncapped, fixed-count, more PPO steps, 8 lanes/GPU, streamed Flora | Claude Code session (tonyzhu163); code on `main`; no Baige access from this session, thu2 has no RoboCasa | **2026-11-02 human decisions:** no all-in action cap (still journaled); fixed 6 iterations, DEV recorded but never stopping; PPO 10 epochs × minibatch 32; 8 lanes/GPU with continuous refill and background resets; F scorers stream grades during the batch (reused only for identical jobs and video bytes, all labels before GAE). **Now also done:** resume for fixed-count S/F slices (set `TASK_RESUME_FROM`/`TASK_RESUME_FROM_SHA256` to the previous slice's `RESUME-REQUIRED.json`; new run id, same source); first-success timing and p99 horizons belong to the native pi0.5 first-success timing row (Baige `job-uiwnmnif0zfz`). **Horizon (human, 2026-11-02):** five PPO tasks at 600 for training, Flora and ID eval; OOD tasks keep native horizons (no Flora, no clock input), with success at control 600 as a secondary number. Pending implementation after the current bug fixes: set the four 750 tasks to 600 in `TASKS`, update the 750-based tests, recalibrate Flora and recheck headroom at 600, re-run the joint before-panels and rebind templates ([FLORA-PPO](../docs/FLORA-PPO.md#benchmarks-in-order)). **Keep for meta-learning:** every `pending-NNN.pt`, `boundary-NNN.pt`, `labels-NNN/` and monitoring record of any arm whose histories feed AD/DPT/DIT (only copy of behavior-policy mean/std and raw Flora scores; [review](../docs/literature/in-context-rl-review.md#methods-beyond-ad-and-dpt-for-a-ppo-history-dataset)). Render stays 256×256 (Flora tiles each view at 320×320; changing it would shift pi0.5 inputs and void calibration). Joint SGD not launched; completed native-horizon hardware qualification is recorded in the following row, and the600-control migration still needs implementation, fresh calibration/headroom and before-panels. |
| Joint lane / streamed-Flora qualification | Owner chat `01a19a8c`; Baige `job-q7okv1fl1gda` SUCCEEDED14:18 Shanghai; runtime`6de858c3`, future source`c0bfd4a4` | **Memory/horizon fix qualified:**82/82 Linux core tests;10full600-control episodes/6000controls/120records/10streamed grades/zeroSGD; collection45.27controls/s/GPU (29.12including native load),85.14GiB peak,9.83GiB minimum free. Flora .75/batch8 leaves23608MiB after warmup, clearing unchanged4lane21973MiB guard by1635MiB;8lane guard still fails. Both future entries enforce4lanes and templates are rebound, but NOT_ADMITTED pending fresh600calibration250 and common-before540 plus actual resource bindings. [Audit](robocasa-joint-lane-qualification-20261102-r5/PARENT-REVIEW.json), [next-stage budget](robocasa-joint-lane-qualification-20261102-r5/NEXT-STAGE-PLAN.json). All owned allocations terminal;5attempt lifetime upper2800GPU-s. Lastcheck14:20; next await separate8GPU55min budget decision. |
| Residual PPO KL stopping and epoch diagnostics | Owner chat `01a19ac5`; core on main `25a2fc38`; Baige existing development allocation, CPU only | **Complete:73/73 Linux integration tests pass** at `461f6334`, including the concurrent600-control migration. Initial72-test qualification includes closed-form KL stopping, actual epoch/gradient-step counts, full-batch metrics, constant-return JSON and strict checkpoint settings. Disabled-KL learner matches previous committed model/Adam/RNG/old metrics exactly. Fresh RoboCasa training/common-before defaults to `target_kl=.02`; historical resumes preserve their setting. Per-epoch JSON/progress and initial/final diagnostics are wired. Check after each epoch; retain that epoch, no hard final-KL bound. No simulator collection, scientific training or new GPU allocation. [Integration qualification](residual-ppo-kl-diagnostics-20261102-r2/QUALIFICATION.json), [reference comparison](residual-ppo-kl-diagnostics-20261102-r1/reference-comparison.json). |
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
