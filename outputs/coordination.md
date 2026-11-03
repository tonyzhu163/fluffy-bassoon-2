# CTM in-context RL: coordination

Live coordination for `fluffy-bassoon-2`, updated **2026-11-03 Asia/Shanghai**.
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

The [2026-11-03 snapshot](archive/coordination/coordination-20261103.md) holds the table before the
2026-11-03 trim (terminal rows, superseded residual/interface rows). The
[2026-11-01 snapshot](archive/coordination/coordination-20261101.md) holds
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
- Budget: **no fixed action cap** (human, 2026-11-03), removing the earlier
  ~1M source actions per learner. Still charge each learner's actions to its
  own ledger and report them.
- Baseline priority (human, 2026-10-31): same-motor DSRL/SAC first, native
  VLRLA PPO afterward. `frozen-vla-poc` (`finish_assembly`/`finish_ppo`) is
  the sole LIBERO launcher; RoboTwin AD GPU work stays deferred.
- Active `codex/*` branches were merged into `main` on 2026-10-31; merge
  further verified work promptly (see [AGENTS.md](../AGENTS.md)).

## Ongoing work

Rows are as last recorded by their owners; terminal and superseded rows moved
to the [2026-11-03 snapshot](archive/coordination/coordination-20261103.md#ongoing-work).

| Work | Cluster / job / source | Last check, progress and next action |
| --- | --- | --- |
| Deterministic steering evaluation repair | Owner chat `01a1a00f`; Baige Pro6K / 算法Z组 `job-23fst4fg3idu`; source `e81a4fa6` | **Nov3 15:54 Shanghai:** 59/59 core tests pass again on merged main; corrected one-GPU native qualification running with 15-minute limit. R1 failed its visibility probe before model/controls, terminal/released with87GPU-s carried into the one-GPU-hour / 10,000-control cap; existing training attempts unchanged. Next verify probe, five-case zero-steering agreement and release. [Plan](robocasa-deterministic-eval-fix-20261103-r1/PLAN.md). |
| Flow-SDE source → learned-KV AD/DPT route | Owner chat `01a1a01f`; design adopted, no new allocation | **Nov3 human decision:** prioritize a stronger action-expert Flow-SDE PPO source, then fit AD/DPT learned-KV bridges through the frozen original VLA using action-derived flow targets. One source serves both students; no inverse steering labels or separate steering-PPO source required. Core docs reconciled; existing direct-action path and differentiable KV primitive identified. Integration/native qualification and improving source remain pending. Existing held runs/contracts/budgets unchanged. [Contract](flow-sde-kv-bridge-20261103-r1/PLAN.md). |
| Steering PPO KL guard ablation | Owner chat `01a19fe9`; Baige Pro6K `job-1nds2rm4v2rp` RUNNING4GPUs; source `12d1f241` | **Nov3 15:38 Shanghai:** all4 actual probes/zero-parity/influence receipts pass; initial500 full600-control episodes complete, all4guarded panels advancing; 716/1500 total complete, no worker failure. Same6GPUh cap incl prior200GPU-s; deadline16:44:09, current throughput fits. Next complete paired panel/artifact and release audit. Full PPO HUMAN HOLD at2/50. [Contract](steering-kl-guard-ablation-20261103-r1/PLAN.md), [state](steering-kl-guard-ablation-20261103-r1/STATE.json). |
| PPO scalar metric relay through Mac | Owner chat `01a19fe2`; Baige read-only SSH → local mirror → W&B; heartbeat every20minutes | **Nov3 14:12 Shanghai:** both current velocity/KV streams mirrored through update2 (5events each,32857bytes total), hashes/identities/sequences checked. Periodic relay active; first authenticated Mac upload still HTTP403 on official API, so cloud sync unverified. Credentials/routing outside Git; no training/job changes. [Status](ppo-metrics-relay-20261103-r1/STATUS.json), [relay](ppo-metrics-relay-20261103-r1/README.md). |
| RoboCasa 16/32/48 environments per GPU and larger PPO scale | Owner chat `01a19d07`; capacity jobs released; four-method pipeline and read-only performance audit complete | **Nov3 14:24 Shanghai:** both source arms paused at2/50 under owner01a19b74; latest human KL-review hold preserved, no job/source mutation here. Real production uses15–33 native calls/rank vs12×B48 qualification; optimizer1.38s/10.91s within375.45s/400.22s phases. Same2064-feature/native32/1152 paths; IO/reset/fragmentation effects unseparated. Source-owner closeout plus remaining48TRAIN/10DEV/final and assumed240s reload+120s release forecasts47.07/49.97GPUh (97.05combined) within newly approved60/arm120combined; serial speedup unproven and this diagnosis non-gating. Pipeline source414ab5d9 passed98LinuxCPU tests/8synthetic fits/4exact resumes. [Performance diagnosis](robocasa-rollout-capacity-3248-20261103-r1/PERFORMANCE-DIAGNOSIS-20261103.md), [pipeline](steering-four-method-pipeline-20261103-r1/RESULT.md), [human budget](steering-ppo-ad-dpt-20261102-r1/SOURCE-BUDGET-60PERARM-20261103-r1.json). |
| RoboCasa365 atomic_seen baseline sweep | Owner chat `01a1a04a`; Baige Pro6K `job-jnlxhvwhrdji`, 8 GPUs; source `42cefb77` | **Nov3 15:42 Shanghai:** all eight allocated GPUs passed the in-job probe, and 29 Linux core tests passed; native workers are loading for the 12-task / 240-episode sweep (147,000 policy controls, zero SGD). Next: verify completed episodes, then report first/end success with Wilson intervals; 55-minute allocation, no automatic retries. [Launch and evidence](robocasa-atomic-baseline-sweep-20261103-r1/SUBMITTED.json), [plan](robocasa-atomic-baseline-sweep-20261103-r1/PLAN.md). |
| Flora rubric factorial replay (proposal) | Claude Code session (tonyzhu163) proposes; no Baige route, **needs a Baige owner**; plan `dd82beeb`, not launched | Replaces untested revision 2, which drops the OpenDrawer "fully open" sentence the audit hypothesis depends on. **Stage A** drawer ground truth for the 183 TRAIN failures: A1 read saved monitor success-step fields (no approval); A2 open-loop replay of saved actions from native seeds on CPU, final drawer openings (**needs human approval**, simulator controls, zero inference/SGD). **Stage B** same 634 videos, four arms O/S/T/R1 (original, +shared text, +task text, revision 1; text byte-for-byte from revision 1), 2,536 labels, **~12.4k GPU-s, ceiling 18,000, needs human approval**. Readout: TRAIN within-case pairwise accuracy primary (baseline O 0.937, R1 0.918), with E/P5/G5 reward-map ablation; decision rules fixed in the plan. Production prompt, map and caches unchanged. Owner: add your own row when submitted. [Plan](robocasa-flora-rubric-factorial-20261102-r1/PLAN.md). |
| Steering PPO histories → AD/DPT convergence and curves | Owner chat `01a199fb`; **explicit human KL-review hold atupdate2;60GPUh/arm remains approved** | **Nov3 14:49 Shanghai:** independently read actual human reply`01a1a00f` in`01a19fe9`:Hold recovery until the KL test is reviewed. Producer cursor91 active on independent Flow pipeline;root hold record confirms0newrecovery executors. No restart-from-zero recipe selected. Preserve both event005 actor/value/Adam/sampler/global+workerRNG/state,768TRAINepisodes/arm,before/DEV0 and27058GPU-s selectedcosts. Human60GPUh/arm120combined and24Linux-core-tested budget code remain valid,but do not override KLhold. No full source/newstudent/nativecurve admitted;no extra source collection,partial-source fitting or otherowner takeover. Monitor explicit human review/recipe/release then exact qualified source recovery/full50completion and downstream independent convergence/readouts;quiet unchanged hold. [Direct hold](steering-ppo-ad-dpt-20261102-r1/KL-REVIEW-HOLD-OBSERVATION-20261103-r1.json),[budget](steering-ppo-ad-dpt-20261102-r1/SOURCE-BUDGET-60PERARM-20261103-r1.json),[registry](steering-ppo-ad-dpt-20261102-r1/REGISTRY.json). |
| Flora prefix first-success scoring | Claude Code session (tonyzhu163) proposes; **human approved 24,000 GPU-s 2026-11-02**; no Baige route, **needs a Baige owner to launch**; not launched | Score the 384 TRAIN + 50 CALIBRATION OpenDrawer videos at each RLinf chunk boundary (50…400 controls) with the original prompt: 3,472 prefix labels, frame-limited collage with no re-encoding, 0 new controls/SGD. Pick (P5 threshold, K consecutive checks) on even cases, report odd cases; adopt as the gate-1 Flora first-success reward if held-out false successes ≤15%, recall ≥0.80 and early fires ≤10%, otherwise use the fixed fallbacks. Needs Stage A1's `success_steps` check first. ~17k GPU-s estimated, **approved ceiling 24,000**, ≤4 GPUs, one allocation, no automatic retry. [Plan](flora-prefix-first-success-20261102-r1/PLAN.md). |
| Joint five-task frozen velocity / all-layer KV PPO | Owner parent `01a19b74`; **HUMAN HOLD at update2 pending KL-test review**, all original allocations released | Root independently verified direct human turn `01a1a00f-157f-7521-a92a-e79d52a0b5fa`: “Hold recovery until the KL test is reviewed.” No recovery/new full-source executor submitted. Approved lifetime60GPUh/arm120combined does not override this hold. Preserve both event005 checkpoints, full actor/value/Adam/sampler/global+worker RNG/state, all768TRAIN episodes/arm, before/DEV0 and27058GPU-s costs. KL owner `01a19fe9` owns copied-state bounded ablation; possible restart is not yet a chosen recipe. Independent Flow four-method work continues. [Controlling hold](robocasa-joint-steering-scale50-20261103-r1/KL-REVIEW-HOLD.json), [pause/state](robocasa-joint-steering-scale50-20261103-r1/PAUSE-UPDATE2.json), [budget](steering-ppo-ad-dpt-20261102-r1/SOURCE-BUDGET-60PERARM-20261103-r1.json). |
| Flow-SDE four-method source preparation | Owner parent `01a19b74`; **R2 full-state fit continuation RUNNING** Baige `job-qbxwsy8jjdsb`, native readout pending | Nov3 15:45: exact new exclusiveGPUprobe passed(host job-qbxwsy8jjdsb-master-0,PCI A4); AD/DPT4700updates and DIT/IQL actor1500/1400 advance above preserved R1 boundaries, finite losses. R1 operational exit15:38:47/allchildren0/released, none converged; complete policy/value/targetQ/Adam/RNG/sampler/plateau retained. R2 submitted once15:42:09,1RPBZZZ6/16CPU/64GiB/8SHM/65min/autoretryOFF, immutablefit e7ead0ee core/launcher match reviewedmain42cefb77. Selected prep1021+R1fit3069=4090GPU-s once; R2 live cost pending release. Whole7.5GPUh cap and prospective50ID+50OOD x5attempts/full-reset/initial-final anchors unchanged. Actual selected-student native qualification/pace/full-cost admission pending; no robot scores/meta-learning claim. [Execution](flow-sde-four-methods-20261103-r1/EXECUTION.json), [r2 admission](flow-sde-four-methods-20261103-r1/FIT-R2-ADMISSION.json), [r2 launch](flow-sde-four-methods-20261103-r1/FIT-R2-LAUNCH.json), [r1 release](flow-sde-four-methods-20261103-r1/FIT-R1-RELEASED.json). |
| Gate readiness and native-criteria Flora rubric | Owner chat `01a19bef-b16f`;Baige frame-edge r1 COMPLETE/REJECTED `job-pb53j1e4jom1`, source `1f29377b`;GPU released13:45:32 Nov3 |40/40grades and4anchor repeats verified; botharms6/10negative score5,3/4false episodes; clause truefullscores4→5/8 and timelyconfirmed1→2/3,128stillmissed,216lateclose1. Actual19processor inputs areBF16 and byte-match CPUcastrefs;40RGB/message audits and54file hashes pass.575workerGPU-s/590scheduler;subtotal7925workerGPU-s excludingearlierterminalvideo work. Offline external latch fixesforgettingofrecognizedsuccesses but not3persistentfalseepisodes; no promotion/cutoffchange or further same-panelwordingfit. Next scientificdiagnostic:separate current-state recognition from temporalfirstsuccess aggregation. No GPU job active/new trial launched. [Verified result and all scores](robocasa-flora-drawer-frame-edge-20261103-r1/RESULT.md). |
| Flora current-state recognition diagnostic | Owner chat `01a1a006`; Baige `job-6zq2cpuq1c7c` SUCCEEDED/released15:30:24 Nov3; source`db2ef630` | All76scores/4repeats/37actualBF16-input matches complete. Prefix grade FP7/12,TP5/6; checklist FP10/12,TP6/6; reset/current pairs TP0/6. All candidates rejected.1003wrapper/1010scheduler GPU-s; r1 CPU failure0GPU. Raw receipts retained on Baige; archive download blocked by organization; inspected aggregates recorded with that limitation. Next: bound/qualify official per-step rubric and camera inputs before another diagnostic. No production/PPO change. [Result](robocasa-flora-current-state-20261103-r2/RESULT.md), [evidence](robocasa-flora-current-state-20261103-r2/OBSERVED-RESULT.json). |
| Research execution / five-update visibility | Coordinator chat `01a19f33`; full PPO HUMAN HOLD2/50; KL diagnostic R2 queued | **Nov3 15:15 Shanghai:** repaired diagnostic `job-1nds2rm4v2rp` submitted once on freshsource `12d1f241`, 4RPBZZZ6/89min; queue screenshot independently inspected, output absent, actual GPUprobe/episodes pending. R1 terminal failed before model/episodes;200GPU-s conservative charge carried,21400 remain in same6GPUh diagnostic cap. Full velocity/KV sources remain held2/50 until KL review;120GPUh budget does not lift hold. No DEV5/completed source. [Check](research-coordination-20261103-r1/CHECK-20261103-1515.json), [hold](research-coordination-20261103-r1/HUMAN-HOLD-KL-REVIEW.json), [state](research-coordination-20261103-r1/STATE.json). |
| Per-task Flow-SDE PPO teacher campaign, 10M controls/task | Claude Code session (tonyzhu163); **plan and static audit only, not launched; needs a Baige owner** | **2026-11-03 human decisions:** simulator first-success reward; full action-expert Flow-SDE PPO, one independent 8-GPU run per task, ~10M controls each; KL penalty and KL stopping off, KL metrics still logged; collect and audit all five corpora before any meta-learner. Supersedes the separate velocity/KV Stage 1 PPO arms as source campaigns ([PAPER decision](../docs/PAPER.md)); those rows stay under their owners' hold. Static audit found blockers to fix before launch. [Plan](robocasa-flow-sde-10m-per-task-20261103-r1/PLAN.md). |
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

## Paused and recently completed

See the [2026-11-03 snapshot](archive/coordination/coordination-20261103.md#paused) and
FINDINGS for paused work and completed runs (OpenDrawer Flow-SDE PPO, AD/DPT
readouts, matched memory controls, Stage 0 screen, joint calibration).
