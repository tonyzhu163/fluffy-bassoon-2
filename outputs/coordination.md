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
| Deterministic steering evaluation repair | Owner chat `01a1a00f`; Baige Pro6K / 算法Z组 `job-mnhfyjktzwi5`; source `7cee5f2e` | **Nov3 16:08 Shanghai:** 59/59 core tests pass; R3 native five-case qualification running, allocated GPU probe passed, 15-minute limit and no SGD. R1/R2 setup failures terminal/released with zero controls; cumulative202GPU-s carried into the one-GPU-hour / 10,000-control cap. Next verify exact cross-arm scenes/noise/batches/actions and release. [Plan](robocasa-deterministic-eval-fix-20261103-r1/PLAN.md). |
| Flow-SDE source → learned-KV AD/DPT route | Owner chat `01a1a01f`; design adopted, no new allocation | **Nov3 human decision:** prioritize a stronger action-expert Flow-SDE PPO source, then fit AD/DPT learned-KV bridges through the frozen original VLA using action-derived flow targets. One source serves both students; no inverse steering labels or separate steering-PPO source required. Core docs reconciled; existing direct-action path and differentiable KV primitive identified. Integration/native qualification and improving source remain pending. Existing held runs/contracts/budgets unchanged. [Contract](flow-sde-kv-bridge-20261103-r1/PLAN.md). |
| Steering PPO KL guard ablation | Owner chat `01a19fe9`; Baige Pro6K `job-1nds2rm4v2rp` RUNNING4GPUs; source `12d1f241` | **Nov3 15:38 Shanghai:** all4 actual probes/zero-parity/influence receipts pass; initial500 full600-control episodes complete, all4guarded panels advancing; 716/1500 total complete, no worker failure. Same6GPUh cap incl prior200GPU-s; deadline16:44:09, current throughput fits. Next complete paired panel/artifact and release audit. Full PPO HUMAN HOLD at2/50. [Contract](steering-kl-guard-ablation-20261103-r1/PLAN.md), [state](steering-kl-guard-ablation-20261103-r1/STATE.json). |
| PPO scalar metric relay through Mac | Owner chat `01a19fe2`; Baige read-only SSH → local mirror → W&B; heartbeat every20minutes | **Nov3 14:12 Shanghai:** both current velocity/KV streams mirrored through update2 (5events each,32857bytes total), hashes/identities/sequences checked. Periodic relay active; first authenticated Mac upload still HTTP403 on official API, so cloud sync unverified. Credentials/routing outside Git; no training/job changes. [Status](ppo-metrics-relay-20261103-r1/STATUS.json), [relay](ppo-metrics-relay-20261103-r1/README.md). |
| RoboCasa 16/32/48 environments per GPU and larger PPO scale | Owner chat `01a19d07`; capacity jobs released; four-method pipeline and read-only performance audit complete | **Nov3 14:24 Shanghai:** both source arms paused at2/50 under owner01a19b74; latest human KL-review hold preserved, no job/source mutation here. Real production uses15–33 native calls/rank vs12×B48 qualification; optimizer1.38s/10.91s within375.45s/400.22s phases. Same2064-feature/native32/1152 paths; IO/reset/fragmentation effects unseparated. Source-owner closeout plus remaining48TRAIN/10DEV/final and assumed240s reload+120s release forecasts47.07/49.97GPUh (97.05combined) within newly approved60/arm120combined; serial speedup unproven and this diagnosis non-gating. Pipeline source414ab5d9 passed98LinuxCPU tests/8synthetic fits/4exact resumes. [Performance diagnosis](robocasa-rollout-capacity-3248-20261103-r1/PERFORMANCE-DIAGNOSIS-20261103.md), [pipeline](steering-four-method-pipeline-20261103-r1/RESULT.md), [human budget](steering-ppo-ad-dpt-20261102-r1/SOURCE-BUDGET-60PERARM-20261103-r1.json). |
| RoboCasa365 atomic_seen baseline sweep | Owner chat `01a1a04a`; Baige `job-jnlxhvwhrdji` SUCCEEDED, all 8 GPUs released 15:52:35 Nov3; source `42cefb77` | **COMPLETE:** 12 tasks × 20 episodes, 147,000 controls, zero SGD/unresolved intents; all 720 videos decoded and 2,940 exact native chunk/noise schedules audited, conservative cost 6,328 GPU-s (1.758 h). Counter→Stove 10/20 is the new pick-and-place candidate; Cabinet 15/20, Sink→Counter 19/20 and Toaster→Counter 15/20 exceed 70%; full first/end rates and Wilson intervals in the [results](robocasa-atomic-baseline-sweep-20261103-r1/RESULT.md), with [audit](robocasa-atomic-baseline-sweep-20261103-r1/FINAL-AUDIT.json) and [cost](robocasa-atomic-baseline-sweep-20261103-r1/TERMINAL-ACCOUNTING.json); human portfolio selection and any fresh 50-case qualification next. |
| Flora rubric factorial replay (proposal) | Claude Code session (tonyzhu163) proposes; no Baige route, **needs a Baige owner**; plan `dd82beeb`, not launched | Replaces untested revision 2, which drops the OpenDrawer "fully open" sentence the audit hypothesis depends on. **Stage A** drawer ground truth for the 183 TRAIN failures: A1 read saved monitor success-step fields (no approval); A2 open-loop replay of saved actions from native seeds on CPU, final drawer openings (**needs human approval**, simulator controls, zero inference/SGD). **Stage B** same 634 videos, four arms O/S/T/R1 (original, +shared text, +task text, revision 1; text byte-for-byte from revision 1), 2,536 labels, **~12.4k GPU-s, ceiling 18,000, needs human approval**. Readout: TRAIN within-case pairwise accuracy primary (baseline O 0.937, R1 0.918), with E/P5/G5 reward-map ablation; decision rules fixed in the plan. Production prompt, map and caches unchanged. Owner: add your own row when submitted. [Plan](robocasa-flora-rubric-factorial-20261102-r1/PLAN.md). |
| Steering PPO histories → AD/DPT convergence and curves | Owner chat `01a199fb`; **KL native panel complete,final audit/review pending; source hold2/50** | **Nov3 16:09 Shanghai:** independent raw diagnostic RESULT SHA`5bc17afb` verifies1500episodes/900,000evalcontrols/0pending. First-success guarded/off:velocity43.2/40.4%,KV40.4/44.4%; off-minus-guarded95%pairedCIs[-8.8,+3.2]/[-1.6,+9.6]points,p=.443/.212. Exploratory first-update1epoch/5IDtasks,not a full-source learning gain;owner artifact/terminal audit still pending. Human KL-review hold remains controlling at2/50;60GPUh/arm120combined approved,alloldstate/data27058GPU-s preserved. No restart/recovery recipe selected,no complete source/newstudent/nativecurve. Monitor final audit+explicit humanreview/release then exact authorized source completion/downstream convergence;do not take KL/Flow owners’ work or treat diagnostics as source. [Preliminary dependency readout](steering-ppo-ad-dpt-20261102-r1/KL-READOUT-PRELIMINARY-20261103-r1.json),[hold](steering-ppo-ad-dpt-20261102-r1/KL-REVIEW-HOLD-OBSERVATION-20261103-r1.json). |
| Flora prefix first-success scoring | Claude Code session (tonyzhu163) proposes; **human approved 24,000 GPU-s 2026-11-02**; no Baige route, **needs a Baige owner to launch**; not launched | Score the 384 TRAIN + 50 CALIBRATION OpenDrawer videos at each RLinf chunk boundary (50…400 controls) with the original prompt: 3,472 prefix labels, frame-limited collage with no re-encoding, 0 new controls/SGD. Pick (P5 threshold, K consecutive checks) on even cases, report odd cases; adopt as the gate-1 Flora first-success reward if held-out false successes ≤15%, recall ≥0.80 and early fires ≤10%, otherwise use the fixed fallbacks. Needs Stage A1's `success_steps` check first. ~17k GPU-s estimated, **approved ceiling 24,000**, ≤4 GPUs, one allocation, no automatic retry. [Plan](flora-prefix-first-success-20261102-r1/PLAN.md). |
| Joint five-task frozen velocity / all-layer KV PPO | Owner parent `01a19b74`; **HUMAN HOLD at update2 pending KL-test review**, all original allocations released | Root independently verified direct human turn `01a1a00f-157f-7521-a92a-e79d52a0b5fa`: “Hold recovery until the KL test is reviewed.” No recovery/new full-source executor submitted. Approved lifetime60GPUh/arm120combined does not override this hold. Preserve both event005 checkpoints, full actor/value/Adam/sampler/global+worker RNG/state, all768TRAIN episodes/arm, before/DEV0 and27058GPU-s costs. KL owner `01a19fe9` owns copied-state bounded ablation; possible restart is not yet a chosen recipe. Independent Flow four-method work continues. [Controlling hold](robocasa-joint-steering-scale50-20261103-r1/KL-REVIEW-HOLD.json), [pause/state](robocasa-joint-steering-scale50-20261103-r1/PAUSE-UPDATE2.json), [budget](steering-ppo-ad-dpt-20261102-r1/SOURCE-BUDGET-60PERARM-20261103-r1.json). |
| Flow-SDE four-method source preparation | Owner parent `01a19b74`; **R2 fit continuation RUNNING** Baige `job-qbxwsy8jjdsb`, native readout pending | Nov3 15:56: AD5700/DPT5800updates, DIT/IQL value1900 and actor2200/2100; finite validation losses107.49/102.35/152.18/170.02, none converged. AD/DPT status files remain R1 until invocation exit; actual metrics/latest advance. Full-state r2 source e7ead0ee,1RPBZZZ6/16CPU/64GiB/8SHM/65min/autoretryOFF; selected prep+R1cost4090GPU-s, liveR2cost pending release. Whole7.5GPUh and prospective50ID+50OOD x5attempts/full-reset/initial-final anchors preserved. SharedfreePFS824.20GiB is below unchanged848GiB native admission guard; no EVAL launched, storage remedy needed before selected-student qualification, no otherowner data/held PPO changed. [Latest bounded check](flow-sde-four-methods-20261103-r1/CHECK-20261103-1556.json), [execution](flow-sde-four-methods-20261103-r1/EXECUTION.json), [r2 admission](flow-sde-four-methods-20261103-r1/FIT-R2-ADMISSION.json). |
| Gate readiness and native-criteria Flora rubric | Owner chat `01a19bef-b16f`;Baige frame-edge r1 COMPLETE/REJECTED `job-pb53j1e4jom1`, source `1f29377b`;GPU released13:45:32 Nov3 |40/40grades and4anchor repeats verified; botharms6/10negative score5,3/4false episodes; clause truefullscores4→5/8 and timelyconfirmed1→2/3,128stillmissed,216lateclose1. Actual19processor inputs areBF16 and byte-match CPUcastrefs;40RGB/message audits and54file hashes pass.575workerGPU-s/590scheduler;subtotal7925workerGPU-s excludingearlierterminalvideo work. Offline external latch fixesforgettingofrecognizedsuccesses but not3persistentfalseepisodes; no promotion/cutoffchange or further same-panelwordingfit. Next scientificdiagnostic:separate current-state recognition from temporalfirstsuccess aggregation. No GPU job active/new trial launched. [Verified result and all scores](robocasa-flora-drawer-frame-edge-20261103-r1/RESULT.md). |
| Flora current-state recognition diagnostic | Owner chat `01a1a006`; Baige `job-6zq2cpuq1c7c` SUCCEEDED/released15:30:24 Nov3; source`db2ef630` | All76scores/4repeats/37actualBF16-input matches complete. Prefix grade FP7/12,TP5/6; checklist FP10/12,TP6/6; reset/current pairs TP0/6. All candidates rejected.1003wrapper/1010scheduler GPU-s; r1 CPU failure0GPU. Raw receipts retained on Baige; archive download blocked by organization; inspected aggregates recorded with that limitation. Next: bound/qualify official per-step rubric and camera inputs before another diagnostic. No production/PPO change. [Result](robocasa-flora-current-state-20261103-r2/RESULT.md), [evidence](robocasa-flora-current-state-20261103-r2/OBSERVED-RESULT.json). |
| Research execution / five-update visibility | Coordinator chat `01a19f33`; full PPO HUMAN HOLD2/50; KL diagnostic R2 queued | **Nov3 15:15 Shanghai:** repaired diagnostic `job-1nds2rm4v2rp` submitted once on freshsource `12d1f241`, 4RPBZZZ6/89min; queue screenshot independently inspected, output absent, actual GPUprobe/episodes pending. R1 terminal failed before model/episodes;200GPU-s conservative charge carried,21400 remain in same6GPUh diagnostic cap. Full velocity/KV sources remain held2/50 until KL review;120GPUh budget does not lift hold. No DEV5/completed source. [Check](research-coordination-20261103-r1/CHECK-20261103-1515.json), [hold](research-coordination-20261103-r1/HUMAN-HOLD-KL-REVIEW.json), [state](research-coordination-20261103-r1/STATE.json). |
| Per-task Flow-SDE PPO teacher campaign, 10M controls/task | Owner chat `01a1a077`; Baige Pro6K,1GPU/task: Open `job-cn2if26i9hth`,Toaster `job-zwstryxe9plb`,Drawer `job-sfm20c7jc9dp`,Coffee `job-49hgr77uuw87`,Stove `job-mrwxr4gq2p7u`; eval source`f30a2a4f` | **Nov4 12:42 Shanghai:** all five allocated probes and zero-error initial replays pass;160 native episodes complete, Drawer130panel64/64 done and the other tasks advancing. Evaluate DEV64 in human-requested130→0 order, lock winners then counterbalanced fresh TEST128;20-minute monitor ACTIVE,22GPUh all-attempt cap incl qualification (0.644GPUh creation-to-check upper,21.68GPUh conservative total reservation). Original10M sources remain closed/failed teacher gates; no training/fitting. [Live state](robocasa-flow-sde-checkpoint-selection-20261104-r1/EXECUTION.json), [plan](robocasa-flow-sde-checkpoint-selection-20261104-r1/PLAN.md), [original results](robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md). |
| Flow-SDE 10M per-dimension drift diagnostic | Owner chat `01a1a4a2`; Baige Pro6K `job-aesultap3qa8` RUNNING/1GPU; source `0dbdafa4` | **Nov4 12:48 Shanghai:** 61/80 forwards: Drawer/Coffee/Stove16/16; OpenDrawer13/16 through100; Toaster pending. All recorded rows reproduce R3. Drawer independent reduction passes: final−137.52, binary24.44%, arm61.43%; neither70% rule fires. Different denoising scales confound raw cross-task drift. Same3GPUh cap,0controls/SGD; clock pending, stage healthy. Next check13:08; after80 receipts finish full CPU audit and scheduler release/cost closeout. [Readout](robocasa-flow-sde-dimension-drift-20261104-r1/RESULT.md), [execution](robocasa-flow-sde-dimension-drift-20261104-r1/EXECUTION.json). |
| Flow-SDE 10M matched-denoising-index drift | Owner chat `01a1a4de`; Baige Pro6K preparation, not launched | **Nov4 12:48 Shanghai:** taking the requested matched-index replay; verify retained policy-zero cohorts on allocated Linux CPU, then launch one GPU only after dimension job `job-aesultap3qa8` releases. Ceiling4GPUh, no controls/SGD or automatic retry. [Plan](robocasa-flow-sde-matched-denoise-drift-20261104-r1/PLAN.md). |
| Student-fitting admission and ID portfolio (decision record) | Claude Code session (tonyzhu163); no job | **Nov4 human:** no meta-learner/student fitting until the checkpoint-selection sweep's fresh TEST shows a selected checkpoint beating policy 0 with Holm-corrected exact McNemar p<.05 (wins>losses); non-admitted sources stay plumbing-only. **Power warning:** simulated from the 64-case discordance, TEST128 detects Toaster's TRAIN-size +8.9pp only 13% of the time at Holm's first step (p<.01; 34% at p<.05); ~512 paired cases give 81%, so a null at 128 is not evidence of no improvement. **Human:** replace ID TurnOffStove with a buttons/knobs task (recommended TurnOnSinkFaucet, fallback TurnOnElectricKettle), pending a 50-case native qualification at horizon 600; TurnOnMicrowave stays OOD. |
| TurnOffStove replacement qualification | Owner chat `01a1a4de`; Baige Pro6K preparation, not launched | **Nov4 12:48 Shanghai:** human requested execution of both50-case Faucet/Kettle qualifications at600controls. Fixing a copied20-case rate denominator before allocated Linux qualification; fixed3–35/50 Faucet-first rule,2GPUs/6000GPU-s, noSGD. [Plan](robocasa-buttons-replacement-qualification-20261104-r1/PLAN.md). |
| 512-case paired endpoint test, Toaster + OpenDrawer | Owner chat `01a1a4de`; Baige Pro6K preparation, not launched | **Nov4 12:48 Shanghai:** human requested execution of policy0vs130 on512fresh paired cases/task; unchanged qualified fixedB8 workers,ABBA/commonRNG and Holm over2tasks. Separate27600GPU-s ceiling,two1GPU jobs; first verify staging/CPU orchestration and native first banks. NoPPO/student fitting. [Plan](robocasa-flow-sde-endpoint-test512-20261104-r1/PLAN.md). |
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
