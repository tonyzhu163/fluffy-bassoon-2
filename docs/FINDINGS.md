# Verified findings

## Rotation sign branch in the policy state on Kinova3 (frozen SmolVLA, exploratory)

The simulator reports the Kinova3 hand quaternion with the opposite sign from Panda and IIWA, so the packed rotation X is near −3.2 where every source arm gives about +3.1. On six drawer-and-bowl layouts with two noise seeds, frozen SmolVLA scored 0/12 on Kinova3 as reported and 4/12 with the same rotation written on the source branch; Panda scored 3/12 as reported and 0/12 when written on the Kinova3 branch. All seven changed outcomes went the predicted way (sign test p = 0.016). Every native π₀.₅ Kinova3 run packed the raw sign; the effect on π₀.₅ is unmeasured. [Native CPU results](../outputs/frozen-shift-smolvla-state-20261124-r1/RESULT.md)

## Native frozen-policy intervention selection

On the four held-out Kinova3 drawer-and-bowl layouts, the factorized and pooled selectors chose different one-shot branches on all four and both succeeded only on layout 39. Continuing, correcting and cueing also succeeded only on 39, leaving zero failures that one-shot selection could rescue. Source-arm forks supplied zero rescues and one loss from cueing. The 116-episode native run verified 596 decision records, including 1,524 executed controls that differed from nominal. [Paired results and independent readback](../outputs/selector-native-launch-20261124-r1/RESULT.md)

## Current drawer state from the fixed BASE camera

A visual cue calibrated on 18 Panda demonstrations removed 20 premature placement cues across 248 archived Kinova3 replans, with zero missed openings. On four controlled Panda layouts it recognized all 16 reclosures. It missed 24 of 192 open frames, all at 0.5 mm beyond the native threshold. [Case-level validation](../outputs/frozen-vla-selector-20261124-r1/drawer-evidence.md)

## Frozen cross-embodiment adaptation on new arms

On six fresh Kinova3 starts from three selected tasks, recent source-trained motion feedback reduced displacement error against its own forecast at the first intervention from 13.73 to 7.38 mm. Native successes fell from 5/6 untreated to 1/6 with feedback, with four losses and zero rescues. A static correction scored 4/6, and stale feedback scored 0/6. The untreated and two feedback branches shared exactly the first 15 physical controls. On a separate test of the same Panda arm with a different native controller, posture-dependent inverse kinematics retained 5/6 successes under joint-position control, versus 6/6 under original Cartesian control and 0/6 with a fixed calibrated inverse. [Feedback and native reward tapes](../outputs/frozen-vla-tracking-feedback-20261123-r1/RESULT.md) · [Controller comparison](../outputs/frozen-vla-native-controller-20261123-r1/RESULT-NATIVE.md)

On drawer layouts 34–37, every arm's own fixed command sequence reproduced its native success label in CPU replay. Kinova3 solved two of four under its own commands and also two under the matching layout's IIWA commands. A complete matrix of 16 cross-arm replays found one rescue among the two remaining starts. IIWA's layout-35 tape solved Kinova3 layout 37 at control 182 of 184. No other donor rescued either failed start. A uniform draw of another layout's IIWA tape had 2 successful pairings out of 12. [Physical replays, controls and tape exhaustion](../outputs/frozen-transfer-donor-20261124-r1/RESULT.md) · [Per-start verification](../outputs/frozen-adaptation-20261124-r1/donor-recheck.json)

When locked before testing on five other Kinova3 layouts, IIWA tape 35 completed 1/5 against 2/5 for the target's own raw commands. The shortest successful source tape also completed 1/5, rescuing one original failure on layout 14 and losing two successes. On eight additional layouts the same two tapes completed 2/8 and 3/8, respectively. Native reward was measured through success or exhaustion of the supplied actions. [Fixed-donor replication and duration controls](../outputs/frozen-transfer-donor-validation-20261124-r1/RESULT.md)

On four new drawer layouts on IIWA and Kinova3, raw frozen π₀.₅ and the tuned Perceiver each finished 6/8 starts. A one-replan placement instruction finished 4/8 whether triggered by the camera or the clock. The camera cue changed the next executed action on all eight starts. It lost two tuned successes and rescued zero (−25 percentage points, paired 95% layout interval −50 to 0). Across this panel and the eight-task motion trial, a hindsight choice between tuned and its two new options rescued none of the four distinct tuned failures. This bound applies to one choice held for the episode. [Paired cue trial](../outputs/frozen-transfer-cue-20261124-r1/NATIVE.md) · [Per-start choice analysis](../outputs/frozen-adaptation-20261124-r1/episode-choices.json)

Across eight official BASE tasks on IIWA and Kinova3, raw frozen π₀.₅ solved 16/16 starts on layout 31, the tuned Perceiver 14/16, fixed temporal averaging 10/16 and averaging aligned to measured TCP motion 12/16. Alignment rescued three starts and lost one against fixed averaging, but lost both drawer starts that tuned solved. The adaptive index departed from fixed on 371/711 decision boundaries. [Native outcomes and verified command traces](../outputs/frozen-transfer-progress-20261124-r1/NATIVE.md)

Across 256 held-out-arm OOD supports on four arm–policy panels, equal-weight tuned-plus-geometric yaw estimates scored 6.133°, versus 6.132° for two independently seeded tuned Perceivers and 6.433° for one. The cross-architecture difference was +0.001° (paired 95% task interval −0.719° to +0.785°). On eight held-out Panda/Sawyer source tasks the same mixed ensemble had a 2.154° advantage. [Matched ensembles](../outputs/frozen-transfer-ensemble-20261124-r1/RESULT.md) · [Omitted-chunk selection](../outputs/symmeta-omitted-chunk-20261124-r1/SCORE.md)

Every one of 600 requested Kinova3 poses on two failed drawer episodes had a static joint-valid witness. On two successful moka-pot episodes, 110 requested poses had certified exclusions at their original wrist orientation. A wrist relaxation chosen from joint encoders and known kinematics rescued neither drawer and reduced fixed-tape Kinova3 completions from 3/6 to 1/6. [Feasibility and physical intervention](../outputs/frozen-transfer-pose-20261124-r1/RESULT.md)

On the original 22 target-arm command tapes, all 253 sustained, strictly identified pickups were compatible with the released task instruction. Of the three original native failures, two had no eligible closed-gripper window and one carried a permitted bowl. This archive supplies no definite distractor pickups with which to test identity recognition. [Role definitions and episode-level exposure](../outputs/frozen-transfer-identity-20261124-r1/RESULT.md)

Under identical commands, IIWA and UR5e differed in whether they held an object on 2 of 8 tasks. The carrying hand had the wider measured aperture on all 14 opposing windows. Across three target arms and five informative tasks, adding central wrist-image motion features produced Brier loss 0.0956 versus 0.0682 from proprioception (paired difference +0.0274, 95% task interval −0.0349 to +0.0756). Those 146 windows came from 17 sustained events. These data support testing which object is held as a separate question. [Physical labels, arm swap and reader comparisons](../outputs/frozen-transfer-object-20261124-r1/RESULT.md)

A source-fitted vertical timing correction following gripper closure gave 0.491 mm/control held-out tracking error, compared with 0.473 for static gain and 0.492 for the same timing correction shifted within the command block. Its paired difference from static was +0.018 mm/control (95% whole-task interval −0.002 to +0.034) across two new arms, three held-out tasks and two layouts. Both static and timing completed 2/12 fixed-tape replays. The result deprioritizes this particular timing rule for a live-policy trial. [Matched simulator interventions](../outputs/frozen-transfer-timing-20261124-r1/RESULT.md)

## Public camera history identifies first drawer opening on fresh Kinova3 simulator resets

A four-patch camera reader fitted on 30 human Panda demonstrations scored 0.0138 Brier on ten held-out demos versus 0.0106 for a source-fitted clock. On four fresh native Kinova3 resets, each executing the same two previously recorded policy command sequences, the reader scored 0.0090 versus 0.0395. The paired Brier reduction was 0.0305 (95% layout bootstrap interval 0.0288–0.0320, conditional on those two sequences). It ranked the open trajectory higher in all 19 same-layout, same-time comparisons with opposite physical labels. Two drawers subsequently reclosed and all eight attempts timed out. A one-replan prompt change passed a native CPU interface check. Its effect on frozen-policy success and the independently registered live-policy starts remain unmeasured. [Results, split and individual predictions](../outputs/frozen-vla-phase-transfer-20261123-r1/RESULT.md)

## Predicting unshifted motion on three simulated robot arms

From the same 15 passive target controls, a source-regularized 3D response model reduced displacement prediction error relative to a planar response on Kinova3 (−0.69 mm per five controls, approximate 95% task interval −1.35 to −0.02) and IIWA (−0.35 mm, −0.50 to −0.19). Error increased on UR5e (+0.65 mm, +0.12 to +1.18). The learned nonlinear model had higher point errors than its linear counterpart on all three arms, with intervals spanning zero. An equal-budget probe that excited three directions increased the 3D model’s error on subsequent policy motions for all three arms. These are offline predictions on eight held-out tasks per arm using trusted metric end-effector positions. [Registered experiments, controls and uncertainty](../outputs/frozen-vla-motion-models-20261123-r1/HANDOFF.md)

On six fresh simulated Kinova3 starts the frozen policy solved 4, the planar adapter solved 4 and the 3D adapter solved 3. Both adapters rescued one drawer start. The full 3D adapter lost both untreated moka-pot successes. The six within-job opening proposals matched across all three arms. In a separate appearance experiment on the same six starts, the untreated policy solved 2 and both colour interventions solved 3. That job reproduced all six opening input/noise hashes but none of the six untreated opening proposals from the action experiment. Its two additional untreated failures were the moka-pot starts. These two baselines establish appreciable variation across processes. [Motion results](../outputs/frozen-vla-native-transfer-20261123-r1/RESULT.md) · [Appearance and repeat evidence](../outputs/frozen-vla-visual-adapter-20261123-r1/RESULT.md)

A privileged Panda render at the physical Kinova grip site changed 1,946–8,244 external pixels after OpenPI preprocessing on ten drawer exposures from two target command replays. The independent Kinova render reproduced all 20 native camera frames exactly. Both camera views matched on RGB and depth with the arm hidden, and the wrist image was unchanged by the substitution. The frozen-policy proposal and score comparison awaits the parent launch. [Measured frames, bounded IK and matched native control](../outputs/frozen-vla-source-view-20261123-r1/RESULT.md)

## A direct Perceiver recovers held-out yaw offline and succeeds on 54.2–79.7% of native held-out retries

On the paper’s 64 recorded OOD supports, the continuous-label Perceiver with a lower peak learning rate and 200 warmup updates scored 5.42°, 5.83° and 5.48° yaw MAE across seeds 41–43. The seven-shift SymMeta checkpoints scored 6.82°, 7.00° and 7.29°. Across all 96 recorded supports, mean planar operator error was 0.146 versus 0.154. At the same 2,500 updates and with the same continuous labels, the original scaling-sweep schedule gave the Perceiver 96.86° mean OOD yaw error. The direct reader estimates all three interface factors without the explicit yaw solve. [Seed scores and training comparisons](../outputs/symmeta-wider-family-20261123-r1/RESULT.md)

The original offline experiment retained scores only. Before rollouts, three exact-recipe Perceiver refits were reloaded and their predictions fixed. These refits averaged 5.82° OOD yaw error on UR5e. Both readers ran fresh on every pair using the same passive support, full reset, noise and opening policy proposal. All 1,152 pairs passed the native-reward readback. [Checkpoint scores](../outputs/symmeta-perceiver-retry-20261123-r1/evidence/train02pro/training-sanity.json) · [Pair-level verification](../outputs/symmeta-perceiver-retry-20261123-r1/evidence/native02g4090/readback.json)

| Held-out native retries | Tuned Perceiver | Fresh seven-shift SymMeta | Perceiver minus SymMeta in points [95% eight-task interval] |
| --- | ---: | ---: | ---: |
| UR5e × π₀.₅ | 141/192 | 153/192 | −6.25 [−15.10, +1.04] |
| IIWA × π₀.₅ | 153/192 | 147/192 | +3.13 [−4.17, +9.90] |
| Kinova3 × π₀.₅ | 129/192 | 126/192 | +1.56 [−6.25, +8.33] |
| UR5e × π₀ | 104/192 | 107/192 | −1.56 [−12.50, +8.85] |

Each denominator uses 64 held-out queries repeated at reader seeds 41–43. Exact paired percentile intervals resample the eight tasks with their shifts and seed repetitions together. They admit advantages for either reader on every OOD panel. Across all 96 queries per seed, the tuned Perceiver and fresh SymMeta succeeded on 209/288 and 226/288 UR5e × π₀.₅ retries (−5.90 points [−12.50, −0.35]). Each method used its own yaw, gain and gripper correction. [Four-panel counts and intervals](../outputs/symmeta-perceiver-retry-20261123-r1/RESULT.md) · [Matched pairs](../outputs/symmeta-perceiver-retry-20261123-r1/evidence/native02g4090/pairs.csv.gz)

In an exploratory check on the same 768 held-out panel–query–seed cases, replacing only tuned Perceiver yaw with the circular midpoint of its yaw and SymMeta’s changed mean yaw error from 6.43° to 6.09°. The paired change was −0.34° (95% eight-task bootstrap interval −1.05° to +0.45°). On UR5e × π₀.₅ compositional shifts, the original SymMeta alone succeeded on 17 retries and the tuned Perceiver alone on four. SymMeta had the closer yaw in 15 of its 17 successes and the closer gain in 12. These are post hoc comparisons on the existing supports. Native outcomes for the blend remain unmeasured. [Analysis and input records](../outputs/symmeta-yaw-blend-20261124-r1/RESULT.md)

On the original seven-shift archive, changing only the Perceiver learning-rate schedule reduced mean OOD yaw MAE from 95.63° to 83.37° (paired change −12.26°, 95% interval −17.04° to −7.39°). Across all 96 supports, planar operator error fell from 0.9665 to 0.8990 (−0.0676, 95% interval −0.1003 to −0.0358). On the 32 familiar supports, yaw MAE rose from 0.65° to 1.88°. The intervals resample eight whole tasks with the three reader seeds fixed. The tuned seven-shift and continuous-label readers shared a peak rate of 0.0003 and 200 warmup updates. Their 83.37° and 5.58° OOD errors compare different source archives (1,344 versus 1,466 traces), interface sampling, positional slots (32 versus 24), minibatches (64 versus 32) and update budgets (3,500 versus 2,500). [Paired seven-shift result and protocol comparison](../outputs/symmeta-seven-schedule-20261123-r1/RESULT.md)

On four new in-family draws per recorded trace, mean operator error was 0.149 for Procrustes, 6.037 for unregularized factored least squares and 0.152 for factored ridge on the same frozen reader. Training a full-map Perceiver across families improved axis swaps (0.413 versus 1.257 for Procrustes). On general maps its mean error was 15.212 and the mean of its within-seed medians was 1.366. The median ratio of the two command singular values was 0.099 on the recorded supports. [Operator errors by family](../outputs/symmeta-wider-family-20261123-r1/gpu01w454/wider-family-summary.json)

## ReGround’s released 90° rotation test on Panda

We ran the complete published rotation cell on Panda, with 35 held-out starts per task, eight tasks, five seeds and a single uninterrupted attempt. On 1,400 assignments per method, frozen π₀.₅, streaming SymMeta and fitted ReGround succeeded on **0, 401 and 1,271**. The released success IQMs were **0.000, 0.186 and 0.974**, and unsafe-episode IQMs were **0.843, 0.200 and 0.339**. ReGround reported 0.98 and 0.29. Our released 95% intervals for fitted ReGround were [0.957, 0.991] and [0.296, 0.390]. The largest per-task success gap was long task 3, 0.577 here versus 0.77 published. ReGround used 100,000 nominal and 10,000 target transitions per fit. Streaming SymMeta used the ongoing episode, passed wrist commands through and had zero separate target-fit transitions. [Full results, resource costs and per-task comparison](../outputs/reground-rotation-reference-20261123-r1/RESULT.md)

The five seed assignments share 280 distinct task–start pairs. The arithmetic mean success advantage for ReGround over streaming SymMeta was 62.14 percentage points (95% start-cluster interval 58.21–65.86). Seven original assignments had different opening proposals somewhere among the three arms, although all initial robot states matched. Rerunning all 32 assignments in both affected shards on shared native openings gave 0/12/31 successes for Base/SymMeta/ReGround versus 0/13/31 on the original shards. [Paired estimates and repeated-start intervals](../outputs/reground-rotation-reference-20261123-r1/evidence/comparison.json) · [Controlled rerun](../outputs/reground-rotation-reference-20261123-r1/common-opening/readback/RESULT.md)

On a smaller release-matched panel, ReGround succeeded on 0/32, 0/32 and 9/32 test starts at 15, 100 and 1,000 target transitions for seed 0. Seed 3 gave 0/32, 5/32 and 19/32. The 10,000-transition fits solved 32/32 and 26/32. Both fits shared the same 32 native starts and all opening checks passed. Relative to 10,000, success changed by −90.6, −82.8 and −46.9 percentage points. Exact 95% paired within-task start-bootstrap intervals were [−95.3, −85.9], [−89.1, −76.6] and [−56.3, −37.5] with tasks and seeds held fixed. At 15 transitions the release uses two trajectory prefixes and reserves one for validation. [Fit and input accounting](../outputs/reground-low-budget-20261123-r1/RESULT.md) · [Paired uncertainty](../outputs/reground-low-budget-20261123-r1/evidence/paired-starts.json)

## Calibrated retry on the native UR5e command interface

On the eight nominal UR5e × π₀.₅ queries in the 96-query BASE retry panel, the uncorrected policy succeeded on 8/8. Each of three SymMeta reader seeds (41–43) succeeded on 6/8 after calibrating on a separate 15-control support. All three estimates are evaluated on the same eight tasks and starts. These are distinct from the 32 original LIBERO-PRO streaming trials below. [Paired task outcomes and archived source fingerprints](../outputs/symmeta-paired-statistics-20261123-r1/nominal-retry.json)

## Streaming on the unmodified Panda command interface in original LIBERO-PRO

On 32 single-attempt queries across eight released tasks and four perturbation families, unadapted π₀.₅ succeeded on 19 and the same policy with the frozen streaming reader succeeded on 3. Sixteen pairs regressed and zero were rescued (paired difference −50.0 percentage points, suite-stratified task-bootstrap interval −59.4 to −40.6). Both arms issued identical commands for their first ten controls. The pilot omits Environment, which contributes to the historical five-family leaderboard. [Native readback and task-family counts](../outputs/symmeta-original-pro-20261123-r1/NATIVE-READBACK.md)

## Probe direction and planar interface recovery on UR5e

On open-loop frozen π₀.₅ action tapes, damping the inverse reduced clipped command error from 0.308 to 0.184 on 64 weak two-direction supports. Isotropic damping scored 0.189. A scaled rotation gave 0.039 on the 192 recorded yaw/gain supports where the damped full inverse gave 0.136. [Task-level intervals, strong-support controls and command replay](../outputs/frozen-vla-factory-20261123-r1/inverse/RESULT.md)

On 384 new simulated UR5e supports with reflected or anisotropic XY interfaces and matched endpoint jitter, an independent second command direction reduced full-map relative forward error from 0.560 to 0.270 at the same 15-control budget (paired reduction 0.290, eight-task 95% interval 0.204 to 0.363). Trusted FK supplied the displacements. On 192 archived yaw/gain supports, scaled rotation gave lower error than full ridge, 0.080 versus 0.262. These screens measure command-map recovery. [Directions, maps and observation assumptions](../outputs/frozen-vla-factory-20261123-r1/excitation/RESULT.md)

## Observation adapters for frozen SmolVLA

An affine joint-to-pose adapter fitted on the opposite robot gave first-five-action RMSE of 0.2346 on UR5e and 0.3193 on IIWA relative to the frozen policy using the measured target tool pose. Dropping state gave 0.1001 pooled RMSE. Fitting the same adapter on 15 joint/pose pairs from three separate target episodes gave 0.0141, improving over opposite-arm calibration on all 16 held-out frames (paired mean reduction 0.2629, 95% eight-task interval 0.2097–0.3262). Unknown source-frame registration and downstream task success remain open. [Leave-one-arm-out and few-shot comparison](../outputs/frozen-vla-factory-20261123-r1/state-transfer/RESULT.md)

On eight standard LIBERO Panda tasks with physical quarter-turn camera rolls, frozen SmolVLA's full-chunk action RMSE relative to the upright reference fell from 0.2069 to 0.0172 with an image-estimated correction. One upright scene was falsely rotated. In a separate experiment with one near-reset Panda/UR5e pose per task, rendering the source arm using only the target tool pose, jaw and installation geometry reduced 50-step translation RMS from 0.0596 to 0.0282 on ten full UR5e images. The displaced-render control scored 0.0922. These CPU forward passes compared proposals at fixed inputs and noise. The native success of these two input adapters remains unmeasured. [Camera controls](../outputs/frozen-vla-factory-20261123-r1/view/RESULT.md) · [Rendered arm and occlusion controls](../outputs/frozen-vla-factory-20261123-r1/rendered-appearance/RESULT.md)

## Matched-source visual yaw on UR5e

On the 64 held-out supports, uniform pooling over all three learned motion vectors gave 6.53° yaw MAE. Selecting the first or final vector gave 10.81° and 12.05°. Weighting the three by magnitude gave 6.18° (paired change −0.36°, 95% eight-task interval −0.81° to +0.03°). All readouts used the same frozen 15-control features. [Temporal pooling experiment](../outputs/frozen-vla-factory-20261123-r1/pooling/RESULT.md)

On the 64 held-out π₀.₅ supports, a learned visual reader scored 6.53° mean absolute yaw error and a six-coefficient linear flow projection scored 9.16°. Both used the same 1,344 labeled source traces and the same geometric solver with equal chunk weights. The 2.63° paired advantage had a 95% task-bootstrap interval of 1.64° to 3.65°. On the 32 unseen-angle cases alone the advantage was 0.30° (−2.40° to +2.90°). [Source fit, per-task estimates and reference intervals](../outputs/symmeta-matched-flow-20261123-r1/OFFLINE.md)

On fresh, reset- and noise-matched second attempts, the learned reader succeeded on 46/64 held-out queries against 39/64 for the linear reader (+10.9 percentage points, 95% task-bootstrap interval +1.6 to +21.9). The gains were 23 versus 17 on 32 compositional shifts and 23 versus 22 on 32 continuous-angle shifts. All 96 settings gave 70 versus 62 successes (+8.3 points, +2.1 to +16.7). The visual features and pooling differ, and both methods use the same equal-weight solver. [Native outcomes and paired cases](../outputs/symmeta-matched-flow-20261123-r1/RESULT.md)

On a separate 96-pair second-attempt UR5e × π₀.₅ run, trusted joint-based yaw with the learned gain and gripper correction gave 78 native successes versus 76 for visually estimated yaw. The kinematic minus visual difference was +2.1 percentage points (95% task-bootstrap interval −2.1 to +7.3). [Native paired outcomes](../outputs/symmeta-fk-ur5e-20261123-r1/RESULT.md)

## SymMeta versus source-calibrated brightness flow across reader seeds

Across the eight tasks and 96 queries per panel, mean native success over seeds 41–43 improved by 19.8 points on UR5e, 24.3 on IIWA and 22.6 on Kinova3 with π₀.₅. All eight task means favored SymMeta in each π₀.₅ panel. On UR5e with π₀, six tasks improved and both LIBERO-Object tasks declined. The mean gain was 9.7 points, with a two-sided whole-task sign-flip p-value of 0.078. Outcomes reuse the same eight task clusters across seeds. [Paired analysis and saved outcomes](../outputs/symmeta-paired-statistics-20261123-r1/RESULT.md)

## SymMeta command-yaw estimates follow camera rotation on held-out UR5e

Across the same 64 out-of-distribution π₀.₅ supports, the frozen seed-41 reader had 6.83° yaw MAE upright and 84.52° / 101.23° after displayed image rolls of −90° / +90°. Source-calibrated brightness flow scored 12.29° / 84.36° / 101.63° for upright / −90° / +90°. Independent joint kinematics gave 2.02° on the same motions. In a separate two-task MuJoCo check, physically orbiting the camera ±30° shifted the estimated command yaw by 28–30° on average with identical actions and physics across views. This was an offline identification test. [Paired controls, moderate rolls, provenance and limitations](../outputs/symmeta-camera-command-20261123-r1/RESULT.md)

The seed-41 second-attempt calibration panels use eight simulated held-out LIBERO-Pro2.0 tasks, each crossed with twelve researcher-imposed command transforms. A 15-control trace on layout 11 precedes a fresh query on layout 13. All arms use public task success. The Perceiver in these original panels uses the historical seven-shift checkpoints. It predicts the transform from history. The command-frame affine control fits local proprioception. Neither is a run of published ICBA or ReGround. The brightness-translation baseline fits one global XY image displacement per frame pair. [Evaluation protocol](../papers/icml-xemb-sysid/main.tex)

- **6-DoF `UR5e`, frozen $\pi_{0.5}$:** Passive V2 `76/96` (`79.2%`, `52/64 = 81.2%` OOD) vs. exact oracle `82/96` (`85.4%`), passive brightness translation `56/96` (`+20.8 pp`, 95% CI `[+13.5, +30.2]`), passive DIS `40/96` (`+37.5 pp`), Perceiver `33/96` (`5/64 = 7.8%` OOD), command-frame affine `6/96`, swapped `0/96`. [UR5e π₀.₅](../outputs/xemb-icml-campaign-20261121-r1/evidence/ur5e-pi05-summary.json)
- **7-DoF KUKA `IIWA`, frozen $\pi_{0.5}$ (`job-3qs72l3fcgi9`, `exit 0`):** Passive V2 `76/96` (`79.2%`, `52/64 = 81.2%` OOD, `84.4%` continuous-angle OOD) vs. exact oracle `85/96` (`88.5%`), passive brightness translation `52/96` (`+25.0 pp`, 95% CI `[+18.7, +31.2]`, `27W/3L`), passive DIS `30/96` (`+47.9 pp`), Perceiver `34/96` (`90.6%` familiar → `5/64 = 7.8%` OOD), command-frame affine `6/96`, swapped `1/96`. [IIWA π₀.₅](../outputs/xemb-icml-campaign-20261121-r1/evidence/iiwa-pi05-summary.json)
- **7-DoF `Kinova3`, frozen $\pi_{0.5}$ (`job-66opmwn2j3a9`, `exit 0`):** Passive V2 `60/96` (`62.5%`, `38/64 = 59.4%` OOD) and passive EquivFlow `53/96` (`55.2%`, `38/64 = 59.4%` OOD) vs. exact oracle `74/96` (`77.1%`), passive brightness translation `42/96` (`+18.8 pp`, 95% CI `[+10.4, +27.1]`, `22W/4L`), passive DIS `28/96` (`+33.3 pp`), Perceiver `29/96` (`81.2%` familiar → `3/64 = 4.7%` OOD), command-frame affine `6/96`, swapped `1/96`. [Kinova3 π₀.₅](../outputs/xemb-icml-campaign-20261121-r1/evidence/kinova3-pi05-summary.json)
- **6-DoF `UR5e`, frozen $\pi_0$ (`job-bhl6rx8ok5t8`, `exit 0`, second VLA):** Canonicalizing the $SO(3)$ quaternion double cover (`q → -q` on the downward-gripper hemisphere to match `pi0_libero`'s `AbsoluteActions` `+π` convention) resolved the earlier `0/96` saturation (`rot_x = -4.0..-5.7 → +0.034`). Passive V2 scored `54/96` (`56.2%`, `35/64 = 54.7%` OOD), compared with `52/96` for the exact oracle (`54.2%`) and beating passive brightness translation `44/96` (`+10.4 pp`, 95% CI `[+2.1, +18.8]`, `24W/14L`), passive DIS `29/96` (`+26.0 pp`), Perceiver `22/96` (`2/64 = 3.1%` OOD), command-frame affine `4/96`, and swapped `0/96`. [UR5e π₀](../outputs/xemb-icml-campaign-20261121-r1/evidence/ur5e-pi0-summary.json)

**Embodiment-diversity & capacity scaling law + sub-second passive convergence (`job-1keh8zstmt2w`, `exit 0`, 105 real MuJoCo models × 3 seeds):** Training on `1,466` real MuJoCo `Panda+Sawyer` trajectories sweeping source embodiment diversity $M \in \{2, 4, 7, 16, 32, 64, 128\}$ and evaluating on the locked 96/96 held-out 6-DoF `UR5e` test set (`n=64` OOD) gave the following transform-recovery results. At the stated training sizes, `equiv_v2_dis_dz` achieved `8.7°` OOD yaw MAE at $M=4$, `7.9°` (`96%` all-correct) at $M=7$, and `5.7°` (`99%` all-correct) at $M=128$; `factored_no_equiv_v2` plateaued at `50.4°` (`41%`) at $M=128$; `perceiver_icba_w192` peaked at `20.0°` (`45%`) at $M=64$; and the 5.31× larger `perceiver_icba_w384` had higher yaw error at each $M$ (`52.1°–106.3°`, `≤20%` OOD). On the locked 96/96 passive support horizon ablation (3 seeds), `equiv_v2_dis_dz` converged in `0.75 s` of passive motion: `12.68°` (5 controls / `0.25 s`), `7.42°` (10 controls / `0.50 s`), and `7.04°` (15 controls / `0.75 s`) vs. brightness translation `12.29°` and the historical seven-shift Perceiver `95.63°`. [Scaling Law](../outputs/xemb-icml-campaign-20261121-r1/evidence/stage3-scaling-summary.json) · [Horizon Ablation](../outputs/xemb-icml-campaign-20261121-r1/evidence/ablation-horizon-ur5e.json)

**Full 3D $SO(3)$ + anisotropic per-axis gains on held-out `UR5e` (`job-y05kcwejh6xt`, `exit 0`, `96/96` live):** Pre-scaling commands by estimated per-axis gains $(g_x, g_y, g_z)$ before differentiable $SO(3)$ SVD Procrustes fixed the earlier `11.73°` anisotropic bias (`<1e-4°` in unit tests). Trained strictly on 10 discrete single-axis 3D transforms on `Panda+Sawyer` and tested zero-shot on `UR5e` across 12 3D transforms (including continuous oblique Euler rotations + unequal gains `(0.65, 1.25, 1.40)` + grip flip), `ours_equiv_so3` achieved `37/96` (`38.5%`; `68.8%` familiar, `37.5%` comp OOD, `9.4%` cont $SO(3)$ OOD; `24.5°` mean 3D geodesic error) vs. exact 3D oracle `85/96` (`88.5%`), nominal `7/96` (`+31.2 pp`, 95% CI `[+19.8, +41.7]`, `31W/1L`), 3D command-frame affine `4/96` (`+34.4 pp`, `34W/1L`), and swapped `0/96` (`37W/0L`). Monocular single-camera depth ambiguity on out-of-plane pitch/roll explains the remaining gap to oracle. [SO(3) Summary](../outputs/xemb-icml-campaign-20261121-r1/evidence/stage2d-so3-summary.json)

**`LIBERO-Pro2.0` visual & camera perturbation stress-test on held-out `UR5e` (`job-y4zq4tz7rvmg`, `gpu14pro2`, `exit 0`, `240/240` queries = `2,160` live episodes):** Evaluating across five selected visual/camera perturbations (`S13` surface appearance, `S14` scene contrast or lighting, `S15` camera extrinsics, `S16` camera intrinsics, `S19` observation corruption), **zero-retraining `BASE`-trained Passive V2** (`ours_equiv_v2_passive_base`) achieved `133/240` (`55.4%`, or `133/192 = 69.3%` on VLA-solvable shifts vs. `146/240 = 60.8%` oracle) and `6.57°` mean yaw error across all 5 shifts: `S13` `75.0%` (`5.87°`), `S14` `87.5%` (`5.85°`, exceeding oracle `85.4%`), `S15` `0.0%` (`6.94°`, while exact oracle scored `0/48` due to shifted wrist RGB), `S16` `35.4%` (`7.86°` vs. oracle `39.6%` and DIS collapse `51.48°`), and `S19` `79.2%` (`6.35°`). Overall it beat passive brightness translation (`109/240 = 45.4%`, `10.19°`) by `+10.0 pp` (95% CI `[+4.2, +15.8]`, `35W/11L`), active DIS + Procrustes (`81/240 = 33.8%`, `24.75°`) by `+21.7 pp` (`[+11.7, +30.8]`, `60W/8L`), `Pro2`-retrained active V2 (`87/240 = 36.2%`) and EquivFlow (`86/240 = 35.8%`) by `+19.2–+19.6 pp`, `Pro2`-retrained Perceiver (`50/240 = 20.8%`, `72.31°`) by `+34.6 pp` (`[+23.7, +42.9]`), nominal (`25/240 = 10.4%`) by `+45.0 pp`, and command-frame affine (`13/240 = 5.4%`, `92.00°`) by `+50.0 pp`. [Pro2 Stress Summary](../outputs/xemb-icml-campaign-20261121-r1/evidence/pro2-stress-ur5e-summary.json)

In the paired S15 development comparison with frozen π₀.₅ on simulated UR5e, active EquivFlow, active DIS and the exact command transform each went from 4/4 native successes on BASE to 0/4 on S15. DIS had lower mean absolute yaw error under both cameras. EquivFlow’s excess error narrowed from 14.20° to 5.16°. The oracle failed every shifted query even with the true transform, which limits attribution of task failure to the readers. The comparison covers two layouts and two oblique transforms on one task. [Camera results and per-case evidence](../outputs/xemb-camera-dev-20261122-r1/RESULT.md)

On the 96-query second-attempt command-calibration panel in LIBERO-Pro2.0 BASE scenes with simulated UR5e and frozen π₀.₅, passive V2 achieved 76/96 native successes (52/64 OOD), compared with 56/96 for passive brightness translation and 82/96 for the exact-transform oracle. V2’s paired advantage over brightness translation was 20.8 percentage points (95% task-clustered interval +13.5 to +30.2). Active EquivFlow scored 54/96 against 47/96 for active DIS (interval on the difference −8.3 to +22.9 points). Single-attempt streaming scored 20/96. Frozen π₀ scored 0/96 for all 12 methods including the oracle, while it succeeded on Panda in a one-start development control. [Official paired results, recovery and source control](../outputs/xemb-live-wave-20261121-r1/RESULT.md)

On one LIBERO Goal 1 development layout, all 14 CPU simulator checks passed across simulated UR5e, IIWA and Kinova3. UR5e and IIWA each exercised BASE and all five visual perturbations. Kinova3 exercised BASE and S15. The checks verified robot identity, matching initial scenes, native camera or appearance changes and measured response to 15 controls. [Scope and simulator evidence](../outputs/xemb-stage2c-cpu-20261121-r1/RESULT.md)

On simulated ManiSkill3 xArm6, the active EquivFlow reader without vertical parallax correction scored 5/48 on held-out OOD native task queries. The corrected and Panda-only variants each scored 3/48. All three scored 0/24 on continuous angles, where the oracle scored 16/24. The earlier V2 reader with vertical correction scored 11/48 on the same seed schedule. Each run retained aggregate counts, so the cross-run comparison has no per-query win/loss estimate. [EquivFlow results](../outputs/xemb-live-wave-20261121-r1/evidence/ms3-equiv-flow-summary.json) · [V2 results](../outputs/xemb-live-wave-20261121-r1/evidence/ms3-v2-summary.json)

With exact 3D motion on the actual 15-control probe schedule and unequal axis gains, the committed SO(3) reader missed the injected rotation by 11.73°. A full linear solve recovered the transform to numerical precision. [CPU reference and controls](../outputs/xemb-stage2d-cpu-20261121-r1/RESULT.md)

**Official LIBERO-Pro2.0 test split + ManiSkill3 cross-morphology visuomotor SysID (`1,116` official test episodes + `144` ManiSkill3 trials):** Freezing our 24-task 7-DoF (`Panda`+`Sawyer`) checkpoints with zero test-set tuning and evaluating one-shot on the strictly untouched **8 official LIBERO-Pro2.0 test tasks** (`SPLITS.json["test"]`, `93/96` queries = `1,116` live episodes across all 4 suites on held-out physical 6-DoF `UR5e`; `job-7uccgsa8ktmi`, `4.80` GPUh; `7/8` shards `12/12`, `libero_goal-6` `9/12`) confirmed every dev finding with even sharper separation. Overall official test success (`n=93`): `privileged_oracle` `79/93` (`84.9%`), `analytic_visual_sysid_passive` `60/93` (`64.5%`), **`ours_lumaflow_active` `52/93` (`55.9%`; `+48.4 pp` vs. `nominal` `7/93 = 7.5%`, 95% CI `[+29.2, +67.8]`, `46W/1L`; `+24.7 pp` vs. `joint_transformer_active` `29/93 = 31.2%`, 95% CI `[+7.3, +41.4]`; `+39.8 pp` vs. `factored_no_equiv_active` `15/93 = 16.1%`; `+40.9 pp` vs. `analytic_proprio_sysid` `14/93 = 15.1%`; `ours_cleared` `7/93`, `ours_swapped_donor` `3/93`)**, and zero-probe **`ours_rgb_passive` `48/93` (`51.6%`; `+20.4 pp` vs. active `joint_transformer_active`, 95% CI `[+6.7, +33.3]`)**. On combined OOD (`n=61`), `joint_transformer_active` plunged `-81.1 pp` (`84.4%` familiar → `3.3%` OOD: `0/32` comp OOD, `2/29` cont OOD) and `factored_no_equiv_active` scored `0/61` (`0.0%`), whereas `ours_lumaflow_active` held `54.1%` (`33/61`, beating `analytic_visual_sysid_passive` `56.2%` on comp OOD at `59.4%` and beating it on `LIBERO-Goal` `16/21` vs. `14/21`) and zero-probe `ours_rgb_passive` held `47.5%` (`29/61`). In an independent physics engine (**ManiSkill3 / SAPIEN**, `job-ffvpyqvj7twq`, `0.09` GPUh, `exit 0`), training strictly on 7-DoF `Panda`+`Fetch` and transferring zero-shot to held-out physical 6-DoF `xarm6_robotiq` (`144` trials, `f0_eq=True`) reproduced the identical ordering and beat even hand-crafted optical flow: `factored_equiv` **`38.02°`** (`35.3°` fam / `35.1°` comp / `43.6°` cont) vs. `analytic_visual_sysid` `67.45°`, `factored_no_equiv` `71.73°`, `analytic_proprio_sysid` `90.06°`, and `joint_transformer` `95.08°` (`112.2°` comp OOD / `104.8°` cont OOD). [Official Test Evidence](../outputs/libero-xemb-ur5e-scale-20261120-r1/evidence/official-test-summary-compact.json) · [ManiSkill3 Evidence](../outputs/maniskill3-xemb-sysid-20261120-r1/evidence/ms3-xemb-summary.json)

On 50 paired simulated Goal 5 BASE development starts per arm, a learned public-observation gate on a frozen source motor scored 30 Panda and 18 Sawyer successes. The concurrent raw policy scored 33 and 20; applying the same learned correction unconditionally scored 30 and 0. The gate avoided five of eight Panda losses and 18 of 20 Sawyer losses, and selected none of five available Panda rescues. The learned correction offered zero Sawyer rescues. [Paired source experiment](../outputs/embodiment-source-correction-gate-20261120-r1/RESULT.md)

**Scaled 8-GPU held-out 6-DoF `UR5e` cross-morphology visuomotor SysID (`1,056` live episodes):** Scaling training 6× across all 24 fitting tasks (`1,344` 7-DoF `Panda`+`Sawyer` trials) with RGB-permutation-invariant `luminance_flow` inputs and evaluating all `96` paired held-out 6-DoF `UR5e` queries × 11 arms (`8.11` Pro6K GPUh across `job-6prgb2hkbe8c` and `job-5s10a9cdqh77`, both `exit 0`, official test split untouched) definitively separated memorization from true latent identification. Monolithic `joint_transformer_active` reached `24/32` (`75.0%`) on familiar regimes (`2.14°` offline yaw error) yet collapsed `-68.8 pp` to `4/64` (`6.2%`) on combined OOD (`0/32` compositional OOD, `4/32` continuous-angle OOD; `107.47°` / `83.60°` offline error); `factored_no_equiv_active` collapsed to `1/64` (`1.6%`) on OOD. By contrast, SO(2)-equivariant `ours_retained_active` (`luminance_flow`) achieved **`50/96` (`52.1%`, `79.4%` of `privileged_oracle` `63/96 = 65.6%`)** overall (`48.4%` = `31/64` OOD) with **`43 wins / 0 losses`** vs. `nominal` (`7/96 = 7.3%`, `+44.8 pp`, 95% CI `[+21.8, +66.7]`), `43W/0L` vs. `ours_cleared` (`7/96`), `48W/1L` vs. `ours_swapped_donor` (`3/96`), and `+22.9 pp` vs. `joint_transformer_active` (`28/96`, 95% CI `[+3.1, +41.7]`). Even passively without an active probe, `ours_retained_passive` reached **`38/96` (`39.6%`; `39.1%` OOD)**, statistically beating `joint_transformer_active` (`+10.4 pp`, 95% CI `[+2.1, +21.9]`). Grayscale optical-flow `analytic_visual_sysid_passive` scored `59/96` (`61.5%`, within `9.4 pp` of `ours_retained_active`, CI `[-28.1, +4.2]`), while `ours_retained_active` beat `analytic_visual_sysid_active` (`39/96 = 40.6%`) by `+11.5 pp`. [Scaled Result](../outputs/libero-xemb-ur5e-scale-20261120-r1/RESULT.md) · [Execution](../outputs/libero-xemb-ur5e-scale-20261120-r1/EXECUTION.md)

**Held-out 6-DoF `UR5e` cross-morphology visuomotor SysID (baseline POC):** Trained exclusively on 7-DoF `Panda`+`Sawyer` (8 fitting tasks, `224` trials), our SO(2)-equivariant factored visual reader transferred zero-shot to held-out physical 6-DoF `UR5e` (`7 != 6` joint DoF mismatch) with a 100% frozen $\pi_{0.5}$ policy across `48` paired queries (`528` live closed-loop episodes, `2.321` Pro6K GPUh, `exit 0`). Overall live success (`n=48`): `privileged_oracle` `39/48` (`81.2%`), grayscale-flow `analytic_visual_sysid_passive` `33/48` (`68.8%`), raw-RGB `ours_retained_active` (`factored_equiv`) `23/48` (`47.9%`), `analytic_visual_sysid_active` `17/48` (`35.4%`), `factored_no_equiv_active` `14/48` (`29.2%`), `ours_retained_passive` `14/48` (`29.2%`), monolithic `joint_transformer_active` `8/48` (`16.7%`; `0/16` comp OOD, `1/16` cont OOD), `nominal` `7/48` (`14.6%`), `ours_cleared` `7/48` (`14.6%`), `analytic_proprio_sysid` `6/48` (`12.5%`), and `ours_swapped_donor` `0/48` (`0.0%`). Every paired 95% bootstrap CI of `ours_retained_active` over `nominal` (`+33.3 pp`), `joint_transformer_active` (`+31.2 pp`), `factored_no_equiv_active` (`+18.8 pp`), `analytic_proprio_sysid` (`+35.4 pp`), `ours_cleared` (`+33.3 pp`), and `ours_swapped_donor` (`+47.9 pp`) excluded zero. The remaining gap to grayscale flow (`68.8%`) isolates raw-RGB arm-appearance overfitting and directly motivates the 8-GPU `luminance_flow` scale-up. [Result](../outputs/libero-xemb-ur5e-sysid-20261120-r1/RESULT.md) · [Execution](../outputs/libero-xemb-ur5e-sysid-20261120-r1/EXECUTION.md)

On 50 simulated Sawyer Goal 5 source-development starts, a fixed scripted expert introduced at PaliGemma's first plate contact rescued 18/30 raw failures and lost 7/20 raw successes. Net native success rose from 20/50 to 31/50. The same expert achieved 37/50 from reset. All 50 takeover states and prefixes matched the raw branch. This establishes a post-contact corrective opportunity under privileged control. [Matched fork and reset control](../outputs/embodiment-sawyer-contact-fork-20261120-r1/RESULT.md)

The controlled OpenDrawer reader used one probe to choose a useful correction on measured native branches. The broader RoboCasa studies found no consistent held-out benefit from retaining history. The results below distinguish selection among executable actions, learned motor control and source-policy competence. [Controlled reader](#native-history-under-a-persistent-calibration-error) · [Multi-task comparison](#corrected-three--versus-six-task-transformer-comparison) · [Corpus diagnosis](../outputs/project-diagnosis-20261116-r1/RESULT.md)

## Synthetic learning histories

A five-condition, three-seed hidden-rule selection grid trained four-layer
Transformers for 20,000 updates per fit, then evaluated frozen weights using
each learner's own attempts on fresh scenes. Execution was perfect. On held-out
rules, the choice-and-outcome learner improved from 45.44% at k=0 to 57.05% at
k=6, while exact rule enumeration on those same histories reached 86.50%.
The training-rule gap was similarly large (58.19% versus 87.17%). All seeds
failed the original near-reference screen. Demo-only reached 45.49%, blind
choices 41.14%, full corrections 67.51%, and linked attempts/corrections 69.62%
at held-out k=6. These curves show partial history use in the synthetic task,
not robotic adaptation or learned information-seeking exploration. The
reference uses a full-family prior rather than each split's sampling prior,
and that grid did not include a same-model feedback intervention.
[Results, seed variation and limitations](../outputs/learning-histories-20261111-r1/RESULT.md).

A subsequent frozen-checkpoint intervention reproduced all six original curves
exactly and evaluated fresh paired histories. On held-out rules with independent
random attempts, real feedback reached 54.65%, masked feedback 43.94%, shuffled
feedback 38.83%, and demo-only context 44.93%. Real feedback helped in all three
seeds; both paired within-fit intervals excluded zero in each seed. Independent
random choices remove the training actor's indirect reward channel. This supports
beneficial feedback dependence in the synthetic task. Training-actor histories
raised held-out accuracy only from 57.38% to 59.15%, still below their 84.49%
enumeration reference; training rules had a similar deficit. History-policy
shift therefore does not explain most of the gap. These are frozen-query input
interventions, not alternate closed-loop rollouts or robotic adaptation.
[Paired diagnostics and limitations](../outputs/learning-histories-feedback-20261111-r1/RESULT.md).

A five-recipe, three-seed screen selected dense multi-query supervision with
Fourier features for correction reading. On fresh frozen-checkpoint evaluation
sessions, held-out k=6 accuracy rose from 67.40% to 75.64%, improving in all
three fits. The same-history enumeration reference reached 94.94%; training
rules also retained a 10.41-point deficit. The revised attempt-and-reward
learner reached 60.50% versus 57.34% originally, with one fit slightly worse
and different learner-generated histories. Every fit still failed the
within-five-points-at-every-prefix screen. This confirms a recipe improvement
on the same rule split, not a solved inference problem or robotic adaptation.
The screen changes attention, supervision and features together and does not
isolate their effects. [Fresh confirmation, all recipes and limits](../outputs/learning-histories-fix-20261111-r1/RESULT.md).

## Sampling variation on standard LIBERO-90

Across 300 matched query poses on 15 fitting tasks, two independently sampled frozen π0.5 rollouts had median wrist separations of 2.21 mm after the first five controls and 1.15 cm after 50. Both succeeded on 74 poses, one succeeded on 20, and neither succeeded on 206. Hindsight selection between complete recorded rollouts raises 28.0% average single-draw success to 31.3% on this snapshot. A per-decision selector and a history-conditioned LoRA policy require live paired evaluation. [Matched trajectories, native outcomes and scope](../outputs/libero-native-noise-20261119-r1/RESULT.md)

In the registered full-action LIBERO-90 study, the outcome and imitation readers finished fitting on 960 practice/query pairs. During development, the single-draw native baseline encountered a different motor input from the compute-matched baseline on task 40, repeat 3. The 30-task test never opened, leaving the effect of cross-episode practice on native retry success unmeasured. [Verified failure and recovery evidence](../outputs/libero90-full-action-20261119-r1/RESULT.md)

## LIBERO-Pro2.0 retry development

In a closed-loop Goal 2 experiment, true X decoding plus a grasp-conditioned regrasp succeeded on 24/24 assignments across six poses, two command conventions and BASE/D17. Correct X decoding with always or never regrasping each achieved 22/24. Regrasp rescued D17 on pose 30 and spoiled BASE on pose 33 under both wire signs. Open-jaw pulses identified X on 24/24 and gave identical wrist/jaw readings in all 12 grasp-condition pairs. Contact evidence was already present by the regrasp decision on all six measured poses. [Native motor and contact evidence](../outputs/libero-command-grasp-20261119-r1/RESULT.md)

On the held-out reversed-X/D17 combination, the factored and capacity-matched joint readers each identified both factors on 8/8 fresh poses for each of three fitting seeds. Both succeeded on 7/8 queries. The free-space probe with the fitting-selected fixed grip scored 4/8. A controller deciding from the live query grasp matched every previous-attempt outcome, including 30/32 across all four conditions. Prior contact and explicit factorization added zero measured successes on this Goal 2 panel. [Held-out reader comparison](../outputs/libero-command-grasp-reader-20261119-r1/RESULT.md)

For the same bottle instruction, released D17 and D18 produced four of twelve pose/goal pairs with two failed practices and disjoint successful query corrections. Retreat and regrasp reached 10/12 D17 queries and zero D18. The archived D18 recovery reached 4/12 D18 and zero D17. Six public wrist/jaw features chose 6/8 correct strict retries, versus 0/8 after swapping histories and 4/8 for the fixed choice. Five-frame π0.5 features chose 4/8. All paired terminal images differed. All four qualifying pairs concerned the cabinet. [Same-goal recovery evidence](../outputs/libero-temporal-recovery-20261119-r1/RESULT.md)

A reader fitted only on 16 bowl practices transferred to six known bottle development poses. With π0.5 features it selected 9/12 correct native retries after two zeros, falling to 3/12 with the matched histories swapped. General PaliGemma PT reached 6/12, pooled RGB 7/12 and a secondary six-feature wrist/jaw control 10/12 on the same fitting data. The respective all-assignment scores were 63, 59, 60 and 64/96. This supports transfer of execution information and retains a compact motion explanation. [Bowl-to-bottle transfer](../outputs/libero-witness-transfer-20261119-r1/RESULT.md)

On the paired bowl test, the LIBERO-tuned π0.5 tower, general PaliGemma PT-224 tower and time-matched pooled RGB classified 16/16, 15/16 and 14/16 development practices. All selected 4/4 strict corrections. Across 32 balanced assignments they scored 23, 24 and 24 native successes. The one-success difference came from an ambiguous dual drop and an unavailable cabinet query. [Frozen encoder comparison](../outputs/libero-witness-encoders-20261119-r1/RESULT.md)

On two of four prespecified development triples, identical bowl commands and two zero rewards called for opposite executable corrections from the same query start. RGB histories selected 4/4, the same histories swapped selected 0/4, and commands or public motion selected 2/4. Final camera views alone also selected 4/4. Across all 32 assignments of outcomes and hidden goals, the final-view reader with native feedback reached 23/32 versus 13/32 when its decision ignored feedback. Four development reset IDs, two goals and a released D18 drop support this controlled result. [Paired physical evidence](../outputs/libero-visual-witness-20261119-r1/RESULT.md)

On the same two-axis diagnostic, all three support-only MLPs that shared an estimator across X and Y recovered both signs on all 60 double-reversal variants. Three joint MLPs missed all 60, despite getting 175–177/180 familiar variants correct. Training samples, supervision and the algebraic command decoder were matched. Command-only and motion-only inputs each had identical empirical distributions across conventions for every source pair. The 60 variants reuse 15 development reset pairs. [Factorization and controls](../outputs/libero-axis-factor-20261119-r1/RESULT.md)

With physical probe commands on fresh support resets, the preregistered shared reader selected corrections yielding 64/64 native successes during open-loop playback of π0.5 tapes across 16 development starts and four constructed command conventions. Cleared support reached 16/64, opposite support 0/64, and the analytic rule 64/64. The withheld double reversal alone gave 16/16 with the reader versus 0/16 cleared. Four full simulator-state hashes differed from the collection archive. All newly paired branches shared identical starts and nominal playback reproduced all 16 archived reward sequences. [Native physical controls and limits](../outputs/libero-axis-native-20261119-r1/RESULT.md)

Across eight project-development BASE/S18 pairs with matched initial physics, S18 offset the public wrist position by approximately (+7.5, −6.0, +4.5) cm while preserving consecutive displacement within 1.21 × 10⁻⁷ m for 80 identical controls. A history reader that only sees command–displacement pairs therefore lacks a useful signal for this released additive calibration case. [Native observability test](../outputs/libero-s18-probe-20261119-r1/RESULT.md)

In a coordinate-calibration diagnostic on archived BASE trajectories, a command–motion sign rule recovered both coordinates of the first frozen π0.5 command on all 15 eligible development reset pairs with both axes reversed. A fresh reader trained on the three other conventions and balanced probe directions recovered both signs on 28, 22 and 23 of the 60 direction variants in three fits. Clearing history or supplying opposite-convention history gave 0/60 in each fit. Each physical query contributes four direction variants. These scores measure offline command recovery. [Calibration results and controls](../outputs/libero-axis-calibration-20261119-r1/RESULT.md)

A linear calibration readout after the two support tokens recovered both signs on 27, 33 and 33 of the 60 double-reversal variants, versus 25, 26 and 15 using the corresponding untrained encoders. The same readout fit at the trained query token reached 9, 11 and 22. This reused the exploratory development panel and the same three training seeds. [Frozen representation analysis](../outputs/libero-axis-readout-20261119-r1/RESULT.md)

The four-mode timing pilot had 32 fitting queries. Only 19 had any successful
action, 12 distinguished successful from unsuccessful modes, and two of
those 12 were delayed. On 32 development queries the retained reader scored
20 in each fit. Clearing or swapping support scored 20, 20 and 19. Swapping
changed up to 16 choices. The fitting-selected fixed mode scored 18, the
best fixed mode in hindsight scored 19, the lag rule scored 21, and the
measured action union scored 22. Ten of sixteen delayed queries failed every
mode. Three paced outcomes changed across workers on matched reset IDs,
making this a poor panel for resolving a one-success advantage. The evidence
establishes sparse decision supervision and limited headroom above the lag
rule on this action menu. [Timing pilot](../outputs/libero-decision-pretraining-20261119-r1/RESULT.md)

A 2,048-update timing adapter fit used 8,192 windows from 1,299 released
demonstrations across 24 fitting tasks. On eight development tasks, true
timing from reset scored 4/16 BASE and 1/16 S20, versus 16/16 BASE and
1/16 S20 for frozen π0.5 and its zero-correction solver control. Fixed
pacing scored 9/16 BASE and 5/16 S20. True timing from reset had zero wins
and one loss against its fitting-selected fixed flag. The public lag rule
identified all 32 timing settings. Held-out demonstration loss improved,
while the learned correction failed to open useful live timing-selection
headroom. [Native S20 motor comparison](../outputs/libero-s20-adapter-20261119-r1/RESULT.md#native-development-benchmark)

On the released-scene development probes, reward-aware retries scored
157/192 versus 81/192 when the selector ignored reward. Adaptive and fixed
second-probe schedules tied at 154/192 when both used two rewards. In the
separate bottle replay, repeat/switch scored 31/48, matching the ceiling
given the observable tape and reward. A motion classifier scored 28/48,
and the privileged hidden-goal ceiling was 34/48. These comparisons show
where a simple use of feedback captures the available gains and where
stronger continuations or more informative interactions are needed.
[Scene probes](../outputs/libero-scene-probes-20261119-r1/RESULT.md) · [Bottle replay](../outputs/libero-failure-diagnosis-20261119-r1/RESULT.md)

On a custom two-instruction LIBERO-Object development study, a bowl-trained
reader of commands, sensed motion and rewards reached 192/288 native successes.
Commands and rewards alone reached the same score, the ceiling of the measured
query alternatives. Motion changed 38 choices and zero native outcomes.
Both instructions placed soup in the soup scene. In the salad-dressing scene
both instructions worked and two rewards sufficed. In the ketchup scene only
the ketchup instruction placed either object. The 288 paired assignments
reused 108 simulated episodes on six reset poses per layout.
[Layout-level outcomes and intervention checks](../outputs/libero-object-history-20261119-r1/RESULT.md)

On a separate Goal 4 BASE bowl panel, opening the gripper during lift allowed
1/10 new practices to complete. The same override during descent allowed 9/10.
A reader trained on 30 earlier practices that added wrist motion around jaw
opening to public timing features earned 237/320 paired native successes versus
224/320 with the timing features. The aggregate difference came from histories
involving one placement where neither trigger fired. On the 256 assignments
where both triggers fired on both practices, wrist and timing readers earned
181 and 182 successes, versus 160.5 and 160 with same-start motion reassignment.
The 320 assignments reuse five practice poses and four query poses in fixed-command
simulator replay. [Physical intervention and transfer checks](../outputs/libero-release-phase-20261119-r1/RESULT.md)

A randomized single-retry scorer averaged 19.0/32 development successes. Giving the same transformer both measured retry outcomes for every fitting history raised the mean to 19.5 and the episode cost from 64 to 96. At equal 48-episode budgets the single, sparsely paired and fully paired fits averaged 18.25, 18.375 and 18.00. A two-row lag table fitted on the same single or full labels reached 20.25 in both equal-budget arms. The best measured candidate reached 22 and the existing lag rule reached 21. All fits reused the same eight development tasks. [Logged and branched comparison](../outputs/libero-logged-retry-20261119-r1/RESULT.md)

### Embodiment and calibration controls

- On 16 fresh paired Panda/Sawyer layouts with independently crossed command yaw and gain, a frozen analytic estimator inferred yaw within 22.5° on all 128 practices for each schedule. Mean angle errors were 2.47° with consecutive directions and 2.34° with shuffled directions. For shuffled probes, matched, yaw-swapped, gain-swapped and common query MSEs were 0.8660, 0.9825, 0.9001 and 0.9825. The ordering contrast crossed zero. Shuffled-probe gain error was 0.183 on Panda versus 0.021 on Sawyer. This measures source-arm response inference. [New factorial panel](../outputs/embodiment-probe-order-20261120-r1/RESULT.md)

- On the archived Panda/Sawyer command-frame panel, a four-step physical estimator recovered the nearer hidden yaw in 27/32 practices versus 20/32 using the current step alone. On separate query resets its normalized XY motion MSE was 0.9508 with matching practice, 1.0988 with swapped practice and 0.9948 with a fixed common correction. The paired interval for the fixed-common comparison crossed zero. Mean absolute angle error was 30.3°, with yaw and gain coupled in the two tested settings. [Run and limitations](../outputs/embodiment-tape-observability-20261120-r1/RESULT.md)

- At fixed training cost on simulated Panda and Sawyer, four source command settings per arm reduced the reader’s held-out motion MSE by 3.75% on a new angle and 4.01% on a new yaw–gain combination versus the two-setting fits. Both missed the registered 10% threshold. Matching versus swapped support gave 0.53594 versus 0.53591 across the two arms. The 256 native practice/query tapes and nine reader fits measured response prediction. The held-out-arm task-success question remains open. [Results and controls](../outputs/embodiment-command-frame-20261120-r1/RESULT.md)

- An explicit hard bypass reproduced all 19,563 issued actions byte-for-byte from 100 native Panda/Sawyer source rollouts. The residual was never called on the closed branch. This is an archived-action check. Learned abstention and closed-loop gains remain open. [Identity audit](../outputs/embodiment-meta-factory-20261120-r1/identity-native.json)

- On 50 new native Goal 5 tabletop placements, one dual-source PaliGemma SFT checkpoint completed 25/50 Panda and 6/50 Sawyer episodes. Extended action-regression SFT on a larger bank completed 34/50 and 20/50. Each row used identical frozen weights across both arms and each controller issued five controls per replan. These are source-motor results on one task. [Native counts and checkpoint evidence](../outputs/libero-paligemma-dualarm-20261120-r2/evidence/source50/native.json)
- On the same 50 placements per arm, continuing the categorical motor with the first generated bank and recovery data changed Panda success from 14 to 23 and Sawyer from 15 to 16. On Sawyer the continuation solved 11 previously failed starts and lost 10 previous successes. The separate recovery fit scored 17 Panda and 15 Sawyer. The waypoint-supervised fit scored 1 on each arm. All six measured candidates missed the source cutoff of 38 successes per arm. [Paired generated-bank evaluation](../outputs/embodiment-probe-order-20261120-r1/EVAL-BROAD.md) · [Five-model native tape audit](../outputs/embodiment-probe-order-20261120-r1/MOTOR-LIVE.md)
- All 30 failed Sawyer starts and all 20 successful starts for the Huber checkpoint reached direct finger-to-plate contact. Median net plate progress toward the goal after first touch was 11.8 cm in failures and 20.5 cm in successes. All 50 native replays reproduced the original controls, motion, episode lengths and rewards. This locates the failure after first contact in this cohort. [Contact definitions, geometry, and results](../outputs/embodiment-sawyer-source-contact-20261120-r1/RESULT.md)
- At the same 6,400 additional updates from the same categorical parent, expanding successful generated training coverage from 95 to 349 layouts changed native success from 23/50 to 9/50 on Panda and from 16/50 to 22/50 on Sawyer. The three-bank fit gained four Panda starts and lost 18. It gained 16 Sawyer starts and lost 10. The smaller of its two arm totals fell from 16 to 9 on this task and training seed. The best single motor under the registered two-arm criterion remains Huber regression at 34/50 Panda and 20/50 Sawyer. [Input manifests and paired native results](../outputs/embodiment-source-bank-scale-20261120-r1/RESULT.md)
- Shifting MuJoCo Panda finger collision geometries by 0.035 m in XY left all precontact 256×256 RGB and public proprioception bitwise identical across 104 CPU replays (0 anomalies, 8/8 BASE, 8/32 open-loop shifted, 8/8 tasks post-contact distinguishable), but failed closed-loop Gate 1 on frozen π0.5 across eight development tasks (`job-gmj5rsb0xd59`, 168 paired episodes): shifted success was 23/48 nominal versus 22/48 grasp-compensated and 20/48 grasp-and-release-compensated, with 8/8 BASE in all three arms (0.447 GPUh). [TCP falsification](../outputs/libero-xemb-tcp-20261120-r1/RESULT.md) · [Execution](../outputs/libero-xemb-tcp-20261120-r1/EXECUTION.md)
- A reader and action adapter trained on simulated Panda and Sawyer reduced frozen π0.5's native success on a held-out UR5e from 4/4 to 2/4 on two development tasks. Empty history scored 4/4, wrong-arm history 0/4 and independent target calibration 3/4. A post hoc nearest-centroid probe identified 0/4 UR5e histories. The policy used history, but the correct history cost two successes against the unmodified motor. [Paired scores, latent readout and source controls](../outputs/libero-crossarm-20261120-r1/RESULT.md)
- A general PaliGemma motor trained on simulated Panda demonstrations scored 0/2 on fresh Panda plate-pushing starts. After 100 source motion-reader updates on Panda/Sawyer and 150 joint reader–adapter updates with the motor frozen, correct practice, cleared practice, wrong-body practice, raw control and privileged calibration each scored 0/2 on UR5e. The joint embedding identified 2/8 independent arm histories, including 0/4 UR5e. The planned two-arm demonstration fit stopped at 11/16 Panda and 0/16 Sawyer native expert successes. [PaliGemma experiment and evidence](../outputs/libero-paligemma-crossarm-20261120-r1/RESULT.md)
- In a fresh 12-episode comparison, the 180°/low-scale/inverted-gripper composition again defeated nominal π0.5 and was solved by learned calibration, analytic image motion and a public-joint Jacobian requiring no RGB or URDF. The nominal 45° case succeeded on this allocation after failing on the first. The two allocations had different opening action hashes despite identical recorded reset and motor-noise inputs. [Fresh paired native scores](../outputs/libero-xemb-sysid-20261119-r1/evidence/scoring/joint.json) · [Joint-angle readout](../outputs/libero-xemb-joint-sysid-20261119-r1/RESULT.md)
- A factored public-history reader rescued both shifted native Goal 1 queries on one development start. Clearing or swapping support, a joint history transformer, and wrist-only calibration each failed both. Closed-form visual calibration, a factored reader without equivariance, and current-prefix calibration also succeeded on both. Across eight tasks the true interface gave 47/48 native successes against nominal 9/48. [Native rescue and controls](../outputs/libero-xemb-sysid-20261119-r1/RESULT.md)
- Across eight development tasks, lowering physical finger friction reduced default frozen-policy tape success from 8/8 to 6/8. The true-material choice between default and paced tapes tied fixed default at 14/16, with one recovery and one loss from switching to pacing on reduced friction. [Paired material replay](../outputs/libero-xemb-friction-dev-20261119-r1/RESULT.md)
- Rotating the physical agentview camera about its optical axis by 0°, +90° or −90° changed the reset image while preserving measured physics and public proprioception. An RGB template fitted on eight training tasks identified 21/24 orientations on eight development tasks, compared with 8/24 for a cleared orientation input. Errors clustered on one task. [Physical camera-roll audit](../outputs/libero-xemb-camera-20261119-r1/RESULT.md)
- On a previously studied Goal 2 bottle-placement pose, reducing realized pad/bottle friction from 2.00 to 0.95 left all measured precontact observations identical and made a regrasp fail. The fixed nominal frozen-policy tape still succeeded under both materials, tying the material-aware oracle at 2/2. Six native 300-control replays used zero GPU-hours. [Friction gate and per-control readback](../outputs/libero-xemb-contact-20261119-r1/RESULT.md)

## LIBERO-Gen motor qualification

The released frozen Behavior Prompting Policy succeeded on 20/20 paired
native initializations for each of two bowl-to-plate choices: the center
bowl and the bowl beside the plate. Both candidates shared exactly matching
RGB/wrist/proprioceptive resets; all 967 saved tensors loaded exactly and
parameters stayed frozen. Initializations 0–4 repeated the engineering panel;
new indices 5–19 succeeded 15/15 for each candidate. This qualifies these two
motor choices on the native task family, not a released benchmark score,
custom relational layouts or history-based adaptation. No history reader
was trained. [Protocol, raw evidence and limits](../outputs/libero-gen-history-pilot-20261111-r1/RESULT.md).

The first custom-layout smoke then failed on all four tested layouts for
each prompt (0/8 physical episodes), despite correct paired resets and
unchanged frozen weights. Several trajectories transported a bowl but
never reached either native plate predicate; other trajectories reached
empty space. A targeted two-episode native-state0 replay through the same
recorder then succeeded with both candidates, matching the prior initial
observations and the smoke prompts exactly. This supports custom scene
transfer as the leading explanation, with a single-state control limit.
This is a development failure, not a history-learning result. [Custom-layout evidence and limits](../outputs/libero-gen-custom-layouts-20261111-r1/RESULT.md).

A bounded demonstration repair then obtained both intended native goals on
custom layout 0, including a corrected teacher that aligned the center bowl
above the plate before descending. In eight paired frozen-policy episodes
across that scene and a nearby scene, original prompts scored **0/4** and
repaired prompts **0/4**. Initial states, observations and per-episode seeds
matched; all 967 checkpoint tensors loaded exactly, and copied original
prompts reproduced their historical tensor hashes. Every paired action
trajectory changed with the new prompt, but neither placement nor the two
intended choices became reliable. This establishes an unsuccessful prompt-only
repair on this development pair, including the demonstration scene. It does
not establish that motor fine-tuning or another checkpoint will fail; no
history reader was trained. Demonstrations and policy episodes use the same
first-native-success endpoint, without an added released-placement persistence
criterion. [Repair results, teacher fixes and evidence](../outputs/libero-gen-prompt-repair-20261111-r1/RESULT.md).

## RoboCasa365

The Atomic-Seen baseline uses the native JAX π0.5 Human300 checkpoint at step **75,000**. Load `pi05_pretrain_human300/multitask_learning/75000` from `robocasa/robocasa365_checkpoints` at revision `c484448a`, including its bundled normalization. OpenPI predicts 50 actions and executes five before replanning, with ten denoising steps. [Pinned checkpoint](https://huggingface.co/robocasa/robocasa365_checkpoints/tree/c484448aba1a9b60a04c9b0ca117241518ea69f3/pi05_pretrain_human300/multitask_learning/75000)

| Measurement | Result | What it establishes |
| --- | --- | --- |
| [Native Human300 atomic SFT reproduction](../outputs/robocasa-sft-reproduction-20261109-r1/ATOMIC-RESULT.md) | 363 successes in 900 episodes (40.33%) across 18 atomic tasks, versus the published 39.6%. | The original JAX evaluator, checkpoint and horizons give a credible atomic reference. The [composite continuation](../outputs/robocasa-sft-reproduction-20261109-r1/CLOSEOUT.md) closed early and is not a composite benchmark score. |
| [Paired RLinf source endpoints](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/robocasa-endpoint512-acceleration-20261104-r1/RESULT.md) | On 512 paired cases per task, OpenDrawer improved from 331 to 369 successes and CloseToasterOvenDoor from 168 to 212. Both passed the two-task Holm correction. | These fixed teachers improved over SFT on their own tasks. In a [separate three-task test](../outputs/robocasa-eight-hour-babysit-20261104-r1/SOURCE5-RESULT.md), Faucet passed (204 → 242/512); Counter and Coffee did not. This does not establish five effective sources. |
| [Frozen AD steering with real prior attempts](../outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/STAGE1-RESULT.md) | On held-out Fridge after five attempts, full-history KV and velocity students succeeded on 33 and 41 of 50 cases; clearing history gave 34 and 39. | Using the frozen expert improved some transfer comparisons with a direct action head. The full/reset comparisons show no consistent benefit from retaining history; one fit per interface. |
| [Additional 10-update PPO pilots](../outputs/robocasa-six-source-decision-20261112-r1/RESULT.md) | Cabinet scored 38→40/64 (nine paired gains, seven losses); Rack 42→41/64 (four gains, five losses); CounterToStove 40→40/64 (seven gains, seven losses). Each preserves 1,280 training episodes and a verified full checkpoint/archive. | These short single-seed pilots establish collection and optimizer execution, without clear evidence of endpoint improvement. Physical/public initial states matched exactly; maximum initial RGB differences were 1, 1 and 2 intensity levels. Explicit DPT teacher-query labels remain uncollected. |
| [Complete four-method frozen KV comparison](../outputs/robocasa-kv-comparison-20261112-r1/RESULT.md) | Across12,000 audited episodes, DIT had the highest sixth-episode full-history success:55.3% ID and56.0% OOD, versus AD46.7% and44.0%. DPT scored47.3%/49.0%, IC-IQL48.0%/44.0%. | DIT exceeded SFT by+8.7pp ID/+12.0pp OOD and fixed PPO130 by+4.0pp/+6.0pp (descriptive gaps). Full-minus-reset was+4.7pp ID (95% interval−2.0 to11.3) and+6.0pp OOD (−3.0 to15.0). The larger fresh paired confirmation below did not reproduce these history gains or establish superiority over SFT/PPO. One fit/method, unequal training durations and first-query replay discrepancies limit causal attribution. Historical headline only: original collector histories are excluded from the corrected matched comparison; closed payloads are in verified cold storage. |
| [DIT first-attempt evidence audit](../outputs/robocasa-dit-diagnostic-20261113-r1/RESULT.md) | All 250 paired first attempts have identical empty entry signatures; 14 outcomes differ. In eight raw Coffee pairs, six match in every compared chunk field. One pair first differs by one RGB component after 100 controls; the success-discordant pair differs by 60 RGB components at reset, each by one intensity level, before features and commands diverge. | No recorded initial history leak was found. Rendering variation is a plausible contributor in the inspected mismatch; this retrospective subset neither establishes its causal effect nor explains all 14 cases. Exact initial pixels alone do not ensure identical later observations. |
| [AD](../outputs/robocasa-eight-hour-babysit-20261104-r1/AD-ID50-RESULT.md) and [DPT](../outputs/robocasa-eight-hour-babysit-20261104-r1/DPT-OOD50-RESULT.md) direct-action baselines | Stronger performance on the three fitted tasks than on held-out Fridge and Microwave. | These students bypass the native action expert. Their performance cannot be attributed to KV steering or taken as proof of adaptation. |
| [Flora calibration](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/robocasa-joint-h600-preparation-20261102-r4/RESULT.md) and [reward mismatch](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/robocasa-ppo-reward-gap-investigation-20261102-r1/RESULT.md) | Native/task-specific AUROC ranged from 0.848 to 0.982 on five fixed 50-episode, 600-control banks. A separate OpenDrawer training audit found maximum grades on 58 of 183 simulator failures. | The rewarder ranks outcomes usefully on these panels, but a high grade is not native success. The matched 50-case residual-PPO comparisons did not establish superiority over simulator reward. |

A [controlled hidden-goal pilot](../outputs/robocasa-meta-feedback-20261109-r1/RESULT.md) executed three frozen π0.5 bowl prompts on each of eight scenes. Two of 12 support rollouts placed and released a fruit, both into the left bowl on one scene. None of the 12 fresh-query rollouts released into either bowl. These development cases provided zero successful query choices for testing reward memory or first-attempt exploration. A separate drawer manipulation check changed lateral miss at closure but yielded no handle acquisition or native success on the two query tapes examined.

In the [compact-table follow-up](../outputs/robocasa-compact-bowls-20261109-r1/RESULT.md), frozen π0.5 grasped in 8/8 trials but all three released fruits went right, regardless of the requested side. A scripted post-grasp carry using simulator coordinates released to either requested side on 3/4 scenes (6/8 continuations). A fixed reward-conditioned rule on these same development traces obtained 3/4 query goals, versus 2/4 with either constant choice and 1/4 with inverted reward. There were only two scene pairs; the original motor did not qualify for held-out evaluation.

On eight fresh compact-bowl scenes, a shared neutral frozen pickup followed by a privileged scripted carry produced both target-specific releases on seven. Under balanced hidden goals, choosing the correct carry reached 14/16 and either fixed carry reached 7/16. A leave-scene-pair-out diagnostic decoded the hidden goal from public support motion and terminal reward on 15/16 cases, versus 8/16 with reward masked. Using the decoded choice to select between the two physically executed query branches scored 13/16 with real reward, 7/16 with reward masked and 1/16 with counterfactual reward. The two ambiguous cases came from one timed-out support. This qualifies the action labels and the information in support for a learned-motor test. [Exact-fork qualification and decoder](../outputs/robocasa-bowl-dpt-labels-20261110-r1/QUALIFICATION.md).

In the subsequent learned-carry test, 24 training scenes supplied 956 masked action chunks, and eight development scenes selected 1,024 updates for both LoRA students. On 16 held-out query scenes crossed with both rewarded sides, DPT reached 3/32 matching-history goals, 5/32 with an opposing-goal donor, 4/32 with counterfactual reward and 5/32 with reward masked. Memoryless LoRA scored 5/32 and frozen π0.5 4/32. Matching history gained zero paired outcomes and lost two against its donor. Both scripted releases were feasible on all 16 scenes. A decoder fitted only on training scenes read 32/32 goals from public support motion and reward, and a linear probe of the selected DPT representation read 17/32. This one-fit binary pilot found no matching-history benefit in physical control. [Locked confirmation and protocol](../outputs/robocasa-bowl-dpt-labels-20261110-r1/RESULT.md).

The property-given LoRA control selected 128 updates on the same development scenes and reached 7/32 confirmation goals. It reproduced 2/8 fixed training goals, and its existing 1,024-update checkpoint reproduced 3/8. A teacher-assisted handoff after the shared lift reached 2/8. Continuing the same instructed fit with restored optimizer and sampler to 4,096 updates reached 8/8 training goals, gaining five and losing zero against 1,024. This establishes reproduction on those four training scenes. Held-out evaluation of the extended checkpoint is the next step. [Motor fitting diagnostics and resumption evidence](../outputs/robocasa-bowl-dpt-labels-20261110-r1/RESULT.md).

In an eight-seed OpenDrawer screen, two scenes opened under the frozen native policy. Both exact nominal replays matched. Both scenes still opened with either sign of a five-control, 4 cm lateral offset and zero correction. Nineteen of 20 branches opened across all five residual choices. The only timeout came after adding a correction that reinforced the offset. All four uncorrected branches succeeded, leaving no failures on which to measure rescue. [Matched drawer results](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/robocasa-drawer-authority-20261110-r1/RESULT.md).

A held lateral correction opened two of six natural OpenDrawer failures. Both rescued at −2 cm, the same sign as their nominal lateral misses. The best fixed offset matched the best per-scene choice. All eight archived command tapes replayed exactly, all eight native outcomes reproduced under fresh inference, and all six matched zero-offset continuations failed. [Paired evidence and videos](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/icrl-idea-checks-20261110-r1/RESCUE.md).

On two OpenDrawer scenes crossed with two noise seeds, normal effort and doubled post-grasp translation each opened 3/4 drawers at normal damping, 3/4 at 8× and 0/4 at 64×. Three pairs still grasped at 64×. The added translation changed no paired native success outcome. [Stiffness measurements](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/icrl-idea-checks-20261110-r1/STIFF.md).

On eight development scenes per task, four native Human300 π0.5 retries from exactly matched starts gave scene ICCs of 1.00 for CloseDrawer, 0.885 for OpenDrawer, 0.222 for CloseFridge and −0.067 for TurnOnMicrowave. All eight CloseDrawer scenes had unanimous outcomes, including three that failed on every attempt. Seven of eight OpenDrawer scenes had unanimous outcomes. This measures repeatability at a fixed starting scene. The effect of support history on a fresh query remains open. [128 native episodes and per-scene counts](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/icrl-idea-checks-20261110-r1/PERSISTENCE.md).

Across eight CloseDrawer development scenes and two noise seeds, the native policy, a stationary wait, and either lateral base movement each closed 10/16 drawers. A forward move closed 7/16, losing three native successes. No movement or wait rescued a native failure. A per-scene oracle also closed 10/16. Of the three recurrent failures, one never made gripper contact, one barely moved the drawer after contact, and one reached about 60% open before stalling. [Matched corrections and camera evidence](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/robocasa-closedrawer-corrections-20261110-r1/RESULT.md).

A memoryless LoRA student trained on the final Drawer130, Toaster130 and Faucet40 PPO actions reduced held-out native flow-matching error from 0.08113 to 0.07887. DPT + LoRA scored 0.08207 with the matching prior SFT attempt and 0.08262 with a different same-task attempt. The paired 95% interval for the own-history improvement included zero across 48 held-out reset scenes. The pilot measured offline imitation with one fit per arm. [Three-teacher RoboCasa pilot](../outputs/robocasa-three-ppo-dpt-lora-20261110-r1/RESULT.md).

On 96 fresh RoboCasa365 scenes with one shared frozen-policy first attempt, two-epoch memoryless LoRA succeeded on 60 retries, frozen π0.5 on 40, and DPT with own versus same-task donor history on 37 versus 54. On the 52 first-attempt failures these four arms recovered 27, 19, 12 and 21, respectively. Selecting training duration on 24 development scenes chose one epoch for DPT and two for memoryless. On the other 72 scenes, matching DPT reached 38, donor DPT 41, and memoryless 42. Own history gained six and lost nine paired cases against donor history. Native checkpoint selection improved DPT success without establishing a memory benefit. [Paired native retries and three-seed learning curves](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/robocasa-three-ppo-retry-20261110-r1/RESULT.md).

For historical Flow-SDE teachers, the [critic parity audit](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/robocasa-flow-sde-critic-parity-20261104-r1/RESULT.md) found that rollout and replay could use different denoising-step values even when likelihoods matched. Record the RLinf revision and critic definition when comparing these runs.

### Frozen DIT paired confirmation

The same frozen DIT checkpoint was evaluated on 256 fresh paired cases per task,
with the sample count fixed before outcomes. All 1,280 cases and 11,520 physical
episodes passed the archive audit. The endpoint is episode-six success; DIT
retains five prior attempts in the full-history arm and starts with empty replay
in the reset control. ID comprises Drawer, Toaster and Coffee; OOD comprises
Fridge and Microwave relative to the meta-training corpus.

| Panel | DIT full | DIT reset | SFT | PPO130 |
| --- | ---: | ---: | ---: | ---: |
| ID | 43.23% | 44.92% | 41.15% | 49.74% |
| OOD | 46.09% | 44.14% | 43.36% | 49.41% |

| Full DIT minus control | ID difference (pp), paired 95% interval | OOD difference (pp), paired 95% interval |
| --- | ---: | ---: |
| Reset: history effect | −1.69 [−4.43,1.17] | +1.95 [−1.17,5.08] |
| SFT | +2.08 [−1.17,5.34] | +2.73 [−1.17,6.64] |
| PPO130 | −6.51 [−10.29,−2.60] | −3.32 [−7.23,0.59] |

The larger panel does not confirm a history benefit or an advantage over SFT.
Full DIT trails PPO130 on ID, with its interval below zero; the OOD comparison
remains uncertain. Every per-task full/reset interval also includes zero.
The earlier small-panel history gain is therefore not replicated, rather than
evidence that the true effect is exactly zero. These are marginal, unadjusted
task-stratified paired bootstrap intervals conditional on one DIT fit and these
five tasks. They do not establish training-seed robustness, explain the original
method ranking, or resolve the corrected matched four-method comparison.
[Audited result](../outputs/robocasa-dit-paired-confirmation-20261114-r1/RESULT.md) ·
[Coverage](../outputs/robocasa-dit-paired-confirmation-20261114-r1/COVERAGE.json) ·
[Figure (PDF)](../outputs/robocasa-dit-paired-confirmation-20261114-r1/figures/frozen-dit-paired-confirmation.pdf).

### Corrected six-task PPO30 source baselines

All six fixed-step30 teachers now have verified 64-case paired comparisons
against SFT, using native success and 600-control horizons. Five panels were
recovered from saved source evaluations; OpenDrawer was evaluated on a fresh
paired bank after its orderly training exit skipped the original final panel.
The endpoint was fixed before outcomes.

| Task | SFT successes /64 | PPO30 successes /64 | Paired gain (pp), descriptive 95% interval |
| --- | ---: | ---: | ---: |
| OpenDrawer (fresh panel) | 37 | 48 | +17.19 [6.25,29.69] |
| CloseToasterOvenDoor | 19 | 15 | −6.25 [−17.19,4.69] |
| CoffeeSetupMug | 10 | 21 | +17.19 [3.13,31.25] |
| PickPlaceCounterToStove | 40 | 38 | −3.13 [−12.50,6.25] |
| TurnOffStove | 7 | 4 | −4.69 [−12.50,3.13] |
| TurnOnSinkFaucet | 24 | 28 | +6.25 [−3.13,15.63] |

OpenDrawer gains14 cases and loses3 relative to SFT on the same fresh64 cases;
its old SFT33/64 panel is separate and is not paired or pooled with this result.
The repair completed128 physical episodes, with every archive read back and all
paired case identities verified. Faucet uses its saved intermediate step30
panel, not its step40 final panel. [Drawer paired evidence](../outputs/robocasa-six-task-scale-20261113-r1/evidence/OPEN-DRAWER-BASELINE-RESULT.json) · [Saved-panel audit](../outputs/robocasa-six-task-scale-20261113-r1/evidence/SAVED-BASELINES-AUDIT.json).

Drawer and Coffee show positive gains with intervals above zero; the other four
intervals include zero. These are one training seed per task and marginal,
unadjusted paired bootstrap intervals, not a six-task superiority claim.
Collection completion alone does not establish uniformly stronger teachers.
These source comparisons do not measure held-out meta-learning or history gains;
the completed fixed-final transformer readout is reported below.
[Six-task baseline summary](../outputs/robocasa-six-task-scale-20261113-r1/SIX-TASK-BASELINES.json).
[Paired baseline figure (PDF)](../outputs/robocasa-six-task-scale-20261113-r1/figures/six-task-baselines.pdf).

### Corrected three- versus six-task transformer comparison

All eight fixed-final AD/DPT/DIT/IC-IQL fits and the complete13,200-episode
comparison finished. The audit verified25 cold archives, all40 paired cases per
task, and no missing slots or outcome-conditioned exclusions. The effective
panels are120 ID cases across Drawer/Toaster/Coffee and80 OOD cases across
Fridge/Microwave; repeated physical episodes are not independent cases.

| Method | Training tasks | Full-history sixth success, ID | OOD |
| --- | ---: | ---: | ---: |
| AD | 3 | 45.00% | 36.25% |
| DPT | 3 | 50.83% | 37.50% |
| DIT | 3 | 41.67% | 36.25% |
| IC-IQL | 3 | 49.17% | 43.75% |
| AD | 6 | 45.00% | 40.00% |
| DPT | 6 | 45.00% | 42.50% |
| DIT | 6 | 46.67% | 40.00% |
| IC-IQL | 6 | 47.50% | 41.25% |

Same-case SFT scored49.17% ID/41.25% OOD and PPO30 scored52.50%/40.00%.
No aggregate model-versus-baseline contrast establishes a positive advantage.
Three-task DIT was below PPO30 on ID by−10.83pp [−19.17,−2.50]. PPO30 is
task-matched on ID; both OOD tasks use the fixed OpenDrawer30 teacher, so this
is a single-source transfer control. These are separate cases from the64-case
source panels above, and separate checkpoints from historical PPO130.

The sole positive aggregate history interval is three-task IC-IQL on ID:
**+7.50pp [0.83,15.00]**, with14 wins,5 losses and101 ties. Its OOD effect is
+3.75pp [−3.75,11.25]; six-task IC-IQL gives−1.67pp [−9.17,5.83] ID and
+2.50pp [−5.00,10.00] OOD. No model has an OOD history interval excluding zero.
All eight three-to-six-task full-success intervals and all eight corresponding
history-change intervals include zero. DIT is not consistently the strongest
method in this corrected comparison.

These are marginal95% task-stratified paired bootstrap intervals, with one fit
per cell and no multiplicity correction or training-seed uncertainty. Budgets
match10,000 actor updates; DIT/IC-IQL additionally receive1,000 Q/V updates.
Adding tasks changes diversity, data volume and per-task exposure together.
OOD is relative to meta-training tasks, not necessarily base-VLA pretraining.
First full/reset outcomes disagree on2–7/120 ID and2–12/80 OOD cases depending
on the fit, despite passing the prescribed state/RGB-tolerance and reset-action
replay checks. The causes were not isolated; these discrepancies limit causal
attribution to history. All cases remain included.

The recommendation is **stop scaling this recipe in its current form**:
the isolated IC-IQL interval does not justify collecting more teachers without
independent confirmation and an advantage over stage controls. This is a
bounded resource decision, not proof against meta-learning generally. All GPU
jobs are closed at219.5/230 conservative GPU-hours; valid models, data and
evidence remain preserved.
[Decision and full comparisons](../outputs/robocasa-six-task-scale-20261113-r1/RESULT.md) ·
[Audited statistics](../outputs/robocasa-six-task-scale-20261113-r1/matched-finalize01/analysis/RESULT.json) ·
[Coverage](../outputs/robocasa-six-task-scale-20261113-r1/matched-finalize01/analysis/COVERAGE.json) ·
[History figure](../outputs/robocasa-six-task-scale-20261113-r1/matched-figures02/matched-performance.pdf) ·
[Scaling figure](../outputs/robocasa-six-task-scale-20261113-r1/matched-figures02/matched-scaling.pdf) ·
[Baseline gaps](../outputs/robocasa-six-task-scale-20261113-r1/matched-figures02/matched-baseline-gaps.pdf).

### Frozen DIT advantage and history diagnostic

The completed CPU diagnostic found uneven PPO quality and weak signals in the
current learner. Each source task contains537–2,179 successful training episodes
out of3,360, so successful behavior is present. First-to-last-ten-update success
changes are modest and vary by task; these collection rates are separate from
paired SFT/PPO30 evaluations.

On all16 validation streams/task at four fixed collection updates, DIT weights
are nearly uniform: effective sample size is99.50–99.86% of the panel size.
They change the all-query successful-example share by−0.453 to+0.188 percentage
points per task. No first-query DIT advantage AUC interval is wholly above0.5;
IC-IQL also has no first-query interval above chance. Some Q estimates predict
eventual success, but Q−V has weaker outcome ranking. This is observational
evidence, not a counterfactual test of individual action quality.

The all-query success label includes the suffix after the one success reward
is spent. Those queries have zero remaining return; their outcome ranking is
not an action-quality test. The later pre-success analysis below qualifies
that interpretation. First-query statistics are unaffected.

Archived native flow checks show DIT full/reset history differences of0.234%
and0.214% of native output RMS for the three-/six-task fits. The corresponding
relative imitation-loss improvements from history are+0.0056% and−0.0147%.
These are flow predictions on128 held-out queries per actor, not new simulator
success rates. Together with the native readout above, this favors qualifying
advantage weighting and useful history influence before more teacher collection;
it does not establish that stronger PPO would fix the result.

All four frozen Q/V models,13,824 model-query records and144 independent
statistic checks passed. Batched inference matched production stepping within
3.58e-7. The panel contains4,608 unique source queries in384 episode records;
uncertainty resamples whole streams, with one fit/model and unadjusted intervals.
No GPU-hours were added. [Diagnostic and limitations](../outputs/robocasa-dit-diagnostic-20261115-r1/RESULT.md) ·
[Complete statistics](../outputs/robocasa-dit-diagnostic-20261115-r1/evidence/analysis01/RESULT.json).

### Project data, training exposure and frozen-value diagnosis

A further CPU audit found 30 distinct reset seeds in every source stream,
while the native evaluation repeats one canonical scene across attempts.
Training and testing therefore differ in the scene structure of their
histories. Current initial-query pooled features identify source task with
99.375% accuracy on 2,880 held-out queries using fixed training centroids.
This measures visible task identity, not absence of useful scene-specific
unknowns or policy competence.

Reconstructing the exact actor sampler, with final RNG agreement against all
eight fits, gives 80,000 sampled queries per fit. Unique-query coverage is
47.54% for three tasks and 27.89% for six. Respectively 21.42% and 19.82% of
samples follow first success; 34.10% have no preceding complete episode.
This establishes unequal data exposure at the fixed update budget, not that
more training or success masking would necessarily improve native outcomes.

Removing post-success queries from the outcome-ranking proxy reveals modest
positive six-task DIT AUC intervals on Toaster and Faucet, and a negative one
on CounterToStove. DIT weights remain nearly uniform (ESS 99.43–99.83%).
In a separate frozen-value intervention, substituting another stream's action
increases observed-return prediction error in all eighteen task/model panels,
with nine positive marginal intervals. Substituting prior history has mixed
effects: one positive and two negative Q-error intervals among eighteen.
Thus some action information is learned, without consistent evidence that the
particular prior history improves prediction. Donor actions lack measured
counterfactual returns, so this does not validate action-value ranking.

All inputs were hash-verified; repeated Q/V predictions agree exactly with the
earlier diagnostic. Independent raw-record recomputation checked 588 statistics
and count identities. Both stages exited zero, used no GPU-hours, and their
1.44 MB of raw evidence was cold-archived with verified readback. The measured
limitations support stopping scale-up of the current recipe and first testing
whether matched history supplies useful information for attainable actions.
[Diagnosis, method differences and proposed next experiment](../outputs/project-diagnosis-20261116-r1/RESULT.md) ·
[Corpus evidence](../outputs/project-diagnosis-20261116-r1/evidence/corpus01/RESULT.json) ·
[Value interventions](../outputs/project-diagnosis-20261116-r1/evidence/value01/RESULT.json).

### Native history under a persistent calibration error

A controlled OpenDrawer pilot establishes useful history for a small causal
correction selector. The frozen Human300 pi0.5 motor receives a hidden signed
4 cm approach offset; the reader chooses left, no correction or right. Eight
nominally successful scenes train the reader and eight are held out. At the
fixed 1,024-update endpoint, averaged over three fit seeds:

| Reader | Prior probe action and reward | Uncorrected-attempt motion history |
| --- | ---: | ---: |
| DPT teacher labels | 95.83% | 100% |
| DIT, tau 1 | 50% | 100% |
| DIT, tau 0.1 | 58.33% | 97.92% |
| Measured-outcome weighting | 92.71% | 100% |
| Memoryless teacher-label control | 50% | 50% |

These are held-out second-attempt success rates scored against measured native
candidate outcomes. Clearing the main DPT reader's history reduces 95.83% to
50%; flipping its support reward reduces it to 4.17%. Main-version critic AUC
is 0.748–0.800, versus 0.998–1.000 with motion history. The latter variant also
changes the support policy, so the difference is not a single-feature ablation.
The results support usable history and identify estimated DIT weighting under
sparse feedback as an unresolved limitation; they do not justify new broad
PPO collection or establish a unique critic failure mechanism.

This is one task family, a hand-designed offset, a privileged geometry trigger,
and a three-action selector, not the earlier large VLA adapter or a multi-task
ID/OOD comparison. The 32 primary constructed trials share only eight held-out
scenes and two dependency blocks. Training data used 4090 and held-out data
Pro6K. All 16 tape replays match exactly; fresh closed-loop repeats preserve
success in all 16 and terminal physics in 14. The audit passed 1,333 checks;
the completed phase charged 6.578 of 8 GPU-hours. The earlier multi-task null
results stand. [Full results, controls and limitations](../outputs/robocasa-feedback-advantage-20261117-r1/RESULT.md).

On a separate OpenDrawer development panel, the full history adapter given the true condition and privileged timing succeeded on **3/8** fresh episodes, the uncorrected SFT motor on **2/8**, and the privileged teacher on **8/8**. This measures the learned motor interface with the answer supplied. The panel and action interface differ from the three-choice reader above. [Execution record](../outputs/robocasa-full-history-adapter-20261117-r1/EXECUTION.md) · [Command diagnostics](../outputs/robocasa-full-history-adapter-20261117-r1/AUTHORITY.json)

## Procedural LIBERO

Eight training layouts fixed before simulation passed construction, initial contact,
settling and object-centre visibility checks; all 48 dropped-bowl controls satisfied
only the intended receptacle predicate. Those controls establish predicate semantics,
not policy execution or robot reachability. Recomputing the 2,912 hidden rules on
each settled layout changed 3,040 of 23,296 nominal labels across five layouts.
Rendered inspection also found three partly cropped bowls. History targets must be
bound to realized reset geometry, and centre visibility alone does not establish
full-object visibility. [Geometry evidence and limits](../outputs/libero-procedural-qual-20261111-r1/GEOMETRY-RESULT.json).

## RoboTwin

### Sidney phone-stand baseline

All three policies start from `SidneyXie/pi05_robotwin` at revision `e49e2ab6`. The frozen evaluation uses the native per-target controller, a 1 cm near clipping plane on both wrist cameras, ten denoising steps, 50-action chunks and a 400-target episode limit. Within each domain, all three policies use the same 100 scenes, prompts and sampling seeds. [Pinned checkpoint](https://huggingface.co/SidneyXie/pi05_robotwin/tree/e49e2ab6c11f07511573b67261bd129e88d0a416) · [Matched evaluation](../outputs/robotwin-sidney-native-endpoints-20261109-r1/RESULT.md)

| Policy | Clean successes / 100 | Randomized successes / 100 |
| --- | ---: | ---: |
| Original SFT | 75 | 87 |
| PPO after 40 Adam updates | 77 | 88 |
| GRPO after 40 Adam updates | 78 | 90 |

**Training volume per algorithm** was two collections × 1,024 trajectories × 400 action slots = **819,200 nominal action steps**, or 16,384 chunks of 50. Each policy received 40 synchronized Adam updates. The training journal counted 842,240 slots including 23,040 setup/reset allowances. Actual delivered motor calls during training were unmeasured. [Training accounting](../outputs/robotwin-sidney-matched-baselines-20261106-r1/RESULT.md)

The subsequent frozen evaluation delivered **34,533 targets for PPO** and **33,768 for GRPO** across 200 episodes each, with zero further updates. The 1–3 point gains over SFT had two-sided exact paired p-values of at least 0.45. [PPO readback](../outputs/robotwin-sidney-native-endpoints-20261109-r1/PPO40-READBACK.json) · [GRPO readback](../outputs/robotwin-sidney-native-endpoints-20261109-r1/GRPO40-READBACK.json)

At the original 10 cm wrist clipping distance, SFT scored 23/100 clean and 31/100 randomized. Restoring the fingertips visible in training recovered 55 clean failures and lost three successes. On randomized scenes it recovered 58 and lost two. The reference loader matched all 813 released tensors. The uploader reports 66/100 clean on a separately evaluated cohort. [Camera experiment](../outputs/robotwin-sidney-gap-20261109-r1/RESULT.md)

On clean phone-stand seeds 100000 and 100001, two independent native SAPIEN replays per seed reproduced the saved Sidney success on action 128. All three camera views, public joints, articulation positions and velocities, and phone and stand poses were bitwise identical at all 129 observed boundaries, including the action-50 fork. [Replay evidence](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/robotwin-e1-replay-20261110-r1/RESULT.md).

### Motus and other RoboTwin results

| Measurement | Result | What it establishes |
| --- | --- | --- |
| [Released Motus native phone baseline](../outputs/robotwin-motus-release-baseline-20261109-r1/RESULT.md) | 27/100 clean and 25/100 randomized successes on `place_phone_stand` under the documented reconstruction. | Below the reported 49%/53% for that task. This is one motor and controller protocol, not the 50-task aggregate. |
| [Corrected Motus source confirmation](../outputs/robotwin-restoration-20261106-r1/MOTUS-SOURCE-GRPO-CONFIRMATION01-READBACK.json) | SFT 7/64 vs. selected GRPO 10/64 on disjoint paired scenes; eight gains, five losses, exact two-sided p = .581. | The predeclared improvement gate failed. Working PPO/GRPO execution is not evidence of an improving source or transferable adaptation. |
| [Teacher-forced action replay](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/robotwin-motus-replay-20261109-r1/RESULT.md) | On 20 clean demonstrations, mean shift-one arm error was 0.0154 rad versus 0.0789 for holding state. Wrist-camera swaps increased error. | The frozen motor predicts useful actions on expert observations. This offline measurement does not explain the closed-loop success gap. |

The [decoded-learner audit](../outputs/robotwin-restoration-20261106-r1/CPUDECODEDFIT01.json) and [four-method steering qualification](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/steering-four-method-pipeline-20261103-r1/RESULT.md) test objectives, causal context and recovery. They are software qualifications, not held-out performance scores.

The [full pre-consolidation evidence ledger](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/24fcc5b016f6d6f15cd7bad09cec9bd723873338/docs/FINDINGS.md) preserves detailed benchmark results, negative findings and citations for retired experiments. Consult a run’s own source and evidence before comparing results across protocols.

## Agent-collected RoboCasa context

A local Codex CLI selecting `gpt-6-astra` controlled native RoboCasa365 through
SSH/MCP and completed OpenDrawer, pretrain, seed 7 with hidden native feedback.
Native success first occurred at control 210 and remained true at the final
control 220. The recorder exported 15 pre-action context/action decisions,
221 observations and 663 camera images with verified alignment. This is one
successful integration example; no downstream VLA training or cross-task
success estimate was measured. [Run, replay and dataset evidence](../outputs/astra-robocasa-demo-20261112-r1/RESULT.md).

The direct Responses API client also passed synthetic and native RoboCasa MCP
qualification with `gpt-6-astra`: five calls per smoke, image tool outputs,
two recorded controls and an aligned causal export. The native smoke checked
all nine camera-image hashes and used 0.016101 conservative GPU-hours. This
qualifies the API transport; it did not attempt manipulation or learner training.
[API qualification evidence](../outputs/agent-sim-api-debug-20261112-r1/RESULT.md).

A paired teacher-update pilot used frozen Human300 pi0.5 on four fresh OpenDrawer
scenes (3/4 native successes under a 500-control budget). Astra abstained in all 72
judgments across current-only, actual-history and donor-history inputs, leaving
zero estimable teacher-change targets. Six native candidate pairs qualified;
one had different success outcomes. Thirteen of sixteen planned branches were
verified, including seven exact baseline continuations. The remainder stayed
unqualified after an initial-reset rejection and action drift on motor restart.
The experiment also reproduced and locally repaired camera corruption caused
by rendering-context cleanup. No TTT learner was trained. This is a negative
result for raw five-control candidate ranking under the tested prompt, not a
test of learned capability transfer. [Pilot results and limitations](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/f03ddbb69933e0cc6656aa54f7c9c900da4d580c/outputs/teacher-update-pilot-20261112-r1/RESULT.md).
