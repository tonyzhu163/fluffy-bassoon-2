# Which changes help a frozen policy on another arm?

The reference is a frozen π₀.₅ or π₀ motor with a tuned Perceiver at its command interface. In the earlier four-panel comparison the tuned reader completed 527/768 held-out command-shift retries and SymMeta completed 533/768. Each panel's paired task interval included zero. Native policy success, measured from the simulator reward, is the deciding comparison for a proposed adapter. This run also uses physical replay and source prediction screens to find out whether a mechanism has enough headroom to warrant a closed-loop experiment. [Earlier native comparator](../symmeta-perceiver-retry-20261123-r1/RESULT.md) · [Literature and registered comparisons](REVIEW.md)

## What happened in closed loop

We ran 96 new LIBERO-Pro2.0 episodes from 24 paired starts on simulated Kinova3 and IIWA arms. Each start had four deployments of the same frozen π₀.₅ policy on the nominal command interface. Every comparison used the simulator's terminal reward. Each new intervention changed what the robot actually did. None recovered a tuned Perceiver failure on the starts where it was tested.

| Experiment | Starts | Raw policy | Tuned Perceiver | First intervention | Second intervention |
| --- | ---: | ---: | ---: | --- | --- |
| Motion alignment, eight tasks on one layout | 16 | 16 | 14 | Fixed averaging 10 | Measured motion 12 |
| Placement instruction, one drawer task on four other layouts | 8 | 6 | 6 | Clock trigger 4 | Camera trigger 4 |

On the drawer panel the visual cue fired earlier than the clock on three starts, together on four, and later on one. It changed the next executed action on all eight starts versus tuned. Against the clock it rescued one start and lost another. Against tuned it lost two and rescued zero (−25 percentage points, paired 95% layout interval −50 to 0). Camera-based phase prediction on recorded motion had previously improved over the clock. Timing this one policy instruction with that prediction did not improve the native reward. [Drawer trial, controls and complete episode records](../frozen-transfer-cue-20261124-r1/NATIVE.md)

After both trials we computed the highest success a chooser could achieve if it knew every recorded outcome and selected one complete deployment at the start. With the tuned reader and the two new interventions available in each trial, hindsight achieved 14/16 and 6/8, exactly the tuned baseline. Adding the unadapted policy gave 16/16 and 6/8. Thus learning a better selector over these whole-episode options cannot recover another success on these starts. Switching actions partway through a trajectory remains a different question. The two failed Kinova3 drawer starts also failed under every tested live mode, while IIWA solved the same layouts under the original instruction. [Per-start results and input hashes](episode-choices.json) · [Linux scoring source](episode_choices.py)

Four Baige distributed jobs each used one RTX 4090. Launchers ran for 3,086 seconds in total (0.86 GPU-hours). The scheduler displayed 3,186 seconds across the four jobs (0.89 GPU-hours). The native readbacks verified 64 motion-alignment and 32 cue episodes. Offline reader fitting was the fourth job. [Scheduler receipt](completed-jobs.jpg) · [Motion trial and paired intervals](../frozen-transfer-progress-20261124-r1/NATIVE.md)

## Replaying a successful trajectory from the other arm

On the four drawer layouts, each arm's own raw command tape reproduced its original success label in native CPU replay (6/8 in total). Swapping in the other arm's tape from the **same layout** gave 4/8, including 2/4 on Kinova3. In a complete four-by-four comparison, one IIWA tape from layout 35 reached reward on Kinova3 layouts 34, 35 and 37. On layout 37 it finished at control 182, rescuing a failure shared by every live target policy condition. Its matching IIWA tape ran out without reward. No IIWA donor solved Kinova3 layout 36. Uniformly selecting a donor from one of the three *other* layouts succeeded on just 2/12 target–donor pairs. Each successful target run is a constructive witness of cross-arm action reuse; choosing the strongest source tape after inspecting all four target layouts gives an optimistic estimate of its performance elsewhere. [Registered swap and native returns](../frozen-transfer-donor-20261124-r1/RESULT.md) · [Independent verification and off-layout results](donor-recheck.json) · [Timing of the off-layout analysis](DONOR-ANALYSIS.md)

## Where an embodied decoder already changes native success

There is a separate positive control for the broader research question. On Panda, changing the control interface from Cartesian actions to joint increments made a decoder necessary. A decoder that recomputed differential inverse kinematics at the current joint pose preserved 5/6 frozen-policy successes. A fixed calibrated inverse and holding the joints each got 0/6. That comparison held the body fixed, used two tasks, and lacked a tuned, pose-informed learned decoder. It shows when current robot configuration can matter to a frozen policy, and leaves open whether geometry helps a matched learned decoder generalize across bodies. [Matched joint-control trial](../frozen-vla-native-controller-20261123-r1/RESULT-NATIVE.md)

## Post-grasp timing

Panda and Sawyer supplied a source estimate of the vertical response after commanded gripper closure. The experiment replayed the source commands on native IIWA and UR5e models, across three held-out tasks and two layouts. Mean vertical tracking error was 0.481 mm/control untreated, 0.473 with static gain, 0.491 with the source timing rule and 0.492 after shifting its corrections by two controls. The timing rule's difference from static was +0.018 mm/control (95% task interval −0.002 to +0.034). The two rules completed 2/12 fixed-tape continuations each. The source result gives no reason to carry this particular timing rule into a live-policy trial. [Motion, rewards and uncertainty](../frozen-transfer-timing-20261124-r1/RESULT.md)

## Detecting a loaded grasp

In matched IIWA–UR5e simulations using identical commands and initial scenes, only IIWA held the pudding on one task and only UR5e held the bowl on another. The remaining six tasks gave no strict carry-versus-empty contrast. Across three target arms and layout pairs, 146 matched observations came from 17 events on five tasks. The hand holding the object always had the greater measured jaw opening within a pair. Absolute widths overlapped. Source-trained central wrist image features plus proprioception scored 0.0956 Brier versus 0.0682 for proprioception and 0.1022 for the corner-image control. The paired center-minus-proprioception interval was −0.0349 to +0.0756. On the original layouts, 337 preterminal windows followed five close commands. Physical labels identified 253 carrying, 3 empty and 81 ambiguous windows. The direct comparison deprioritizes a camera-based empty-grasp recovery on this shared-gripper benchmark. [Paired readouts and controls](../frozen-transfer-object-20261124-r1/RESULT.md)

## Choosing between yaw estimates

The two readers were refitted with eight entire Panda/Sawyer tasks held out. On 448 physical traces from those tasks, a selector that scored an omitted command against its visual motion reduced yaw error by 2.278° compared with one tuned Perceiver (paired 95% interval −4.144° to −0.600°). A fixed mixture chosen on the calibration tasks was 0.390° better than the selector (adaptive minus fixed interval −0.415° to +1.705°). Picking a single reader by the visual score selected the better yaw 47.9% of the time versus 64.5% for always picking geometry. These are source-arm angle measurements on seven familiar shifts. The fixed blend is worth testing against a second independently trained Perceiver; these results do not establish an advantage for per-trace selection. [Protocol and independent scoring evidence](../symmeta-omitted-chunk-20261124-r1/SCORE.md)

## Holding an unintended object

A follow-up used the released task definitions to distinguish allowed pickups from distractors. Across the 22 original-layout episodes, the monitor found 253 sustained windows carrying an allowed object and zero carrying a definite distractor. Of the three policy failures, one contained a sustained correct pickup and two had no eligible closure window. This leaves approach errors and intervening contacts unobserved. The archive supplies no negative examples for a loaded-object identity classifier. [Task roles, coverage and failure readback](../frozen-transfer-identity-20261124-r1/RESULT.md)

## An independent second Perceiver explains the target-arm yaw gain

With two equally weighted yaw estimates, the geometry–Perceiver ensemble reached 6.133° and the two-Perceiver ensemble reached 6.132° on 256 target OOD supports. The same single-reader reference scored 6.433°. The mixed-minus-homogeneous difference was +0.001° (95% paired task interval −0.719° to +0.785°). Weights learned on reserved source tasks moved both estimates to 6.643° and 6.132°, respectively. Across 448 source-test supports, the equal-weight mixed model did beat the homogeneous ensemble by 2.154° (−3.561° to −0.904°). Source-task accuracy favored geometry, but its added value failed to carry over beyond an ordinary two-reader ensemble on these target panels. Each method used two reads, with potentially different arithmetic cost and training exposure. The target panels had been inspected previously. [Matched two-reader test](../frozen-transfer-ensemble-20261124-r1/RESULT.md)

## Wrist feasibility and actual failure

Static kinematics admitted all 600 goals that Kinova3 itself requested on the two failed drawer runs. By contrast, 110 goals were provably infeasible at their requested orientations on the two moka-pot runs that originally succeeded. Allowing wrist tilt can reduce instantaneous position error under a joint-speed envelope. Acting on that calculation rescued neither drawer. On fixed-command replays the wrist intervention reduced Kinova3 successes from 3/6 to 1/6 and Panda successes from 6/6 to 0/6. One Kinova3 moka-pot replay also missed the historical native success under raw commands. The physical geometry is real; choosing a wrist change by positional feasibility alone damaged task execution. [Kinematic certificates, native replays and intervention controls](../frozen-transfer-pose-20261124-r1/RESULT.md)

## Reusing the previous action plan on new arms

The frozen π₀.₅ policy made a fresh prediction every five controls. Two variants averaged its current translations with the previous prediction. One advanced five positions by the clock. The other chose the position whose predicted displacement best matched how far the robot had moved. The initial support, task policy, random noise and control budgets matched across modes. Sixteen paired starts covered the eight official task identities on one new closed-loop layout of each arm. The simulated command interface was nominal.

| Native successes | Raw π₀.₅ | Tuned Perceiver | Tuned + fixed averaging | Tuned + measured progress |
| --- | ---: | ---: | ---: | ---: |
| IIWA | 8/8 | 7/8 | 6/8 | 6/8 |
| Kinova3 | 8/8 | 7/8 | 4/8 | 6/8 |
| Both | 16/16 | 14/16 | 10/16 | 12/16 |

Against tuned, progress alignment lost the drawer task on both arms and rescued zero starts. Against fixed, it gained three starts and lost one. The paired difference was −12.5 percentage points against tuned (95% task-bootstrap interval −37.5 to 0) and +12.5 against fixed (−12.5 to +37.5). The adaptive index differed from the clock on 371 of 711 replan boundaries. The raw policy solved every start on this panel. These comparisons establish that the intervention changed actual execution and that it fell short of the tuned baseline despite outperforming fixed averaging on this sample. More diverse starts would be needed to measure rescue of raw failures. [Paired native evidence](../frozen-transfer-progress-20261124-r1/NATIVE.md)
