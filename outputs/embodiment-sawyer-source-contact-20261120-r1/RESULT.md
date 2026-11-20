# Where the Sawyer source episodes failed

**All 30 failed starts reached direct hand-to-plate contact. All 20 successful starts did too.** The plate was about 25 cm from the center of the goal site at first touch in both groups. From that touch to the terminal control, the median reduction in distance was **11.81 cm on failures** and **20.54 cm on successes**. Twenty-nine of the 30 failures moved the plate closer overall. On these starts the motor usually made contact and advanced the plate, then ran out of controls before satisfying the native goal. [50-case result](evidence/native01/replay/cases.json)

**Replay admission was 50/50, with zero rejected cases and zero exceptions.** Every episode reproduced its original opening physics, issued controls, per-control rewards, termination flags and length. The cohort is the Huber step-6,400 checkpoint on the 50 previously scored Sawyer development layouts in `large-panel-v3/dev`. Its original native outcomes were 20 successes and 30 failures. [Replay validation](evidence/native01/replay/report.json) · [Source report provenance](../embodiment-probe-order-20261120-r1/MOTOR-LIVE.md)

| Measurement | Failed starts (30) | Successful starts (20) |
| --- | ---: | ---: |
| Direct hand contact / no observed hand contact | 30 / 0 | 20 / 0 |
| Palm contacted the plate | 13 | 11 |
| Left finger contacted the plate | 30 | 20 |
| Right finger contacted the plate | 30 | 20 |
| Arm link contacted the plate | 1 | 1 |
| First hand-contact control | 49 [44.5, 58.75] | 47.5 [44.25, 51.25] |
| Controls with hand contact at a substep or endpoint | 246 [220, 251.75] | 64 [52.5, 89.25] |
| Time with an active hand contact in the solver, s | 11.46 [10.36, 11.74] | 2.59 [1.99, 3.91] |
| Plate-to-goal distance at first hand contact, cm | 25.05 [24.34, 25.89] | 25.13 [24.50, 25.87] |
| Best reduction in distance after first touch, cm | 15.53 [9.07, 17.94] | 20.54 [19.94, 21.34] |
| Net reduction at episode end, cm | 11.81 [6.60, 16.88] | 20.54 [19.94, 21.10] |
| Terminal plate-to-goal distance, cm | 13.03 [7.88, 17.64] | 4.22 [3.96, 4.99] |

Continuous entries are medians [25th, 75th percentile] across episodes. Group contact counts can overlap. No episode started in hand contact. Every episode had hand contact during integration and also at a control boundary. The control count includes contact at the final boundary, which can be the first solver sample of the next control. This adds one control on five starts and two controls on one start compared with counting solver-active controls alone. The duration in seconds counts only solver samples. Success ends an episode immediately. All 30 failures continued to control 300, giving them more time to accumulate contact. Raw duration therefore cannot establish whether a longer push would help. [Group statistics and individual measurements](evidence/native01/replay/cases.json)

## What counted as a touch

The replay followed the R2 recorded-action stride replay procedure. For the generated panel it called the pinned large-panel tape validator, generated-state physical reset, camera-free source session and native Sawyer step. The observer used R2's active-contact criterion, `efc_address >= 0`, with the opposing MuJoCo geoms classified by compiled body ancestry. Each case retains the geom IDs, owning bodies, collision masks and actual observed pairs. Multiple simultaneous contacts count once per group and control. An active contact indicates a native contact constraint. It does not measure a useful push, force magnitude or grasp quality. [Replay](replay.py) · [Contact observer](geometry.py)

The moving plate was `plate_1_main`, body 36, collision geoms 158–167. Palm geom 75 belonged to body 19, `gripper0_right_gripper`. The left finger and pad were geoms 77 and 78 on bodies 21 and 22. The right finger and pad were geoms 80 and 81 on bodies 23 and 24. The two arm-link episodes involved geom 73 on body 17, `robot0_right_l6`. Failed seed 740038 contacted plate geom 161, and successful seed 740039 contacted plate geom 162. Both episodes also had finger contact. [Failed case](evidence/native01/replay/validated/740038.json) · [Successful case](evidence/native01/replay/validated/740039.json)

Each 20 Hz control comprised 25 native 2 ms Euler physics steps. The observer read the contact pairs and plate-to-site distance from the position stage used for each integration step. It independently evaluated contacts at reset and at each returned control boundary using scratch MuJoCo data, leaving the live integration state intact. The trace records boundary plate XYZ, public end-effector XYZ, goal-site XYZ, all substep distances, and the substep or boundary at which each pair was active. First touch is the earliest observed active pair. Contact duration is active solver samples multiplied by 2 ms. The measurement has 2 ms resolution. The saved end-effector position is the public observation returned for that control. Its sensor timing can differ from the recomputed endpoint plate pose. All contact and progress measurements use native geom pairs and native plate/site positions. [Trace layout](geometry.py) · [Example trace](evidence/native01/replay/validated/740000.npz)

Progress uses horizontal distance from the plate root to native site 5, `main_table_stove_front_region`, on body 1. Positive net progress means that distance decreased between first touch and episode end. Best progress uses the smallest distance attained after first touch, including within-control physics samples. The native `On(plate_1, main_table_stove_front_region)` predicate supplied the terminal reward. A distance to the site center alone cannot determine the predicate for this rectangular region. [Native site and compiled model](evidence/native01/replay/validated/740000.json) · [Independent trace recount](summarize.py)

## What reproduced

The diagnostic bound each of seeds 740000–740049 to its original layout, donor-state file, scene, opening receipt and action tape. Every issued action matched the corresponding five-control chunk prefix and native step index. All 50 raw opening physics states and opening public wrist/jaw hashes matched exactly. The original same-arm limits were 10 µm and 100 µrad for position and rotation, and 0.1 mm/s and 1 mrad/s for velocity. Measured opening errors were zero in all four categories. All 50 openings also matched the archived collision-admission record, including the number of settled contact points. [Per-case validation](evidence/native01/replay/report.json)

All **11,497 native motion vectors** and **2,304 replanning proprioception hashes** matched exactly. Independently integrating the archived float32 motion vectors reproduced the public wrist position within 2.01 nm, the wrist orientation within 0.019 µrad and the jaw opening within 0.24 nm. Every native terminal reward and length matched. The native interpreter used MuJoCo 3.2.3 and robosuite 1.4.0 with the original evaluator at `fc61bf2a`. The original camera hashes remain bound to the source tapes. Replay omitted image rendering because the controls were already recorded. The source tapes have no independent per-step plate or contact channel. [Runtime, digests and trajectory checks](evidence/native01/replay/report.json) · [Execution](EXECUTION.md)

The registered approach-miss explanation had zero examples in this panel. The result directs the next diagnostic toward the push after first touch. Direction, interruptions and loss of progress remain plausible mechanisms. A matched continuation from the same first-contact state would distinguish which of them changes native success. The present comparison describes these 50 checkpoint-selected development starts. [Registered question](PLAN.md)
