# Sixteen planar commands identify a hidden rotation on both source arms

**The frozen analytic estimator recovered yaw on every probe in this panel.** All 128 estimates per schedule were within 22.5° of the imposed yaw. Gain remained harder on Panda. Both physical factors affected predictions on a fresh query when only the corresponding practice donor was swapped. The registered comparison found no advantage for holding each direction for four steps. [Paired results](evidence/native01/score/report.json) · [Decision rule](PLAN.md)

## Fresh paired experiment

On simulated Panda and Sawyer, we varied +45°/+90° yaw independently of translation gain 0.65/1.0. Both probe schedules issued the same sixteen planar movements at equal amplitude, duration and energy, with the same gripper command. A third tape measured an independent, mixed-command query for both schedules. The study covers 16 shared support/query layout pairs, 128 independent-of-support query tapes, 384 tapes total and 6,144 native controls. Analysis reused the old physical and learned coefficients without fitting on the new panel. All 384 tapes were validated; the Linux synthetic scoring test passed. Baige job `job-owxei8dm5dvv` ran on 4 CPUs with zero GPU cards. [Collection manifest](evidence/native01/collection/collection.json) · [Frozen scoring](SCORING.md) · [Run log](evidence/native01/run.log)

| Frozen estimator and practice order | Mean absolute yaw error | Mean absolute gain error | Matching query MSE | Yaw-only swap | Gain-only swap | Fixed common correction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Analytic, consecutive directions | 2.47° | 0.0916 | 0.8689 | 0.9873 | 0.9024 | 0.9825 |
| Analytic, shuffled | 2.34° | 0.1018 | 0.8660 | 0.9825 | 0.9001 | 0.9825 |
| Supervised ridge reader, consecutive directions | 12.28° | 0.2312 | 0.8691 | 0.9594 | 0.9073 | 0.9825 |
| Supervised ridge reader, shuffled | 21.29° | 0.2292 | 0.9177 | 0.9725 | 0.9380 | 0.9825 |

The analytic estimator's consecutive-minus-shuffled matched MSE was +0.00282, with a paired 95% bootstrap interval of [−0.00160, +0.00758]. The consecutive ordering also had a slightly higher point estimate on each arm. The registered preference for consecutive commands therefore failed. This interval quantifies uncertainty in the difference; we did not run an equivalence test. The ridge reader's own difference was −0.0486 [−0.1142, +0.0068]. Its mean yaw error was lower with consecutive commands, but its predictive ordering advantage also remains uncertain. [Per-layout differences and intervals](evidence/native01/score/report.json)

## Which part of practice was useful?

Yaw donors came from the same arm, layout, probe schedule and gain, and differed only in yaw. Gain donors likewise matched on yaw. With shuffled practice, the analytic estimator's matching-minus-control improvement expressed as control MSE less matching MSE was 0.1164 [0.0971, 0.1358] for the yaw swap, 0.0341 [0.0286, 0.0398] for the gain swap and 0.1164 [0.0548, 0.1873] for the fixed common correction. Both arms individually favored matching over each control. These are paired intervals over 16 layout indices. The 128 query cases share commands and layouts within each block. [Exact contrasts and resampling scheme](evidence/native01/score/report.json)

| Frozen analytic estimator, shuffled practice | Mean absolute yaw error | Mean absolute gain error | Within 0.175 of true gain | Matching / common query MSE |
| --- | ---: | ---: | ---: | ---: |
| Panda | 2.88° | 0.183 | 34/64 | 0.7176 / 0.8636 |
| Sawyer | 1.81° | 0.021 | 64/64 | 1.0145 / 1.1013 |

At fixed gain, the pooled estimated yaw change was 44.86° for a true change of 45°. At fixed yaw, the pooled estimated gain change was 0.396 for the true change of 0.35. The physical estimator approximates the within-arm command mapping. It does not assign a name to an unseen arm. The true simulated mapping through the shared temporal response yielded query MSE 0.8728. This diagnostic is allowed to know the imposed angle and gain, but it shares the approximate wrist dynamics model used by the readers. Its MSE need not bound a fitted effective correction. The earlier mixed-action experiment used different practice and query layouts. Comparing the earlier 30.3° error with this panel cannot isolate which difference accounts for the improvement. [Prior panel](../embodiment-tape-observability-20261120-r1/RESULT.md)

**Decision.** Retain a 16-step axis-only probe and the analytic estimator as a source-calibrated baseline. Each schedule worked; the primary evidence gives no reason to prefer consecutive commands. The learned reader has yet to add value over analytic calibration on these shifts. PaliGemma issued no commands in this panel. A native reward claim requires a source-qualified motor, an oracle controlling the same bounded action interface, and then a paired comparison of raw, matched, swapped and analytic control. The independent source workflow verifies candidate and scorer hashes before any further evaluation. [Source-motor audit](MOTOR.md) · [Mounted development reports](MOTOR-STATUS.md)
