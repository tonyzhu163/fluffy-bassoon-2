# The existing practice has usable information about the command mapping

**A 16-command practice sequence on Panda or Sawyer carries information the original reader largely missed.** A physical estimator that allows the wrist to respond over four successive control steps placed the forward rotation closer to the true one of two angles in 27 of 32 cases. Matching practice improved prediction on an independently reset query relative to command-matched practice from the other setting in all eight layout blocks, averaged over arms and settings. However, the average absolute angular error was 30.3°, and the relative benefit over one fixed common correction was inconclusive overall. This is evidence for distinguishing the tested interface conditions on familiar arms. [Per-case results](evidence/cpu01/estimate/report.json) · [Plan](PLAN.md)

## What changed when the estimator used motion timing?

The physical estimator learns a shared temporal response from 24 nominal source support sequences. It estimates one rotation and gain per new support sequence. The supervised readout uses all 192 labeled source tapes to fit a separate prediction of the complex rotation and gain. Both use the same 32 evaluation supports. Both are blind to arm and condition identifiers during inference. All eight layout indices are shared across the two arms and two settings. Only scoring uses the two possible evaluation angles. All reported estimates came from a verified Baige Linux CPU run at `b1eb9dd7`, with zero new simulator steps or GPU-hours. [Physical estimator](METHOD.md) · [Supervised readout](FACTOR.md) · [Execution](EXECUTION.md)

| Estimator | Nearest of the two angles | Mean absolute yaw error | Mean absolute gain error |
| --- | ---: | ---: | ---: |
| Physical response using four time steps | 27/32 | 30.3° | 0.448 |
| Physical response using the current step only | 20/32 | 47.2° | 0.598 |
| Factor-supervised readout using four time steps | 20/32 | 32.4° | 0.479 |
| Factor-supervised readout using the current step only | 21/32 | 37.6° | 0.439 |
| Constant source mean used by the supervised readout | 16/32 | 67.5° | 0.412 |

Lower continuous error and higher nearest-angle accuracy are different outcomes. The two target gains were 1 and 0.65, a gap of 0.35. An absolute gain error of 0.448 leaves substantial uncertainty about correction magnitude. Labels, architecture, temporal aggregation and estimation assumptions differ from H1. These results do not isolate a single cause for its failure. [Supervised per-case estimates](evidence/cpu01/factor/report.json) · [H1 results](../embodiment-command-frame-20261120-r1/RESULT.md)

## Does the estimate matter on a fresh query?

| Four-step physical estimator | Matching practice MSE | Swapped practice MSE | Fixed common MSE | Mean absolute yaw error |
| --- | ---: | ---: | ---: | ---: |
| Panda | 0.7160 | 0.8924 | 0.8085 | 20.4° |
| Sawyer | 1.1856 | 1.3052 | 1.1810 | 40.3° |
| Equal weight across both arms | 0.9508 | 1.0988 | 0.9948 | 30.3° |

The matched-support prediction error was 13.5% lower than the swapped-support error. The paired swapped-minus-matched MSE was 0.1480, with a 95% bootstrap interval of [0.1217, 0.1793] over the eight shared layout indices. For the stronger fixed common baseline, the corresponding common-minus-matched difference was 0.0440, interval [−0.0606, 0.1407]. On Sawyer the fixed correction edged out the history-conditioned estimate. The metric is normalized XY wrist-motion MSE on the last 13 rows of each query. H1 scored a different output and time window, so absolute losses cannot be directly ranked across studies. [Paired queries and bootstrap](evidence/cpu01/estimate/report.json)

Across the sixteen distinct arm/layout donor pairs, swapping history changed the physical estimator's inferred rotation by a median 37.85° in absolute value. The ideal separation was 45°. H1's median inverse-direction change was 0.21°. Absolute separation can hide a reversal: the mean **signed** high-minus-low gap was +55.3° on Panda and +10.1° on Sawyer, with two wrong-direction Sawyer pairs. H1 computed its correction direction through a fitted action matrix; the present estimator constrains the map to a planar rotation and gain. These statistics diagnose sensitivity within each method. [Physical report](evidence/cpu01/estimate/report.json) · [H1 inverse readbacks](../embodiment-command-frame-20261120-r1/RESULT.md)

## Decision for the paper

Keep the existing stop on adding robot families. A reader can use evidence in these sequences, yet the two evaluated settings couple +45° with gain 1 and +90° with gain 0.65. The panel was examined previously, both arms were present in source data and the experiment issued no adapted task action. Neither accurate independent factor identification nor improved manipulation on another arm follows from this analysis.

The next fresh source-simulation comparison should **cross yaw and gain independently**. Evaluate all four combinations of +45°/+90° and gain 1/0.65 on each arm, with both probe schedules on matched fresh starts. Compare four consecutive commands in each of the +X, −X, +Y, −Y directions against a shuffled sequence of exactly the same commands. Hold duration, translation energy, gripper and starting pose fixed. Freeze the four-step analytic estimator before collection; compare it with the learned readout and the fixed common correction. Score signed yaw with gain held fixed, gain with yaw held fixed, and matching versus swapped practice on the same independent queries. This distinguishes lack of excitation from a calibration problem and tests whether temporal structure adds value at constant practice cost. This fresh panel is proposed, and no data from it enter the result above.

For a PaliGemma task-success claim, independently qualify the source motor and show that an oracle inverse through the same bounded action interface rescues paired failures. The recorded source checkpoint scored 34/50 on Panda and 20/50 on Sawyer against the 38/50 per-arm development gate. With headroom established, compare matching practice, swapped practice, raw motor and equal-budget analytic calibration on the identical query starts. A shift a simple calibration solves makes a useful control for a later task in which interaction must also reveal a contact property. [Motor evidence and decision rules](../embodiment-meta-factory-20261120-r1/DECISION.md) · [Validity review](REVIEW.md)
