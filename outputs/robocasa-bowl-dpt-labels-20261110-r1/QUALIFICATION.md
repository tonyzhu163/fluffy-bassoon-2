# Compact-bowl action labels qualified on seven of eight fresh scenes

Frozen Human300 π0.5 grasped on all eight layouts using a single neutral instruction. Each continuation replayed the same pickup controls and matched the saved physics, pixels from all three cameras and public state at the first grasp. On seven scenes a privileged scripted carry could release into either requested bowl. The eighth grasped at control 428, leaving 72 controls. Neither carry reached a released placement before the 500-control limit. The outcome is the custom one-fruit goal scored with RoboCasa's receptacle and gripper-far predicates.

| Scene | Controls to grasp | Left carry | Right carry |
| --- | ---: | --- | --- |
| 987100 | 200 | Left | Right |
| 987101 | 199 | Left | Right |
| 987102 | 258 | Left | Right |
| 987103 | 274 | Left | Right |
| 987104 | 254 | Left | Right |
| 987105 | 428 | Timeout | Timeout |
| 987106 | 194 | Left | Right |
| 987107 | 228 | Left | Right |

The per-goal oracle scored 14/16 across balanced hidden goals. Either fixed carry scored 7/16. Fourteen uniquely successful continuations supplied 351 overlapping action chunks. Each starts at an actual query observation and contains up to 50 delivered controls plus a validity mask. On the first scene, the action targets differed after a common initial lift. At the last shared replan, 30 controls after grasp, the two branches still had bit-identical RGB and state. [Coverage and physical outcomes](qualification-coverage.json) · [Geometry admission](geometry.json) · [Normalization and mask check](mask-check.json).

A ridge classifier trained on other scene pairs used public lateral motion, executed lateral commands and the terminal binary reward. A balanced permutation assigned the support probe independently of the hidden side. Each physical support was scored against both goals, and each query used a different scene. The classifier identified the goal on 15/16 held-out goal cases, 14/14 following a released support and 1/2 following the timeout. Masking reward yielded 8/16. The query-observation control also scored 8/16. As each query observation occurs under both goals, every deterministic observation-only classifier is limited to half on this balanced comparison. These counts reuse eight physical supports in four disjoint scene pairs. The classifier predicts a side; learned-policy release success is the next measurement. [Fold assignments and predictions](qualification-history.json).

Looking up the physically executed query branch selected by each held-out classifier gave 13/16 correct fresh-query placements with support and reward, 7/16 with reward masked, and 7/16 with the current observation. Feeding the counterfactual reward from the same support trace gave 1/16. The per-goal oracle reached 14/16. Two goals shared the query whose late grasp left too little time, and one query followed a timed-out support with an ambiguous reward. The actuator for all of these scores was the privileged scripted carry. [Choices and executed outcomes for every pair](qualification-routing.json).

The 4090 development allocation ran the eight pickups and sixteen replayed continuations in 1,463 seconds, or 0.406 GPU-hours, from the worker clock. It qualified with the repository's explicit-host GPU probe. Initial headless geometry screening needed the Ubuntu graphics dispatch libraries installed under this run's private graphics directory. No shared environment was modified. The collection used commit e6d6e3d0 and the audit used 704b9f6d. On Baige the raw traces, images, native coordinate labels and masks are under this run's qual-01 and read-qual-01/targets directories. [Protocol](PLAN.md) · [Mask and target export](export_targets.py).

The reusable constructed-query command reproduced both opposite winning targets from the same public query in the first scene, then matched all 351 native-coordinate targets and their validity masks on the 14 physical continuations. A unique release supplied the label without a miss measurement. The constructed-query, causal-decoder and flow-history suites passed 23 tests on Baige Linux at 7fd34720. [Paired command and numeric evidence](query-target-evidence.json).
