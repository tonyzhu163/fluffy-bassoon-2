# Native-arm residual action model

**Ready for Kinova3 native reward evaluation.** Feed the unchanged frozen π0.5 seven-component proposal and the selected row-major correction from the matrix file into `motion.map_action`. The Jacobian method uses `matrix`, the planar comparison uses `planar_matrix`, and unmodified π0.5 issues its original proposal. Map every query action after unnormalization and normal policy processing. Calibration is already complete, independently for each arm/task, on layout 11. The native evaluator owns fresh query layouts and matched resets, frozen proposal noise and rewards.

* Native launcher input on Baige is `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-motion-models-20261123-r1/cpu01/matrices.json`, SHA-256 `20a9d6800e10404ee23a542c70a3ba1fb077daeccd56cf7fea93555c7add64cf`. The verified local copy is [matrices.json](evidence/matrices.json). Keys include `Kinova3/libero_goal/3`, `Kinova3/libero_10/8`, `Kinova3/libero_goal/6`. Each value identifies its own tape SHA, physics, task, seed, reset, collector commit and source fit SHA.
* The Linux CPU qualification and offline benchmark ran from isolated Git snapshot `761f4209f4838099e4f80fad2937ad1e0f0cfcc5`. Reproduce it on Baige with `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-motion-models-20261123-r1/source/run.sh /mowenjie/fluffy-bassoon-2/outputs/frozen-vla-motion-models-20261123-r1/cpu-repeat`. Supply a fresh output directory. Numerical checks verify identity, row ordering, saturation and preservation of wrist and gripper. The 15 target controls produce three displacement observations.
* The source linear law predicts measured displacement from current and previous normalized translation commands. A source-centered target residual updates the current-command effect with three new target observations. For deployment we invert the estimated steady-state translation Jacobian, penalizing deviations from issuing the frozen action by empirical residual power from source data. This assumes stable dynamics and trusted public metric pose during calibration. The preceding command is zero for the initial chunk. Source-only selection used both leave-one-source-arm-out directions, with different support/query tasks and no target inputs, labels or rewards.

| Held-out arm | Source, mm | Planar residual | 3D residual | Target-only ridge | Wrong-arm residual | 3D minus planar, approx. 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Kinova3 | 3.748 | 3.742 | 3.056 | 17.652 | 3.896 | −0.69 [−1.35, −0.02] |
| IIWA | 2.579 | 2.738 | 2.391 | 17.578 | 2.802 | −0.35 [−0.50, −0.19] |
| UR5e | 3.919 | 2.695 | 3.343 | 19.491 | 4.166 | +0.65 [+0.12, +1.18] |

Endpoint is the mean Euclidean displacement prediction error per five-control chunk when one task supplies calibration and another supplies the query. Eight unique tasks per arm supply 56 ordered task pairs and three early chunks per pair. The approximate t intervals use a delete-one-task jackknife over **both** roles. They do not account for source/model uncertainty or multiplicity. Wrong-arm support uses the same task and start from the two other robots; issued actions may differ across robots. The intervals comparing correct and wrong arms cross zero on IIWA and Kinova3. The larger 3D model loses to planar on UR5e. Sparse source arm coverage and correlated early policy motions limit extrapolation. These measurements support a targeted Kinova3 and IIWA task-success experiment. Equal or lower paired success than unchanged π0.5 limits claims for control, even if the forward fit improves.

[Input provenance and source-only tuning](evidence/source-fit.json) · [All paired predictions aggregated by task pair](evidence/paired-scores.json) · [Arm metrics and uncertainty](evidence/summary.json) · [Linux CPU output](evidence/cpu01.log)

## Source-trained neural residual and active calibration

A four-hidden-unit tanh network learned residual forward dynamics on source robots. The three-chunk target fit can update native translation sensitivity and sensitivity along four hidden coordinates. Ridge regularization shrinks the update toward the fitted source dynamics. The shared-source-only and linear-target-head variants isolate how much the neural predictor and its adaptive head contribute. The frozen VLA, input signals, target budget and comparisons match the linear experiment. Three optimization seeds were fixed in advance. Leave-one-source-arm-out fits trained on only one physical arm each. The pooled fits trained on two. A broad claim about architectures trained across a large range of robot dynamics exceeds this panel.

| Arm | Linear residual, mm | Neural fixed source | Neural with linear target head | Neural with adaptive hidden head | Adaptive neural minus linear, approximate 95% interval |
| --- | ---: | ---: | ---: | ---: | --- |
| Panda, validated after training on Sawyer | 4.285 | 4.954 | 4.388 | 4.584 | +0.300 [−0.039, +0.639] |
| Sawyer, validated after training on Panda | 5.168 | 5.649 | 5.166 | 5.375 | +0.206 [+0.026, +0.387] |
| Kinova3 | 3.056 | 4.052 | 3.289 | 3.328 | +0.272 [−0.311, +0.855] |
| IIWA | 2.391 | 2.760 | 2.462 | 2.485 | +0.095 [−0.257, +0.446] |
| UR5e | 3.343 | 4.228 | 3.405 | 3.470 | +0.127 [−0.172, +0.426] |

Each source arm supplies 24 tasks with four starts each. All nine source network fits converged. An initial Linux qualification used a 700-iteration cap that five fits reached. Repeating with a 5,000-iteration cap and a relative function tolerance of 10⁻⁸ preserved the observed ordering. Optimizer seeds, fitted weights and diagnostics are available in [the source models](evidence/neural-cpu02/models.json). Full intervals, per-seed errors and support spectra are in [the neural predictions summary](evidence/neural-cpu02/neural-summary.json).

**Command coverage and transfer.** The median stable rank of a three-chunk passive support is 1.01 on every native arm. The weakest direction has 0.7% to 1.5% of the strongest direction's singular value. Typical early policy commands provide little independent information about all nine Jacobian coefficients. The archive also contains an identical three-direction, 15-control active command sequence on all 216 nominal source and target conditions. Its stable rank is 1.68. Each active/passive pair had bitwise-identical initial public images and state. We used source-only tuning for the active-support model and retained the same passive target queries.

| Arm | Active 3D residual, mm | Active planar | Wrong-arm active 3D | Active minus passive 3D, approximate 95% interval |
| --- | ---: | ---: | ---: | --- |
| Kinova3 | 5.160 | 3.242 | 5.467 | +2.104 [+0.572, +3.636] |
| IIWA | 3.869 | 3.029 | 5.378 | +1.478 [+0.376, +2.581] |
| UR5e | 4.519 | 3.207 | 7.231 | +1.176 [+0.099, +2.252] |

The broader probe made the 3D fit worse at predicting passive frozen-policy motion in every arm. Its translation directions, time courses and gripper commands changed together. Their individual contributions remain unresolved. Matching the robot under identical issued probes improved motion prediction over using another arm on IIWA and UR5e; the Kinova3 interval included zero. Controlling commands leaves differences in arm configuration and contact. These exploratory results support taking the original passive linear and planar models into the paired native task trial. The forward-motion diagnostic alone cannot qualify their steady-state inverse during transient motion, clipping or contact.

Verified on Baige CPU from isolated Git snapshot `09ddd1f1179bd9e2be3c6523cc94db087dac5c7c`. From `/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-motion-models-20261123-r1/ablations-source`, run `./neural.sh /mowenjie/fluffy-bassoon-2/outputs/frozen-vla-motion-models-20261123-r1/neural-repeat` and `./active.sh /mowenjie/fluffy-bassoon-2/outputs/frozen-vla-motion-models-20261123-r1/active-repeat` in new output directories. The fixed-command readback additionally verified all eight UR5e task identities, tape hashes, support seeds and reset fingerprints against the independently locked historical panel. [Readback](evidence/active-cpu01/readback.json) · [Active-panel scores](evidence/active-cpu01/active-summary.json) · [Source-only selection on matched probe episodes](evidence/active-cpu01/active-source.json)
