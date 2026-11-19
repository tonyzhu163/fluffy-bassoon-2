# Offline readout

Run `python analyze.py OUT > OUT/analysis.json` with the directory containing `fit/`, `eval/`, `weights/` and `frozen.json`. The scorer reads native rewards from all 72 fitting and 192 evaluation episodes. It checks pose and seed assignments, distinct physical starts, shared reset inputs, the ten public pulses and the fitting-selected grip. Within each physical case, differences in post-probe motor inputs, proposals or public states appear in `post_probe_differences`. Each branch keeps its measured return.

For seeds 41, 42 and 43, `cases` identifies each donor, the selected physical branch, and its native result and control count. The supplied-factor policy enables recovery on D17. The current-only arm uses its own live query. An absent closure decision has a null grip and contributes to every success and cost denominator. Grip accuracy uses the number of actual predictions.

`by_seed` reports the held-out reversed-X/D17 cell and the equally weighted four-cell panel. Paired wins count factored-retained successes on queries where the comparator failed. Losses count the reverse. The two swap rows also count changed sign and grip choices, so a donor that leaves a choice or native outcome intact remains visible. Support costs and union successes use the recipient’s actual practice, including for donor controls. The standalone analytic probe counts the first ten support controls and their rewards. Standalone current-only uses zero prior controls. Both include all query controls.

**Limits.** Each fitting cell has eight assignments. Positive wire and BASE each occur in 16 of 24 histories, and reversed-X/D17 is absent. The fixed grip reflects those frequencies. Evaluation reuses eight fresh pose pairs across 32 assignments and all three fits. A held-out tie leaves the prior-contact advantage unresolved. The panel covers one task and robot morphology.

**Static validation.** Mac arm64 in-memory compilation passed. Source checks matched the scorer’s cells, seeds, reader signature and episode fields against the driver, readers and runtime. Linux execution and numerical validation belong to the orchestrator. GPU use was zero.
