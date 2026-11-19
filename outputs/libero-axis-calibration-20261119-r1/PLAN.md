# Can a fresh reader combine two command corrections?

Use the same frozen LIBERO π0.5 motor for every comparison. Train on the nominal command convention, reversed X, and reversed Y. Reserve the double reversal in all fitting examples. On a different reset of a disjoint development task, score the signs of the corrected first planar motor command. An existing single-convention controller can get at most one of the two signs right on the double reversal. A reader that estimates an independent sign for each axis can get both right.

## Inputs and split

Use the BASE tapes from [menu02](../libero-decision-pretraining-20261119-r1/RESULT.md), collected at `b90ee5a0c049d301529d0bf661d0a682b858ed78`. Eight fitting tasks and eight development tasks each supply two reset pairs. No official test tasks enter this diagnostic. Use the public commands and measured end-effector positions from the support reset and the frozen motor's first proposal on its paired query reset. Each convention relabels the support command coordinates and the required query coordinates. The executed physical motions stay the same under this exact change of coordinates.

From the first 80 recorded support controls select one X-dominant and one Y-dominant command by greatest absolute command on the respective axis. Use the recorded following displacement, including motions that disagree with the command. No reward, current task ID, calibration ID or query outcome enters the learner. Report missing or numerically zero components and the resulting coverage.

## Fit and comparisons

Use a fresh joint causal Transformer, two binary outputs, three fixed seeds (41–43) and 400 optimizer steps per fit on Baige CPU. Use both command signs correct as the primary endpoint. Report per seed and convention, plus individual axis accuracy and familiar-convention accuracy. Count native task/reset groups separately from coordinate relabelings.

Pair each query with its matching support, no support, and support expressed in the opposite convention. Include the analytic command–displacement sign estimator and the true-convention decoder. Check whether motor-command direction alone predicts the convention by also reflecting both support commands and measured displacements together, independently by axis. This changes probe direction while preserving calibration and the query target. Evaluate all four such directions. Make a paired fit with those reflections balanced in training to test whether the distribution of probe directions explains any shortcut. Apply reflections only to the supports. They are offline transformations, not additional observed trajectories.

An interpretable result requires reporting familiar-convention performance, the ability of measured motion to identify the correction, and the gap between correct and mismatched histories. Low familiar scores limit what a double-reversal miss says about composition. A gain that disappears on direction-balanced probes can be explained by support command directions. Even perfect command recovery in this test cannot establish native manipulation success or attribute the update rule to π0.5's original pretraining.

## Budget and decision

Use one CPU worker with at most two compute threads, 15 minutes wall time per attempt, zero GPU-hours. Save fixed-seed predictions and checkpoints with the run. Only a transferable learned correction warrants moving this hypothesis to fresh simulator execution. The comparison of pretrained initializations requires a separate matched motor experiment. The official benchmark panel stays sealed.
