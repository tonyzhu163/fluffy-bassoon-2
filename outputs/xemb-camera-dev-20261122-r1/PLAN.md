# UR5e camera comparison on LIBERO development

Does released S15 increase the frozen active EquivFlow reader’s absolute yaw error relative to source-calibrated DIS compared with BASE? An unchanged or smaller paired EquivFlow-minus-DIS error gap on S15 counts against that hypothesis. Compare terminal native successes alongside yaw errors. A low S15 privileged-oracle success count limits interpretation of live failures because the motor then struggles even with the true transform. Four queries per camera, from two layouts on one task, are descriptive.

Use frozen π₀.₅, UR5e and development `libero_goal` task 1. Cross support/query layout pairs 11/13 and 12/14 with `cont_45_nom` and `cont_m45_075` under BASE and official S15. Support seeds are 31810100 plus support layout and query seeds are 41810100 plus query layout. Collect the existing 15-control probe fresh for each of eight cells. Load the Stage 1 active `equiv_flow_homog_nodz` seed-41 checkpoint. Fit the Stage 2 active nominal gain and DIS projection once on nominal Panda/Sawyer training tapes. Freeze both readers before target collection. Use their predictions, nominal identity and privileged truth on four matched query episodes per cell, for 32 terminal outcomes.

Discard one initial native create to warm the simulator. Before scoring, read each native plan and verify the released row, scene and physics agree across cameras while both rendered views change. Require disjoint support/query fingerprints. A zero-control reference inference fixes each query’s reset, noise bank and opening proposal. Verify these again before each rollout’s first action. Store support tapes, issued commands, raw predictions, absolute circular yaw errors, reset and opening hashes, native rewards and terminal outcomes in the attempt directory. Each completed episode updates its case JSON atomically. The complete summary gives counts, paired wins/losses and mean absolute circular yaw errors by camera. A failure or timeout leaves partial evidence. A complete panel requires 32 terminal outcomes, summary.json and exit zero.

The parent validates and commits the scoped files, then deploys a fresh detached checkout at that full SHA. Submit one Pro6K on Baige Distributed Training for 25 minutes (at most 0.417 GPU-hours). Use the validated SHA as an external job argument and a fresh attempt name. From that checkout, run

```bash
bash outputs/xemb-camera-dev-20261122-r1/run.sh "$VALIDATED_COMMIT" gpu01
```

Evidence is written under shared storage at `/mowenjie/fluffy-bassoon-2/outputs/xemb-camera-dev-20261122-r1/gpu01/`. The launcher checks the project and release pins, probes one GPU and uses the qualified OpenPI interpreter, cv2 overlay and NVIDIA EGL runtime. The same project checkout supplies the evaluator, readers, simulator worker, ArmEnv and motor.
