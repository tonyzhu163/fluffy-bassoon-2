# Simulation pilot on held-out arm kinematics

Can a short interaction on a new arm improve the native task success of frozen OpenPI π0.5? Fit the encoder and action adapter on the Panda and Sawyer families. Reserve the UR5e family for evaluation. These are separate MuJoCo robot models on the same LIBERO-Pro2.0 tasks. Use the same two-finger gripper and Cartesian command convention on all three. The holdout applies to this experiment's trained components. The frozen checkpoint's training provenance is separate.

## Experiment

1. Qualify the three model identities, joint counts, image and gripper observations, physical response, and task-object placement on allocated Linux. Use a matching Panda control. Record any mount, camera, or start-pose changes necessary for the alternate robots.
2. Pretrain a reader to predict observed motion on a fresh episode from commands and earlier interaction with the same training arm. Jointly fit the reader and a small residual action adapter on source bodies with fixed π0.5 proposals. Keep all target commands, rewards and targets outside fitting, normalization and model selection.
3. Measure native success on fresh matched query starts. Compare unmodified π0.5, inferred calibration, cleared or mismatched support, and a separately supplied calibration. The calibration can correct the action interface only. It supplies no expert grasp plan. If the known correction fails, report the ceiling for that interface and diagnose the controller or perception before spending on a larger fit. If history fails to beat controls, report that outcome.

The official task test split stays sealed. Checkpoint selection uses source bodies only. The user reserved hardware for later. No hardware or physical-robot datasets are part of this pilot. Integration starts from main `af1dd6e0`, LIBERO-Pro2.0 `e5500c69`, and OpenPI `215abfb2`. Start the pilot with one GPU per Baige Distributed Training job within an initial cumulative allowance of 8 GPU-hours. Native resets and CPU qualification can run on the existing Baige development allocation.

## Ownership

The simulator worker owns arm construction and reset equivalence. The learner worker owns the two losses and inference. The runner worker owns collection and native comparisons. The coordinator owns qualification, deployment, training jobs, results and the aggregate budget.
