# Compatible AD/DPT data from frozen-motor steering

Keep the entire motor fixed at theta0. SAC changes a separate noise policy;
the executed physical action is `a = motor_theta0(observation, z)`. Teacher
and eventual student therefore use the same mapping from noise to action.

| Later use | Required target and retained evidence |
| --- | --- |
| Algorithm distillation | Predict actual executed z from preceding ordered observation/noise/reward histories. Keep failures, warmup, intermediate policies and across-episode learning order. A chronological history is not an expert-only dataset. |
| DPT-style pretraining | Predict a strong final SAC noise teacher's action for a saved query observation, conditioned on interaction context. Generate these distinct labels later from the immutable teacher checkpoint/config/normalization/RNG. Competence must be measured before calling the teacher expert. |
| Action-space learning from native PPO | Retain physical actions and raw observations. PPO changes the motor action expert, so its sampling noise is not automatically an equivalent steering action for theta0. Keep this dataset's motor/version and role separate. |

Existing qualified collection retains raw current/next agent and wrist images,
proprioception and instruction; actual32D noise and its motor tiling; executed
physical action arrays and duration; primitive rewards and steering rewards;
success/termination/truncation; case/task/episode/step order and behavior-policy
update clocks. It also journals execution inputs before RPC and outcomes before
SGD, and saves learner/target/Adam/replay/sampler/Python/NumPy/JAX/motor RNG.
Actual duration and row order determine masks and time indices losslessly.
The new profile additionally binds the state-population cursor and query role.

This means no extra interaction is required just to create later same-motor
teacher labels. Optional VLA-prefix feature extraction can also use saved raw
observations later. Neither is a prerequisite for ordinary DSRL collection.
AD/DPT training should wait for a useful source learning process; an execution
receipt is not evidence that the histories contain an effective algorithm.

This task57 target-online pilot is separate from the all40 source corpus;
no90/Plus/Pro trajectory enters that pretraining bank.

Sources: [Algorithm Distillation](https://arxiv.org/abs/2210.14215),
[DPT](https://arxiv.org/abs/2306.14892),
[DSRL](https://arxiv.org/html/2506.15799v2), and the
[existing capture inspection](../dsrl-frozen-pi05-data-readiness-20261031-r1/READINESS.md).
