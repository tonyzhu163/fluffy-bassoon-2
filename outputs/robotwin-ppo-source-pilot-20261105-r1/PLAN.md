# RoboTwin source PPO pilot

2026-11-05, Asia/Shanghai. User requests trying PPO after restoring the
RLinf/VLRLA benchmark arm. Start with `place_phone_stand`, a VLRLA seen task
with shipped RLinf TRAIN/evaluation seed banks. No held-out-task training.

Use the common Motus pi0.5 checkpoint
`motus-robotics/pi0.5_robotwin2@effc6e1a99bd4d0edf9447e8ba313def67649cfe`,
including its own normalization and ALOHA delta-joint transforms. This is a
declared initializer reconstruction, not the unidentified VLRLA author motor.
It was fine-tuned on 50 RoboTwin tasks; held-out claims concern history training.
Restore the verified cold cache into a fresh project cache; preserve its archive.

First allocate one GPU for at most 20 minutes: allocated GPU probe, existing
core native-PPO/robot-environment tests, then a real native H50/400-control
episode on `place_phone_stand`. Bind inference/rendering, verify three cameras,
native action parity and unchanged motor weights. Save exact seed/action
counts and qualification output. No SGD in this first qualification job.

After this passes, run RLinf action-expert Flow-SDE PPO on the same task,
with the VLM frozen and two complete iterations before expanding. Preserve
the VLRLA collector's H50, five denoising steps, noise .3, 256 environments x
four rounds, 400 controls, five optimization epochs, minibatch 2048 and
microbatch32. Use eight ranks if capacity allows; four ranks may increase
accumulation without changing the global batch. Compare paired independent
128-case native before/after panels and retain full iteration-boundary recovery
(Adam/scheduler, worker samplers and RNG), actual chronological histories and
all failed-attempt costs. Two updates qualify execution; they do not establish
convergence or source expertise.

Runtime: reuse clean historical RLinf `c605b9d55ed93a180119b099e6f836abff293b08`
and RoboTwin `dca9ec682688821c42944eacd7f5dd9bfb15396f`, with the retained
`robotwin-vlrla-20261029-r1` overlay. Current upstream was audited in the
[re-entry plan](../robotwin-rl-baselines-20261105-r1/PLAN.md). The inherited
runtime's compatibility is checked on the actual device; no active checkout
is modified. Bind the current reviewed project-main commit per allocation.

Initial qualification plus two-update pilot budget: at most 4 GPU-hours
including setup/failures. Use separate attempt/output identities. If measured
runtime cannot fit, preserve a complete checkpoint and record a new contract
before continuing within the repository-authorized cumulative 50 GPU-hours.
Do not stop another owner's work or change an active run's contract.

Before submission: verify queue capacity/limits/actual type and storage
capacity/quota/write access. Monitor every 20 minutes after launch. Report
job ID, progress, last check, next action and cumulative allocated GPU time.

Current state: preparing model restoration and the first native qualification;
no new allocation has been launched yet.
