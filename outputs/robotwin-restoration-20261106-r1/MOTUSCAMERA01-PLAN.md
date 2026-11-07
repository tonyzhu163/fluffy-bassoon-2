# Paired source-motor camera and horizon probe

The current pinned runtime uses RoboTwin camera near0.1m. Current RLinf upstream
adds0.01m for SAPIEN3.0.1 because gripper fingers are otherwise clipped from wrist
images. Separately, our Motus pilot allowed only192 requested targets. Before
more PPO, compare these factors under frozen joint Motus with no student model
and zero SGD. This is a diagnostic, not an effective PPO/GRPO claim.

Use the first8 precommitted source DEV cases from MOTUS-DEV-CASES.json, without
outcome filtering. Four panels: default near0.1/H192, fixed near0.01/H384,
fixed near0.01/H192, default near0.1/H384. Native chunk32,5 ODE steps, fixedB2,
same per-case/chunk noise addresses and frozen bytes in every panel. Every panel
gets new owned simulator processes, same reset identities and one shared motor.
Within a near setting the common short/long horizon command/public-observation
prefix must be exact. Across near settings initial state/scene/instruction must
match; image differences are measured, not rejected. Record actual camera
construction near parameters, full initial PNGs, every intent/physical transition,
rewards/success, terminal/timeout and requested/delivered/physics accounting.
Never edit external native source or closed training worktrees.

32 complete episodes, at most9216 requested targets, at most8GiB project BOS,
16GiB free local scratch;1 GPU/16CPU/128GiBhost/16GiB SHM. Reserve1800GPU-s
including startup and release; worker25min, scheduler27min, watchdog30min,
retries off. Worst cumulative217649GPU-s=60.4581h under approved100h cap.
Fresh queue/device/storage checks and CPU preflight required before admission.
Qualify hook idempotence/default preservation/reset isolation and actual native
config composition on Linux without constructing a model or simulator, then
check actual GPU/render binding, full episodes, frozen bytes, reserve and cleanup.
Independent cold artifact readback must precede a result or protocol migration.

Interpretation: camera changes may alter policy observations but need not improve
success; the longer horizon measures additional time, not PPO learning. Small
paired8-case counts are exploratory. Use these results plus the actual learning-
signal audit to choose the next named PPO control. Preserve group8 checkpoint1
and its original cost/optimizer state. Do not change several learning parameters
simultaneously and attribute success to one. No held-out task is trained.
