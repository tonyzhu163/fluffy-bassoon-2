# Repair the frozen motor by changing its demonstration

The user requested repair after the custom-layout study found 0/8 native
successes while a native replay through the same recorder succeeded 2/2.
The first repair changes demonstration data, not BPP weights. This run has
an initial 4 GPU-hour cap including failed allocations; prior task-wide
conservative use was approximately 17.75 GPU-hours. Use allocated Baige
resources, latest reviewed main and immutable deployments. No numerical
success cutoff or automatic training/panel escalation is imposed.

## What changes, and what stays fixed

Retain the released checkpoint, native reward predicates, controller,
camera preprocessing and the previous custom-layout states. The candidate
behaviors are the two bowl-to-plate choices. A demonstration is supplied
for each candidate independently of any hidden rule; this stage has no
history reader or hidden-rule inference. Privileged simulator positions may
help a demonstration teacher but never enter BPP at evaluation.

Source inspection found an earlier prose error: the checkpoint's actual
`cfg.task.shape_meta.action.rep` is `delta`, and the unchanged official
runner uses `control_delta=True`. Absolute control was used only for the
geometry preflight's stationary settling. The earlier motor/control results
remain valid; their raw code and receipts are unchanged. New prompt actions
must be executed delta commands, not absolute Cartesian targets.

## Obtain demonstrations on one development layout

Use cached native demo 0 for each candidate. Both source demos have states,
delta actions and image/proprioception records. Do not trust the near-plate
file's `abs_actions`: its pose is constant despite actual motion.

In one short simulator-only allocation, replay both source demonstrations
on their native states, then attempt both retargeted demonstrations on
custom layout 0. The native replays check whether the extracted controls
and state conventions reproduce the source before modifying them.

Construct source Cartesian targets with the installed controller and
recorded states/delta actions. Shift the grasp segment by the bowl's
source-to-layout displacement; transition toward the plate's displacement
during closed-gripper carry, preserving the demonstrated rotations and
gripper sequence. Execute through the same bounded delta controller and
record actual actions and real simulator observations. Record the exact
phase rule before executing it. This is a teacher attempt, not a claim that
retargeting is guaranteed to work. Retain failures and diagnose them before
changing the recipe or output identity.

The source dataset generation scripts differ on whether observations are
saved before or after actions. Compare the source fields with restored
pre/post states during the native replay and record the actual convention.
HDF5 images are stored upside down; the official dataset loader flips them.
Preserve that convention so the prompt and live policy inputs agree.

A trajectory becomes a candidate prompt only if its intended native goal
was actually reached. This is the meaning of a successful demonstration,
not a minimum percentage for admitting a model. Preserve all attempted
teacher trajectories and their outcomes separately.

## Compare original and repaired prompts

If both demonstrations are obtained, package one per candidate using the
official HDF5/dataset/prompt path. Keep the released normalizer and all
checkpoint tensors unchanged. Check the data conversion against an original
demo through the official loader; this protects the action/image interface.

Evaluate original versus repaired prompts on the same two scene states:
layout 0 (the demonstration scene) and layout 1 (a nearby scene, with swapped
physical IDs, not used to tune the teacher). Two candidates times two scenes
times two prompt conditions gives eight physical policy executions. Use
matched per-episode policy random seeds and the existing 562-control limit.
Report each native outcome, selected object and placement behavior. A gain
only on layout 0 is reproduction; layout 1 provides a small transfer check,
not a held-out benchmark or evidence of general rule learning.

Use a separate process and fresh one-demo asset root for each condition.
The original root contains an unchanged copy of each original `demo_0`;
its official-loader prompt tensors must hash to the previously measured
native/control prompts. Disable the checkpoint's training-split filter for
both one-demo roots (`is_seen=False`), because the repaired demonstration
was never in that split. This changes dataset selection, not the frozen
model. Per-episode policy seeds are `91001 + 10000*candidate + layout_index`;
the two conditions must have identical scene states, public observations,
selected physical objects and seeds. Simulator validity and input parity
checks do not impose a minimum success score.

No weights are trained in this first repair. Further data collection or
motor training must answer the concrete question left by these results,
with an explicit new recipe and remaining budget. Do not repeat a larger
panel just to satisfy a checklist or turn a score into a pass/fail verdict.

## Revision after teacher03

Both native source replays succeeded. Both retargets grasped and lifted the
intended bowl. Candidate 0 descended with a roughly 4 cm wrist tracking lag
before horizontal alignment; the bowl pushed and tilted the plate and never
reached its native goal. Candidate 1 reached its native goal first at action
index 333, remained successful for several controls after gripper reopening,
then crossed the native predicate's 3 cm horizontal boundary at index 356.
Its full 372-control trajectory is retained, including that later failure.
Post-action source observations fit the replay much better than pre-action
observations, so prompt packaging uses `post` alignment.

The teacher04 revision replays only candidate 0's successful 133-control
grasp/lift prefix, then holds its actual controller orientation, aligns the
carried bowl above the plate at the measured high lift, and descends with
bounded delta controls. Document finite phase budgets and servo parameters
before execution. No new native replay or candidate 1 simulation is needed.

Repaired prompts end at the first intended native success, matching the
policy's existing episode endpoint. This replaces the provisional demand
that the full open-loop teacher schedule remain successful at its final
frame; that extra persistence requirement was not the benchmark definition.
Candidate 1 uses teacher03's actual successful prefix (334 controls), and
candidate 0 will use teacher04 if its native goal is reached. Each packaging
receipt records the full trajectory's length/final reward and the selected
endpoint. This establishes native-goal execution, not stable released
placement beyond the scored endpoint. No model success-percentage gate is
introduced.
