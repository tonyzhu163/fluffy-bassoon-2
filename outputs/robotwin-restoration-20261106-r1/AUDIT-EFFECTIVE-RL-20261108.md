# Effective PPO/GRPO audit —8 November2026

No effective RoboTwin PPO or GRPO control has been established. A working
learner, exact checkpoint recovery and a completed collect/update cycle are
engineering evidence. They are not learning gains. The user requires an
effective source control before further in-context student work; those later
experiments now wait. The cumulative cap is100 GPU-hours, explicitly approved
in the reply to call_9991d141f4c9474cb3b968465bf752a5. Closed use is233110 GPU-seconds=64.753h after the failed continuation;
126890 remain. No GPU allocation is active; the likelihood audit is in CPU
preflight. Historical costs are retained.

Verified endpoints: earlier Sidney base/PPO/GRPO5/16,5/16,2/16; earlier recovered
Motus base/updated4/16,2/16; latest fresh Motus base/updated3/16,4/16. No positive
control is claimed. The last job completed only one fresh collection plus15
Adam steps per rank, then timed out during collection2.

The exact published experiment was not reproduced: task-specific adjust_bottle
SFT became joint Motus on place_phone_stand; native50-target chunks became32;
200-target wrapper became192; PPO groups changed1→8, reducing distinct reset
scenes per1024-episode collection to128. The pinned phone training bank has
1000 scenes; a group1 control must explicitly cycle24 repeats. Matched group8 is a valid
separate comparison, not the unchanged released PPO recipe. Held-out task roles
remain protected. The1000-epoch YAML ceiling is not evidence of the reported
checkpoint's actual training duration.

Actual first-cycle aggregate logs: approximate KL0.029, clip fraction0.159,
ratio1.008, policy loss−0.108, critic explained variance−0.086, value clip
ratio0.447, value loss0.056, logged pre-clip gradient norm11.353. These are
rounded native aggregates over the collection/five epochs, not final-checkpoint
critic measurements or per-update trends. Negative explained variance flags
poor value prediction but does not alone establish causality. Actor trajectory
receive time1388.1s versus training195.5s also identifies collection as the
main measured wall-time cost. The full raw-batch diagnostic below has now completed.

The independent raw-reward GRPO reduction passed on1024 actual policy0 episodes:
128 groups,73 tied/55 mixed-score,136 positive-score episodes, worst advantage
error1.141e−7 against1e−5. Tied groups give exact-zero advantages. This shows
nonzero available signal and substantial tied-group dilution, not a completed
GRPO training run. No PPO critic timeout bootstrap was used. See
GRPORAWGROUPAUDIT02-COMPLETE.json.

A further simulator mismatch needs an isolated check: current RLinf upstream
explicitly fixes SAPIEN3.0.1 wrist-camera clipping by changing near plane0.1m
to0.01m. The installed RoboTwin camera source uses0.1m, and the pinned RLinf
wrapper lacks that fix. Nonblank-camera checks do not detect missing gripper
fingers. This is a concrete upstream difference, not yet a demonstrated cause
of our low base success or an authorized silent migration of closed runs.
Primary sources: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html
and https://github.com/RLinf/RLinf/blob/main/rlinf/envs/sim/robotwin/robotwin_env.py
The original model's expected cameras and actual paired images must be checked
before choosing a fresh training protocol.

Next decisions follow evidence: independently quantify actual raw rewards,
PPO timeout bootstrap, values/targets and advantage separation; inspect horizon
and actual observation fidelity; then choose a fresh qualified continuation or
an explicitly named group1 source positive control. Preserve all old contracts
and costs. Require advancing fresh on-policy cycles and a larger paired fixed
source DEV panel; select on DEV only and confirm a claimed gain on disjoint
source evaluation cases. One16-case fluctuation is insufficient. Do not change
critic, noise, horizon, reward and grouping simultaneously and call it a fix.

The corrected actual-batch learning audit completed on all8 actors. Native
shared-channel transport maps actor ranks to env ranks0,6,5,3,4,7,1,2; all192
batches of commands and public observations matched the resolved donors exactly.
All8 independent GAE references passed. There are136 positive raw-score episodes
out of1024 and5958 valid decisions. Every positive-score episode decision has
positive normalized advantage (per-rank minima1.60–4.41). Added timeout bootstrap
is exactly zero on this collection; omitting it changes no advantage signs.
Initial critic explained variance against raw Monte Carlo returns is near zero
(range−.055 to.019). Thus the audit does not support a reward mismatch,
bootstrap-dominated signal or absent positive-advantage explanation here. It does
not establish final-checkpoint critic quality or a causal failure mechanism.
See MOTUSLEARNINGAUDIT02-COMPLETE.json and its full raw-result hash.

The camera/horizon probe CPU preflight passed at9624f831; actual GPU/render
qualification and paired frozen-policy outcomes are pending. Storage write/fsync
and cold readback passed; live queue reports60 free RPBZZZ6 GPUs,2774 CPU cores
and38210GB memory at01:24–01:26 Shanghai. The probe requests1GPU/16CPU/128GiB,
zero new worker PFS output and at most8GiB BOS. These are dated admission
observations, not persistent hardware guarantees.

Frozen camera/horizon probe and independent315-artifact readback completed.
On the same8 DEV cases, near0.1/H192 gives2/8, near0.01/H192 gives2/8,
near0.1/H384 gives3/8, near0.01/H384 gives2/8. Camera images change (wrist
mean absolute pixel difference about3/255). Same-camera common horizon prefixes
are exact, all32 physical episodes complete, and all owned workers close.
This small panel does not establish a camera fix or justify migrating the old
optimizer state. Finish the unchanged8rank/group8 checkpoint1 continuation to
complete cycle2. Then run the separately named group1 PPO control and matched
native group8 GRPO under an explicit remaining-budget plan. Keep student work
waiting. Camera565GPU-s closeout raises closed use to216414GPU-s=60.115h;
resume reserves21600GPU-s, worst238014GPU-s=66.115h under100h.

The unchanged continuation restored all24 workers, collected fresh policy1
trajectories on all8 actors and completed actor updates. It then failed in
our checkpoint-signature validator before checkpoint2 could be archived or DEV2
run. canonical_layout line36 rejects the optimizer-state keys of a zero-length
unowned FSDP parameter; the actual rejected keys must be measured before
changing the rule. This is a concrete checkpoint engineering failure, not a
PPO formula diagnosis. Checkpoint1 and the fresh policy1 collection remain
preserved; complete checkpoint2 is absent. See MOTUSRESUME01-CLOSEOUT.json.

The next read-only check compares unchanged checkpoint0 likelihoods against
the original behavior-policy probabilities at B32 andB256. Native chunk-level
reduction sums448 coordinates. Existing coordinate-MAE and zero-ratio checks
against a replaced actor reference are insufficient. Hold the matched Motus
GRPO launch until this is measured; no numerical root cause is yet established.
