# Sidney RLinf integration and PPO pilot

User selected Sidney and asked to continue toward PPO on2026-11-05, then explicitly confirmed broad multi-task SFT with no RL tuning as the intended clean checkpoint. Keep Sidney; do not switch to task-specific RLinf adjust_bottle SFT. Use
Sidney's published SFT scores as the baseline reference; no broad SFT rerun.
The earlier local screen is not admitted because reset instructions described
stale scene metadata. Prior scheduler usage:2561GPU-seconds(0.7114GPUh).

Fix reset language using the actual constructed task.get_info() and protect
changed-seed/unstable-retry behavior with core environment tests. Keep external
RLinf c605b9d and RoboTwin dca9ec6 checkouts immutable. Record actual object IDs
and effective instructions at reset. Never replace a selected bank seed silently.

Convert pinned Sidney e49e2ab6 with RLinf's released key/layout conversion,
overriding its tied-embedding assumption with Sidney's actual input embedding.
Retain saved mean/std,14 absolute joints and native Hugging Face tokenization
through a named checkpoint-specific model-boundary adapter. No Motus delta,
calibration or quantile transforms. Native H50/K10 first; compare saved pixels,
state and token IDs plus same-noise actions in native/RLinf. Check FP32 reference
arithmetic (decoded maxabs<=5e-4, normalized maxabs<=1e-3) and report BF16 drift
separately. These are integration checks, not SFT benchmark estimates.

One1GPU/30min bridge allocation runs the core reset tests, conversion, three
aligned source-observation replays and one RLinf episode. A full-FP32 comparison
is diagnostic only, invoked if mixed-BF16 outputs need numerical localization;
training retains Sidney's native mixed BF16 with full-FP32 vision and norms.
Then migrate explicitly to the
VLRLA K5/.3 Flow-SDE sampler for a two-iteration action-expert PPO pilot on the
source task place_phone_stand, simulator reward, frozen VLM. Reuse the native
RLinf learner/recovery wrappers; keep global batch2048, microbatch32, five
epochs,256envs x4rounds and H400. Verify rollout/actor replay before SGD,
measure capacity/time, retain complete optimizer/sampler/RNG boundaries, and
check advancing updates. This establishes execution, not convergence. No
held-out-task selection. Scheduler resource limits and actual GPU type are
checked before launch. Budget up to4 additional GPUh for bridge+pilot, within
the repository-authorized50GPUh task scope; record all attempts and migrations.

Execution migration: the eight-GPU group was not schedulable. Use four ranks
with microbatch32 and accumulation16, retaining global2048,256 environments
and five epochs. A50-minute four-GPU allocation replaces25 minutes on eight.
The model bridge is complete; train01 failed before model initialization on
Hydra inheritance. The retry uses the same recipe as a standalone primary
configuration. Completed scheduler charge through train01:3761GPU-seconds
including the earlier qualification;920 bridge plus280 failed-PPO seconds
are part of the current additional envelope. No SGD has run at submission.

Measured migration after the first live source chunks:256-scene reset costs
about7 minutes and a50-control vector chunk about40 seconds on four GPUs.
The50-minute allocation cannot reliably complete two full iterations. Preserve
the complete step0 native actor/Adam/sampler/RNG boundary, stop only this owned
job, and resume it in a fresh output with a3-hour maximum allocation. Keep
the same four ranks, batch, source seeds, sampler and learner. Charge the old
partial prefix in the same action journal under its original attempt; it is
not an extra PPO update or admitted complete source history. Increase the
self-imposed bridge+pilot envelope from4 to14 additional GPUh, including all
attempts; this is within AGENTS.md authorization up to50GPUh for the task.
No user hard cap changes. Preserve all checkpoint/cache inputs.

BF16 execution migration: actual64-environment rollout logprobs replay exactly
at batch64, but batch32/16/8 replays differ by up to0.04197 per scalar.
Keep BF16 and the original1e-3 replay guard. Qualify a full64-example backward
pass, then use actor micro64 and accumulation8 on four ranks; global2048,
five epochs, sampler, loss and optimizer schedule stay unchanged. The earlier
64-to32 continuation was stopped before any SGD. Retain its restored-state
receipt and charge its prefix; resume the complete original step0 boundary.
The next four-GPU allocation is170 minutes so all attempts stay within the
14 additionalGPUh envelope. No optimizer updates have occurred yet.
