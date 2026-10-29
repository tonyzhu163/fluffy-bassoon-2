# Saved-trajectory PPO diagnostic, separate from the matched comparison

2026-10-30 Asia/Shanghai. Within the100seconds left of the original45GPU-minute
envelope, request **one GPU, maximum one scheduler minute**, automatic retries
off. No simulator, VLA weights or new robot actions. The Python process has a
45second alarm (including the allocation probe), an outer50second timeout
with5second kill grace, and refuses to begin updating without15seconds left.
Previous test allocations measured1m10s of scheduler runtime including probe,
Python/tests and teardown; their CUDA probe plus process overhead was about
51seconds when run separately. This diagnostic shares the probe/update process
and CUDA context and skips pytest, aiming to fit the remaining100seconds.
A timeout is an execution-cost result, not evidence against PPO or meta-RL.

Read-only CPU checkpoint inspection verified exactly three complete K4
persistent-PPO source trials: place_phone_stand44chunks/1333actions,
place_object_stand44/1340, pick_diverse_bottles52/1600. Total12episodes,
140noise decisions,4273previously charged actions. Each requested/completed
four episodes, has no budget truncation, and terminates its trial only on the
last recorded decision. All were sampled from the unchanged zero-update
policy. The real allocation will check saved likelihoods and values again.

Input is the immutable r4 ppo-recovery.pt, SHA
722618543005ffcac5591593f67f2f8eaa3bd9fcc1be5cebf66995e9487306a9.
The GTrXL core, PPO objective and optimizer are unchanged; the sole later
collection change in vla_noise_ppo.py is an optional metadata callback.
Review of latest origin/main found no new commits beyond the includedacca706.

This is an **explicit batch-three fork**, using the saved five-epoch objective,
minibatch16, AdamW/learning rate and all original settings. It is not a batch-
eight main-comparison update and cannot be resumed into that comparison. It
must produce finite metrics, nonzero gradients, changed weights, exactly one
PPO update/fiveoptimizer steps, and a full optimizer/sampler/RNG recovery that
retains the consumed raw PPO tensors. Never overwrite or consume the original
source checkpoint; its complete recovery is the fallback if an in-progress
private update must be discarded on timeout. Require a completed result and
verified checkpoint SHA before counting a retained update.

There is no robot performance evaluation and no SAC update in this diagnostic.
SAC's139replay rows remain below its declared500-decision warmup and256batch;
its recipe is not reduced to force learning. The prepared two-hour, matched
32k/64k/128k comparison still requires separate resource approval.
