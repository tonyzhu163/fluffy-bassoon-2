# Residual PPO KL stopping and diagnostics

User requested KL early stopping and more diagnostics, then explicitly selected
Baige for testing. This is core learner qualification, with no simulator
collection, policy deployment, or change to an active run.

The optional core setting is `target_kl`; old omitted/None settings retain the
fixed-epoch optimizer and historical checkpoint configuration shape. Fresh
RoboCasa training/common-before profiles default to 0.02; explicit settings
override it. Resumes must bind the saved setting exactly. Stop both actor and
critic at an epoch boundary when full-rollout sampled KL exceeds the target;
keep the completed epoch. No rollback or hard KL-bound claim.

Record initial, per-epoch and final full-rollout metrics, pre-clipping gradient
norms, actual gradient steps, sample passes and timings. Preserve the old
optimization-minibatch averages separately. The driver saves each epoch before
starting the next and streams it to the existing progress log.

Qualification covers the closed-form KL of a known Gaussian mean shift,
unequal final minibatches, actual early-stop counts, no-op/large-target numerical
equivalence including Adam and RNG, constant-return JSON handling, checkpoint
threshold binding, and existing learner/rollout/recovery regressions. Run on
the existing Baige development allocation with CUDA hidden and bounded CPU
threads; no GPU model computation or additional allocation is required.

Linux results and exact source/file identities will be saved here. Mac checks
are syntax/static only.
