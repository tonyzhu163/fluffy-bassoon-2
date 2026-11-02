# OpenDrawer AD/DPT pilot on completed simulator-reward PPO

Human request, 2026-11-02: try AD and DPT on the existing OpenDrawer PPO data
and compare performance with the original PPO. This is an explicitly
simulator-reward development experiment. It is not the Flora method result;
the separately owned Flora PPO run remains unchanged.

Use the immutable S8-r2 history (SHA256
`7d66810302a46ddf95477ad108a64ca031d69b0784ae211980c32714e5cc4912`):
384 episodes, six behavior versions, 5,760 decisions, 2088 features and six
pre-tanh residual controls. Pin its final teacher checkpoint
`84b4a25f1f0e55a22f4fdf265766fa5f6295504db2cf9a0a3ba8a0a28cc44f4d`.
The original zero-residual checkpoint is
`0bf0a89a34ae957c7d4802f2d103e43e9ef56a0391af03ff7720a79019f2aab7`.

Group each of 64 collection slots across the six PPO versions. Every stream
contains six complete episodes (different scenes, not same-scene retries).
Select eight validation slots with NumPy PCG64 seed 90211 before windows;
56 streams/336 episodes train, eight streams/48 episodes validate. Validation
states were used by the original source PPO: this measures held-out student
imitation, not independent teacher training. New physical evaluation cases
are separate from every source episode and previous DEV/ID/OOD panel.

AD labels are actual recorded u. DPT-inspired labels are the pinned final
PPO actor's deterministic means on those exact saved features, using its
original normalization. Every non-target context array stays identical.
Mean labels match the original PPO's deterministic evaluation convention;
they are candidate-teacher labels, not independently qualified optimal actions.

Train AD/full, AD/static, DPT/full and DPT/static with the existing plain
causal decoder learner: seed0, width128, two layers, four heads, context128,
window64, batch8, AdamW LR3e-4, 500 updates, Gaussian NLL, final endpoint
selection, validation every100 updates/four fixed batches. Normalization is
TRAIN-only. Same initialization, sampled windows and optimizer schedule.
Each fit has a 900-second operational cap; a partial endpoint cannot enter
evaluation. No architecture or hyperparameter sweep.

Primary readout: 32 prospectively fixed new native constructor seeds
880110000..880110031, same native target split, paired H50 motor innovations
and full H750 episodes. Compare original PPO initializer, final PPO, and all
four students, deterministic means throughout. Each case starts empty student
memory. Report success counts, paired wins/losses and exact paired sign tests;
one seed and 32 cases are preliminary.

Secondary mechanism check: first eight cases, each full student gets one
complete support episode on the same reset, then a separate query with the
primary query's exact motor draws. Actual simulator terminal support reward
is installed before the query. Source histories do not train same-scene
retries. The matched context-reset query equals the already measured primary
K0 query: same weights, empty memory, episode-start token, reset and noise;
there is no episode-index input. Reuse those results, retaining support cost
for both comparisons. K1 success is separate from support success. No weight
updates, rewarder substitution, evaluation-driven checkpoint selection or
automatic scale-up.

Resources: one Baige job, two allocated GPUs, at most three wall hours
(six GPU-hours), at most 200,000 all-in new actions, 32GiB new outputs. There
are at most224 physical episodes; worst-case224*(750+128)=196672 controls.
CPU qualification and export use allocated Linux CPUs with CUDA hidden.
Run the actual GPU probe inside the new allocation, qualify zero residual on
the first retained initializer episode, and preserve journals/records/clips
and full optimizer/sampler/RNG checkpoints. Stop at bounds; no hidden retries.
