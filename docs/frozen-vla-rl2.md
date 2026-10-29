# Direct RL² steering of a frozen VLA

The performance question is held-out robot success versus deployment episodes
and primitive simulator transitions. RL² is the direct on-policy comparator to
AD and DPT distillation; each method controls the same compact motor noise
coordinate. A small qualification run or finite PPO loss establishes an interface,
not useful adaptation or competitive robot performance.

`vla_rl2.py` reuses `GatedTransformerXLCore`, `RecurrentActorCritic`,
`update_ppo`, and atomic complete-trial learner checkpoints. The policy is generic
in the fixed public feature dimension and defaults to a 128-wide, two-layer
GTrXL. Its standard squashed Gaussian head emits 32 bounded steering coordinates.
The frozen motor adapter broadcasts that coordinate over its native horizon.
The collector rejects further action remapping because its recorded likelihood
must describe the actually executed bounded coordinate.

At query t, input is the current fixed public observation feature, previous
executed noise, previous actual source reward, previous episode-done flag, and
current episode-start indicator. The first query has zero previous feedback.
Task/scene identities and reset seeds are metadata. The same frozen feature
encoder is used during all source collection and held-out evaluation. Feature
normalization is fixed in saved buffers and is identity by default.

A scene trial contains K complete episodes from the identical initial state and
reset seed. Initial observation hashes must agree. Recurrent state and terminal
feedback persist across physical episode boundaries in `full` mode. Only the
last transition of the last episode ends the RL² return. GAE and the PPO loss
therefore include future episodes in the trial; backpropagation replays the whole
trial without detaching at episode boundaries or using TBPTT. Gamma and lambda
are explicit in the saved training contract, defaulting to one.

Each PPO batch is one complete, unpadded scene trial. This handles varying
success times without introducing padding into the shared PPO critic loss.
The default is four PPO epochs and one whole-trial minibatch. Measure simulator
and update time during qualification before selecting the interaction/update
budget. `episode` mode trains a separate matched comparator whose recurrent
state and previous feedback are cleared at every episode start; it retains the
same trial-return objective. Resetting a full-trained policy only at evaluation
is a distinct ablation.

`train_rl2` samples training initial-state indices and reset seeds, collects only
real environment feedback, verifies old-policy likelihood replay before every
update, and records primitive steps, benchmark settling actions, forward samples
and optimizer steps. Explicit held-out indices can be supplied and must be
disjoint. The budget reserves one worst-case complete trial before collecting;
early success is charged at its actual executed primitive count. A shared
feature/prior training cost belongs in `source_contract` and is reported separately
from new RL² source interactions and held-out deployment interactions.

Recovery includes actor, critic, optimizer, sampler and Python/NumPy/Torch/CUDA
RNG states at a whole-trial boundary. The fixed feature encoder and frozen motor
must be provided from the same pinned source/interface; the caller qualifies
allocation, simulator and motor execution. Exported candidates include all policy
weights and fixed normalization, but no trial state. `FrozenRL2Controller` exposes
the same `act`, `observe` and `reset_trial` protocol as AD/DPT deployment and
changes no weights. It uses the checkpointed global Torch RNG, seeded per paired
held-out scene by the evaluator.
