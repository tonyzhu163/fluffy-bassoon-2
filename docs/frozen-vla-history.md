# Frozen VLA history-controller contract

The question is robot success and deployment sample efficiency on ordinary
benchmark tasks. A causal Transformer predicts a compact continuous noise
coordinate for a frozen motor. Distillation loss is an implementation diagnostic;
only held-out simulator outcomes establish performance. This controller does not
update parameters during deployment.

## Public experience and labels

A complete trial is a chronological sequence of complete episodes. Store these
arrays per trial, with no privileged state in the observation feature vector:

| Field | Shape and dtype | Meaning |
| --- | --- | --- |
| `observations` | `[T,F]`, float32 | Public features before the decision |
| `executed_noise` | `[T,A]`, float32 | Actual coordinate used by the frozen motor |
| `target_noise` | `[T,A]`, float32 | Supervision label, never policy input |
| `rewards` | `[T]`, float32 | Actual reward after executing this row |
| `dones` | `[T]`, bool | Episode ended, including time limit |
| `episode_starts` | `[T]`, bool | First row and each row immediately after done |
| `teacher_steps` | `[T]`, int64 | Nondecreasing source teacher checkpoint/update identity |

At query t the token contains only `observations[t]`, `executed_noise[t-1]`,
`rewards[t-1]`, `dones[t-1]`, and the current episode-start indicator. Previous
feedback is zero at a fresh trial. Neither the current reward, current target,
future outcomes, teacher identity nor task/scene seeds enter this token.
Collection metadata should record task/scene seeds, teacher checkpoint hashes,
teacher updates, frozen motor identity, feature/noise contracts, environment
transition cost, and source commit. A metadata field is not an input feature.

DPT uses separate teacher targets on actual imperfect behavior contexts.
Behavior cloning uses the executed coordinate as its target. AD additionally
requires a chronological learning trajectory with multiple teacher-step
identities within each trial. This identity stays constant while the behavior
weights stay fixed; increasing the collection clock alone does not count.
Fixed-teacher cloning is not labeled AD. This schema checks chronology and label equality, but whether the teacher actually
improves must be established from its recorded robot outcomes.

`save_dataset` writes concatenated ragged trials to NPZ and a hash-bound JSON
manifest. The trainer requires disjoint explicit `metadata.trial_ids` across
training and validation, and matching `motor_sha256`, `feature_contract`, and
`noise_contract`. Trial IDs alone do not establish scene or task independence;
the collection plan must define and audit the split.

## Controller, fitting and deployment

`HistoryNoisePolicy` reuses the repository's GTrXL relative-attention core with
functional state. Its diagonal Gaussian predicts unbounded motor-noise
coordinates. The motor adapter owns a fixed map such as clipping or broadcasting
to the motor horizon. For a clipped adapter the Gaussian NLL is a distillation
surrogate over recorded coordinates, not the exact clipped-action density.
Observation normalization is fitted on training observations and then fixed.
Training samples full trials or windows with random endpoints and context lengths;
right padding contributes no loss. Targets are never used to construct feedback.

Run `python -m fluffy_bassoon.meta_rl.vla_history_train --help` on the allocated
Linux host. Training requires pinned dataset hashes, a clean exact source commit,
and the explicitly identified allocation hostname. It exports candidate weights
and a full optimizer/sampler/RNG checkpoint. Select model architecture and budget
before looking at held-out robot success; validation NLL selects an imitation
candidate and cannot select a claimed successful robot method by itself.

`FrozenHistoryController` requires `act(public_observation)` followed by
`observe(actual_executed_noise, actual_reward, done)` for each real transition.
`reset_trial()` removes all previous task experience. Modes are:

- `full`: retain within-trial context across episode boundaries.
- `episode`: clear state and previous feedback at every episode start.
- `static`: clear state and all previous feedback at every query; retain only
  the current observation and current episode-start indicator.

A static model must also be trained from scratch on the same labeled data and
budget. Resetting a history-trained model at inference is a distinct ablation.
Use paired held-out task/scene/noise plans to compare native motor, ordinary
trained noise policy, independently trained static student, and full-history
student. Report success versus deployment environment steps/episodes, final
success, and the data/training cost used to produce each controller. Persistence
alone does not demonstrate feedback-based adaptation or better robot performance.

## Same-scene source adaptation

`vla_retry_teacher.collect_retry_teachers` branches independent full source
learners from one immutable shared DSRL checkpoint, including critic, replay,
optimizer and sampler/actor RNG. Every branch learns across K episodes on exactly
the same initial state and reset seed; initial observation hashes must match.
Training, supervised-validation and final-evaluation initial-state indices are
disjoint. Each source episode is saved with the behavior update identity before
collection, update identity afterward, actual executed noise/rewards and a
complete learning checkpoint. An explicit worst-case additional interaction
budget is checked before collection. The shared prior's training cost is counted
once and added to actual retry-teacher transitions.

AD labels each row with its original executed noise. DPT labels the same rows
with the branch's final scene-adapted teacher. Both retain the original past
noise and outcome fields. Every branch's observations must be encoded by the
same fixed shared-prior encoder; scene-adapted feature encoders would create
inconsistent input coordinates. Deployment uses that same shared-prior encoder.
This helper produces source data; held-out robot success versus K deployment
episodes remains the test of whether a frozen student acquired a useful process.
