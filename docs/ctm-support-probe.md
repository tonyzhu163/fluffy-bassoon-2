# Balanced native support and fresh-probe qualification

This bounded stage qualifies the native interface, history dataset and evaluation
protocol for one ManiSkill placement task with the existing frozen pi0.5 motor.
It does not automatically authorize a larger PPO run, new task family or new
architecture. Run sizes and allocations belong in the dated launch record.

## Explicit data coverage

`meta_rl.pi05_balanced_plan.make_balanced_history_manifest` constructs the
private simulator plans. The initial qualification has six training and six
validation trials: every cell of theta `{-45,0,45}` degrees × object `{0,1}`
appears once in **each** split. One six-lane batch per split avoids the accidental
coverage gaps of the earlier two-lane random pilot. Theta and object remain
fixed within a lane across four attempts; overlay, position and orientation
vary. Each trial starts with fresh learner memory.

Both history splits use overlays 0–11. Validation holds out complete history
seeds and scene identities, while theta/object support remains shared. It is
an in-distribution history holdout, not held-out-task or held-out-angle
generalization. The first three attempts provide support and the fourth is a
training-distribution probe. Training labels never use overlays 12–15.

Plans retain the existing `theta_degrees`, `scenes[attempt][lane]`, `reset_seeds`
and `split` interface. Additional private fields record object indices, complete
history seeds, trial identities, attempt roles and canonical hashes. The manifest
stores explicit `train_indices` and `validation_indices`, per-theta/object
coverage, complete trial sets and exact sample budgets. Do not infer a split
from list position or a modulo heuristic in downstream training. Validate these
fields before collecting histories or fitting either architecture. Canonical
content hashes differ from the byte hashes of pretty-printed JSON files; retain
both when writing artifacts.

Six trials per split × four attempts × 80 steps means 1,920 simulator steps per
split, 3,840 total, 768 policy chunks and 768 additional native label calls.
These counts exclude interface-oracle evaluation, fresh probes and offline
optimization. Record those costs separately. Raw training and validation metrics
must retain the theta/object cells rather than only an overall average.

## Current action and supervision interface

The [feature bridge](pi05-feature-bridge.md) reads detached native final-layer
KV and public action/reward/boundary/clock feedback. The native action expert and
decoder remain frozen. The [stochastic PPO policy](pi05-bridge-policy.md)
samples Gaussian residual flow velocities on seven active model channels,
with one recurrent transition per observation chunk and four native Euler
passes. The complete native model horizon is eight; the simulator executes
only the decoded five-step prefix.

DPT-style targets must therefore correct and supervise only the five executed
steps, preserve the model-action tail and keep all labels separate from the
executed behavior history. Sigma, bounded residual scale, loss support and
native source hashes remain explicit compatibility fields. A privileged
inverse-command labeler supplies task labels; it is neither a learned ICRL
teacher nor evidence that our learned bridge already controls the motor.
Qualify actual bridge steering headroom against paired zero-bridge and external
correction references before interpreting lower flow-matching loss.

## Held-out support and fresh probes

`make_heldout_probe_plan` returns the evaluator wrapper: a three-attempt
`support_plan`, flat-batch `probe_plans`, shared per-replicate `probe_seeds`, and
`opposite_indices`. Four lanes pair theta `[-30,30,-30,30]` with objects
`[0,0,1,1]`; opposite-history partners share their object and differ in theta
sign. These angles are held-out **interpolation** between training angles, not
extrapolation. Support scenes use overlays 0–11. Fresh fourth-attempt probes
use overlays 12–15, exclude every support episode identity, and share exact
scene/reset/native-noise plans across evaluation conditions. The bounded
qualification defaults to one probe replicate; it cannot establish a reliable
generalization effect from four lanes.

Clone the state after support attempt three, including recurrent tensors **and
pending previous action, terminal reward, boundary and clock feedback**. Each
probe consumes this feedback once at the ordinary fourth-attempt clock. Each
probe starts from the same copied support state; it must not learn from earlier
probe episodes. Within-probe observation processing remains enabled. This keeps
the trained K4 contract intact instead of adding an untrained fifth attempt.

Compare normal history, memory wiped, reward blind and valid opposite-condition
history. Wrong-history state must include the corresponding public feedback,
not only permuted recurrent tensors. A trained episode-reset/static adapter is
a stronger baseline than a same-checkpoint memory lesion. Visual transitions
can reveal actuator dynamics without reward, so reward-blind improvement is
possible and does not itself imply leakage.

Report per-attempt support success and independent probe success, not cumulative
“at least one success.” Keep paired binary outcomes indexed by condition,
theta, object, complete support-trial identity and probe replicate. Replicates
branching from one support history are clustered observations, not independent
training seeds. Four lanes and one learner seed qualify plumbing only; stronger
claims need repeated independent histories, training seeds and grouped uncertainty.

## Admission

On allocated Linux resources run `tests/test_pi05_balanced_plan.py` alongside
the bridge, distillation and fresh-probe suites. The planner itself initializes
no model, simulator or CUDA runtime. The bounded launch must measure one actual
six-lane batch before collecting the full planned histories, preserve every
finished phase under its hard cap, and save per-cell competence/fit/probe
results. Portable CTM/XL warm starts are candidates until their live behavior
passes the stated checks; passing this stage is not a downstream PPO benefit
or an architecture-superiority claim.
