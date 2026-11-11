# Next stage: custom-layout qualification

This is a preparation contract, not a launched reader experiment. The
current allocation only qualifies the released two-skill motor. No result
on the custom relations, hidden-rule inference, or learned history reader
is implied by native motor success. Freeze the final motor receipts before
turning this contract into code or assigning a new collection budget.

## Physical and information contract

Retain the two black bowls, plate, distractors, robot, cameras, controller,
checkpoint and successful demonstration library from the native motor
qualification. Define two observable hypotheses: leftmost bowl in the
fixed agent-view image, and bowl nearest the plate by tabletop distance.
Privileged geometry may generate and audit scenes, but is never a reader
input. Camera projection and distance thresholds must be fixed before
query collection; reject visual ties, occlusion and collisions.

Build two layout strata: agreement, where both hypotheses select the same
bowl, and disagreement, where they select different bowls. The initial
demonstration comes from agreement. Attempts and fresh queries come from
disagreement. Randomize positions within each stratum and counterbalance
which physical bowl instance satisfies each relation. Do not assume the
released center/near-plate prompts retain the same object identity after
positions change: verify what each prompt actually grasps on every layout
family. In disagreement scenes both different candidate placements must
be executable. If their demonstrated behavior collapses to the same bowl,
that family fails qualification; it is not a usable two-choice task.

Hide task names, BDDL paths, native goal predicates, object-of-interest
fields, rule labels, target-bearing language, and privileged object IDs
from the reader. A fixed library can enumerate feasible candidate behaviors
without indicating which one is the hidden goal. Reader observations are
RGB/wrist RGB, proprioception and a neutral candidate index; only the
explicit true-feedback condition receives the hidden native outcome bit.

## Native rewards without a stopping leak

The released runner stops when its configured native goal succeeds.
Replacing that goal with the hidden rule would let hidden success change
trajectory length and late actions, even when outcome tokens are masked.
Therefore keep motor stopping dependent only on the selected candidate
skill and a fixed execution cap. Record every attempt to the same declared
horizon, padding after selected-skill termination with an explicit common
recording convention. Padding masks, timestamps, filenames and termination
metadata must not depend on the hidden reward.

Evaluate the hidden goal's native predicate separately on those same
realized states. A concrete implementation candidate is a non-rendering
sidecar for each of the two goal variants: identical physical BDDL, restore
the primary simulator state, forward physics, and call the official native
success evaluator. The BPP wrapper exposes `get_sim_state`, `set_state`,
`regenerate_obs_from_state` and `check_success`; qualify restoration and
predicate parity before relying on this route. It must reproduce the
primary environment's native reward for the matching goal, leave primary
physics untouched, and distinguish the alternate goal correctly. A
handwritten distance threshold is not a replacement for native reward.

## Admission and readout

Proposed next admission panel: 16 development layouts, eight agreement
and eight disagreement, with the relation-selected physical bowl identity
balanced within each stratum. Execute both unchanged candidate prompts
from every layout: 32 physical episodes. The five-scene native panel took
13m51s including cold imports and prompt caches; 32 episodes fit a provisional
one-GPU, sixty-minute cap with a fresh output identity. This is a next-stage
proposal, not an allocation under the current motor-only stage. Freeze the
layout seeds, relation margins, target mapping and gate before execution.

Keep all failures. Record candidate selected, bowl actually grasped,
selected-skill success, both native hidden-goal outcomes, resets, videos,
execution time and peak memory separately. Require at least seven successes
out of eight for each candidate in each stratum, plus distinct intended bowl
choices in at least seven of eight paired layouts in each stratum. Agreement
means the two hidden rules favor the same candidate; the motor action set
must still offer two executable choices. All 16 physical
resets must match exactly across candidate evaluations. Every sampled state
used for sidecar validation must give the same native success bit as the
primary environment for its matching goal; any mismatch blocks collection.
These are engineering admission thresholds, not precise reliability claims.
Custom layout selection may use development seeds only; test layouts are
locked afterward. No tuning on test failures. If the released prompts cannot
reliably express both alternatives after layout changes, stop and choose a
separately qualified motor or task family rather than relabeling outcomes.

Audit leakage by replaying a fixed candidate schedule independent of
hidden rewards and verifying that changing the hidden goal changes only
the reward channel. The primary true/masked/shuffled comparison reuses
identical scenes, actions, video and metadata. Shuffling preserves the
positive marginal and reports the number of labels actually changed.
Closed-loop reward-aware collection is a separate result, because its
later selected actions can encode previous feedback.

Only after this admission step should an offline-trained, frozen learned
reader select motor prompts. Keep whole histories in the predefined
train/validation/test scene namespaces, with both binary rules represented
in training and fresh layouts at test. This remains one-bit task
identification with execution noise. Multi-step evidence accumulation and
new-rule generalization require a later design with at least three
overlapping hypotheses and held-out rule compositions.
