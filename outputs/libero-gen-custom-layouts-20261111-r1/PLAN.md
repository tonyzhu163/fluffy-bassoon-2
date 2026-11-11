# Custom-layout physical and information admission

This stage follows the completed frozen BPP motor qualification. It has a
hard additional cap of **4 GPU-hours**, with all startup/failure allocations
counted. Prior task-wide conservative use was approximately17.50GPU-hours.
No reader training is authorized before admission passes. Use latest main,
immutable deployments, a fresh identity per allocation and native rewards.

## Scientific contract

Two hidden rules are LEFTMOST and UPPERMOST in the fixed agent-view image.
Nearest-to-plate was rejected because it always selected the same fixed
motor skill, making the image unnecessary once the binary rule was known.
The motor candidates remain the two qualified, frozen demonstration prompts:
center bowl and near-plate bowl, each placing its chosen bowl on the plate.
The intended physical bowl ID for each candidate is assigned before rollout.
ID swapping cannot turn a wrong candidate execution into success.

The first allocation is geometry only: no BPP checkpoint or learned policy
is loaded. Four fixed near-plate bowl/plate placements surround a fixed
center-bowl location. Four independently seeded jittered variants per
quadrant give16 layouts. Seeds3100–3115, position jitter±4mm; alternating
physical bowl IDs within every quadrant. Camera, robot and fixtures stay
fixed. The cookies and ramekin move to declared fixed safe locations.
All coordinates are preregistered in `geometry.py:templates` before any
simulator/model execution; there is no rejection sampling or adaptive search.

Geometry admission requires: at least6pixels separation on each projected
image axis; both centers inside an8pixel image border; no initial contact
between movable objects or fixtures; at most5mm xy settling drift after
20 fixed, stationary, open-gripper controls; neither native bowl-on-plate
goal initially true; independent sidecars match official native predicate
calls on primary states with no primary-physics changes. Counterfactual
positive placement fixtures are geometry diagnostics, never motor successes.
Inspect the rendered contact sheet to confirm projection/visibility.
Each rule must choose each candidate4/8 times within each stratum; rewarded
physical bowl IDs must be independently balanced4/8. Both strata have8 cases.
A geometry failure is preserved and diagnosed before a revised identity.

Only after geometric and visual admission passes do we freeze the16-state
manifest. Then run a small fixed-prompt smoke and the32-episode motor panel
(two candidate commands from every layout). No tuning on panel outcomes.
Require≥7/8 selected-skill native successes for each candidate in each
stratum and distinct intended executions on≥7/8 paired layouts in each
stratum. Repeated smoke cases are marked and never counted as fresh evidence.

## Outcome and recording contract

Evaluate both native bowl-on-plate predicates through LIBERO's official
`_eval_predicate` on the realized primary state, with geometry-stage sidecar
crosschecks. This is a native predicate call, never a handwritten distance
reward. Only the preassigned selected-skill predicate may control motor
stopping; the persistent hidden rule has no access to actions or termination.
Record the same562-control horizon (including12 initial gripper controls),
padding after selected-skill termination with a fixed convention. Public
records contain RGB/wrist/proprioception, executed actions, a neutral candidate
index and a padding mask. Exclude task/BDDL/goal/physical-ID metadata. Hidden
rule variants reuse the identical public record and differ only in outcome.
Audit those bytes under true/masked/shuffled outcomes; independently scheduled
histories are the primary feedback diagnostic. No reader experiment is
claimed by this physical/information qualification.
