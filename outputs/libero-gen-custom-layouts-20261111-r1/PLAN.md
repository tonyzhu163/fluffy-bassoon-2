# Custom-layout physical and information study

This stage follows the completed frozen BPP motor qualification. It has a
hard additional cap of **4 GPU-hours**, with all startup/failure allocations
counted. Prior task-wide conservative use was approximately 17.50 GPU-hours.
This stage measures the motor and recording interface; it does not train a reader. Use latest main,
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
quadrant give 16 layouts. Seeds 3100–3115, position jitter ±4 mm; alternating
physical bowl IDs within every quadrant. Camera, robot and fixtures stay
fixed. The cookies and ramekin move to declared fixed safe locations.
All coordinates are preregistered in `geometry.py:templates` before any
simulator/model execution; there is no rejection sampling or adaptive search.

The completed geometry preflight used: at least 6 pixels separation on each projected
image axis; both centers inside an 8 pixel image border; no initial contact
between movable objects or fixtures; at most 5 mm xy settling drift after
20 fixed, stationary, open-gripper controls; neither native bowl-on-plate
goal initially true; independent sidecars match official native predicate
calls on primary states with no primary-physics changes. Counterfactual
positive placement fixtures are geometry diagnostics, never motor successes.
Inspect the rendered contact sheet to confirm projection/visibility.
Each rule must choose each candidate 4/8 times within each stratum; rewarded
physical bowl IDs must be independently balanced 4/8. Both strata have 8 cases.
A geometry failure is preserved and diagnosed before a revised identity.
The exact pixel/drift cutoffs were development heuristics, not calibrated
reliability criteria. Preserve what was run; do not automatically impose
these numbers on future work. Visibility, initial collisions, stable
physics and native predicate correctness are the underlying concerns.

The geometry and visual review fixed the 16-state manifest before motor
execution. Report native outcomes and distinct intended executions for
each candidate and stratum, with no arbitrary minimum success fraction.
Repeated smoke cases are marked and never counted as fresh evidence.
The original proposed 32-episode panel is not an automatic next step:
additional rollouts must resolve a concrete uncertainty left by the smoke.

## Outcome and recording contract

Evaluate both native bowl-on-plate predicates through LIBERO's official
`_eval_predicate` on the realized primary state, with geometry-stage sidecar
crosschecks. This is a native predicate call, never a handwritten distance
reward. Only the preassigned selected-skill predicate may control motor
stopping; the persistent hidden rule has no access to actions or termination.
Record the same 562-control horizon (including 12 initial gripper controls),
padding after selected-skill termination with a fixed convention. Public
records contain RGB/wrist/proprioception, executed actions, a neutral candidate
index and a padding mask. Exclude task/BDDL/goal/physical-ID metadata. Hidden
rule variants reuse the identical public record and differ only in outcome.
Audit those bytes under true/masked/shuffled outcomes; independently scheduled
histories are the primary feedback diagnostic. No reader experiment is
claimed by this physical/information qualification.

## Geometry02 correction, frozen before allocation

Geometry01 (`03261779`, job `job-a3laid6l40hi`) completed layouts 0–9,
then exhausted robosuite's Python episode counter because restoring MuJoCo
state does not reset that counter. Native sidecar parity and both positive
fixtures passed in all 10 completed cases. Layout 2 had bowl/plate contact;
layouts 8–9 had 57–59 mm drift, and layout 9 also had bowl/cabinet contact and
insufficient image-axis separation. The original evidence is retained.

Geometry02 resets and seeds the native environment before each restored
case/diagnostic. Its fixed hold command follows BPP's -90° controller-frame
rotation; recorded EEF drift must be ≤2 cm and ≤10° after 20 controls. Original
RGB is retained, with projected 0/1 markers in separate diagnostic images.
The seeds, jitter, balance and native checks remain unchanged.

The corrected center is (-.09,.115). Near-bowl templates are
(.005,.255),(-.185,.255),(.005,-.025),(-.21,.04); plate templates are
(.17,.255),(-.28,.375),(.17,-.025),(-.35,.105). Cookies move to (.17,.115);
ramekin remains (-.34,.24). This geometry-only correction moves the lower
templates away from the fixed cabinet and separates near bowl/plate rims.
Static inspection explicitly accounts for flat_stove.xml's burner offset
of +.15 m along local x and .095 m box half-width: fixture-origin distances
alone understated its collision extent. Actual simulator admission remains
required for that allocation. These coordinates were frozen before any
BPP motor outcome on a custom layout was observed.

Geometry02 passed all 16 cases and independent visual review. The immutable
admission manifest is `admission.json`, bound to the result and 16 physics
state hashes. The motor smoke uses indices 0,5,8,13 with both fixed prompts
(8 episodes, including swapped IDs in each stratum); the originally proposed
full panel uses 0–15 (32 episodes). All runtime-valid motor failures remain
in the evidence. The repeated smoke cases are development
evidence, not additional independent trials.

The recorder observes raw controls below the official vector/chunk wrapper.
It finalizes each episode before the next initializer and the runner's extra
video-cleanup reset. Native selected-skill stopping matches LIBERO's original
raw success timing when IDs match; after ID swaps it uses the preassigned
intended bowl. Both native predicates are read-only diagnostics. Paired
distinct execution is counted using opposite one-hot `native_any` vectors;
moving both bowls onto the plate is reported separately because it cannot
distinguish the hidden rules. The prompt tensor hash must
remain identical across all layouts for each candidate. Geometry-compatible
128×128 cameras, no render resizing, and 562 controls are asserted explicitly.

## User-directed interpretation change after smoke01

On November 11 the user rejected unnecessary fixed pass rules and tests.
The 7/8 motor cutoff had no calibrated or statistical basis and is removed
from future analysis. Smoke01 remains immutable at `62246070`; its original
audit output is retained without rewriting the executed contract. Future
readouts report the same counts without a binary motor-admission verdict.

Smoke01 reported zero native successes in all eight executions. Inspect
the trajectories before deciding whether more of the near-identical
layouts would be informative. There is no automatic 32-episode launch.
The reason to investigate is the observed absence of successful alternatives
and informative outcomes, not falling below a chosen percentage.

No new test suite is needed for this change. Keep run-local integrity
checks for matched resets, actual native rewards, fixed motor weights,
correct recording/padding and hidden-rule-independent public inputs: those
protect the experiment's interpretation. Do not add checks solely for
bookkeeping or to enforce an unsupported performance threshold.
