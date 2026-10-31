# Frozen Motus steering evaluation

## Event teacher04 gate failure and independent saved-data diagnoses

The event-aligned capacity job `job-2rkimbvypowd` stopped before fitting.
Expected target: all12 teachers succeed, then fit the reactive KV/action heads.
Actual: aligned02/00/01/03 each succeed193 actions; aligned04 times out at its
natural400 cutoff after consuming its140 source frames and final holds. The
unchanged fixed-H32 teacher02 control also fails400. No cache, loss, optimizer
step or learned-controller result exists. All demonstrations remain retained.
[Closed result](../../outputs/frozen-pi05-endtoend-20261031-r1/EVENT-CAPACITY-TERMINAL-RESULT.md).

The allocated CPU diagnosis (`job-nnjau49cnk40`,15/15 tests,77 inclusive
CPU-worker-seconds) verifies all33 inspected chunks against their full submitted
plans, masks and captured native noise. Both02 resets match their recorded actor
poses and measured joints. Failed04 lifts the phone briefly after chunk4, then
loses elevation after chunk5, before opening starts in chunk6. At chunks4/5 its
measured fingers reach0 against the same closing drive for which successful02
retains a nonzero finger gap. This is consistent with a missed or unstable grasp
and loss during early transport. Contact forces, substep timing and source
measured poses are unavailable, so friction, cadence and grasp causality remain
unresolved. It does not support a diagnosis based solely on late release.
[Physical report](../../outputs/frozen-pi05-endtoend-20261031-r1/EVENT-DIAGNOSIS-CPU-RESULT.md).

A separate support audit (`job-zyg23qydopt3`,13/13 tests,73 CPU-worker-seconds)
finds that the unused action head's actual ±0.05rad arm corrections and zero
gripper corrections cannot exactly reproduce the four captured successful
TRAIN plans. Across896 submitted slots, the coordinatewise arm MSE floor is
0.00277498rad² and raw-gripper floor0.04808329; combined14-coordinate floor is
0.00924760. An explicitly hypothetical ±2rad arm interval removes the arm floor
but leaves the gripper floor. The large active-left discrepancy differs from
the much smaller inactive-right discrepancy. These are support bounds for
specific plan-imitation targets, not physical task impossibility or a trained
head's attainable loss. The expert replay bypasses this unused head, so the
bound does not explain04's replay failure.
[Support report](../../outputs/frozen-pi05-endtoend-20261031-r1/ACTION-SUPPORT-CPU-RESULT.md).

Independent suffix job `job-4352gzlcnq9f` is now terminalSucceeded/exit0:
all original05–11 succeed once,193 actions each,1351 new actions. The expected
7/7 diagnostic coverage target is met. Across the fixed identities,04 remains
the sole failure; this does not pass the all12 fitting gate or silently assemble
a mixed-run expert bank. No cache or fit ran. Frozen721a final signature
matches, full270181B recovery hashes to5c0829d3, and producer canonical totals
are21522 actual/charged with zero pending/uncertain. The owner checked all49 new
chunk metadata and7 manifests and independently summed1351 actions; no full
historical journal replay or checkpoint deserialize is claimed.
[Terminal receipt](../../outputs/frozen-pi05-endtoend-20261031-r1/EVENT-SUFFIX-TERMINAL-RECEIPT.json).

The suffix costs318 inclusive GPU-seconds, releasing1482 of its1800 reservation.
All targeted GPU attempts are closed at2652GPU-seconds/21522 actions. The
prospective wider action head (`de667c81`) separately passes40/40 synthetic
head tests on allocated zero-GPU2CPU/16GiB hardware,74 CPU-worker-seconds.
It declares2rad requested arm edits and1.1 raw gripper edits with source arm
headroom; the old0.05/zero-gripper head and runtime stay unchanged. This is
unit qualification, not VLA/physical qualification, target support or fitting.
No actual cache/fit/source admission follows.
[CPU receipt](../../outputs/frozen-pi05-endtoend-20261031-r1/WIDE-ACTION-CPU-RECEIPT.json).

The later matched prelift pair closes with both conditions failing400,
800 actions. Requested grip contrast.321608 and identical preludes/other13
parameters were verified; measured closure moved fingers about2.3mm more,
but the phone still lost elevation before opening. Native C3 retiming exposes
only9 of32 nominal grip slots, a mechanism hypothesis rather than proof.
The failed pre-model launcher costs103 GPU-seconds and its qualified retry233,
336 total inside the original600 ceiling; original failures stay retained.
[Closed pair](../../outputs/frozen-pi05-endtoend-20261031-r1/PRELIFT-CLOSEOUT.md).

The prospective matched four-case follow-up is also terminal: one versus five
repeated native holds, each with current versus close-endpoint grip, all
**four fail400**. All preludes and same-grip targets across doses match;
within-dose contrasts are nondegenerate (.472362). It costs310 GPU-seconds
and1,600 actions, with frozen721a unchanged and all four post-branch recoveries
saved. No candidate04 exists, so the fixed bank stays11/12 and fitting remains
closed. These specific interventions did not rescue04; this does not reject
all reactive policies or establish a unique contact/approach/transport cause.
Repeated native holds also consume TOPP/RNG operations, so this is not a pure
elapsed-time contrast. [Four-case receipt](../../outputs/frozen-pi05-endtoend-20261031-r1/FACTORIAL-TERMINAL-RECEIPT.json).

Saved scalar diagnosis finds that the five-close finger coordinates nearly
plateau after hold2: (.035652,.037593)m versus (.035659,.037552)m after hold5.
The extra change beyond one hold is only about.27/.97mm; the five-current
endpoint largely converges despite the requested contrast. Both fingers then
reach0/0 after lift in every condition, while phone height falls before release.
Observable settling occurred; adequate contact or a stable grasp was not
established. This weakens simple insufficient-settling rescue, without proving
a unique approach/contact/path/transport cause or author-scene identity.
No further blind hold retry follows.
[Scalar report](../../outputs/robotwin-prelift-factorial-diagnosis-20261101-r1/REPORT.md).

The targeted phase closes these jobs at3,452 GPU-seconds/23,922 actions with
no active holds. Old-phase CPU is now1,775/1,800,25 unused after the separately
bound qualification below.
One composite qualification passes29 partial-wrapper and33 factorial cases plus
real04 schedule preflight on zero-GPU2CPU/16GiB,88 inclusive CPU-worker-seconds.
The original bank44, wide40, resource3 and parallel14 checks are retained, not
repeated. The zero-action PARTIAL11 cache also closes successfully:11 cases/77 rows
(49TRAIN/28VAL),1,377,989,403 bytes and154 GPU-seconds. The execution owner
checks all77 payload hashes, immutable cursor/RNG-recovery bytes and unchanged
721a motor. The canonical action tail stays unchanged. It is preparation, not
a full12 seal, successful04 or fitting/source admission.
[CPU proof](../../outputs/frozen-pi05-endtoend-20261031-r1/PARTIAL-FACTORIAL-CPU-RECEIPT.json),
[cache closeout](../../outputs/frozen-pi05-endtoend-20261031-r1/PARTIAL11-TERMINAL-RECEIPT.json).

The required diagnostic42 plus curve42 tests pass on exact06 in a zero-GPU
Linux allocation. The optional fitted-bridge suite has5 passes/3 fixture
failures; the overall wrapper exit1 and93-second charge are retained. A
separate preservation job on d442 passes33/33 new recovery tests; three
optional bridge fixtures fail on another missing fake-backend attribute,
and the optional readout does not run. That wrapper also retains exit1,
charging98 of its separate120 CPU-worker-seconds. These failures do not
override independently qualified cache preservation or authorize a fit.
[Recovery qualification](../../outputs/frozen-vla-preservation-software-qual-20261101-r1/RECEIPT.json).

The d442 cache04 GPU job closes13 original04 rows and immutable before/after
whole-case RNG/cursor recoveries. All13 payload byte hashes match; the paid11
manifest is unchanged. Frozen721a qualification passes and no robot action
runs. The subsequent full12 sealer fails at its combined mask/label/context
schema guard, so the allocation remains Failed/exit1 despite completed
encoding. Its173 GPU-seconds bring the phase to3,625, with23,922 actions
unchanged and no active GPU hold. Checkpoint byte/ZIP-CRC checks do not imply
an actual restore. [Terminal evidence](../../outputs/frozen-pi05-endtoend-20261031-r1/CACHE04-TERMINAL-RECEIPT.json).

Expected target: admit the complete retained command-plan cache before any
fit. Actual: encoding/recovery succeed while consumer validation fails.
Allocated CPU inspection of the first paid00, first04 and last04 rows confirms
float32 official state shaped(1,32). The only failed predicate in each is the
sealer's hardcoded(1,14) state expectation; every other enumerated schema,
context, mask, label and cost predicate passes. This isolates the observed
failure in those three rows; it does not validate the full90 rows.
[Measured predicate report](../../outputs/frozen-vla-readout-software-qual-20261101-r1/evidence/schema-inspection.json).

Consumer400eed69 changes that one expectation to(1,32), preserving full model
state and every other guard. Two independent static reviews pass. The correction job-qzpsryt5bul9 now reports12/12 focused passes and a successful
existing full90-row/all12 CPU seal, retaining paid77+new13 and failed04. This
is a repair of a concrete ABI mismatch, rather than another physical hypothesis.
The allocation closes exit0/103 inclusive CPU-worker-seconds. Copied JUnit,
zero-GPU/clock and terminal metadata hashes pass parent review; all90 tensor
validation remains the allocated producer's proof, not a Mac tensor audit. No payload conversion, VLA, GPU or robot action
ran. [Copied CPU proof](../../outputs/frozen-vla-cache-readout-correction-cpu-20261101-r1/PARENT-REVIEW.json).

Independent f3 readout qualification passes56/57 tests. The sole failed fixture
compares an empty unresolved-decision mapping with integer0 after its three
synthetic400-step episodes; production behavior is unchanged. Fixture235
corrects the comparison to{}. Its one corrected selector now passes in the same correction job. The three
previously failed bridge fixtures also pass; old component passes are carried,
not rerun. The old wrapper remains Failed/exit1 at69/120 CPU-worker-seconds.
The new correction uses103/300, separate from old1,775/1,800,98/120 and69/120;
no old cap is reset. Optional outcomes did not gate the independently
qualified cache seal.
[Readout receipt](../../outputs/frozen-vla-readout-software-qual-20261101-r1/RECEIPT.json).

This new diagnostic keeps whole8TRAIN/4VAL and failed04 explicitly false; it
imitates known submitted command plans rather than claiming all teachers
succeed. The old successful-teacher gate remains unchanged. No learned head,
chronological RL source corpus, competent noise teacher or AD/DPT result
exists yet. There is no filtered or reduced bank. Two independent fixed5000
command-fit children now have900 inclusive GPU-seconds each reserved inside
the original A1 residual, with actual runtime/resource admission still pending.
Each will measure update time/capacity inside the retained scientific run;
insufficient time must leave a full incomplete recovery, not a truncated
endpoint or a favorable validation choice. A conditional1,500-second readout reservation now fits the same old A1
residual:15 fixed native/action5000/KV5000 episodes,6,000 controller actions,
with qualification embedded in the first native episode. It may launch only
after both actual5000 endpoints and runtime/resource admission; no early
checkpoint substitution or additional episode is admitted. At05:20 the independent fits had real optimizer progress on distinct
physical GPUs: action1,409/KV499 with full recovery hashes matching. Both
subsequently stop at the existing inclusive time guard, Failed/exit1. Action
retains3,399 complete updates; KV2,655. All0/100/500/2000 milestones and full
recoveries survive; neither5000 endpoint existed at that closeout, so final-only
robot readout was held. Each attempt charges782/900 GPU-seconds, leaving118. Closed
original phase cost is now5,189 GPU-seconds/23,922 actions. No independent
terminal full-motor hash was published on this guard failure; per-update
frozen checks remain recorded.
[Terminal proof](../../outputs/frozen-pi05-endtoend-20261031-r1/COMMAND-FIT-TERMINAL-PARENT-REVIEW.json).

The initial allocations were too short. Saved action intervals500→2000 and
2000→3399 are228.4s/333.7s; KV500→2000 and2000→2655 are332.2s/180.1s.
These include recovery/evaluation work. The guard repeatedly stats a growing
immutable recovery tree, a plausible I/O contributor without profiling proof;
filesystem latency and milestone work remain alternatives. No resource or
learner optimization is introduced by this diagnosis, and incomplete prefixes
are not negative capacity results.

Continuation must restore the same optimizer/samplers/global and private RNG,
then finish the fixed5000 schedule. Existing400 resource admission cannot
represent a larger continuation while preserving the old900 rows: it compares
historical lineage caps to one current head cap, and118 cannot admit setup.
A narrow operational successor is being prepared to validate old and new
credits separately while preserving every old row and exact scientific400
contract. An explicit linked3,600-second credit (two1,800 children) and a new
200-second focused CPU gate are funded under the per-job standing authority.
They use the same original global28,800 clock/cap; original A1 charges and
its1,500-second readout hold stay separately inside7,200. No old cap or ledger
is reset. The operational successor3d35 passes29 focused tests plus actual small
predecessor inspections on allocated zero-GPU Linux, charging77/200 CPU-worker
seconds. It retains protected400 mathematics/strict scientific contract and
records the new execution provenance separately. Actual resumed action3,614
and KV2,906 updates exceed their saved3,399/2,655 boundaries; new full recovery
hashes and distinct allocated GPU probes pass. No prefix replay, new robot
action or final5000 endpoint is claimed.
[Resume proof](../../outputs/frozen-pi05-endtoend-20261031-r1/LINKED-RESUME-PARENT-REVIEW.json).

At06:20, the linked action continuation completes the fixed5,000 updates,
Succeeded/exit0, charging727 GPU-seconds after its original782. The endpoint
and full recovery hashes match; all90 cached-native parity and source postfit
frozen motor721a checks pass. Training/validation decoded14D plan MSE are
0.00547043/0.03020526; validation gripper MSE0.15009219 dominates that aggregate.
The final head gradient is finite and nonzero. These are command-plan fit
measurements, not physical competence. The primary endpoint remains5,000;
validation does not select an earlier checkpoint. KV is still running at4,727
in this saved observation. Closed phase GPU cost is5,916, with1,800 KV and1,500
readout seconds held separately. Endpoint CRC is null in the saved receipt;
there is no new terminal restore claim.
[Action endpoint and parent review](../../outputs/frozen-pi05-endtoend-20261031-r1/ACTION5000-0620-PARENT-REVIEW.json).

The unchanged235 readout guard rejects the linked-family resource names.
Resource-only successor949c88fe has independent static PASS and preserves
policy/episode/RNG mathematics. Its one allocated CPU gate, job-4jq7u25x1sv6,
qualifies exactly43 new and11 affected legacy resource cases under a new
180-worker-second total. Optional cached wide-head error analysis may use
only remaining time after these54 cases; it cannot hold readout or create a
new allocation. After both actual5,000 endpoints and required CPU PASS, the
funded unchanged15-episode physical readout may launch with actual linked
closeouts/runtime admission. No additional confirmation or favorable prefix
selection is needed. No source teacher, chronological RL corpus or AD/DPT
eligibility follows from these offline fits.


## Closed replay diagnosis: requests exact, physics unresolved

The one corrected4bb inspection succeeded on Baige `job-a3luoqs9bwkn`:
13/13 focused tests, actual0GPU/2CPU/16GiB,81 all-in CPU-worker-seconds.
All608 valid teacher00/01/02 requests exactly match admitted HDF5 commands,
including final-target holds; all seeds, instructions and14 initial drive
targets match. This rules out a wrong reference index or altered issued command
in these inspected records. Failed02 nonterminal arm end-drive residuals are
at most0.00036648rad and gripper residual0. Successful00/01 terminal comparisons
are partial-command qualified. Drive targets are not measured joints or object
poses. No cache, fit, simulator or model ran in this inspection.

Reference02 closes the gripper at frames50..63 while the arm is stationary
47..70; opening109..122 occurs in stationary106..128. Successful00/01 have
similar hold/transition patterns. Native arm compression/TOPP retiming and
separate original-slot gripper timing differ from author dense execution;
this is a candidate mechanism, not an explanation established by these traces.
The saved reset head images appear closely matched, while wrist views differ.
Native actor poses, measured joints, camera matrices and merged task_config
were not saved, so scene identity remains unproven. Final images compare source
frame128 with actual command400, without synchronized physics timing.

The fixed8TRAIN/4VAL demonstrations, including failed02, remain intact. The
next step is a narrow successor proposal with retained/repeated episodes,
recovery and costs before dispatch. There is no automatic teacher exclusion,
refit, executor change, GPU retry or source admission. Whole phase CPU spend
is827/1800 with973 remaining; GPU/action spend stays2036/18599. Earlier gates
and their failures remain below. [Report and exact evidence](../../outputs/frozen-pi05-endtoend-20261031-r1/REPLAY-DIAGNOSIS-RESULT.md),
[parent saved-artifact review](../../outputs/frozen-pi05-endtoend-20261031-r1/REPLAY-DIAGNOSIS-PARENT-REVIEW.json).

## Current capacity run stopped before fitting

Baige `job-471ns3j0v2n4`, source `8fb991c8`, is terminalFailed/exit1. Actual
one-GPU/own-PID checks pass. At qualification, motor fingerprint721a remains
fixed, decoded zero/native maximum error is0, real KV gradient0.00309944 and
motor gradients0. Repeated paired prefix/mask/features/cache are exact. The
complete400-action qualification episode times out. No fit or optimizer update
runs; a fresh terminal full-motor rehash is not claimed.

| Fixed teacher replay | Actions | Outcome |
| --- | ---: | --- |
| Phone00 |104|Success|
| Phone01 |104|Success|
| Phone02, seed2 |400|Timeout|

The code stops before cache creation or either100-update fit. Matching seed
and initial drive targets do not prove object poses or measured joint state
match. Cause is unestablished, not a negative learned-steering result. Keep all
fixed8TRAIN/4VAL demonstrations, including02. The next bounded allocated CPU
comparison checks delivered commands/indices/masks, initial scenes/cameras and
author-versus-native timing. No teacher filtering, fitting fallback or blind
retry is admitted.

This attempt costs266 all-in GPU-seconds/1008 actions; cumulative2036/18599
leave5164 GPU-seconds/11401 actions in A1 and26764/281401 in the whole phase.
Pending/uncertain actions are0. The41381-byte driver/RNG recovery is preserved
at SHA `c59c91799938c928bdd683a6ce681cfa3f9853fe0c829c05fe6657fe30af5d5b`.
CPU totals746/1800,1054 remaining. [Terminal evidence](../../outputs/frozen-pi05-endtoend-20261031-r1/CAPACITY-GPU-RESULT.md).

## Completed restricted screen, 19:18 Shanghai

Successor `job-zp47pliu429f`, exact source `969fb44c`, is terminal Succeeded,
exit0. The complete conditional grid contains51 episodes:16 preserved native,
27 search and8 confirmation. No episode, pending action or uncertain charge
is missing. All51 manifest hashes/counts and paired reset identities match;
the final recovery is stable and the terminal full-motor guard passes.

| Phone development case | Noise search | KV search | Action search | Confirmation noise/KV/action | Matched native |
| --- | --- | --- | --- | --- | --- |
|1|0/3|0/3|0/3|Not triggered|Not triggered|
|2|0/3|0/3|0/3|Not triggered|Not triggered|
|3|1/3|2/3|1/3|1/2,1/2,0/2|1/2|

No arm meets the predeclared requirement of2/2 intervention successes against
0/2 native on at least one case. No teacher or source collection is admitted.
The search maximum is selected on the development outcomes; it cannot replace
the independent confirmation result. This is a restricted fixed4D random-basis
probe, not a test of the complete learned reactive KV/action policy classes.

R2 costs1138 conservative GPU-seconds/12236 new controller actions. With R1
counted once, cumulative cost is1770 GPU-seconds/17591 actions, leaving5430
GPU-seconds/12409 actions inside A1 and27030 GPU-seconds/282409 actions in
the whole phase. CPU remains331/1800 worker-seconds. No budget is replenished.
Original raw/native records and both recoveries remain immutable. See
[terminal evidence](../../outputs/frozen-pi05-endtoend-20261031-r1/A1-R2-TERMINAL-RESULT.md)
and [fixed rule](../../outputs/frozen-pi05-endtoend-20261031-r1/SELECTION-RULE.json).

The next discriminating test is the original same-phone supervised capacity
check:8 whole TRAIN/4 VAL released demos, full learned reactive KV/action
controllers, with the32 locked cases unopened. Archive timing, camera and
reset semantics plus actual executable/cost admission remain pending. The
provisional5400 GPU-seconds/10000 actions fit the remaining A1 caps but are
not a launch reservation. No capacity model or result exists yet.

A positive result for the full reactive class would not qualify the existing
fixed4D coefficient collector or its AD/DPT labels. The interface bridge is
unresolved: source collection requires an actually qualified teacher and
compatible labels under that same interface. Released supervised demos are
capacity diagnostics, not a replacement for chronological RL histories.

## Actual phone data qualification and public-state limitation

Baige `job-rh3ldkdapfv9`, exact source `9e1953e6`, succeeded with11/11 tests,
actual0GPU/2CPU/16GiB and86 conservative CPU-worker-seconds. Phase CPU use is
417/1800,1383 remaining; GPU/action cost is0. All12 whole phone candidates
are saved with actual seeds, scene metadata and fixed first-seen instructions.
All explicit success flags are absent. Temporal action alignment and simulator
replay remain unqualified; this is no expert/capacity/performance claim.

Saved A1 commands have17760 issued slots and17591 actual valid slots. Raw
native grippers6/13 are finite but exceed[0,1]: valid minima
[-.0101426,-.0069030], maxima[1.0090805,1.0086253], outside counts2362/5913.
The actuator's existing clipping is canonical. Any calibration headroom guard
must check transformed arms while preserving raw native grippers bitwise.

Pinned RoboTwin get_obs stores commanded drive targets plus gripper values;
vector_env.update_obs assigns that joint_action/vector to public state. This
is not measured articulation qpos. Along with the previous requested command,
current target feedback may expose calibration bias after one chunk away from
clipping/transients. Both signals are already in the static student's2076D
input. A gap against one fixed blind teacher does not establish history or
reward necessity. Retain first-query/static/reward-lesion controls and report
system identification honestly if reward feedback is unnecessary. No condition
redesign is authorized by this finding.

[Exact CPU evidence](../../outputs/frozen-pi05-endtoend-20261031-r1/PHONE-STAGE0-CPU-RESULT.md),
[metadata/source conventions](../../outputs/frozen-pi05-endtoend-20261031-r1/phone-data-preparation/source-conventions.json).
[Complete public-state source chain](../../outputs/frozen-pi05-endtoend-20261031-r1/PUBLIC-STATE-SOURCE.md).

## Software qualification retains failed attempts

| CPU attempt | Tests | Outcome | Worker-seconds |
| --- | ---: | --- | ---: |
| e5f685d1 / job-mjw4gq2e94d3 |91|87pass/4fail|134|
| 88bf7018 / job-nneqgcplclpy |42|32pass/10blocked|121|
| 8fb991c8 / job-qh4pau3eh6v8 |10 affected|10pass|74|

The first gate found three fixture mapping assertions and a real missing-method
budget call, repaired in both CLI/driver paths without changing the canonical
budget API. All10 affected source/pipeline cases pass in the second gate.
Its22 capacity passes remain; ten further cases were blocked before assertions
because my combined tree omitted the required metadata fixture. The final child
restores the exact existing inventory bytes, changes no runtime/tests, and passes
those ten, including the actual-author JPEG codec check. All32 capacity cases
are qualified across this retained evidence, not a fresh full-suite rerun.

All attempts attest0GPU/2CPU/16GiB, with no VLA or simulator. These329 seconds
share the original600-second envelope;271 unused seconds return within the
same phase. Together with earlier417, actual CPU use is746/1800. Software
qualification admits no source teacher or memory/performance claim.
[Failed gate](../../outputs/frozen-pi05-endtoend-20261031-r1/COMPOSITE-CPU-RESULT.md),
[capacity rollup](../../outputs/frozen-pi05-endtoend-20261031-r1/CAPACITY-CPU-ROLLUP.json).

## Repair qualification (historical)

The repair passed108 allocated CPU checks plus an actual-artifact migration
dry-run, without changing the old journal. Re-encoding saved original RGB/state
with its actual original prompt reproduces public features, prefix mask and
all18 caches exactly. Native/zero decoding is exact; real KV gradient norm is
0.00363427, motor gradients0, and the721a motor fingerprint remains unchanged.
The16 native episodes were reused. The first four steering timeouts were an
interim observation; the completed grid above supersedes that status.

## Initial interface qualification and stopped R1


Baige `job-hm6fgvu3p4a6`, exact source `8248b1ce`, uses the frozen Motus
pi0.5 motor through pinned RLinf/RoboTwin. This is runtime qualification,
not a trained steering, robotic-performance or in-context-learning result.
The allocated synthetic CPU gate passed71/71 checks on actual0GPU/2CPU/16GiB,
charging79 of the phase's1800 CPU-worker-seconds.

The saved17:54 Shanghai receipt verifies:

| Check | Observed result |
| --- | --- |
| Native cached decode with identical noise | Maximum absolute error0 |
| Zero learned KV, fixed KV and physical-action edits | Bitwise equal decoded arrays |
| Real frozen-suffix gradient into the KV adapter | Norm0.0023523415632070587 |
| Gradients into motor parameters | 0 |
| Full motor fingerprint before/after | Identical `721a7ed…` |
| Native KV shapes |18 layers, each K/V `[1,968,1,256]` |
| First complete real episode | Phone-stand native success,134 controller actions |

The first run is terminal (exit1), after all16 native development episodes:
phone stand1/8 and object stand3/8,5355 controller actions in total. The first
search reset was rejected before any intervention action because the same
seed, identical RGB arrays and identical14D state came with a different
instruction. Frozen public features also differed. The original instruction
was “Carry the thin metallic phone to the flat base phonestand.”; the repeated
reset supplied “Carry the lightweight phone to the light brown phone holder.”
The strict guard is retained; instruction reuse must be repaired before an
explicit fresh-child migration. All16 original native instructions remain
constant throughout their175 raw chunks. Static official-source inspection
shows constructor-cached episode descriptors participate in instruction
generation before the current setup; this is not safely described as only
global RNG jitter. The repair preserves each case's actual original prompt. All32 locked confirmation cases remain closed.
At that stopped R1 stage there was no search/confirmation result or admissible source teacher.
This is an execution defect, not a negative result for any steering family.

The initial qualification motor hashes match. The exception occurs before
the normal final hash block, so an independently recomputed terminal motor
fingerprint is not claimed. Original raw/reset/episode artifacts and the
104609-byte recovery remain intact. The settled632 GPU-seconds leave6568
inside the original A1 limit;5355 actual actions leave24645. Neither limit
is replenished.

Evidence: [initial receipt](../../outputs/frozen-pi05-endtoend-20261031-r1/A1-INITIAL-RECEIPT.json),
[terminal diagnosis](../../outputs/frozen-pi05-endtoend-20261031-r1/A1-TERMINAL-RECEIPT.json),
[language constancy](../../outputs/frozen-pi05-endtoend-20261031-r1/A1-NATIVE-LANGUAGE-CONSTANCY.md),
[instruction source](../../outputs/frozen-pi05-endtoend-20261031-r1/A1-INSTRUCTION-SOURCE.md),
[CPU result](../../outputs/frozen-pi05-endtoend-20261031-r1/A1-CPU-RESULT.md),
[bound grant](../../outputs/frozen-pi05-endtoend-20261031-r1/GRANT.json).
Current costs and next checks belong in [coordination](../../outputs/coordination.md).

## Rendering observation and resolved pairing check

Repeated SAPIEN OIDN invalid-handle messages are preserved. Nonblank-RGB
checks passed and no rendering exception was observed. They do not explain
the paired-input mismatch: both saved reset NPZs were independently rehashed,
and all image/state NPY bytes match exactly. The confirmed recorded input
change is the instruction; the repaired saved-original-prompt re-encode subsequently reproduced the
original prefix exactly, as described above. The guard was preserved.

The live checks establish a usable gradient path and zero-edit compatibility.
They do not establish that bounded edits improve success, that a source
teacher is competent, or that history/reward improves a fixed-weight query.
