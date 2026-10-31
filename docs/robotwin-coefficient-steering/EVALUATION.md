# Frozen Motus steering evaluation

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
