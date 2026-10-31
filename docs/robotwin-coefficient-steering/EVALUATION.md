# Frozen-Motus coefficient steering evaluation

## Live interface qualification, 2026-10-31

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
There is no search/confirmation result and no admissible source teacher.
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

## Rendering observation and next discriminating check

Repeated SAPIEN OIDN invalid-handle messages are preserved. Nonblank-RGB
checks passed and no rendering exception was observed. They do not explain
the paired-input mismatch: both saved reset NPZs were independently rehashed,
and all image/state NPY bytes match exactly. The confirmed recorded input
change is the instruction; a saved-original-prompt re-encode and repeated
identical-input prefix check on an allocation will separate that change from
any numerical jitter. No guard is weakened and no fresh steering action is
admitted until the repaired prepared reset matches the original case.

The live checks establish a usable gradient path and zero-edit compatibility.
They do not establish that bounded edits improve success, that a source
teacher is competent, or that history/reward improves a fixed-weight query.
