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

The first five closed phone development episodes contain one success and four
400-action timeouts,1734 closed actions in total. This is an incomplete,
predeclared development bank; it is not a population success estimate or a
steering gain. All32 locked confirmation cases remain unopened. The fixed
four-coefficient noise/KV/joint screen is still running, and no new teacher,
AD/DPT student or memory comparison has completed.

Evidence: [initial receipt](../../outputs/frozen-pi05-endtoend-20261031-r1/A1-INITIAL-RECEIPT.json),
[CPU result](../../outputs/frozen-pi05-endtoend-20261031-r1/A1-CPU-RESULT.md),
[bound grant](../../outputs/frozen-pi05-endtoend-20261031-r1/GRANT.json).
Current costs and next checks belong in [coordination](../../outputs/coordination.md).

## Unresolved rendering observation

The console repeatedly reports SAPIEN OIDN invalid handles. Saved backend
checks passed on nonblank RGB, a complete native episode succeeded, and no
exception or exit was observed. Harmless denoiser logging, subtler rendering
variation and ordinary policy failures are competing explanations for the
observed warning/failure combination; no cause is established. The already
planned matched native repeats, initial-state identities and stored RGB
provide the next discriminating evidence. Preserve the messages when
interpreting the completed screen; no extra run or recipe change is admitted
by this observation.

These checks establish a usable gradient path and zero-edit compatibility.
They do not establish that bounded edits improve success, that a source
teacher is competent, or that history/reward improves a fixed-weight query.
