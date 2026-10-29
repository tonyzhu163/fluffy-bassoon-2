# Meta-World noise screen: visible reach condition

Independent saved-artifact audit: **2026-10-29 19:24 Asia/Shanghai**.
Baige `job-orgyjb0z9h6d` remains active on its original source
`4d4f46f4cdcd8cfd7c7056e95bf85d0a90e0ed59`. The visible-goal reach condition
closed with all **544 episodes**; the hidden-goal condition has started.

| Goal | Native success | Best fixed-arm success |
| --- | ---: | ---: |
| 0 | 25% | 50% |
| 1 | 0% | 0% |
| 2 | 100% | 100% |
| 3 | 0% | 0% |

Native success is **10/32 (31.25%)**. Selecting noise arms on one half of the
scenes and scoring on the other gives **37.5%**, a **6.25 percentage-point**
gain, below the prespecified 15-point G1 gate. G2 gains **0 points** over the
reference-selected static arm, below its 10-point gate. Two goals were never
reached by any fixed arm or native sampling, so the goal-coverage check fails.
These visible-goal results do not admit noise-arm learner training.

The G3 rank-consistency mean is .85878 over the two informative goals; it is
undefined for the two all-failure goals. The G4 log-score gain is .09082.
Passing those two aggregate diagnostics does not compensate for failed G1,
G2 or goal coverage. All 256 fixed-arm pairs match on binary outcomes; none
match on the complete issued-action hash. Fixed-repeat first-chunk distance
averages .02655, versus .42669 between arms. This is not an exactly
deterministic trajectory cache.

[audit_condition.py](audit_condition.py) independently reconstructs the complete
goal/scene/arm/repeat grid, success rates, split-half arm selection, G1–G4,
threshold decisions, coverage, repeat counts, finite first-chunk arrays and
artifact hashes without importing the production gate implementation. Its
[audit receipt](checks/20261029-1918-visible/audit.json), raw outcome rows,
original gate output and first-chunk arrays are mirrored together in that
directory. Undefined correlations are represented as null in the audit.

This is one frozen motor on four predeclared goals and four fixed scenes per
goal. It establishes a partial interface-screen result, not meta-learning,
held-out task transfer or an architecture comparison. The final motor
fingerprint check remains pending until the complete job ends. Continue the
already authorized hidden-goal and subsequent push conditions within the same
175-minute cap; no learner, motor fine-tune or new allocation is launched.
