# Does the order of 16 self-issued movements improve command calibration?

The previous archived estimator distinguished two settings on Panda and Sawyer and improved wrist-motion prediction versus swapped practice. The two evaluation settings coupled yaw and gain, and Sawyer's matched estimate had similar error to one common correction. This experiment crosses those factors and changes only the order of practice commands. [Archived comparison](../embodiment-tape-observability-20261120-r1/RESULT.md)

Design base `4bd494d188bfdcb757c554bb335b5c19d3bc97e6`. Native simulation uses LIBERO-Pro2.0 BASE Goal 5, fixed arm and gripper choices as in H1. The learned factor estimator, analytic temporal response, normalization and fixed common multiplier come from the archived report at the design base. Freeze their parameters before viewing this panel. PaliGemma is outside this motion prediction test. A separate workstream checks whether an existing source checkpoint qualifies for a later native task-success trial.

| Comparison unit | Prespecified allocation |
| --- | --- |
| Arms and conditions | Panda and Sawyer crossed with yaw +45°/+90° and translation gain 0.65/1.00. These are known source arms. |
| Placement | 16 independent support/query layout pairs. Support seed 991200+i, independent query seed 991300+i. Share each layout across conditions, schedules and arms. |
| Practice | Four +X, four −X, four +Y and four −Y translations of amplitude 0.39. First schedule holds each direction for four consecutive control steps, in that order. The other shuffles the exact same multiset with seed 2991200+i. Other motion components stay zero and gripper command stays −1. |
| Query | A single 16-step mixed H1 query per arm/setting/layout from independent reset, seed 2991300+i. Reuse the query for both schedules and every model and control. |
| Cost | Two practice tapes and one query tape per cell, 384 tapes, 6,144 simulator controls and no model training. |
| Workers | Native collector, frozen evaluation, Baige execution and independent source-motor qualification audit. |

Score the held-out XY motion of subsequent query commands, using the frozen temporal response and the same source normalization and scoring window for every comparison. For each estimator, condition and schedule, compare correct practice with a donor differing only in yaw, a donor differing only in gain, and the same fixed common correction. Also score the true interface through the shared response model. Report mean absolute circular yaw error and gain error, and paired yaw and gain changes while the other factor is held constant. Keep per-arm summaries visible. Weight cells equally. Treat the 16 shared layout indices as the resampling units for paired intervals.

**Decision.** Use blocked minus shuffled performance as a paired comparison. Prefer blocked for both arms if the frozen physical estimator has lower matched motion MSE on each arm and the pooled paired 95% interval for the improvement excludes zero. Evaluate usefulness separately from ordering: matched practice should beat the fixed common correction and the one-factor-swapped controls. The angular and gain half-spacings (22.5° and 0.175) show whether continuous estimates are accurate enough to separate neighboring settings. Report mixed outcomes as such. A query-motion gain alone cannot qualify policy success. Keep UR5e and hardware out of this run.

Staging and computation use the Baige distributed job surface, checked project storage and the qualified headless runtime. Stop on unrecoverable reset mismatches or terminal episodes before completing a tape, keeping partial evidence. On recoverable infrastructure failure record the cause and retry under a fresh attempt. [Earlier native runtime](../embodiment-command-frame-20261120-r1/EXECUTION.md)
