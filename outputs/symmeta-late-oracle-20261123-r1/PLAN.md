# Does losing the first ten controls explain lower single-attempt success?

The published seed-41 UR5e × π₀.₅ reader succeeded on 20 of 96 uninterrupted queries and on 76 of 96 calibrated retries. On a retry, both the information available at the start and the timing of correction change. The experiment below holds the query fixed and intervenes on each factor once.

| Fresh arm | Controls 1–10 | Control 11 onward |
| --- | --- | --- |
| Exact from 1 | True yaw, gain and gripper sign | Same exact inverse |
| Exact from 11 | Identity | True yaw, gain and gripper sign |
| Streaming from 11 | Identity, accumulate both five-control measurements | Seed-41 online estimate, updated on its own trajectory at the original cadence |

Every arm executes one continuous native episode on BASE layout 13. Each policy call proposes ten actions. Only the first five are eligible for execution. The next query observes the resulting state and consumes the next original policy noise tensor. The two delayed arms share the first ten issued controls, motor inputs, proposals, reward sequence and resulting observations. Every arm shares the complete initial reset and the first full motor proposal. A pre-action warm create and the original bounded reset-fingerprint callback qualify the native setup. Once an action is issued, the episode runs through native success or horizon without a reset.

**Timing hypothesis.** If the initial uncorrected actions cost successes, exact-from-1 should outperform exact-from-11 on the same task–transform assignments. A signed difference close to zero with a sufficiently narrow interval would weaken the timing explanation. A signed reversal would favor delaying exact correction on this panel.

**Inference hypothesis.** If residual inference errors cost successes after the opening, exact-from-11 should outperform streaming-from-11. Its advantage measures the effect of replacing the entire post-opening estimate trajectory by ground truth, including subsequent closed-loop consequences. The control-11 estimate and later five-control estimates are saved so early errors can be inspected. A narrow interval around zero would weaken the inference explanation. A broad interval leaves it unresolved. This comparison alone cannot assign the entire difference to the first estimate.

**Analysis.** The primary endpoint is native final task success over all eight tasks × twelve command settings. The eight held-out settings per task form a secondary 64-query subset. For each split, report success counts, directional discordant pairs, ties, signed percentage-point differences and the eight task-level effects. The 95% percentile interval counts all eight-to-the-eighth ordered resamples of eight whole task clusters with replacement. All shifts and all intervention outcomes on each task stay together. The exact-from-1 minus exact-from-11 and exact-from-11 minus streaming differences sum to exact-from-1 minus streaming on the same 96 rows. Intervals are marginal for each contrast. Shared resets, noise, prefix checks, and the nominal identity control test the comparison’s validity. Divergence of the fresh and historical totals is reported separately.

All three arms run fresh. Original rows establish the paper reference and lack per-query prefix traces that could certify an archived arm for a new pair. The 76 calibrated retries use evidence from layout 11 and remain historical context. The parent owns scheduler submission, in-job qualification, monitoring and outcome interpretation.
