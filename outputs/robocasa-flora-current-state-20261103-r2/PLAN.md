# Current-state recognition before another reward revision

Status: r2 corrects the scorer-configuration path after the zero-GPU r1 CPU failure. Scientific selection, four arms, prompts and rejection rules are unchanged.

Owner: chat `01a1a006-2d76-7b13-8660-399c625aee67`.
Human instruction: “keep fixing these”, following review of the five-task
Flora-versus-simulator ROC curves and the recommendation to separate native
condition recognition, temporal aggregation, and reward mapping.

The goal remains a qualified Flora reward aligned with simulator completion
across the five RoboCasa reward tasks. This run is the next diagnostic, not a
substitute for unused-case validation or a declaration that the reward is fixed.

## Question and factors

Does judging the current state with initial target context improve recognition
of the requested drawer and its native opening condition? Use the same 18
endpoints from the completed frame-edge trial, plus four verbatim reference
repeats. These are reused, error-enriched development cases.

Cross two evidence representations with two reporting contracts:

| Factor | Control | Alternative |
| --- | --- | --- |
| Evidence | Official 64-frame causal prefix | Reset and current frame, each duplicated for a temporal patch pair: `[0,0,t,t]` |
| Reporting | Official one-to-five output grammar, current-state criterion | `Target: identified/uncertain` and `Opening: meets/short/uncertain` under the same current-state criterion |

Both evidence arms judge the endpoint, not whether success ever occurred.
Both use the same native instruction and explicit static target/extent rubric.
The reporting factor includes its output instruction and decomposition; it
does not isolate token spelling from reasoning structure. The anchor evidence
factor changes frame count and temporal context together, and is not a pure
camera or resolution ablation. No frame after the query control is permitted.
All three synchronized cameras remain in their original order and resolution.

The top-drawer description is supported for these inspected layouts only.
Native TOP_DRAWER is a height selector, not a universal highest-drawer rule.
The native opening threshold is 0.95 × 0.55 = 0.5225 of cabinet-fixture depth,
not 95% of mechanical travel. Holding the handle and less than maximum
extension are allowed. Numeric wording defines the predicate; it does not
establish that the model can visually resolve that boundary.

## Frozen selection and labels

Use the 18 candidate rows in the completed frame-edge `PREFIX-REFERENCE.json`.
There are **12 current-state negatives and 6 current-state positives**, unlike
the old first-success grouping of ten negatives and eight post-success rows.
The closed-again 64@750 and 216@750 are current-state negatives. The complete
18-row selection and source hashes are frozen in `INPUTS.json` and `SPEC.json`.
Labels must match the corresponding hash-verified retained native replay.
Replay identity lacks a saved full MuJoCo initial state; retain that limitation.

The inference worker reads public jobs and input references only. Per-episode
native success, opening values, first-success clocks, hidden poses and joints
are absent from model messages and requests. The evaluator reads truth after
scoring. No new policy calls, simulator episodes, optimizer steps, or model
weight updates are authorized by this diagnostic.

## Decision fixed before scoring

Graded acceptance is exactly Score 5. Structured acceptance is exactly target
identified AND opening meets. Uncertainty is reported separately and counts as
non-acceptance for recall; it must not disappear from the denominator.
No cutoff, wording, or endpoint is selected after seeing results.

A development candidate passes only with zero false accepts on all 12 current
negatives, all six current positives accepted, all four reference hard grades
reproduced, all input/source audits passed, and all 76 outputs complete.
Report per-case predictions and paired changes, precision/recall, uncertain
counts, and graded AUROC. A pass only admits a larger predeclared validation,
not production. A fail determines whether the next work concerns target/depth
recognition or evidence representation; do not resume lexical fitting without
that diagnosis. External latching is a separate offline readout and cannot
rescue false recognition or promote a different threshold.

## Execution and cost

Use latest reviewed main, commit and deploy through Git into a new detached
checkout. Preserve all old runs and caches. Run preparation on allocated Baige
development CPUs, with CUDA masked and no weight loading. Verify source and
video hashes, causal frames, rendered messages, actual processor references,
parsers, label joins and free storage. Keep the official runtime unchanged.

Then request one actual 96-GiB GPU, 32 CPU cores, 128 GiB memory, 16 GiB shared
memory, automatic retry disabled. Verify current queue limits and GPU type,
then run the repository GPU probe in the actual job step. One persistent
official engine, batch at most eight; model computation stays on Linux.
The allocation is capped at **1,200 GPU-seconds (0.333 GPU-hours)**, including
startup, with child and wrapper deadlines and terminal cost receipts. This is
new incremental scope under the user's under-eight-hour rule, not a transfer
of another run's remaining budget. Record failed attempts and cumulative spend.

## Remaining work after this run

If recognition improves, freeze the candidate and evaluate unused retained
drawer cases at the intended deployment horizon and temporal definition.
Then apply predicate-specific diagnostics to Coffee and the other reward
tasks, preserving target, placement/contact, gripper-distance and appliance
exceptions. Compare reward maps only after recognition, using separate fit
and held-out splits. No ordinal score or grade-token probability is assumed
to be a calibrated success probability. Qualify first-success timing separately
from terminal success before any new PPO reward contract.
