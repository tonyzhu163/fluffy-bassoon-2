# First-rubric iteration on twenty selected trajectories

Completed Baige `job-nun3gy1c0tk0`, source `5b9d6a54`, exit0. The first
tested revision is the byte-identical baseline and reproduces all20 cached
hard grades. This is a new measured iteration on that rubric, after inspecting
its errors and pinned simulator task/fixture source. Only20 existing videos
were scored,14 native failures and6 native successes,100labels in total.

| Rubric / output | False successes out of14 failures | Success controls accepted out of6 | Baseline false successes corrected | New false successes | Baseline accepted controls lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| First revision /1–5 |12 |4 |0 |0 |0 |
| Revision1.1 /1–5 |7 |4 |6 |1 |1 |
| Revision1.1 /0–1 |11 |5 |2 |1 |0 |
| Revision1.2 /1–5 |10 |3 |3 |1 |1 |
| Revision1.2 /0–1 |10 |4 |3 |1 |0 |

Full success means Score5 for graded arms and Success1 for binary arms.
Binary uses its own consistent system and output grammar. Its hard labels
are not calibrated success probabilities. Graded expected scores and raw
binary text are retained in [PAIRED-SAMPLE.json](PAIRED-SAMPLE.json).

Revision1.1 is the better development direction, but has seven false
successes and is not qualified for PPO. Its equal4/6 control total conceals
one loss and one recovery. Direct binary labels do not solve the problem:
they accept eleven failures under1.1, versus seven with the same criteria
and graded output. The explicit numerical-predicate variant1.2 is also
worse. Do not promote either ablation or run a full-bank replay on this basis.

## What the individual failures show

Revision1.1 fixes all four sampled Stove false successes, while retaining
one of the two Stove success controls. The added language follows the named
burner's knob into its off region and rejects absent flame or partial knob
rotation as sufficient evidence. Because several clauses changed together,
this does not establish which clause caused each correction.

Drawer false successes only fall from8/10 to7/10. Two are corrected, but
one previously corrected failure becomes a false success again. Five of
the six persistent drawer errors survive **every** tested wording/output
combination (tiles01,03,04,05,06); both highest-score examples still receive
Score5 or Success1 throughout. Asking for binary output is particularly
permissive here: revision1.1 accepts9/10 failed drawers.

The diagnostic images show substantial drawer opening in some failures,
and poor final visibility in others. They do not prove whether each failure
is due to target identity, insufficient slide displacement, or a completed
state that was later undone. Final joint values and native first-success
timestamps were not retained in this old bank. The next iteration should
inspect these remaining examples at full resolution and establish which
predicate is missing before adding more wording. Another inference batch
should stay small; no further batch was launched in this run.

The native source adds concrete task detail: target is TOP_DRAWER; left/right
is robot-relative at initialization; one drawer slide joint must reach
>=0.95 after normalization by0.55*fixture depth. Opening an entire drawer
stack or withdrawing the hand is not required. Source and limitations are
in [INSPECTION.md](INSPECTION.md) and [SOURCE-AUDIT.json](SOURCE-AUDIT.json).

## Per-trajectory readout

Indices correspond to the reset/end contact sheets. Scores are literal
grades or binary labels, not probabilities. IDs below omit only the shared
dataset/task prefix, preserved in the paired JSON.

| Tile | Native episode ID | Simulator | R1 grade | R1.1 grade | R1.1 binary | R1.2 grade | R1.2 binary |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 01 | `flora-single-case-000112-rank-06-episode-000001` | failure | 5 | 5 | 1 | 5 | 1 |
| 02 | `flora-single-case-000272-rank-02-episode-000004` | failure | 5 | 4 | 1 | 5 | 1 |
| 03 | `flora-single-case-000064-rank-00-episode-000001` | failure | 5 | 5 | 1 | 5 | 1 |
| 04 | `flora-single-case-000376-rank-07-episode-000005` | failure | 5 | 5 | 1 | 5 | 1 |
| 05 | `flora-single-case-000256-rank-00-episode-000002` | failure | 5 | 5 | 1 | 5 | 1 |
| 06 | `flora-single-case-000216-rank-03-episode-000001` | failure | 5 | 5 | 1 | 5 | 1 |
| 07 | `flora-single-case-000208-rank-02-episode-000001` | failure | 5 | 1 | 1 | 1 | 1 |
| 08 | `flora-single-case-000088-rank-03-episode-000005` | failure | 5 | 5 | 1 | 1 | 0 |
| 09 | `flora-single-case-000336-rank-02-episode-000000` | failure | 4 | 5 | 1 | 5 | 1 |
| 10 | `flora-single-case-000024-rank-03-episode-000005` | failure | 4 | 2 | 0 | 1 | 0 |
| 11 | `flora-single-case-000128-rank-00-episode-000006` | success | 3 | 5 | 1 | 1 | 0 |
| 12 | `flora-single-case-000072-rank-01-episode-000000` | success | 3 | 1 | 0 | 1 | 0 |
| 13 | `flora-single-case-000264-rank-01-episode-000001` | success | 5 | 5 | 1 | 4 | 1 |
| 14 | `flora-single-case-000144-rank-02-episode-000004` | success | 5 | 5 | 1 | 5 | 1 |
| 15 | `TurnOffStove-qualification-case-000032-rank-05-episode-000000` | failure | 5 | 4 | 1 | 5 | 1 |
| 16 | `TurnOffStove-qualification-case-000026-rank-04-episode-000003` | failure | 5 | 1 | 0 | 1 | 0 |
| 17 | `TurnOffStove-qualification-case-000014-rank-02-episode-000005` | failure | 5 | 4 | 0 | 5 | 0 |
| 18 | `TurnOffStove-qualification-case-000026-rank-04-episode-000005` | failure | 5 | 4 | 1 | 5 | 1 |
| 19 | `TurnOffStove-qualification-case-000044-rank-07-episode-000002` | success | 5 | 5 | 1 | 5 | 1 |
| 20 | `TurnOffStove-qualification-case-000038-rank-06-episode-000001` | success | 5 | 4 | 1 | 5 | 1 |

## Scope, validation and cost

The sample was intentionally enriched for known revision1 errors: twelve
false full successes, two corrected failures, two downgraded true successes
and four retained true successes. These are development examples, with
repeated Stove case identity in two failure trajectories, not a random
population accuracy, AUROC or first-success precision estimate. Old videos
retain their native horizons; this is not a fresh600-control calibration.
The model received native task text, generic criteria and untouched RGB
videos only, never labels, error-group names, seeds or hidden simulator state.

All100labels,64 sampled frames per video, rendered prompts,15 batch hashes
and receipts are retained. Eight Linux core scorer tests and the official
binary-template check pass; copied batch hashes match the Linux receipts.
One persistent TP1 engine, batch8, temperature0; model loaded once. No
simulator calls, robot actions, PPO updates or production reward/cache changes.

The finite worker records713 GPU-seconds; the scheduler reports12min12s
running,20:26:58–20:39:10 Shanghai. Charging even the entire submission-to-end
interval gives749 GPU-seconds, below the1800 cap. No retry or successor job.
Raw evidence: [RESULT.json](RESULT.json), [CPU checks](CPU-CHECKS.txt),
[engine receipt](scoring/ENGINE-READY.json), [terminal charge](terminal-charge.json),
and [completed scheduler view](completed-job.png).
