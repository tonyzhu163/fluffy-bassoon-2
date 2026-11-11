# Frozen BPP on custom relational layouts

The eight-episode smoke produced **zero native bowl-on-plate successes**.
Both fixed motor prompts failed on all four tested custom layouts, despite
their earlier 20/20 success per candidate on native initializations. This
stage has not supplied the executable alternatives needed for the proposed
history experiment. No history reader was trained or evaluated.

The controller did move: six trajectories lifted a bowl by 5.6–13.0 cm.
The traces and videos show wrong-object choices, transport away from the
plate, and empty-space reaches. These are substantive physical failures;
the result is not explained solely by an incorrect reward label. It does
not establish why the frozen policy fails, or rule out every interaction
between the new reset/recording path and the policy.

## What was run

Sixteen geometrically valid development layouts balanced LEFTMOST versus
UPPERMOST rule choices and physical bowl identities. The geometry was frozen
before policy execution. Smoke01 used layouts 0, 5, 8 and 13: one per
quadrant, two agreement and two disagreement cases, including swapped IDs
in each stratum. Both fixed demonstrations were executed from each exact
same state. Candidate 0 denotes the center-bowl prompt and candidate 1 the
near-plate prompt; intended physical IDs were assigned before rollout.

| Layout | Stratum | Candidate 0 intended bowl | Candidate 1 intended bowl | Native outcomes, both candidates |
| --- | --- | --- | --- | --- |
| 0 | Agreement | 1 | 2 | `[0,0]` |
| 5 | Disagreement | 2 | 1 | `[0,0]` |
| 8 | Disagreement | 1 | 2 | `[0,0]` |
| 13 | Agreement | 2 | 1 | `[0,0]` |

Both candidates succeeded 0/2 times in each stratum. There were zero
opposite one-hot outcome pairs. All eight episodes reached the fixed
562-control horizon; neither native predicate was ever true. Both hidden
rules therefore receive the same all-zero outcomes on these records, which
provides no rule-identifying feedback for a reader in this sample.

## Evidence and diagnosis

The allocated NVIDIA RPBZZZ6 probe passed. All 967 saved checkpoint tensors
loaded exactly; weights stayed frozen. The two prompt hashes are distinct
and each stayed fixed across its four scenes. Paired initial physics and
RGB/wrist/proprioceptive hashes match the admitted geometry. The run-local
audit verified public-array allowlists, shapes, hashes, recording/padding,
native stopping and hidden-rule-independent public bytes. Its immutable
smoke output has `motor_admission_passed: null`; the unsupported proposed
7/8 performance cutoff was subsequently removed from future readouts at
the user's direction.

Candidate 0 lifted physical bowl 2 in all four scenes, including layouts
0 and 8 where its intended bowl was 1. Candidate 1 lifted physical bowl 2
in layouts 0 and 5, but neither bowl moved materially in layouts 8 and 13.
No transported bowl reached the plate predicate. The [descriptive motion
readout](DIAGNOSIS.json) retains all quantities without a new success cutoff.
The [sequence sheet](evidence/smoke01/smoke01-sequences.png) samples all eight
videos; original videos and RGB remain unchanged.

The planned 32-episode panel was not launched after the smoke. More nearly
identical layouts would not by itself resolve the observed failure. A
targeted next question, if needed, is whether a previously successful native
state remains successful through the new reset/recording path. That control
has not been run. Geometry or prompts have not been tuned on these outcomes.

## Provenance, cost and retention

Smoke01 ran immutable commit `62246070fa11bc834608703279d15d52b8d4d44c`,
job `job-yv8dtfoqo5kb`, one Pro6K GPU, with exit 0. Submitted November 11
19:19:48 Shanghai; scheduler completion 19:24:51, duration 291 seconds.
Submission-to-completion conservative cost is 303 GPU-seconds.

Including the failed first geometry allocation, this stage used **532
scheduler GPU-seconds (0.148 h)** and **724 conservative GPU-seconds
(0.201 h)** of the additional 4 GPU-hour cap. No job is active.
[Accounting](EXECUTION.json) preserves all allocations and failures.

[Raw small receipts](evidence/smoke01/audit.json), private native traces,
configurations and logs are retained in this directory. Full public NPZs
and their 193 MB archive remain on Baige; two Jupyter downloads failed.
The verified 1.93 MB archive containing videos and all small receipts was
retrieved. [Artifact manifest](ARTIFACTS.json) records every public file and
array hash, both archive hashes, remote locations, and all local video
hashes. Videos stay outside Git; the sequence sheet is committed.

This is an adapted development study, not a released LIBERO-Gen benchmark
score, held-out generalization result, or evidence of learned history use.
