# Frozen BPP on custom relational layouts

The eight-episode smoke produced **zero native bowl-on-plate successes**.
Both fixed motor prompts failed on all four tested custom layouts, despite
their earlier 20/20 success per candidate on native initializations. This
stage has not supplied the executable alternatives needed for the proposed
history experiment. No history reader was trained or evaluated.

The controller did move: six trajectories lifted a bowl by 5.6–13.0 cm.
The traces and videos show wrong-object choices, transport away from the
plate, and empty-space reaches. These are substantive physical failures;
the result is not explained solely by an incorrect reward label. A targeted
native replay through the same recorder then succeeded with
both candidates, supporting custom geometry transfer as the leading
explanation. This single-state control does not rule out every interaction
between the reset/recording path and a changed scene.

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
identical layouts would not by itself resolve the observed failure. The targeted
native replay below resolved the immediate reset/recorder
question on one previously successful state. Geometry or prompts have not
been tuned on these outcomes.

## Targeted native replay

Control01 replayed native initialization 0 through the same raw recorder
with both original fixed prompts. Candidate 0 reached native `[1,0]` after
111 raw controls, and candidate 1 reached `[0,1]` after 195. Both initial
outer observation hashes matched the earlier Pro6K panel03 state 0 exactly;
both prompt hashes matched custom smoke01 exactly. The native source bank
hash was unchanged, and all 967 checkpoint tensors again loaded exactly.
The two recorded initial physics states were identical. No hidden rule,
custom geometry admission or performance cutoff was used in this control.

The [raw control result](evidence/control01/motor/result.json) and
[video sequence sheet](evidence/control01/control01-sequences.png) show
successful executions of the two distinct intended native goals. This
supports scene transfer as the leading explanation for the custom failures.
It establishes native replay through this wrapper, not absence of every
wrapper/scene interaction or failure on every possible custom layout.
The second candidate's policy RNG stream may differ from the previous
20-episode-per-candidate panel; no trajectory-equality claim is made.

The parent agent independently checked both raw native traces, exact prior
observation and smoke prompt hashes, paired states and all 967 tensors,
and visually reviewed both native videos. Its [review image](evidence/control01/parent-native-control-review.png) agrees with the two native goal outcomes.
No separate release requirement is inferred from the final video frames.

## Provenance, cost and retention

Smoke01 ran immutable commit `62246070fa11bc834608703279d15d52b8d4d44c`,
job `job-yv8dtfoqo5kb`, one Pro6K GPU, with exit 0. Submitted November 11
19:19:48 Shanghai; scheduler completion 19:24:51, duration 291 seconds.
Submission-to-completion conservative cost is 303 GPU-seconds.

Control01 ran commit `21dbdd675eaa2accdb0d485167fd34098d32c644`,
job `job-zgcjkfcb2rtz`, also exit 0. Submitted 19:37:44 Shanghai; completed
19:40:35, 151 scheduler seconds and 171 conservative seconds.

Including the failed first geometry allocation, this stage used **683
scheduler GPU-seconds (0.190 h)** and **895 conservative GPU-seconds
(0.249 h)** of the additional 4 GPU-hour cap. No job is active.
[Accounting](EXECUTION.json) preserves all allocations and failures.

[Raw small receipts](evidence/smoke01/audit.json), private native traces,
configurations and logs are retained in this directory. Full public NPZs
and their 193 MB archive remain on Baige; two Jupyter downloads failed.
The verified 1.93 MB archive containing videos and all small receipts was
retrieved. [Artifact manifest](ARTIFACTS.json) records every public file and
array hash, both archive hashes, remote locations, and all local video
hashes. The 167 KB native-control small archive was also verified and
retrieved; its two public NPZs (44.85 MB) remain remote with tracked hashes.
Videos stay outside Git; both sequence sheets are committed.

This is an adapted development study, not a released LIBERO-Gen benchmark
score, held-out generalization result, or evidence of learned history use.
