# Visual completion wording applied; scoring blocked by Baige service

The user-approved completed-state rule is saved in
[revision1_3.json](revision1_3.json). For the existing OpenDrawer bank it
requires a visibly fully open target before Score5. A gap, exposed contents
or obvious partial extension is insufficient;1–4 still describe progress.
The approved flush-or-nearly-flush closing definition is saved under
CloseDrawer, but is not sent to an opening-task scorer.

The14-video comparison is prepared and deployed through Git at `745f10f1`:
the same10 failures and4 success controls, revision1.1 versus1.3,28 planned
graded labels. No new selection, trajectories, binary-output arm or full-bank
rescore. Model,64-frame sampling and expected-grade mapping are unchanged.
The production reward and active runs have not been modified.

Allocated Linux CPU preparation passes: exact input hashes, unchanged shared
text and baseline, correct official1–5 system, one video placeholder, inactive
CloseDrawer reference, zero simulator calls. The existing terminal worker is
byte-identical to the prior eight-test qualification. Raw validation is in
[CPU-CHECKS.json](CPU-CHECKS.json).

GPU scoring is **not complete**. The first UI submission produced no verified
job ID; the filtered task list subsequently reported no matching resource.
A second submission, left open to observe its response, explicitly failed:
"系统服务异常，请稍后重试" (system service exception, retry later), error
`02e0566f-c186-42f9-843a-5bf090b7bb42`. The shared output contains CPU input
artifacts only, with no allocation/scorer results or observed GPU charge.
This is a platform failure, not a rubric score or a GPU inference failure.
The exact observed UI text is retained in [SERVICE-ERROR.json](SERVICE-ERROR.json);
the saved screenshot shows the form after the transient toast disappeared.

No further submission was made after that explicit failure. Any later retry
must first verify that no matching job exists, then use the prepared one-GPU,
15-minute/900GPU-second contract. Do not claim zero possible charge from an
unverified first request, or treat this draft as a measured improvement.
Previous20-video results remain in the r1 folder unchanged.
