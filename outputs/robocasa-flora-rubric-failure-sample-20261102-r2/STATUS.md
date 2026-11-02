# Visible drawer completion: measured result

Baige `job-of3hcwyigjtu` succeeded and released its GPU at23:04:39 Shanghai
on Nov2, source `cbcffb71`. All28 grades on14 OpenDrawer videos are complete.
The new visual-completion rule gives8/10 false successes versus7/10 for prior1.1,
with3/4 success controls retained by both: zero corrections and one new false
success. It does not improve these selected examples. See [RESULT.md](RESULT.md).

The user-approved completed-state rule is saved in
[revision1_3.json](revision1_3.json). For the existing OpenDrawer bank it
requires a visibly fully open target before Score5. A gap, exposed contents
or obvious partial extension is insufficient;1–4 still describe progress.
The approved flush-or-nearly-flush closing definition is saved under
CloseDrawer, but is not sent to an opening-task scorer.

The14-video comparison uses the same10 failures and4 success controls,
revision1.1 versus1.3,28 completed graded labels. No new selection, trajectories,
binary-output arm or full-bank rescore. Model,64-frame sampling and
expected-grade mapping are unchanged. Production reward and active robot
runs were not modified.

Allocated Linux CPU preparation passes: exact input hashes, unchanged shared
text and baseline, correct official1–5 system, one video placeholder, inactive
CloseDrawer reference, zero simulator calls. The terminal worker is unchanged
from the prior eight-test qualification. The new detached source was reviewed
on current main; launch/scoring code, rubrics and engine are byte-identical to
the previously qualified745f10f1 version. All42 video hashes were checked,
and no allocation, scorer directory or model result predated this retry.
Raw validation is in [CPU-CHECKS.json](CPU-CHECKS.json) and
[RETRY-CPU-CHECKS.json](RETRY-CPU-CHECKS.json). Copied batch hashes match
the Linux receipts; all14 baseline hard grades reproduce the earlier sample.

Earlier submissions failed before any verified allocation. The first request
produced no job ID and the filtered list returned no matching resource. The
second explicitly returned "系统服务异常，请稍后重试" (system service exception),
error `02e0566f-c186-42f9-843a-5bf090b7bb42`. Those observations remain in
[SUBMISSION-ATTEMPT.json](SUBMISSION-ATTEMPT.json) and
[SERVICE-ERROR.json](SERVICE-ERROR.json); the saved screenshot shows the form
after the transient toast disappeared. They are not model inference failures
or rubric scores. No impossible zero-charge claim is inferred from an
unverified request.

This is the only verified GPU allocation for r2. It used531 worker GPU-seconds
and at most563 submission-to-terminal GPU-seconds, below the900 cap. No
successor was launched. Previous20-video results remain in r1 unchanged.
The candidate is not qualified for production reward or first-success timing.
