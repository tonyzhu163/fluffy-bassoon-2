# First real PPO iteration verified

Baige Pro6K job`job-rxih2makcwv4`, immutable source`bb4c905a`, completed
`global_step_1` at2026-11-06 02:34:50Shanghai. All four saved actor sidecars
contain20optimizer steps. The17recorded checkpoint data files (5nativeDCP,
12worker sidecars) total11,177,195,667bytes and match all recorded SHA256s.
The boundary JSON is the18th file. DCP metadata contains675model entries,
594optimizer-state fields,26optimizer-group fields,7scheduler fields and RNG.

All four first-replay logprob maxabs errors are zero. Every rank's first
backward has190finite policy and8finite critic gradients. All four frozen-VLM
before/after signatures match. Native mixedBF16, fullFP32vision/norms,
micro64/accum8/world4/global2048/fiveepochs and original loss/schedule remain.

Four source rounds charged409,600requested/conservative motor commands; no
measured valid-action mask is available on this native fast path. These are not
measured executed actions or physics integration substeps. [Accounting scope](CONTROL-ACCOUNTING.md).
Native timings report
2778.9sec collection and532.2sec actor-training wall time (530.6sec native
run_training). The13.935sec interval between first DCP-file completion and
boundary completion omits work before that file completed, so it is not a full
I/O latency measurement. The qualification wrapper will time that operation.

The training metric`success_once=.17773438` is a rollout metric, not heldout
evaluation or a comparison against Sidney's published SFT baseline. This
engineering pilot is not six-task corpus admission or the matched group8 arm.
The second PPO iteration is still running; only complete boundaries support
recovery, with a fresh source reset rather than mid-physics continuation.

[Full-state integrity](CHECKPOINT1-INTEGRITY.json),
[boundary](checkpoint1/pilot-boundary.json), [native metrics](iteration1-metrics.txt).
