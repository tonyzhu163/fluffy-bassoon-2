# Target wording separates the missed success from the wrong-drawer rejection

Baige `job-vzmz8wa87siv` succeeded and released on Nov3 04:55:20 Shanghai.
Source `db1fc5c8ec32eea8550b768f9213629175ebf17f`. All109 new outputs,
140 paired labels,16 batch receipts,109 actual renderer message audits and62
exported artifact hashes verify. All4 verbatim1.6 repeats match exactly.
No new trajectories, robot controls, motor calls or SGD.

| Task wording | False Score5 /20 negative endpoints | Score5 /15 post-success endpoints | Valid raw first detections /5 | Valid two-probe >=4 /5 | False two-probe >=4 episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both clauses, cached1.6 |0|5|3|4|0|
| Base native instruction + shared holding permission |15|15|2|5|1|
| Target block only |3|9|1|4|0|
| Opening-depth clause only |0|0|0|5|1|

The prospective primary rule requires two scores>=4 exactly50 controls apart,
accepts at the second, and latches. Valid means first acceptance after the
native crossing and within100 controls. None passes the frozen screen of
zero false first acceptances and5/5 valid successful episodes. All4 conditions
preserve the same permission to hold the handle. Base is not byte-exact native.

The target-description block is associated with both the desired112 rejection
and the unwanted128 miss on this panel. With that block,112 has no confirmed
acceptance and128 scores1 at350/400/450/750. Without it,128 is confirmed at450
(native384), but112 is falsely confirmed at200. This holds both with and
without the depth clause. Thus removing the opening-depth sentence alone
does not recover128; relaxing target identification recovers it at the cost
of the diagnosed wrong-drawer failure. This is a controlled block ablation,
not proof of an internal visual mechanism or of one offending word.

The opening-depth clause changes score calibration markedly. Depth-only gives
Score4 to all15 post-success endpoints and to112 from150 onward, with no5
anywhere. Its zero false full scores therefore does not imply a useful reward:
threshold4 confirmation still accepts the wrong drawer. Removing depth from
the target-only block also gives premature raw5s to72/144/264 at200; two-probe
confirmation delays their first acceptance to250 on the available sparse grid.
It cannot establish that earlier unsampled probes would be safe.

The next targeted wording trial should retain an explicit wrong-drawer rule
while removing the presupposition that the target has other drawers below it.
The current target block says 'highest drawer in that stack, above every other
drawer';128 visually has a single drawer above cabinet doors. That is a
hypothesis to test, not an established explanation. Earlier composite rewrites
also missed128, so this result does not justify merely deleting that sentence
and declaring a fix.

All grades use the same6 videos/35 causal prefixes, full history, official
messages/sampler/views/model/parser and temperature0. Only112 has the complete
early50-control grid. The other5 cases each have4 selected probes; all are a
reused, error-enriched development panel. Native clocks retain their exact
scene/public-state replay checks and missing-full-MuJoCo-state caveat. There
is no independent validation, population precision or online first-success
qualification. No rubric, threshold or production code is promoted.

Worker charge756 GPU-s; scheduler running770; submission-to-terminal upper788,
below900. Prior3144 prefix/chunk workerGPU-s remain separate; cumulative3900.
No allocation remains live.

Evidence: [metrics](RESULT.json), [paired grades](PAIRED-SAMPLE.json),
[CPU qualification](CPU-CHECKS.json), [actual messages](scoring/PROMPTS.json),
[transfer verification](COPIED-HASHES.json), [scheduler](SCHEDULER-TERMINAL.json).
