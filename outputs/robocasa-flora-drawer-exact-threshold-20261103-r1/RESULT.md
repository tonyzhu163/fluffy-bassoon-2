# Exact numerical opening language suppresses successes as well as failures

Baige `job-ylc4xu1qmk6n` succeeded and released on Nov3 at06:35:41 Shanghai.
Source `41a0e3c76d7d60debd712837bf85c32f1d702226`. All104 new grades,
135 paired labels,15 receipts,104 actual rendered-message audits and60
exported artifact hashes verify. All4 repeated reference grades reproduce.
The sample contains10 retained trajectories and45 causal prefixes, not the
full634-video bank. There were0 new trajectories, policy controls, motor calls
or optimizer updates.

| Task wording | False Score5 /27 negative probes | Score5 /18 post-success probes | Correctly timed raw first detections /6 | Correctly timed confirmed Score5 /6 | Diagnostic confirmed >=4: valid /6; false episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Verbatim1.6,35 cached plus10 new |1|5|3|1|4;2|
| Direct fraction |0|0|0|0|0;1|
| Explicit boundary |0|0|0|0|2;3|

The frozen primary rule requires two Score5 probes exactly50 controls apart,
accepts the second and latches. A pass requires no false Score5 endpoints or
false accepted episodes, and all6 ever-success trajectories accepted after
native success and within100 controls. Neither candidate passes. Both reject
all true full successes as well as all false full successes. Score>=4 remains
a diagnostic, not a newly selected success threshold.

The exact candidate texts are:

> Pull the top {side} drawer out by at least 52.25% of its front-to-back depth.
> Holding the handle is allowed.

> Open the top {side} drawer. Success requires that same drawer to extend at
> least 52.25% of its front-to-back depth beyond its closed position. Opening
> it only halfway is still short of the goal. Holding the handle is allowed.

These candidates change several parts of1.6 together; this comparison does
not isolate the effect of the numerical token alone. Official system/final
messages,1–5 output grammar, parser, sampler, views, full-prefix reader, model
and temperature remain unchanged. All10 targets were audited as uppermost in
their own stacks, including the4 new scene resets in this run. That geometry
equivalence is limited to these layouts.

The added failures show why the older reference and a lower success cutoff
also remain unsafe:

| Case | Native outcome | Controls | Reference1.6 | Direct fraction | Explicit boundary |
| --- | --- | --- | --- | --- | --- |
|256|Never succeeds; target stops at50.22% depth|700,750|4,4|3,3|3,3|
|336|Never succeeds; target stops at33.96% depth|700,750|5,2|3,3|4,4|
|376|Never succeeds; another drawer opens|700,750|4,4|4,4|4,4|
|216|First succeeds at212, then closes|200,250,300,750|3,4,2,1|3,3,3,4|3,3,4,4|
|128|First succeeds at384|350,400,450,750|1,1,1,1|2,3,3,3|2,3,3,3|

For216, the terminal failure is not an ever-success false positive: native
success already occurred at212. For256, 'roughly half' is an insufficient
binary specification because native success requires52.25%, although this
does not establish Flora's internal reason for the grade. The exact texts
correct its full-score classification by withholding5 everywhere; they do not
repair discrimination. Wrong-drawer376 still receives4 throughout under both.

No rubric, cutoff or production reward is promoted. A useful next rubric test
would need visually stated distinctions between wrong-target movement,
partial opening and successful opening, with explicit meanings for the graded
outputs. It must retain the near-boundary and wrong-drawer failures alongside
success controls; zero false full scores alone is insufficient. This is an
untested direction, not an established fix, and the exact predicate must not
be silently replaced by 'any visible opening' or 'halfway'.

This reused, error-enriched panel gives no population precision estimate or
independent validation. Only112 has the complete50-control probe grid; other
cases have selected probes, and the3 new never-success cases have only700/750.
Replay qualification checks saved scene metadata and public robot states,
not a saved full MuJoCo state. No online first-success detector is qualified.

Worker charge748GPU-s; scheduler running750s; submission-to-terminal upper767s,
all below900. Prior5285 workerGPU-s remain separate; cumulative6033 for the
prefix/chunk/clause/wording attempts. No allocation remains live.

Evidence: [metrics](RESULT.json), [paired grades](PAIRED-SAMPLE.json),
[actual messages](scoring/PROMPTS.json), [CPU checks](CPU-CHECKS.json),
[transfer verification](COPIED-HASHES.json), [scheduler](SCHEDULER-TERMINAL.json).
