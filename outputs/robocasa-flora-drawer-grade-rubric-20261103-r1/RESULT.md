# Explicit grade meanings recover successes but also accept every failure case

Baige `job-cocatpwa8lsp` succeeded and released on Nov3 at06:59:28 Shanghai.
Source `2cf35e607db0a8b32f6d33c1a931588ab24791b2`. All94 new grades,
135 paired labels,13 receipts,94 actual rendered-message audits and54
exported artifact hashes verify. All4 verbatim reference repeats match.
Only10 retained videos/45 causal prefixes were used;45 reference labels
were recovered,90 candidate labels and4 repeats were newly scored.

| Rubric | False Score5 /27 negative endpoints | Score5 /18 post-success endpoints | Valid raw first detections /6 | Valid confirmed Score5 /6 | False confirmed Score5 episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference1.6, cached |1|5|3|1|0|
| Visual criterion, generic grades |18|11|3|4|4|
| Same criterion, explicit grades1–5 |19|15|3|5|4|

Adding the grade-anchor block raises post-success full scores11→15 and adds
one false full score18→19. Both candidates falsely confirm all4 never-success
trajectories: wrong-drawer112/376,50.22%-open256 and33.96%-open336. Neither
passes the prespecified screen. Do not report the gain in correctly timed
confirmed successes without these false acceptances and premature endpoints.

Both candidates preserve the native user instruction and use the same system
success criterion: identify the top requested-side drawer, judge its box's
outward displacement, require52.25% depth, allow holding and less than maximum
extension, and count the best state achieved in any view. Both align generic
messages to first success. Only visual_grades adds meanings for scores1–5.
This within-run contrast isolates that complete grade-anchor block; it does
not isolate the cause of differences from1.6 or the previous numerical tasks.
The model, processor, score grammar/parser, sampler, views, temperature and
production worker are unchanged. The run-local message adapter is the tested
intervention, not an official unmodified prompt.

| Case | Native first success | Probed controls | Visual criterion scores | Explicit grade scores |
| --- | --- | --- | --- | --- |
|112|Never|50–750, every50|1,2,1,2,2,5,5,5,5,5,5,5,5,5,5|1,2,1,5,2,5,5,5,5,5,5,5,5,5,5|
|256|Never|700,750|5,5|5,5|
|336|Never|700,750|5,5|5,5|
|376|Never|700,750|5,5|5,5|
|128|384|350,400,450,750|1,1,1,1|1,1,1,5|
|144|222|200,250,300,750|2,3,5,5|2,5,5,5|
|216|212|200,250,300,750|2,5,5,5|2,5,5,5|

Explicit grading recognises128 only at750,366 controls after native384;
there is no timely detection or pair50 controls apart. This is a late terminal
recognition, not a repaired first-success detector. Both candidates give5 at
200 for72 and264, before native208/216. Their250 confirmations therefore use
one premature positive. The primary screen separately requires0 false full
scores, so these cannot be hidden by reporting only the later acceptance time.

The primary rule remains the second of two Score5 probes exactly50 controls
apart, then latch; require0 false Score5 endpoints,0 false accepted episodes,
6/6 successes accepted after native first success and within100 controls,
and4/4 matching repeats. The diagnostic >=4 rule yields4/6 and5/6 valid cases,
both with4 false episodes. No threshold was retuned or promoted.

No candidate fixes the reward, and none enters PPO. The accompanying
[CPU visual audit](../robocasa-flora-drawer-visual-audit-20261103-r1/RESULT.md)
shows poorly framed/occluded target regions in selected stills, including the
near-boundary failure. It does not prove missing information across the entire
sampled video or explain Flora's internal score. The next cheap diagnosis is
to inspect the actual64 sampled collages for these failures and128 before
another wording search. This conclusion changes no frozen input or result.

This reused, error-enriched panel is not independent validation or a population
precision estimate. Only112 has the complete50-control probe grid; the3 added
never-success cases have700/750 only. Retained-action replay matches scene
metadata/public robot states but lacks saved full MuJoCo initial state. No
online first-success detector or production reward is qualified.

Worker charge643GPU-s; scheduler running651s; submission-to-terminal upper661s,
all below900. Prior6033 workerGPU-s stay separate; cumulative6676 for the
prefix/chunk/clause/wording attempts.0 new trajectories, policy controls,
motor calls or optimizer steps. No GPU allocation remains live.

Evidence: [metrics](RESULT.json), [paired grades](PAIRED-SAMPLE.json),
[actual messages](scoring/PROMPTS.json), [CPU checks](CPU-CHECKS.json),
[hash verification](COPIED-HASHES.json), [scheduler](SCHEDULER-TERMINAL.json),
[exact shared criterion](visual_criterion.json), [grade meanings](visual_grades.json).
