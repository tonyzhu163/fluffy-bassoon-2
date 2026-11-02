# Neither positive location nor negative exclusion repairs the drawer tradeoff

Baige `job-ksbimyzk0wkf` succeeded and released Nov3 05:42:22 Shanghai.
Source `274ec5ecb2ed965d184d5f8e7602a533461c4908`. All74 new grades,175
paired labels,11 receipts,74 actual renderer audits and47 exported artifact
hashes verify. All4 verbatim1.6 repeat grades match exactly. The2×2 conditions
share the exact native instruction, half-depth sentence and holding permission.

| Location sentence | Exclusion sentence | False Score5 /20 negative endpoints | Score5 /15 post-success endpoints | Valid raw first detections /5 | Valid two-probe >=4 /5 | False two-probe >=4 episodes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Absent | Absent, cached depth_only |0|0|0|5|1|
| Present | Absent, new location_only |0|0|0|4|1|
| Absent | Present, new exclusion_only |5|4|3|4|1|
| Present | Present, cached compact_target |2|3|2|4|1|

The original1.6 reference remains4/5 valid two-probe detections with0 false
accepted episodes, still missing128. No new condition passes the frozen screen
of0 false accepted episodes and5/5 successful cases accepted after native
success and within100 controls. Confirmation requires two scores>=4 exactly50
controls apart, accepts at the second and latches; no thresholds were retuned.

The location sentence independently suppresses128 under this fixed base:
350/400/450/750 scores change from2/4/4/4 without either sentence to1/2/3/3
with location alone. It still falsely confirms112 at200. Thus the loss does
not require the negative exclusion sentence. This establishes the effect of
this complete location sentence in these prompts, not an internal visual
explanation or an effect of one token.

Exclusion alone gives128 scores1/5/3/4: it recognises the400 prefix once after
native success384, but the next450 grade falls to3, so there is no confirmed
detection. It falsely confirms112 at350 and gives that wrong-drawer video5
at500/550/600/650/750. Location plus exclusion misses128 throughout the valid
time window and also confirms112 at350. The failure is therefore not repaired
by either deleting the negative clause or using it without explicit location.

The cached factorial conditions have identical public video/prefix references,
exact task bytes and verified source hashes; their source runs reproduce the
same four1.6 anchors. Only the two new conditions are fresh inference in this
allocation. Do not describe this as175 new grades or an independent replication.

No rubric or production threshold is promoted. The next evidence gap is the
failure panel itself:112 is its only never-success trajectory. Diagnose four
additional retained terminal failures from the original14-video sample on
Linux CPU, checking all saved scene/state boundaries and distinguishing never
success from success later undone. This adds no GPU work or new motor calls
and should precede further broad wording search.

The reused6-video/35-prefix panel remains sparse before native success in all
cases except112; each successful case has4 selected probes. Native clocks have
metadata/public-state replay qualification but no saved full MuJoCo state.
There is no independent validation, population-precision or online first-
success qualification. No full634 replay, new robot controls, trajectories,
motor calls or optimizer updates were performed in this scoring run.

Worker charge698GPU-s; scheduler running710s; submission-to-terminal upper721s,
all below900. Prior4587 workerGPU-s remain separately charged; cumulative5285
for prefix/chunk/clause/wording attempts. No GPU allocation remains live.

Evidence: [metrics](RESULT.json), [all paired grades](PAIRED-SAMPLE.json),
[actual messages](scoring/PROMPTS.json), [CPU checks](CPU-CHECKS.json),
[transfer verification](COPIED-HASHES.json), [scheduler](SCHEDULER-TERMINAL.json).
