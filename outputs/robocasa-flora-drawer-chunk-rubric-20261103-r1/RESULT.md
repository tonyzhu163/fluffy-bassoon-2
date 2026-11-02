# Recent chunks worsen recognition under the fixed explicit drawer rubric

Baige job `job-5fr1u0s7esch` succeeded and released its single GPU on Nov3
04:20:16 Shanghai. Source `ee9bd8358202745b01aa97bfa02c86f18c3aafee`.
All74 new outputs completed:35 endpoints under each window plus4 unchanged
full-prefix1.6 repeats. All4 hard grades match exactly. The70 exported artifact
hashes,11 batch receipts,74 unique IDs,74 actual prompt hashes and74 actual
frame-read records match locally. Annotated records preserve the original
worker grades; original raw outputs are retained separately. All8 CPU window
decodes match direct source pixels exactly. No new robot controls, trajectories,
motor calls or SGD.

| Fixed1.6 visual input | False Score5 /20 negative endpoints | Score5 /15 post-success endpoints | Raw valid first detections /5 | Two-probe valid detections /5 |
| --- | ---: | ---: | ---: | ---: |
| Cached full causal prefix |0|5|3|1|
| Last50 controls (2.5 seconds) |0|1|0|0|
| Last100 controls (5 seconds) |0|1|1|0|

Valid means first acceptance after native first success and within100 controls.
The predeclared primary rule requires2 Score5 probes exactly50 controls apart,
accepts at the second and latches thereafter. Neither chunk confirms any of
the5 successful trajectories. Neither passes the development screen.

The50-control window gives its sole5 to264 at750, far beyond native success216.
The100-control window gives its sole5 to264 at300, a valid raw delay84 controls,
but no second5 confirms it. All other success trajectories receive no full
score at their sampled post-success endpoints.112 remains rejected at every
sampled endpoint. These zero false positives come with severely reduced
recognition, so they are not a successful reward repair.

128 remains unresolved. Its350/400/450/750 scores are2/1/1/1 for window50 and
1/4/1/1 for window100 (native first success384). A post-hoc CPU check of two
Score>=4 probes recovers0/5 cases for window50 and3/5 for window100, with no
observed false trigger. It still misses72 and128. These fitted-panel checks
do not qualify a different threshold. The original full-prefix1.6 threshold4
confirmation had4/5 valid detections on this same sparse panel.

The tested change removes early trajectory context and changes the duration-
aware sampler's total frames to24 or50, from up to64 for full prefixes.
It does not isolate context length, initial-view context, movement evidence
or frame density as a cause. Actual reads are bounded by each causal endpoint;
the negative result is not an accidental future-frame or full-prefix read.
The exact official grading messages, explicit task text, model and source
videos are unchanged. Only the run-local reader is adapted; production code
is unchanged.

This selected six-trajectory panel still has a full early50-control grid only
for112; the other5 cases have4 probes each. Native clocks retain the earlier
scene/state replay checks and their missing-full-MuJoCo-state limitation.
There is no population precision, online first-success or production reward
qualification. No rubric, window or threshold is promoted.

The next wording experiment should return to full prefixes and separate the
target-identification clause from the opening-depth clause. The current
combined1.6 criterion rejects112 but also loses128; system placement and
recent chunks did not resolve that tradeoff. Testing those clauses separately
on the existing small failure panel is more informative than another combined
rewrite. This is an untested next step, not a demonstrated repair.

Worker charge598 GPU-seconds; scheduler running610 and submission-to-terminal
upper629, below900. Prior2546 prefix-attempt worker GPU-seconds remain separate;
all5 prefix/chunk attempts total3144 worker GPU-seconds. No live allocation.

Evidence: [metrics](RESULT.json), [paired grades](PAIRED-SAMPLE.json),
[CPU checks](CPU-CHECKS.json), [threshold follow-up](CPU-FOLLOWUP.json),
[actual prompts](scoring/PROMPTS.json), [expected frame mapping](EXPECTED-FRAMES.json),
[copied hashes](COPIED-HASHES.json), [scheduler closure](SCHEDULER-TERMINAL.json).
