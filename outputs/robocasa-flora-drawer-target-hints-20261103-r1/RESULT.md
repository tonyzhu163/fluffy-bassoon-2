# Minimal target hints do not repair OpenDrawer first-success scoring

Baige `job-ojya2orpcxik` succeeded and released at 08:07:16 Shanghai on Nov 3.
Source `74efd591fc0286f57691836927b0ebaaa015d7d0`. All 82 new outputs,
104 paired labels,13 receipts,82 actual rendered-prompt audits,82 actual raw
RGB-input audits and 67 exported artifact hashes verify. All 4 reference
repeats match. Only 10 retained videos / 26 causal prefixes were used, with 26
cached reference labels; 78 candidate labels and 4 repeats are newly scored.

| Rubric | False Score 5 /14 negative endpoints | Score 5 /12 post-success endpoints | Timely raw first 5 /6 | Timely confirmed 5 /6 | False confirmed 5 episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
|revision1_6|1|4|3|1|0|
|native_boundary|0|3|3|0|0|
|initial_target_lock|1|6|5|1|0|
|shallow_target_type|2|4|3|1|0|

None passes the prespecified two-Score 5 confirmation screen. Replacing only
1.6's approximate depth sentence with the native 52.25% threshold removes the
reference's false 5 at 336@700, but leaves only 3 true full-score endpoints and
no confirmed success. Adding initial-frame target tracking separately raises
true full scores 3→6 and timely raw detections 3→5, while introducing a false5
for wrong-drawer 376@700. It confirms only case 64 within 100 controls; success is
not repaired by this sensitivity gain. The shallow-drawer hint adds false 5s
at 72@200 (8 controls early) and 376@700 and loses case 72's post-success recognition.

128 remains 1/1/1 at 350/400/450 under every candidate, including the hint
that the target can be a shallow drawer directly above cabinet doors. Its
native crossing is 384. The actual 64 raw RGB inputs during scoring reproduce
the Linux CPU reference hashes exactly. For 128@450 that reference also
matches all 64 frames of the preceding visual audit, which contains 10 source
frames after native success. This excludes different raw decoded inputs as
an explanation for this run's miss. It does not establish the information
retained in post-processor tensors or Flora's internal interpretation.

| Case | Native first success | Controls | Reference1.6 | Exact boundary | Initial target lock | Shallow target type |
| --- | --- | --- | --- | --- | --- | --- |
|112|Never|300,350|1/1|3/3|3/3|1/3|
|256|Never|700,750|4/4|4/4|4/4|4/4|
|336|Never|700,750|5/2|4/4|4/2|4/4|
|376|Never|700,750|4/4|4/4|5/4|5/4|
|64|542|500,550,600|1/5/5|1/3/5|1/5/5|1/5/5|
|72|208|200,250,300|4/5/1|4/5/2|4/5/4|5/1/1|
|128|384|350,400,450|1/1/1|1/1/1|1/1/1|1/1/1|
|144|222|200,250,300|4/4/4|4/4/4|4/4/5|4/4/5|
|216|212|200,250,300|3/4/2|3/3/2|3/4/5|1/1/1|
|264|216|200,250,300|4/4/5|4/4/5|4/4/5|4/4/5|

Exact boundary is a single replacement in1.6: 'roughly half its front-to-back
depth' becomes 'at least 52.25% of its front-to-back depth, just over half'.
The other two variants each add one sentence to that exact base: tracking
the initial drawer front, or its possible shallow-above-cabinet-doors form.
These texts were frozen in the preceding audit before this inference. The
official system/final messages, 1–5 grammar/parser, model, sampler, three views,
processor, temperature 0 and production worker remain unchanged. The run-local
read-only frame wrapper passes the same objects onward after hashing them;
it changes no pixel or sampling index. No per-case simulator answer enters
any candidate message.

The primary rule accepts the second of two Score 5 probes exactly 50 controls
apart, then latches. It requires 0 false Score 5 endpoints, 0 false accepted
episodes, 6/6 successes detected after native crossing and within 100 controls,
and 4/4 matching repeats. A sparse panel with isolated false 5s can report 0
false confirmed episodes while still failing the endpoint condition. The
>=4 diagnostic yields 3/6, 5/6, 3/6 timely detections for the 3 fresh variants,
with 3, 2, 3 false episodes respectively. It is not an adopted fallback.

This is a reused, failure-enriched early screen, not the full 45-prefix panel,
independent validation, a population precision estimate or an online detector.
112 has only 300/350 here, so its earlier behavior remains unscreened. No
candidate is promoted, enters PPO or justifies scoring the full 634 videos.
The native height-filter/uppermost equivalence is qualified only for these
selected layouts; replay still lacks saved full MuJoCo initial state.

The next useful cheap diagnosis is the official processor path for 128: keep
these frozen source frames, record the actual processor configuration and
video grid, and reconstruct or inspect what target detail is retained before
another wording change. This is an input diagnostic, not a proposed reward
or proof that rescaling caused the miss. A repair remains unverified.

Worker charge 674 GPU-s; scheduler running 690s; submission-to-terminal upper 709s,
all below 900. Prior 6676 worker GPU-s stay separate, cumulative 7350 for the
prefix/chunk/clause/wording attempts. 0 new trajectories, policy controls,
motor calls or optimizer steps. No allocation remains live.

Evidence: [metrics](RESULT.json), [paired grades](PAIRED-SAMPLE.json),
[actual messages](scoring/PROMPTS.json), [CPU references](FRAME-REFERENCE.json),
[CPU qualification](CPU-CHECKS.json), [artifact verification](COPIED-HASHES.json),
[scheduler and costs](SCHEDULER-TERMINAL.json), [completion screenshot](completed-job.png).
