# System placement and first-success alignment still do not repair OpenDrawer

Baige job `job-vrj2vlclboxi` succeeded and released its single GPU at
03:46:18 Shanghai on Nov3. Source `b20f3672f6fbec3457341ea7d2b394f3cc958500`.
All74 new outputs completed:35 prefixes for each candidate plus4 unchanged
1.6 repeats. Recovering140 reference grades gives210 paired grades plus4
separate repeats. All4 repeated hard grades match exactly. The47 exported
artifact hashes,11 batch receipts,74 unique IDs,frame counts and74 actual
renderer message hashes match. Actual structured prompts equal the CPU
prepared prompts. No new trajectories, simulator controls, motor calls or SGD.

| Criterion | False Score5 /20 negative prefixes | Post-success Score5 /15 | Correctly timed successful episodes /5 |
| --- | ---: | ---: | ---: |
| Cached1.1 |10|14|3|
| Cached1.6 |0|5|3|
| Cached1.9 |1|8|3|
| Cached1.10 |3|9|1|
|1.11: native task plus verbatim1.6 criterion in system |5|12|3|
|1.12: native task plus verbatim1.10 system criterion, aligned first-success messages |4|11|1|

Correctly timed means the first sampled5 occurs after native first success
and within100 controls. The frozen screen additionally requires0 sampled
false5s and all5 successful episodes detected. Neither candidate passes.

1.11 recognises successful128 at450 (native384), but still grades its400 and
750 prefixes1. It reintroduces wrong-drawer112 full scores at450/550/750 and
premature scores on72/264 at200 (native208/216). Its valid detections are
64/128/144. 1.12 still misses all sampled128 successes, accepts wrong-drawer112
at200, and accepts72/144/264 at200 before native208/222/216. Only64 has a
correctly timed first sampled5. The changes are composite prompt/context
changes; this does not isolate a role-only or single-clause effect.

The production worker source is unchanged. These are custom grading messages
using the official terminal head, sampler,1–5 output grammar and parsing;
they are not the exact official prompt. The renderer audit confirms the new
messages actually reached scoring. This rules out an unapplied adapter as the
explanation for this experiment; it does not explain the model's reasoning.

CPU-only post-hoc checks also leave128 unresolved. Requiring2 Score5 probes
exactly50 controls apart gives0 observed false accepted episodes and4/5
correctly timed successes under either candidate, missing128. These are
separate overlapping video-prefix judgments, not consecutive video frames.
Selecting a P5 cutoff above every sampled negative full score gives4/5 under
1.11 (>0.4598) and3/5 under1.12 (>0.7453), again missing128. P5 is the official
score-token probability calculation, not a calibrated success probability.
The thresholds and confirmation analysis use this same development panel;
they are not independent validation or production choices.

Only112 has a complete50-control early-prefix grid; successful cases have
four selected probes each. The native clocks retain the earlier exact
scene/state-boundary replay checks and missing-full-MuJoCo-state limitation.
No population precision or online first-success qualification follows.

The next justified small experiment holds the explicit1.6 criterion fixed
and tests recent causal video chunks.128's1→5→1 response across longer
prefixes motivates checking the visual evidence presented with the rubric.
Compare the last50 and100 controls on the same35 endpoints, retain the same
views/model/sampler/score grammar, and explicitly qualify absolute frame
bounds. Read out both raw scores and a predeclared two-probe confirmation
rule with a latched first success. This is an untested next step; no new
allocation is active and neither current candidate is promoted.

Worker charge670 GPU-seconds; scheduler running691 and submission-to-terminal
upper703, all below900. Prior1876 prefix-attempt worker GPU-seconds remain
separate; the four prefix attempts total2546 worker GPU-seconds.

Evidence: [metrics](RESULT.json), [paired grades](PAIRED-SAMPLE.json),
[actual prompts](scoring/PROMPTS.json), [CPU post-hoc checks](POSTHOC.json),
[transfer verification](COPIED-HASHES.json), [scheduler closure](SCHEDULER-TERMINAL.json).
