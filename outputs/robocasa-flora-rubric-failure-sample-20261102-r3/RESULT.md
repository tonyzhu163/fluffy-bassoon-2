# Uppermost-drawer and endpoint rubric: no improvement

Baige job-v8f8eobei5an succeeded and released its single GPU at23:56:29
Shanghai Nov2. Frozen source8c3eea04 completed42 full-video grades,6 causal
prefix grades and4 official compact rationales. All copied raw-file hashes
match Linux. Cost603 worker /612 running /619 all-in upper GPU-seconds,
below900. Zero new robot actions or learner updates.

| Criterion | Full-score terminal failures /10 | Accepted success controls /4 |
| --- | ---: | ---: |
| Verbatim1.1 control |7|3|
|1.4 explicit uppermost target |7|2|
|1.5 uppermost + native opening + endpoint |7|2|

Both candidates correct failure88 (5→1), introduce false success272 (4→5),
and lose true success128 (5→1). They emit the same hard grade on all14 videos.
Neither fixes the diagnosed wrong-lower-drawer case112 or the undone opening
in64. Baseline hard grades reproduce13/14: case24 changes2→1, leaving every
full-success decision unchanged. Do not claim bit-exact14/14 grade reproduction.

The four baseline rationale calls all give Score5. For112,64 and128 the text
is the same: “The robot opened the left drawer completely.” These generic
posthoc explanations do not diagnose the simulator error and are not truth.

The six1.5 causal prefixes are more informative:64 is rejected at500 controls,
accepted at550 (native first success542), but still accepted at650 after native
success has been undone.144 is below full score at200 and receives5 at250 and
750 (native first success222). Both sampled first-crossing brackets contain
the true crossing, but this is only two development trajectories. The650
prefix demonstrates a remaining endpoint error. This does not qualify Flora
first-success precision or timing.

No tested candidate is promoted. The next small iteration will put the exact
target and final-state condition directly in a short task instruction, instead
of appending more rules to “Open the left/right drawer.” Start with112,64 and
the four retained success controls. No full634-video bank is rescored.
