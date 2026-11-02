# Test the exact native opening boundary with shorter target language

The previous goal turn made verified progress: four added action replays show
wrong drawer376, target-only insufficient opening256/336, and success later
undone216.256 stops at50.22% fixture depth, below native52.25%, so 'roughly half'
is not a sufficient success definition. The preceding component ablations also
show that countertop-location wording suppresses recognition of128; deleting
the negative clause alone did not fix it.

Test the two exact texts frozen in the prior NEXT-RUBRICS.json: a short active
'Pull the top {side} drawer out by at least52.25%...' instruction, and an explicit
same-drawer/beyond-closed-position boundary with a halfway-is-short contrast.
No countertop location or wrong-drawer conditional is used. Holding permission
stays. This changes multiple wording components together; interpret only the
resulting candidates, not an isolated cause. Validate that 'top' matches the
native target in the four new layouts; freeze the original six-scene audit.

Use10 retained videos/45 causal prefixes: all35 old endpoints,700/750 for each
new never-success376/256/336, and200/250/300/750 for ever-success216(native212).
There are27 pre-success/never-success endpoints and18 post-success endpoints,
4 never-success and6 ever-success trajectories. Negative cases other than112
have sparse late probes. These are reused/error-enriched development cases,
not independent validation or a population precision estimate.

Recover35 exact1.6 labels. Score10 new reference labels,4 old1.6 repeats and
2×45 candidate labels:104 new outputs,135 paired labels plus4 repeats. Keep
all official messages,1–5 grammar/parser, sampler, three views, full history,
model/processor, temperature0 and seed fixed. Audit all104 actual rendered
messages and per-output frame counts. Public jobs contain only task/video/
prefix fields; native clocks and geometry stay outside model inputs.

Primary success uses two Score5 probes exactly50 controls apart, accepting the
second and latching. Require0 false Score5 endpoints,0 false first acceptances,
all6 successful cases detected after native first success and within100 controls,
and all4 reference repeats matching. Score4 can be legitimate partial progress
near the threshold, so the prior two-score>=4 rule is a diagnostic only here.
Report it for every arm, alongside raw Score5 and full-success confirmation;
do not retune either threshold after observing results. A candidate pass still
needs denser early probes and a new small validation panel before production.

Linux CPU qualification first resets the4 new scenes with CUDA disabled,
checks full metadata and top geometry,0 policy controls. Then verify all30
video hashes,751-frame/20Hz clocks,45 replay references and actual prefix decodes
including the new videos. Production worker must remain byte-identical to the
qualified source. Use reviewed latest main through Git in an immutable checkout.
One Baige NVIDIA RPBZZZ6 GPU,32CPU/128GiB RAM/16GiB SHM,15-minute maximum,
auto retry off, actual GPU probe and900GPU-s ceiling; finite780/810/860-second
deadlines. Prior5285 prefix/chunk/clause/wording workerGPU-s stay separate.
No full634 replay, new trajectories, robot policy controls, motor calls or SGD.
