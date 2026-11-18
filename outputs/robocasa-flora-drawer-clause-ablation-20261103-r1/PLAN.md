# Separate the target and opening-depth clauses on full prefixes

The combined1.6 criterion rejects wrong-drawer112 but misses128. Moving the
rubric into system grading and using recent chunks did not repair this tradeoff.
Use a2×2 component ablation to identify which clause causes the loss.

All four conditions share permission to hold the handle. Base is the native
"Open the {side} drawer." plus that shared permission. Target-only uses the
exact first3 target sentences of1.6 and the shared permission. Depth-only
adds the exact "The target must finish open by roughly half its front-to-back
depth." sentence to base. Both is verbatim1.6, recovered from the verified
full-prefix cache. The target factor also necessarily changes the imperative's
target phrase; it tests that target-description block as a whole. It is not
an isolated word or token-length effect.

Use the same6 retained videos and35 causal endpoints, restoring the full
history from reset to each endpoint. Three new conditions×35 plus4 unchanged
both-clause repeats give109 new outputs and140 candidate/reference grades.
No full634 replay, new trajectories, simulator controls, motor calls or SGD.
Keep three views, official sampler/grading messages, model/processor/tokenizer,
score grammar/parser, temperature0 and seed0 fixed. Only native left/right
extraction enters each task string; no outcome, joint state or clock enters
scoring. Audit all109 actual renderer messages against prepared messages.

Report every raw Score5 and first sampled positive, and two-probe rules at
Score>=4 and Score5, requiring probes exactly50 controls apart and accepting
at the second probe with a latch. Based on prior full-prefix threshold4
evidence, the prospective development screen uses the threshold4 rule:
all4 repeat grades match,0 false first acceptances, and5/5 successful cases
accepted after native first success and within100 controls. Score5 confirmation
is a prespecified sensitivity readout. Do not retune thresholds to this result.

These are diagnostic ablations. Dropping a stated native requirement cannot
be justified merely by passing this small panel. A useful candidate still
needs both correct target and sufficient opening, a denser early clock and
a new small validation panel before any production qualification. Only112
has a complete early50-control grid; the other5 cases have4 probes each.
Preserve the existing replay qualification and its missing-full-state limit.

Use reviewed latest main, deployed through Git into a new immutable checkout.
Linux CPU checks frozen input/video hashes, recovered grade identity, unchanged
production worker source,109 prompts and actual24/50/64-frame decoding at
50/100/750 controls. One Baige NVIDIA RPBZZZ6 GPU,32CPU/128GiB RAM/16GiB SHM,
15-minute maximum, auto retry off. TP1/batch8/.78 memory and actual allocation
probe. Prior74-output full-prefix trials completed within705 all-in GPU-s;
the extra35 outputs add5 bounded batches. Keep the same780/810/860-second
internal deadlines and900GPU-s ceiling; preserve partial results on failure,
without automatic retry. Prior3144 prefix/chunk workerGPU-s remain separate.
