# Exact-input motor replay

Baige `job-ptt7fgi29ebx`, source
`b517a91826f1d49400d871523d30a33cf5ffbad2`, completed exit0.
Five Linux checks and the actual allocation GPU probe passed.
64 motor predictions took121.03s, with zero simulator actions or optimizer
updates. All saved prediction arrays were independently hashed.

Two target cases, two identical-input origins, typed/legacy JAX keys and
two repeats were crossed at four stages: plain motor, prefix attached,
prefix qualified and fresh SAC initialized. Every within-process comparison
had action delta0. The full motor parameter state stayed unchanged, SHA256
`c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9`.

The new process still differed from the old native and DSRL reference action
chunks, despite identical saved inputs/noises. Maximum deltas versus old
native were0.002441/0.001953, versus DSRL0.003417/0.001953 for tasks0/1.
The root cause remains unproven. Key type and these initialization stages do
not explain it. HLO fingerprint extraction was unavailable; no identical-HLO
claim follows. Old within-run full/reset controls remain valid; strict
cross-process motor equivalence is unsupported.

The next robot comparison loads both source banks, native and frozen final
target-SAC into one motor process. This controls the observed numerical
nuisance without claiming it has been diagnosed.

Evidence: [compact independent audit](independent-artifact-audit.json),
[Linux checks](focused-tests.log). Full raw index and prediction arrays remain
under `/mowenjie/fluffy-bassoon-2/outputs/libero40-motor-replay-20261030-r1/`;
full index SHA256
`578afb638b76fde8b709eace7da1722aad583f7721a03e42a99f7dee67fa2796`.
