# Completed two-case robot comparison

Baige `job-mjadtzoxnmy6`, run `libero40-subset-target-20261030-r2`, source
`cbd7064b1bdf8f666451a761901a83203b6b68f8`, exited0. All28 primary groups
completed:84 actual episodes,34,369 charged simulator actions within35,046,
zero deployment optimizer updates.250 Linux checks passed/two JAX skips in
the probe environment; actual allocation GPU and motor qualification passed.
Optional DPT25/100 groups were not launched because the remaining physical
reservation could not fit a complete group.

[Independent audit](independent-artifact-audit.json) verifies every group-index
hash, all84 trace hashes, chunk/primitive costs, final boundaries and fixed
weights. [Root result](index.json), [progress](progress.json) and
[paired context audit](paired-context.json) retain compact evidence. Large
raw camera/action traces, all complete recoveries and checkpoints remain
under the immutable run on Baige.

Cases were predeclared LIBERO90 tasks0/1, state0, scene seed123. On task0,
AD/full500 failed the empty-history query and its support, then succeeded in
329 primitive actions. The same weights with episode memory erased failed.
Actual supports and empty-history queries matched exactly; query RNG and
initial nominal states matched, as did all initial RGB/wrist/proprio arrays.
The first retained-context query noise differed by max0.402480781. This is
one controlled history effect. Task1 failed for every method, and every other
zero/one-support query failed. It does not establish reliable adaptation,
reward-mediated learning, broad transfer or strong robot performance.

The frozen source pixel-SAC uses image64/proprio8 while students use frozen
VLA-prefix4104 features. It is a source-transfer control, not ordinary
target-online DSRL. That required comparison remains unfinished.

See [verified table](../../docs/frozen-vla-libero/EVALUATION.md) and the
[predeclared protocol](../libero40-subset-target-20261030-r1/PROTOCOL.md).
