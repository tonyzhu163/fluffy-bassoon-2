# Six-state paired memory replication

Baige `job-8wtz8wd2dhbz`, run `libero40-memory-replication-20261030-r1`,
source `3ab93f96db79524ac6485326a0563c8401c3b2bb`, exited0.45 allocated Linux
checks passed; real GPU, native motor parity and feature parity passed.
Six predeclared LIBERO90 task0 nominal states1–6, scene seed123; same fixed
AD500 full policy as the first case. Zero deployment optimizer updates.

Each case has one actual failed support, followed by two real fresh queries:
retain support context versus erase it, with the same policy and query RNG.
This is a paired counterfactual probe with physically shared support, charged
once. It is not two independent deployment trials.

| States | Support | Retained-context query | Reset-context query |
| --- | ---: | ---: | ---: |
|1–5, each |failure|failure|failure|
|6 |failure|success,111 primitives|failure|
|Total |0/6|1/6|0/6|

Actual total7,091 simulator actions within8,000, no interrupted/uncertain
charge. State6 support cost410, full query121, reset query410. Per-method
support+query cost531 versus820; unique physical triple941. Other cases cost
1,230 unique actions each. Per-method costs count support for each branch;
the joint execution ledger counts its single physical execution only once.

[Independent audit](independent-artifact-audit.json) verifies all18 complete
trace hashes and all18 retained context-file hashes, exact initial public
RGB/wrist/proprio arrays within each triple, matching query RNG, and six
distinct nominal-state hashes. Full boundary recovery hash and primitive/
settling costs were also checked. [Root result](index.json) and
[final progress](progress.json) retain the compact evidence. Full raw RPCs,
episode traces and context/motor/global RNG recoveries remain on Baige.

This repeats a history-dependent success once on new states of the same
goal. With5/6 retained-context queries still failing, it does not establish
reliable adaptation, high topline performance, broad task transfer or a
causal benefit from scalar reward. Ordinary target-online DSRL remains
unfinished. The original state0 result is separate discovery evidence;
do not pool it into the prespecified six-state replication denominator.
