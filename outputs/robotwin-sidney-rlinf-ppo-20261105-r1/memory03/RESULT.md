# Low reset peak; dynamics comparison failed

Job`job-dx2w7coo3lr3` measured per-scene-close trimming on64 native scenes.
Its reset peak was56.120GiB process RSS and58.842GiB cgroup usage; after reset,
RSS55.550GiB/live glibc53.248GiB. The previous native batch-end-only reset peak
was106.183GiB. All three raw RGB inputs and14D public joint states were exactly
identical across the paired resets (maxabs0), and both rounds passed retained
observation, RNG, scene metadata and motor-input identity around manual trim.

The post50-control comparison failed. The exact expression was
`error<=1e-6 and torch.equal(result[1],paired['after_hold_rewards'])`.
The diagnostic saved the reference state/reward but asserted before saving the
candidate final state, reward and numeric error. The reference reward is all0.
It is therefore impossible to tell which conjunct failed from these artifacts;
no exact failed-state reconstruction or dynamics-parity claim is made.
Production remains held. This is not an OOM or a model/learner failure.

The original harness compared RNG only immediately around malloc_trim inside a
reset; it did not control cross-reset hold RNG. Pinned task setup seeds NumPy
and Torch, while its Python random.seed call is commented out. Threaded reset
completion order can also affect the final process RNG state. This is a
hypothesis to test, not an established cause of the observed mismatch.
A fresh diagnostic will compare two native resets plus the candidate, record
exposed qvel/root/object poses/velocities/contacts and task/wrapper bookkeeping,
restore identical diagnostic hold RNG, and save actual inputs/outcomes plus
per-environment errors before unchanged1e-6/exact-reward gates.

All6,400physical controls and1,920setup/reset allowances remain charged to
originalrank1 under freshmemory03, totaling8,320for this attempt and69,120shared
rank1. Scheduler1,309GPU-seconds; creation to release1,328; worker1,285. Source
`7e19ae79cf53bfbe9aad9da6fac6e848c4b9d006`; no VLA or learner was loaded.

[Raw result](RESULT.json), [reference states/rewards](paired-reference.json),
[exit](EXIT.json), [GPU probe](gpu-probe.json), [log](profile.log).
