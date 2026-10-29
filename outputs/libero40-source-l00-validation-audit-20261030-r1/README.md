# Independent second-goal validation stream

Baige `job-aehuoen0j6a5`, run
`libero40-1m-sac-20261030-l00-validation-r2`, source
`261bd4317b71b0605e642d23af18e1a089a770d8`, exited0.263 Linux checks passed,
two probe-environment JAX skips; actual GPU and motor PRNG qualification passed.

36 episodes/32 successes,9 complete K4 cases and distinct nominal states40–48,
1,047 query chunks,11,400 SAC updates.10,344 primitives+370 settling=10,714
actions within the original12,000 validation slot. Worker788.33s. Source
learner/replay and RNG are separate from training states0–6; no target data.

[Independent saved-data audit](independent-artifact-audit.json) checks all36
raw trace hashes/costs/boundaries, paired AD/DPT datasets, actor checkpoint
and complete source recovery. [Source index](index.json) SHA256
`6b49b3d642e3292d5fa9d18d190ebd7f2d6dbdad513ce9a715d8b9312fa08927`.
Full recovery SHA256 `b45e129cc4926f37b7c648beab95b26001d342d7e95ee311ed9c327204c368e6`.
Large raw/recovery files remain on Baige.

This completes the missing independent validation input for the two-goal
source subset. Combined training is1,551 chunks (742+809), validation1,664
(617+1,047). It is not all40 coverage.32/36 mixes warmup and changing source
actors; it does not establish final-actor competence or a causal training gain.
