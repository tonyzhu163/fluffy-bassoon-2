# Original Pi0 DSRL task57 evaluation

The original `pi0_libero` seed0 pilot is running on Baige as
`job-psgxs9n5b08w`, pinned to main commit
`00eca3fae5cb8052cee445f87359e26aa18e1a17`.
The [method contract](../dsrl-pi0-replication.md) fixes the original DSRL,
OpenPI and LIBERO sources and distinguishes this work from Pi0.5.

| Verified panel | Optimizer updates | Replay query rows | Successes | Episodes | Executed panel actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial native Pi0 | 0 | 520 | 4 | 10 | 3520 |

The initial native panel completed at2026-11-01 02:18:55 Shanghai, after the
authors' replay>500 warmup and before the first SAC update. It uses the original
random reset population,400-action horizon, independent H50x32 Gaussian noise,
execute20 interface, and shared training/evaluation environment. No extra
settling or scene selection was introduced. The40% ten-episode baseline is
above the paper's roughly20% aggregate; one panel and one seed cannot establish
or refute the reported three-seed result.

At02:19:58,27 complete training episodes and800 optimizer updates were
verified, with540 replay rows and14,320 executed actions including the native
panel. Full warmup recovery checkpoints and the initial evaluation policy/RNG
snapshot are retained. Learned-policy success and replication of the reported
improvement remain pending; evaluate10 episodes at each exact10k updates and
compare at replay query counts, separately from primitive actions.

All10 Linux CPU checks passed with the pinned Policy API and next-update
optimizer recovery. On sm_120 the original JAX0.5 Triton compiler aborts at
first inference; the active run disables Triton GEMM while preserving package
pins, bfloat16 weights and the scientific recipe. Official GCS Pi0 weights
passed file hashes; byte equality to the removed original S3 release remains
unverified. The shared phase ceilings remain36,000 GPU-seconds,1M physical
actions and64GiB output, including660 conservatively charged GPU-seconds and
400 unresolved actions from failed startup identities.

[Initial receipt](../../outputs/dsrl-pi0-task57-20261101-r1/INITIAL-NATIVE-RESULT.json),
[observed curve](../../outputs/dsrl-pi0-task57-20261101-r1/curve-through-0220.json),
[raw log](../../outputs/dsrl-pi0-task57-20261101-r1/GPU-R4-CHECK-0220.log),
[launch plan](../../outputs/dsrl-pi0-task57-20261101-r1/GPU-LAUNCH-PLAN.json).
