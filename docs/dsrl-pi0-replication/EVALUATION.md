# Original Pi0 DSRL task57 evaluation

The original `pi0_libero` seed0 pilot is running on Baige as
`job-psgxs9n5b08w`, pinned to main commit
`00eca3fae5cb8052cee445f87359e26aa18e1a17`.
The [method contract](../dsrl-pi0-replication.md) fixes the original DSRL,
OpenPI and LIBERO sources and distinguishes this work from Pi0.5.

The single-seed pilot has improved from 4/10 native successes to 10/10 learned
successes in four consecutive panels from 40k through 70k optimizer updates. The full run is still active;
this does not establish the paper's three-seed aggregate or its sample-efficiency
confidence interval.

| Panel | Optimizer updates | Replay query rows | Training primitive actions | Successes | Episodes | Panel actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial native Pi0 | 0 | 520 | 10400 | 4 | 10 | 3520 |
| Learned SAC | 10000 | 1005 | 20074 | 0 | 10 | 4000 |
| Learned SAC | 20000 | 1504 | 30048 | 2 | 10 | 3604 |
| Learned SAC | 30000 | 2003 | 39833 | 8 | 10 | 2421 |
| Learned SAC | 40000 | 2507 | 49664 | 10 | 10 | 1980 |
| Learned SAC | 50000 | 3011 | 59490 | 10 | 10 | 2308 |
| Learned SAC | 60000 | 3504 | 68990 | 10 | 10 | 1821 |
| Learned SAC | 70000 | 4009 | 78704 | 10 | 10 | 1751 |

The initial native panel completed at 2026-11-01 02:18:55 Shanghai after
replay >500 warmup and before the first SAC update. It uses the original random
reset population, 400-action horizon, independent H50x32 Gaussian noise,
execute20 interface, and shared training/evaluation environment. No extra
settling or scene selection was introduced. Its 40% baseline is above the
paper's roughly 20% aggregate. The learned panels use the original stochastic
SAC noise policy; record all panels, including the initial decline.

The first 10/10 panel occurred at 2,507 queried replay records, after 49,664
training primitive actions. Sample, primitive-action and optimizer-update
counts are separate units. This is directional evidence for the reported
steering gain, with one seed and only 10 evaluation episodes per panel; it is
not a claim of better sample efficiency than the published aggregate.

At the direct allocation audit (`2026-11-01T03:07:07.076926+08:00`), training had reached
76,880 updates, 342 training episodes and 4,344 replay rows. The complete
recovery at 70,780 updates and all eight retained evaluation policy/RNG snapshots
(0 through 70k) matched their saved SHA256 hashes. The current process still
held the allocated GPU's compute and graphics contexts. Current-run output was
1,452,942,913 bytes. The physical ledger had 106,601 known actions plus 400
pending actions; retain the additional 400 pending actions from failed R3,
for 107,401 conservatively charged phase actions at that snapshot. The final
before/after motor hash comparison remains pending the run's terminal cleanup.

All 10 Linux CPU checks passed with the pinned Policy API and next-update
optimizer recovery. On sm_120 the original JAX 0.5 Triton compiler aborts at
first inference; the active run disables Triton GEMM while preserving package
pins, bfloat16 weights and the scientific recipe. Official GCS Pi0 weights
passed file hashes; byte equality to the removed original S3 release remains
unverified. The shared phase ceilings remain 36,000 GPU-seconds, 1M physical
actions and 64 GiB output, including 660 conservatively charged GPU-seconds and
400 unresolved actions from failed startup identities.

[Initial receipt](../../outputs/dsrl-pi0-task57-20261101-r1/INITIAL-NATIVE-RESULT.json),
[current audit](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0305.json),
[direct checkpoint and metrics receipt](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0305-DIRECT.json),
[observed curve](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0305-CURVE.csv),
[console evidence](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0245-EVIDENCE.log),
[launch plan](../../outputs/dsrl-pi0-task57-20261101-r1/GPU-LAUNCH-PLAN.json).
