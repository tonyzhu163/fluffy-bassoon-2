# Original Pi0 DSRL task57 evaluation

The original `pi0_libero` seed0 pilot is running on Baige as
`job-psgxs9n5b08w`, pinned to main commit
`00eca3fae5cb8052cee445f87359e26aa18e1a17`.
The [method contract](../dsrl-pi0-replication.md) fixes the original DSRL,
OpenPI and LIBERO sources and distinguishes this work from Pi0.5.

The single-seed pilot improved from 4/10 native successes to its first 10/10
learned panel at 40k optimizer updates. It then scored 10/10 in 16 consecutive
panels through 190k. The latest panel scored 9/10 at
200k updates, after 10,501 replay query rows and 204,372 training primitive
actions. A single 10-episode panel is too small to establish a decline. The full
run is still active; this does not establish the paper's three-seed aggregate
or its sample-efficiency confidence interval.

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
| Learned SAC | 80000 | 4500 | 88202 | 10 | 10 | 1756 |
| Learned SAC | 90000 | 5005 | 97961 | 10 | 10 | 1909 |
| Learned SAC | 100000 | 5507 | 107628 | 10 | 10 | 1656 |
| Learned SAC | 110000 | 6001 | 117111 | 10 | 10 | 1750 |
| Learned SAC | 120000 | 6503 | 126791 | 10 | 10 | 1602 |
| Learned SAC | 130000 | 7005 | 136528 | 10 | 10 | 1464 |
| Learned SAC | 140000 | 7500 | 146174 | 10 | 10 | 1456 |
| Learned SAC | 150000 | 8008 | 155981 | 10 | 10 | 1622 |
| Learned SAC | 160000 | 8503 | 165565 | 10 | 10 | 1606 |
| Learned SAC | 170000 | 9005 | 175372 | 10 | 10 | 1533 |
| Learned SAC | 180000 | 9505 | 185059 | 10 | 10 | 1629 |
| Learned SAC | 190000 | 10002 | 194715 | 10 | 10 | 1764 |
| Learned SAC | 200000 | 10501 | 204372 | 9 | 10 | 1829 |

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

At the direct allocation audit (`2026-11-01T04:25:41.182206+08:00`), the last published
progress was 202,720 updates, 1,070 training episodes and 10,636 replay rows. The
complete recovery at 201,920 updates and all 21 retained evaluation
policy/RNG snapshots (0 through 200k) matched their saved SHA256 hashes.
The current process still held the allocated GPU's compute and graphics
contexts. Current-run output was 3,517,528,954 bytes. The physical ledger,
read separately during a live collection/update boundary, contained
250,007 known actions and 400 pending actions. Retain
the additional 400 pending actions from failed R3, for 250,807 conservatively
charged phase actions at that ledger read. Counting conservatively from job
creation through this audit, plus prior failures, bounds cumulative GPU time
at 8,681 seconds; the final worker lifetime is pending. The final before/after
motor hash comparison remains pending the run's terminal cleanup.

All 10 Linux CPU checks passed with the pinned Policy API and next-update
optimizer recovery. On sm_120 the original JAX 0.5 Triton compiler aborts at
first inference; the active run disables Triton GEMM while preserving package
pins, bfloat16 weights and the scientific recipe. Official GCS Pi0 weights
passed file hashes; byte equality to the removed original S3 release remains
unverified. The shared phase ceilings remain 36,000 GPU-seconds, 1M physical
actions and 64 GiB output, including 660 conservatively charged GPU-seconds and
400 unresolved actions from failed startup identities.

[Initial receipt](../../outputs/dsrl-pi0-task57-20261101-r1/INITIAL-NATIVE-RESULT.json),
[current audit](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0425.json),
[direct checkpoint and metrics receipt](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0425-DIRECT.json),
[observed curve](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0425-CURVE.csv),
[console evidence](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0245-EVIDENCE.log),
[launch plan](../../outputs/dsrl-pi0-task57-20261101-r1/GPU-LAUNCH-PLAN.json).
