# Original Pi0 DSRL task57 evaluation

The original `pi0_libero` seed0 pilot is running on Baige as
`job-psgxs9n5b08w`, pinned to main commit
`00eca3fae5cb8052cee445f87359e26aa18e1a17`.
The [method contract](../dsrl-pi0-replication.md) fixes the original DSRL,
OpenPI and LIBERO sources and distinguishes this work from Pi0.5.

The single-seed pilot improved from 4/10 native successes to its first 10/10
learned panel at 40k optimizer updates. It then scored 10/10 in 16 consecutive
panels through 190k. The latest panel scored 10/10 at
360k updates, after 18,501 replay query rows and 359,418 training primitive
actions. The 200k panel scored 9/10; the full curve retains this dip and the
subsequent panels. The run is still active, with one seed and 10 evaluation
episodes per panel. It does not establish the paper's three-seed aggregate or
its sample-efficiency confidence interval.

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
| Learned SAC | 210000 | 11011 | 214269 | 10 | 10 | 1423 |
| Learned SAC | 220000 | 11504 | 223837 | 10 | 10 | 1588 |
| Learned SAC | 230000 | 12000 | 233512 | 10 | 10 | 1576 |
| Learned SAC | 240000 | 12506 | 243221 | 10 | 10 | 1644 |
| Learned SAC | 250000 | 13000 | 252847 | 10 | 10 | 1411 |
| Learned SAC | 260000 | 13503 | 262642 | 10 | 10 | 1574 |
| Learned SAC | 270000 | 14006 | 272445 | 10 | 10 | 1500 |
| Learned SAC | 280000 | 14505 | 282136 | 10 | 10 | 1634 |
| Learned SAC | 290000 | 15003 | 291754 | 10 | 10 | 1452 |
| Learned SAC | 300000 | 15504 | 301398 | 10 | 10 | 1372 |
| Learned SAC | 310000 | 16006 | 311130 | 10 | 10 | 1430 |
| Learned SAC | 320000 | 16505 | 320769 | 10 | 10 | 1513 |
| Learned SAC | 330000 | 17001 | 330408 | 10 | 10 | 1508 |
| Learned SAC | 340000 | 17502 | 340090 | 10 | 10 | 1460 |
| Learned SAC | 350000 | 18002 | 349704 | 10 | 10 | 1387 |
| Learned SAC | 360000 | 18501 | 359418 | 10 | 10 | 1528 |

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

At the direct allocation audit (`2026-11-01T06:05:39.407696+08:00`), the last published
progress was 360,940 updates, 2,070 training episodes and 18,547 replay rows. The
complete recovery at 353,220 updates and all 37 retained evaluation
policy/RNG snapshots (0 through 360k) matched their saved SHA256 hashes.
The current process still held the allocated GPU's compute and graphics
contexts. Current-run output was 6,059,848,187 bytes. The physical ledger,
read separately during a live collection/update boundary, contained
427,293 known actions and 400 pending actions. Retain
the additional 400 pending actions from failed R3, for 428,093 conservatively
charged phase actions at that ledger read. Counting conservatively from job
creation through this audit, plus prior failures, bounds cumulative GPU time
at 14,679 seconds; the final worker lifetime is pending. The final before/after
motor hash comparison remains pending the run's terminal cleanup.

During the 04:45 check, one NVML query reported compute-only activity for the
correct model PID. Six bounded observations then showed C followed by five
C+G readings on the same allocated UUID, while physical actions advanced and
EGL binding variables remained correct. The subsequent strict audit passed
C+G. This transient telemetry observation did not require a process or source
change. See the [context samples](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0445-GPU-SAMPLES.json).
The 05:25 single-query check again missed combined C+G; a bounded metadata
recheck immediately verified both contexts on the correct UUID. The
[metadata reader](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0525-READER.sh)
now allows up to six observations spaced two seconds apart, requiring the
correct PID/UUID on every observation and combined C+G before passing. The
active scientific source and process remain unchanged.

All 10 Linux CPU checks passed with the pinned Policy API and next-update
optimizer recovery. On sm_120 the original JAX 0.5 Triton compiler aborts at
first inference; the active run disables Triton GEMM while preserving package
pins, bfloat16 weights and the scientific recipe. Official GCS Pi0 weights
passed file hashes; byte equality to the removed original S3 release remains
unverified. The shared phase ceilings remain 36,000 GPU-seconds, 1M physical
actions and 64 GiB output, including 660 conservatively charged GPU-seconds and
400 unresolved actions from failed startup identities.

[Initial receipt](../../outputs/dsrl-pi0-task57-20261101-r1/INITIAL-NATIVE-RESULT.json),
[current audit](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0605.json),
[direct checkpoint and metrics receipt](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0605-DIRECT.json),
[observed curve](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0605-CURVE.csv),
[console evidence](../../outputs/dsrl-pi0-task57-20261101-r1/CHECK-0245-EVIDENCE.log),
[launch plan](../../outputs/dsrl-pi0-task57-20261101-r1/GPU-LAUNCH-PLAN.json).
