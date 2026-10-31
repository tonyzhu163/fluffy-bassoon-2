# Original Pi0 DSRL task57 evaluation

The seed0 pilot completed successfully on Baige (`job-psgxs9n5b08w`) at
2026-11-01 07:35:01 Shanghai, after 500,080 optimizer updates. The original
`pi0_libero` native panel scored **4/10**; the final learned panel scored
**10/10** at 500k updates, 25,504 replay query rows and 495,165 training
primitive actions. The first 10/10 panel occurred at 40k updates, 2,507 replay
query rows and 49,664 training actions. This is a single-seed qualitative
reproduction of the reported steering gain.

The [paper, Section 5.5](https://arxiv.org/html/2506.15799v2#S5.SS5) reports
roughly 20% to 100% LIBERO success around 10,000 online samples, averaged over
three seeds. Our 40% native baseline differs, and each panel has only ten
evaluation episodes. This pilot does not establish the published three-seed
aggregate, its uncertainty, better sample efficiency, or the comparisons
against other algorithms.

The [method contract](../dsrl-pi0-replication.md) and source
`00eca3fae5cb8052cee445f87359e26aa18e1a17` preserve original Pi0 H50/execute20,
LIBERO90 task57 random resets, the shared training/evaluation environment and
original SAC chronology. This work is separate from Pi0.5 and all
source40/meta-learning data.

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
| Learned SAC | 370000 | 19004 | 369176 | 9 | 10 | 1713 |
| Learned SAC | 380000 | 19505 | 378945 | 10 | 10 | 1450 |
| Learned SAC | 390000 | 20000 | 388531 | 10 | 10 | 1396 |
| Learned SAC | 400000 | 20504 | 398242 | 10 | 10 | 1503 |
| Learned SAC | 410000 | 21005 | 407998 | 10 | 10 | 1489 |
| Learned SAC | 420000 | 21502 | 417648 | 10 | 10 | 1516 |
| Learned SAC | 430000 | 22001 | 427267 | 10 | 10 | 1504 |
| Learned SAC | 440000 | 22500 | 436953 | 10 | 10 | 1528 |
| Learned SAC | 450000 | 23000 | 446677 | 10 | 10 | 1477 |
| Learned SAC | 460000 | 23506 | 456490 | 10 | 10 | 1389 |
| Learned SAC | 470000 | 24005 | 466153 | 10 | 10 | 1445 |
| Learned SAC | 480000 | 24500 | 475726 | 10 | 10 | 1572 |
| Learned SAC | 490000 | 25005 | 485500 | 10 | 10 | 1573 |
| Learned SAC | 500000 | 25504 | 495165 | 10 | 10 | 1477 |

The native panel ran after replay >500 warmup and before the first SAC update.
It used independent H50x32 Gaussian noise; learned panels used the original
stochastic SAC noise policy. The early 0/10 and 2/10 panels, and later 9/10
panels at 200k and 370k, are retained. Replay queries, primitive actions and
optimizer updates are separate units.

The terminal audit verified the complete recovery at 500,080 updates and all
51 evaluation policy/optimizer/RNG payloads from 0 through 500k against saved
SHA256 hashes. The final recovery is `recovery-0.pkl` (678,360,350 bytes), SHA256
`e67a7991ec05622e68c2c747e37ed5a2c62eb5c011d6f5ca0cec94315dda9cb5`.
Original DSRL `7f48937`, OpenPI `a6d2400`, LIBERO `87edbd1` and the runner
commit remained pinned with no tracked source changes. All ten allocated
Linux qualification checks passed, including next-update recovery.

The frozen motor matched before and after training: 50 parameter leaves,
3,238,048,528 bfloat16 elements, SHA256
`2b563cd3ae77e837c1c5034316aaaa6c808174ca74444733b9909d17cfbfc903`.
Same-noise qualification passed; model and renderer were attested on the
allocated GPU. Intermittent NVML compute-only readings were resolved by
bounded observations of the same PID/UUID returning C+G; no scientific
process or source was changed.

Physical accounting reconciles every evaluation panel with its ten captured
episodes, actions and successes. Training used 495,165 actions across 2,962
episodes; the native and learned panels used 88,013 actions across 510
episodes. Current-run known actions total 583,178, with no active unresolved
reservation. Retaining 400 unresolved actions from failed R3 gives **583,578**
conservatively charged phase actions under the 1M ceiling.

The launcher charged **20,081 cumulative GPU-seconds** (5 h 34 m 41 s),
including 660 seconds for failed allocations and its 60-second scheduler
allowance, under the unchanged 36,000-second cap. The final CPU-only retrieval
job `job-8gadv5dju8jx` used no GPU, model loading or physics; its creation-to-
terminal lifetime was 43 seconds and is conservatively charged 60 seconds.
Preparation plus retrieval is therefore **1,800/1,800 CPU worker-seconds**.
All owned remote outputs total 8,333,888,847 bytes; including local receipt
copies and a 16 MiB compact-artifact margin bounds retained output at
8,352,437,532 bytes, below 64 GiB. Complete checkpoints and physical traces
remain in the immutable remote run folder.

Official GCS Pi0 weights passed file hashes. Byte equality to the removed
legacy S3 release remains unverified. On Baige sm_120, Triton GEMM was disabled
with `--xla_gpu_enable_triton_gemm=false --xla_gpu_triton_gemm_any=false`;
JAX 0.5, bfloat16 weights and the scientific recipe stayed pinned. These
hosting/compiler differences remain replication limitations.

Both task-owned allocations finished successfully. The recurring monitor was
[removed after verification](../../outputs/dsrl-pi0-task57-20261101-r1/MONITOR-CLOSED.json).

[Final result](../../outputs/dsrl-pi0-task57-20261101-r1/FINAL-RESULT.json),
[full curve](../../outputs/dsrl-pi0-task57-20261101-r1/FINAL-CURVE.csv),
[checkpoint/motor manifest](../../outputs/dsrl-pi0-task57-20261101-r1/FINAL-CHECKPOINT-MANIFEST.json),
[direct terminal receipt](../../outputs/dsrl-pi0-task57-20261101-r1/FINAL-HARVEST-RECEIPT.json),
[CPU retrieval settlement](../../outputs/dsrl-pi0-task57-20261101-r1/HARVEST-SETTLEMENT.json),
[initial native receipt](../../outputs/dsrl-pi0-task57-20261101-r1/INITIAL-NATIVE-RESULT.json),
[launch plan](../../outputs/dsrl-pi0-task57-20261101-r1/GPU-LAUNCH-PLAN.json).
