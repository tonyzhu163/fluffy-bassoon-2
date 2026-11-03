# Paused scale50 source performance diagnosis

Read-only diagnosis requested by coordinator `01a19f33`. It does not change a
source profile, checkpoint, scientific setting, implementation or GPU job.
Latest steering is **hold recovery until the KL test is reviewed**, relayed from
human turn `01a1a00f-157f-7521-a92a-e79d52a0b5fa` in owner chat `01a19fe9`.
The separate lifetime budget approval is 60 GPU-hours per arm / 120 combined;
it does not remove that hold. This report is non-gating performance evidence.

Both original sources completed update 2/50, with 768 TRAIN episodes per arm.
Source owner verified scheduler terminal status, complete resumable boundaries
and child release. Those checks and migration remain with that owner; this
audit does not duplicate or supersede their accounting.

## What differs from qualification

Qualification and production use native source `8b9e1bbb`, runtime `8a1233c3`,
the same model/config/motor-normalization-asset hashes, original H50/10-step/12D motor,
2,064 pre-action features and width256/two-layer steering/value heads. Steering
widths remain 32 for velocity and 1,152 for all-layer K+V. Fractions .125/.64,
learning rates .0003/.00003, minibatch32, ten requested epochs and exact joint
Gaussian KL .02 checked per optimizer step agree. Q2 used the actual frozen VLA
and the same `collect_bank`, raw-history, three-camera recording and persistence
paths, rather than a smaller substitute head or native motor.

The material execution difference is the initial reset barrier. Q2 explicitly
sets `synchronize_initial_lanes=True`; production dispatch omits it, and the
worker defaults to false. Q2 therefore made 12 native B48 calls for each
48-episode, 600-control bank. Actual production starts inference as resets
arrive and drains staggered episode tails. Each rank still serves exactly
576 physical chunks and reaches the same prescribed episode boundaries.

| Arm / update | Native calls across eight ranks | Average native batch | Request→completion proxy | Complete phase |
| --- | ---: | ---: | ---: | ---: |
| Velocity / 1 | 210 | 21.94 | 321.84–350.89s | 355.32s |
| Velocity / 2 | 170 | 27.11 | 333.35–372.08s | 375.45s |
| All-layer KV / 1 | 169 | 27.27 | 339.07–371.19s | 388.09s |
| All-layer KV / 2 | 193 | 23.88 | 343.48–386.97s | 400.22s |

Per-rank time subtracts request-file mtime from the worker's recorded completion
time. It includes waiting and reset work, uses filesystem/worker wall clocks,
and is not a separately instrumented inference or kernel duration.

Q2 needed 222.23s / 221.23s for velocity / KV, with twelve B48 calls on one
GPU and the arms run sequentially. Across production ranks there are 15–33
calls, versus twelve in Q2; the hypothetical eight-rank B48 count is 96.
Fragmentation is observed, but the causal time penalty is unmeasured. More
calls do not monotonically predict slower ranks: velocity update2 rank3 made
17 calls and took372.08s, while rank2 made31 and took352.35s. Q2 also has
different cases, policy initialization and a cached zero/influence check inside
its timed bank; it is not a paired eight-GPU performance experiment.

## Collection, CPU and storage limits

The optimizer accounts for 2.13s / 1.38s in the first two velocity updates and
13.81s / 10.91s for KV. Update2 optimization is only .37% / 2.73% of the full
phase. Most elapsed time is collection, resets, native inference, video/raw
history persistence, aggregation and cleanup. The retained result does not
separately time these components, so it cannot identify a dominant bottleneck.

Q2 had 21 CPU cores / 208GiB; production requested and checked 168 cores /
1,664GiB, the same per-GPU ratio. One model thread and one ffmpeg thread per
camera are already bounded. Production peak container RAM was 1,253.60 /
1,263.99GiB, below 1,664GiB; Q2 sampled peak GPU memory was 73,802 / 73,719MiB.
There is no observed OOM or allocation-capacity failure in these two banks.
The source's resource snapshots occur after a bank; their near-zero GPU
utilization cannot establish utilization during collection. Q2's sampled GPU
utilization averaged14.60% / 12.54%, also including reset, simulation and IO.
No retained production CPU-throttling, IO-pressure or fsync-latency trace was
found in the audited phase metadata. Per-host CPU contention is unproven.

Shared PFS is a plausible pressure point. Each arm's escrow uses one locked
file across eight ranks. Every physical chunk reserves and resolves under that
lock, with four durable JSON writes in total. A 384-episode bank has4,608
physical chunks:18,432 escrow JSON fsyncs plus9,216 pre/post raw-chunk writes,
their hashes, episode checkpoints, canonical evidence and three-camera videos.
The two arms have separate locks but share project PFS. A single raw chunk
sample per rank is .75–.97MB compressed; these samples are not a full-bank byte
census. Serialization and the intra-arm lock remain even if arms run serially.

The first audit's recursive artifact-stat walk hit its330s CPU deadline in
`cxiWaitEventWait`; this is a diagnostic read-cost observation, not a measurement
of historical production IO. That failed attempt is retained. The replacement
audit avoids recursive artifact/journal scans, reads only closed result/request
metadata and bounded samples, and completed in5.70s on the allocated Linux
development CPUs with CUDA hidden. It performed zero model, optimizer or
simulator computation. No source artifact was modified.

## Allocation-only remaining cost

Use the [source-owner scheduler closeout](../robocasa-joint-steering-scale50-20261103-r1/UPDATE2-SCHEDULER-CLOSEOUT.json),
not overlapping process clocks. Selected lifetime spend is13,217 /13,841 GPU-s,
including preparation once, or7.5161 GPU-hours combined. The original guard
counted20–21 minutes of unallocated queue time. Removing it changes49.32 /
52.05 to46.52 /49.39 GPU-hours,95.91 combined, before a new migration/reload
allowance. This correction alone would not have met the former40/80 cap.

From update2,48 TRAIN updates, ten DEV panels and one final paired panel remain.
The following extrapolates update2 and measured DEV0/before timings; it assumes
unchanged speed and uses the before panel as the after-panel proxy. It is a
forecast, not a completion or spend claim.

| Cost / constraint | Velocity | All-layer KV |
| --- | ---: | ---: |
| Remaining TRAIN | 40.05 GPUh | 42.69 GPUh |
| Remaining DEV + final | 2.55 GPUh | 2.64 GPUh |
| All remaining work before reload | 42.60 GPUh | 45.33 GPUh |
| Lifetime total, adding240s reload +120s release per arm | 47.07 GPUh | 49.97 GPUh |
| Headroom under60 GPUh | 12.93 GPUh | 10.03 GPUh |
| TRAIN seconds/update needed for former40 cap, same overhead | ≤309.13s | ≤306.73s |
| TRAIN seconds/update compatible with60 cap, same overhead | ≤496.63s | ≤494.23s |

The combined forecast with those explicit allowances is97.05 GPU-hours, leaving
22.95 under120. The240s reload allowance is an assumption, not a fresh allocation
measurement. Later bank, optimizer, evaluation and recovery times must update
the forecast. Full formula, rank arrays and input hashes are in the
[Linux cost/rank summary](performance-20261103-r2/COST-AND-RANK-SUMMARY.json).
Its scheduler-closeout-based costs supersede the provisional13:14/wrapper-exit
cost sensitivity retained in the raw audit JSON.

## Serial execution and comparability

Running one eight-GPU arm at a time can reduce peak shared-PFS traffic from two
arms, but there is no measured serial-eight-GPU result. At unchanged phase
times, GPU-hours do not improve; elapsed campaign time approximately doubles.
Under the old40 cap it would have required about18% /23% shorter TRAIN phases,
assuming the same evaluation/reload overhead. A return to one-GPU Q2 speed is
not established because a serial eight-GPU arm still has eight model/reset/video
streams and one shared accounting lock. The current120 cap does not require
this unverified optimization.

Per-case/chunk native and actor Gaussian draws come from explicit hashed seeds,
so their random samples are independent of rank, lane arrival and batching.
However, changing native batch shapes can change BF16 kernel/reduction results;
seed invariance does not prove identical decoded actions or robot trajectories.
The barrier flag is already part of the persistent request binding. Enabling it
during exact recovery would require an explicit source/execution migration and
numerical qualification; do not silently edit an old request, profile or bank.
Preserve learned normalization, optimizer, worker/global RNG and all raw histories.

**Next bounded option:** keep this diagnosis non-gating and make no execution
change. Retain the human KL-review hold. After that review and source-owner
authorization, the owner can choose exact update2→3 recovery under the qualified
budget-only migration. Read the naturally produced update3 phase/rank metadata
and allocation-only cost; do not add episodes, replay old work or start a separate
performance job. Initial-lane coalescing and lock/IO profiling remain future
execution work requiring their own reviewed migration, not prerequisites asserted
by this audit.

Evidence: [bounded Linux metadata audit](performance-20261103-r2/AUDIT.json),
SHA256 `03a83d9e13bd4e1b4aeaa45e0cd8549f18e0f1c39c85a5daf3b1b27ddfdf830c`;
[first attempt failure](performance-20261103-r1/FAILURE.json);
[human budget receipt](../steering-ppo-ad-dpt-20261102-r1/SOURCE-BUDGET-60PERARM-20261103-r1.json).
