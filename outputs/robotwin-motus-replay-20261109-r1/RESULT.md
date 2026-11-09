# Released Motus pi0.5 teacher-forced replay

Completed on Baige Pro6K, job `job-x42r92kv2ls1`, with frozen weights and no
simulator calls or optimizer updates. All 20 expert demonstrations (episodes
0–19, stride 4) completed: **644 queries and 3,220 policy inferences** across
five variants sharing each query's noise. The source was
`d59199917ac51eb6c0ab83c4a0926d9c3e6836c5`, with the qualified reference
OpenPI runtime, released checkpoint, H32 and K10 ODE sampler.

The base 32-step arm MAE was **0.015408 rad**, versus **0.078913 rad** for
holding the current state: **80.48% lower**. RGB-channel swapping increased
this error by 2.63%, the scene prefix increased it by 26.25%, and exchanging
the wrists increased it by 169.81%. Unseen instructions changed it by −0.62%.
These are descriptive paired reconstruction errors on expert frames.

All means below use the prospective shift-1 alignment. Gripper error is in the
dataset's 0–1 units; arm error is in radians.

| Variant | Arm MAE, first target | Arm MAE, first 8 | Arm MAE, all 32 | Gripper MAE, all 32 |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.007164 | 0.011051 | 0.015408 | 0.020155 |
| RGB channels exchanged | 0.007149 | 0.011376 | 0.015813 | 0.021708 |
| Unseen instruction | 0.007161 | 0.010996 | 0.015311 | 0.019996 |
| Scene prefix | 0.007356 | 0.012403 | 0.019452 | 0.020600 |
| Wrists exchanged | 0.007314 | 0.017836 | 0.041571 | 0.154637 |
| Hold current state | 0.006085 | 0.027068 | 0.078913 | 0.128217 |

The base policy tracks the demonstrated action chunks substantially better
than holding state, and none of the tested input variants provides a substantial
improvement. At the first target, holding state has lower error than the base
prediction. This diagnostic narrows the input-mismatch investigation; the cause
of the native 27%/25% versus published 49%/53% success gap remains unresolved.
Expert-frame reconstruction alone cannot establish training-pipeline identity
or closed-loop success.

The independent Linux CPU audit, source `b1ca0582`, reconstructed the complete
episode/stride grid, both target alignments and every arm/gripper horizon
metric directly from the archived expert states and saved prediction arrays.
Every row passed its float32 serialization envelope; maximum difference was
`5.3387540832616764e-08`. All aggregate summaries passed. The downloaded archive
passed ZIP CRC checks and every scientific file matched the audit's SHA-256
and byte count.

The scheduler reported success at **2026-11-09 16:45:27 Shanghai**, with its
sole master completed and zero restarts; the GPU is released. Worker time was
662 GPU-seconds, including setup, with 580.46 seconds in the replay loop.
Conservative charge from scheduler creation through terminal success, including
queue time, is **740 GPU-seconds (0.20556 GPU-hours)**. The 2,700-second replay
reservation is closed; concurrent Sidney SFT remains separately reserved in
the shared Epoch2 ledger.

Evidence: [raw result](RESULT.json), [independent readback](READBACK.json),
[worker exit](EXIT.json), [allocated GPU probe](gpu-probe.json),
[scheduler closeout](SCHEDULER-CLOSEOUT.json), and [execution](EXECUTION.json).
The saved predictions remain locally in `predictions.npz` and in the original
BOS run directory; their byte count and SHA-256 are recorded in the readback.
The complete server output is
`/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1/motus-replay-20261109-r1/replay-01`.
