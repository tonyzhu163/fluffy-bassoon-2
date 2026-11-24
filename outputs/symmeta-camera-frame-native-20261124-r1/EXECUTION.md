# Native execution

Parent submitted Baige Distributed Training job `job-t8kuarphk38v` on November 24, 2026 for one RTX 4090, attempt `gpu01`. The requested scheduler duration is 115 minutes and payload wall cap is 6,800 seconds. The fixed code commit is `5000e5ee526d0ee3b34162440327cfc3e4831482`. The shared output root is `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-frame-native-20261124-r1`.

| Shanghai time | Observation |
|---|---|
| 13:31 | CPU03 exited 0. Both layouts per task and all eight true maps passed exact checks. |
| 13:33 | Prepare01 exited 0 with 64 physical starts, each with A, B, shipped challenged and true_map. |
| 13:35 | Parent reported the job ID. Waiting for scheduler allocation and the first paired native record. |
| 13:40 | The job passed its exclusive RTX 4090 evaluation probe. Source and score-table checks precede the first native pair. |
| 13:47 | First complete quartet verified. The producer checked exact actions within the job. The independent CPU verifier reconstructed all physical commands against a float64 reference with a derived float32 rounding bound. The GPU and CPU-prepared panel hashes matched. [First quartet](evidence/gpu01/first-native-pair.json) |
| 14:07 | 24/64 four-arm starts complete, 96 native episodes. Next check 14:27. |
| 14:27 | 57/64 four-arm starts complete, 228 native episodes. The remaining starts belong to the final task. |
| 14:31 | Exit 0. All 64 starts and 256 episodes complete. Independent CPU readback reproduced all 64,249 native controls, 12,899 policy queries, four success counts, 21 A/B discordances and the exact paired task-bootstrap interval. |

One RTX 4090 ran for 3,345 launcher seconds, or 0.9292 GPU-hours. Native ran once, on one allocation. The CPU qualification and post-close readback used no GPU. [Exit](evidence/gpu01/exit.txt) · [Probe](evidence/gpu01/gpu.json) · [Execution log](evidence/gpu01/run.log.gz) · [Independent readback](evidence/gpu01/independent-readback.json)
