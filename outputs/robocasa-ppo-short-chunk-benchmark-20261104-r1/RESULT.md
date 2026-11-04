# One real-weight PPO update with ten executed controls

The shorter-chunk candidate finished one full update on Baige Pro6K job `job-rfd4ton01vj3`. At 76,800 TRAIN controls, it took **283.165 seconds**, 1.80× the original Stage 1 PPO first update. Both used 128 TRAIN episodes, eight GPUs, two actor ranks, microbatch 16 and full sampling records. Each recipe has one measured first update. The original recipe's median interval over updates 2–7 was 168.85 seconds.

| Measurement | Stage 1 PPO, H50/K10 | Candidate, execute 10/K5 |
| --- | ---: | ---: |
| Predicted / executed controls | 50 / 50 | 50 / 10 |
| Action calls per 600-control episode | 12 | 60 |
| TRAIN chunk samples | 1,536 | 7,680 |
| Global minibatch, chunks | 256 | 512 |
| Adam steps per actor rank | 6 | 15 |
| End-to-end first update | 156.909 s | 283.165 s |
| Rollout collection | 120.1 s | 139.2 s |
| Policy inference within collection | 22.173 s | 77.831 s |
| Actor training | 35.472 s | 140.0 s |

The candidate used noise 0.5. On real weights, both actor ranks replayed a [16, 10, 12] likelihood tensor covering 120 executed coordinates per chunk. The maximum absolute chunk log-ratio and VLM-prefix value differences were both zero before the first gradient update. Both actor ranks finished 15 Adam steps with finite metrics and unchanged frozen VLM weights.

The maximum sampled device memory was 61,153 MiB, about 59.7 GiB. Actor peak allocated PyTorch memory was 15.94 GiB per actor rank. The complete run occupied 42.59 GiB including its initial and final checkpoints and raw records. Its 64-case before panel scored 41 successes and the training collection scored 88/128. Both collections preceded SGD. The job stopped at the requested update-1 checkpoint and all 22 checkpoint files verified. This run measures feasibility and time. A post-update score was outside its scope.

Baige recorded 22:14:08–22:24:57 Shanghai, or 1.442 scheduler GPU-hours including startup. The worker clock recorded 1.371 GPU-hours. The benchmark and matched DEV64 consumed 2.135 scheduler GPU-hours combined. [Benchmark audit](BENCH-AUDIT.json) · [resolved contract](FLOW-CONTRACT.json) · [scheduler proof](BAIGE-FINAL.png).
