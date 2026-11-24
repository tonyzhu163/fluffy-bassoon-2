# Native retries with a learned command reader

The tuned Perceiver learned to correct held-out command shifts without SymMeta’s explicit Procrustes yaw solve. It succeeded on 54.2–79.7% of out-of-distribution (OOD) retries across four arm–policy panels. Its lower recorded yaw error on UR5e did not predict higher native success there. On UR5e π₀.₅ it completed 17 fewer retries than fresh, same-seed SymMeta overall and 12 fewer on the OOD subset. All four OOD intervals include zero. The experiment establishes that a learned reader can perform this correction on these tasks. It leaves formal equivalence and the size of any advantage on new tasks unresolved. [Official summary](native02g4090/summary.json) · [Independent readback](evidence/native02g4090/readback.json)

## Paired native success

Each panel has 96 cases per reader seed, including 64 OOD cases. Differences are tuned Perceiver minus fresh SymMeta in percentage points. Brackets give paired 95% whole-task bootstrap intervals, rounded to two decimals.

| Arm and frozen policy | Tuned, all | SymMeta, all | Difference [95% interval] | Tuned, OOD | SymMeta, OOD | Difference [95% interval] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UR5e π₀.₅ | 209/288 | 226/288 | −5.90 [−12.50, −0.35] | 141/192 | 153/192 | −6.25 [−15.10, +1.04] |
| IIWA π₀.₅ | 230/288 | 223/288 | +2.43 [−2.78, +9.03] | 153/192 | 147/192 | +3.13 [−4.17, +9.90] |
| Kinova3 π₀.₅ | 199/288 | 197/288 | +0.69 [−5.56, +6.60] | 129/192 | 126/192 | +1.56 [−6.25, +8.33] |
| UR5e π₀ | 156/288 | 166/288 | −3.47 [−12.15, +4.86] | 104/192 | 107/192 | −1.56 [−12.50, +8.85] |

The eight BASE task identities were `libero_10/{5,8}`, `libero_goal/{3,6}`, `libero_object/{1,8}` and `libero_spatial/{0,8}`. Each contributed four familiar, four compositional OOD and four continuous-angle OOD settings. The interval uses the exact percentile distribution over all 8⁸ ordered resamples of eight tasks with replacement, retaining both readers, all included shifts and seeds 41, 42 and 43 together. It is conditional on those seeds and the released query layouts. The independent record gives counts and paired differences by seed in `per_seed`, intervals in `per_seed_task_cluster_ci95_pp`, and sample SDs across seeds in `seed_sd_pp` for every panel and split. [Per-seed results and intervals](evidence/native02g4090/readback.json) · [Compressed paired outcomes](evidence/native02g4090/pairs.csv.gz)

## What the training comparison establishes

The historical seven-shift Perceiver checkpoints achieved 2.1–5.7% native OOD success across these panels (three-seed means). The tuned reader predicts yaw, planar gain and gripper sign directly and uses the same factorized command interface as SymMeta. On 64 recorded OOD UR5e supports, changing the schedule on the original 1,344 seven-shift traces reduced yaw error from 95.63° to 83.37°. On 1,466 continuously relabeled Panda/Sawyer traces, lowering the peak rate and adding warmup reduced it from 96.86° to 5.58°. The two within-dataset comparisons isolate their respective schedule changes. Across datasets the source traces, interface sampling, positional slots, batch size and update budget also changed, so the contribution of continuous labels alone remains unidentified. [Historical three-seed records](../libero-xemb-ur5e-scale-20261120-r1/evidence/) · [UR5e π₀.₅ seed 41](../xemb-icml-campaign-20261121-r1/evidence/ur5e-pi05-summary.json) · [Seven-shift control](../symmeta-seven-schedule-20261123-r1/RESULT.md) · [Continuous-label control](../symmeta-wider-family-20261123-r1/RESULT.md)

The original tuned run retained aggregate scores only. Before native outcomes were available, the first completed refit (`train02pro`, RTX PRO 6000D, training revision `ba3c41e2`) supplied all predictions. Its three reloaded checkpoints reproduced their pre-save outputs exactly and averaged 5.82° OOD yaw error against 7.04° for the original SymMeta readers. A later RTX 4090 refit was scored offline. Each native retry applied its reader’s full correction, so these success differences cannot be attributed to yaw alone. [Training recipe](TRAINING.md) · [Refit measurements](evidence/train02pro/training-sanity.json) · [Frozen predictions](evidence/train02pro/predictions.json)

## Execution and independent verification

Baige Distributed Training job `job-2s6q9a9xe96e`, attempt `native02g4090`, ran evaluator `d568535b758f2b97b382f9b3e280376e769b5942` on four RTX 4090s. It exited 0 after 8,543 seconds, or **9.492 launcher GPU-hours**. Both readers ran fresh on all 1,152 pairs (2,304 native episodes, 768 OOD pairs). Within each pair the passive layout-11 support tape, frozen policy, layout-13 query, full simulator reset, prompt, policy-noise bank and first uncorrected policy proposal matched. [Launch configuration](native02g4090/config.txt) · [Device probe](native02g4090/gpu.json) · [Exit receipt](native02g4090/exit.txt)

The independent CPU verifier checked every raw pair against the frozen predictions and query schedule, verified reset and opening hashes, and reconstructed success from the per-control native rewards, returns and terminal flags. All four panels reproduced the official summary. The full records, including both reward traces, are at `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/symmeta-perceiver-retry-20261123-r1/native02g4090/pairs/`. [Verifier](readback.py) · [Independent receipt](evidence/native02g4090/readback.json) · [Pair-level evidence](evidence/native02g4090/pairs.csv.gz)
