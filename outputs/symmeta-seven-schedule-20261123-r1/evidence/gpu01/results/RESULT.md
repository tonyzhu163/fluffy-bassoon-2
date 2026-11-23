# Seven-shift schedule control

Three paired Perceiver seeds on recorded UR5e supports. Differences are candidate minus original. Negative differences favor the candidate for errors. Positive differences favor it for accuracies. Intervals resample eight whole tasks and keep seeds paired.

| Panel | Seed | Original yaw MAE (°) | Candidate yaw MAE (°) | Paired difference [95%] |
| --- | ---: | ---: | ---: | ---: |
| ood_combined | 41 | 97.08 | 88.06 | -9.02 [-14.33, -3.68] |
| ood_combined | 42 | 92.34 | 79.03 | -13.32 [-21.64, -5.93] |
| ood_combined | 43 | 97.48 | 83.03 | -14.45 [-19.12, -9.54] |
| overall | 41 | 64.90 | 59.36 | -5.53 [-9.09, -1.99] |
| overall | 42 | 61.88 | 53.09 | -8.79 [-14.37, -3.85] |
| overall | 43 | 65.13 | 56.16 | -8.97 [-12.44, -5.22] |

## familiar (32 traces)

| Seed-averaged metric | Original | Candidate | Paired difference [95%] |
| --- | ---: | ---: | ---: |
| yaw_mae_deg | 0.6451 | 1.8756 | +1.2305 [+0.4273, +2.3768] |
| operator_error | 0.0635 | 0.0856 | +0.0221 [-0.0023, +0.0473] |
| scale_rel_err | 0.0757 | 0.0755 | -0.0001 [-0.0271, +0.0298] |
| grip_acc | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] |
| all_correct | 0.8542 | 0.8125 | -0.0417 [-0.1354, +0.0521] |
| yaw_correct | 1.0000 | 0.9792 | -0.0208 [-0.0625, +0.0000] |
| scale_correct | 0.8542 | 0.8333 | -0.0208 [-0.1146, +0.0625] |
| yaw_median_deg | 0.3469 | 0.6932 | +0.3463 [+0.1583, +0.6499] |

## compositional_ood (32 traces)

| Seed-averaged metric | Original | Candidate | Paired difference [95%] |
| --- | ---: | ---: | ---: |
| yaw_mae_deg | 108.7329 | 97.1693 | -11.5636 [-15.6858, -7.1813] |
| operator_error | 1.6006 | 1.5246 | -0.0760 [-0.1424, -0.0129] |
| scale_rel_err | 0.4450 | 0.4792 | +0.0342 [+0.0242, +0.0463] |
| grip_acc | 0.9583 | 0.9167 | -0.0417 [-0.1042, +0.0000] |
| all_correct | 0.0000 | 0.0000 | +0.0000 [+0.0000, +0.0000] |
| yaw_correct | 0.0312 | 0.1042 | +0.0729 [+0.0312, +0.1146] |
| scale_correct | 0.0000 | 0.0104 | +0.0104 [+0.0000, +0.0312] |
| yaw_median_deg | 89.6309 | 88.4913 | -1.1396 [-1.5510, -0.6955] |

## continuous_angle_ood (32 traces)

| Seed-averaged metric | Original | Candidate | Paired difference [95%] |
| --- | ---: | ---: | ---: |
| yaw_mae_deg | 82.5298 | 69.5707 | -12.9591 [-18.9874, -7.2004] |
| operator_error | 1.2356 | 1.0868 | -0.1488 [-0.2212, -0.0807] |
| scale_rel_err | 0.2350 | 0.2441 | +0.0091 [-0.0031, +0.0213] |
| grip_acc | 0.9792 | 0.9479 | -0.0312 [-0.0729, +0.0000] |
| all_correct | 0.1354 | 0.1562 | +0.0208 [+0.0000, +0.0521] |
| yaw_correct | 0.1458 | 0.2292 | +0.0833 [+0.0417, +0.1250] |
| scale_correct | 0.4167 | 0.4271 | +0.0104 [-0.0625, +0.0833] |
| yaw_median_deg | 77.5722 | 40.9277 | -36.6445 [-44.2754, -20.8767] |

## ood_combined (64 traces)

| Seed-averaged metric | Original | Candidate | Paired difference [95%] |
| --- | ---: | ---: | ---: |
| yaw_mae_deg | 95.6314 | 83.3700 | -12.2614 [-17.0357, -7.3872] |
| operator_error | 1.4181 | 1.3057 | -0.1124 [-0.1600, -0.0647] |
| scale_rel_err | 0.3400 | 0.3616 | +0.0216 [+0.0138, +0.0302] |
| grip_acc | 0.9688 | 0.9323 | -0.0365 [-0.0833, +0.0000] |
| all_correct | 0.0677 | 0.0781 | +0.0104 [+0.0000, +0.0260] |
| yaw_correct | 0.0885 | 0.1667 | +0.0781 [+0.0417, +0.1094] |
| scale_correct | 0.2083 | 0.2188 | +0.0104 [-0.0312, +0.0521] |
| yaw_median_deg | 89.5707 | 86.3530 | -3.2177 [-8.2793, -1.2699] |

## overall (96 traces)

| Seed-averaged metric | Original | Candidate | Paired difference [95%] |
| --- | ---: | ---: | ---: |
| yaw_mae_deg | 63.9693 | 56.2052 | -7.7641 [-11.1084, -4.3843] |
| operator_error | 0.9665 | 0.8990 | -0.0676 [-0.1003, -0.0358] |
| scale_rel_err | 0.2519 | 0.2663 | +0.0144 [+0.0031, +0.0260] |
| grip_acc | 0.9792 | 0.9549 | -0.0243 [-0.0556, +0.0000] |
| all_correct | 0.3299 | 0.3229 | -0.0069 [-0.0382, +0.0243] |
| yaw_correct | 0.3924 | 0.4375 | +0.0451 [+0.0174, +0.0694] |
| scale_correct | 0.4236 | 0.4236 | -0.0000 [-0.0417, +0.0451] |
| yaw_median_deg | 52.5159 | 33.6884 | -18.8276 [-27.9368, -15.2585] |
