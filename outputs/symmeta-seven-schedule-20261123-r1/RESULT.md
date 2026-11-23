# Seven-shift Perceiver schedule control on recorded UR5e supports

Changing the seven-shift training schedule reduced mean held-out yaw error from **95.63° to 83.37°** on the same 64 out-of-distribution (OOD) supports. The paired difference was **−12.26°** (95% whole-task bootstrap interval **−17.04° to −7.39°**). All three seeds improved. Averaging over seeds, all eight tasks improved, by 2.49° on `libero_10/8` through 22.04° on `libero_goal/3`. Absolute error remained large. A quarter-turn is 90°, and only 16.7% of the candidate's OOD predictions fell within the historical 22.5° yaw tolerance, versus 8.9% for the original. [Eight task estimates and independent readback](evidence/verification.json)

| Reader seed | Historic OOD yaw MAE | Schedule-only OOD yaw MAE | Schedule-only minus historic, paired 95% interval |
| --- | ---: | ---: | ---: |
| 41 | 97.08° | 88.06° | −9.02° [−14.33°, −3.68°] |
| 42 | 92.34° | 79.03° | −13.32° [−21.64°, −5.93°] |
| 43 | 97.48° | 83.03° | −14.45° [−19.12°, −9.54°] |

On the 32 compositional OOD supports the mean yaw difference was −11.56° [−15.69°, −7.18°]. On the 32 continuous-angle OOD supports it was −12.96° [−18.99°, −7.20°]. On the 32 familiar supports, yaw MAE increased from 0.65° to 1.88°, a difference of +1.23° [+0.43°, +2.38°]. The intervals use 20,000 draws of eight whole tasks with replacement (RNG seed 20261123). Each draw keeps both predictions, every included shift and all three training seeds together. The intervals describe variation across these eight tasks conditional on the three fitted readers per protocol. [Full production estimates, including gain, gripper and joint correctness](evidence/gpu01/results/summary.json)

## Separate training comparisons

| Perceiver training protocol | OOD yaw MAE, 64 supports | Planar operator error, all 96 supports |
| --- | ---: | ---: |
| Historic seven shifts, archived checkpoints rescored | 95.63° ± 2.33° | 0.9665 ± 0.0225 |
| Same seven shifts, lower rate and warmup | 83.37° ± 3.69° | 0.8990 ± 0.0292 |
| PR7 continuous labels, lower rate and warmup | 5.58° ± 0.18° | 0.1461 ± 0.0045 |
| PR7 continuous labels, original scaling-sweep schedule | 96.86° ± 13.98° | 1.3799 ± 0.1102 |

The table gives the mean and population SD of three seed means. Operator error is the dimensionless residual after correcting the true planar command map with the predicted inverse, measured as $\lVert A\widehat A^{+}-I\rVert_F/\sqrt{2}$. Zero means exact correction. With gain correct, a 60° yaw miss scores 1 and a 90° miss scores √2. The paired seven-shift change was **−0.0676** [−0.1003, −0.0358] across all 96 supports, including familiar cases. The experiment measured offline calibration. [Operator definition](../../src/fluffy_bassoon/xemb/wider_family.py) · [Per-seed operator estimates](evidence/verification.json)

- **Within seven-shift training**, both arms used the original 1,344 passive Panda/Sawyer traces, the same seed-specific initialization and minibatch draws, 32 positional slots, batch 64 and 3,500 updates. The peak rate fell from 0.001 to 0.0003, the cosine endpoint fell proportionally and the new schedule warmed up for 200 counts. The comparison estimates those changes together. [Run plan](PLAN.md) · [Runtime contract](evidence/gpu01/results/contract.json)
- **Within PR7 continuous training**, changing the peak from 0.0015 to 0.0003 with 200 warmup counts reduced OOD yaw MAE by 91.28°. Both PR7 arms used the same 1,466 Stage 3 source traces, continuous interface relabeling, minibatch sampling, 24 positional slots, batch 32 and 2,500 updates. [PR7 evaluator](../symmeta-wider-family-20261123-r1/eval_wider_family.py) · [PR7 seed records](../symmeta-wider-family-20261123-r1/gpu01w454/wider-family-summary.json)
- **Across protocols**, the tuned continuous reader averaged 77.79° less OOD yaw error than the seven-shift schedule-only reader and 90.05° less than the historic reader. The first gap persisted with the same peak rate and warmup count. Source traces, interface sampling, positional capacity, minibatches and decay horizon differed. The individual contributions remain unresolved. PR7 retained seed aggregates for these readers, so cross-protocol task intervals are unavailable. [PR7 readback](../symmeta-wider-family-20261123-r1/gpu01w454/verification.json)

## Evidence and provenance

Baige Distributed Training job `job-v9luygbnvwq2` ran pin `ba3c41e25575d07d7bd8bbfaae2797e7cf9dc0a9` on one RTX PRO 6000D allocation (CUDA reported `NVIDIA RPBZZZ6`). The launcher exited 0 at 16:10:08 Shanghai on November 23 after 267 seconds (0.0742 GPU-hours). All three 3,500-update checkpoints were exported, reloaded and scored. [Original launcher](launch.sh) · [Launch receipt](evidence/gpu01/launch.txt) · [Device probe](evidence/gpu01/gpu.json) · [Run log](evidence/gpu01/run.log) · [Exit](evidence/gpu01/exit.txt) · [Remote checkpoint directory](ssh://baige-dev/mowenjie/fluffy-bassoon-2/outputs/symmeta-seven-schedule-20261123-r1/gpu01/results/checkpoints/)

Independent CPU readback verified all six checkpoint hashes and metadata, the ordered 1,344 source and 96 test tape hashes, and the same 288 seed–support pairs. Recomputing the errors and correctness indicators from all 576 saved predictions reproduced all 160 published metric records, including paired intervals and within-draw medians, to a maximum absolute discrepancy of 1.1 × 10⁻¹⁴. Against PR7's direct historic-checkpoint scores, the largest per-seed discrepancies were 0.00060° OOD yaw and 0.0000054 all-panel operator error. A separate reconstruction from the 16 original campaign shards, which store rounded predictions, gave OOD yaw means of 97.07234°, 92.33875° and 97.47281°. Each was within 0.00372° of its PR7 seed score. [Hash and archive receipt](evidence/receipt.json) · [Independent calculation](evidence/recompute.py) · [Seed 41 records](evidence/gpu01/results/paired-cases-41.json.gz) · [Seed 42 records](evidence/gpu01/results/paired-cases-42.json.gz) · [Seed 43 records](evidence/gpu01/results/paired-cases-43.json.gz)
