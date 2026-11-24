# A second tuned reader accounts for the pooled target yaw improvement

On 256 held-out-arm supports, adding a geometric SymMeta reader to a tuned Perceiver at equal weight gave **6.133°** yaw MAE. Adding an independently seeded tuned Perceiver gave **6.132°**. The paired difference was **+0.001°** (95% task interval **−0.719° to +0.785°**). Transferring the weights fitted on source calibration tasks produced a **+0.511°** difference (**−0.549° to +1.970°**). Both target intervals allow a gain or a loss. On the separate Panda/Sawyer source test, geometry reduced error against the same two-reader controls by **2.154°** at the midpoint and **2.457°** after calibration. The source advantage did not carry through to the pooled held-out-arm comparison. Close this screen without a native run or automatic replication. [Pre-scoring registration](PLAN.md) · [Full estimates and paired intervals](score01/metrics.json)

## Target and source comparisons

Each row reports yaw MAE in degrees. “Two tuned” covers both the fixed and calibrated homogeneous mixtures because source calibration selected exactly ½. “Mixed cal” assigns 63.4967% to the geometric auxiliary. Both differences are mixed minus two tuned, so a negative value favors geometry. Brackets give joint 95% intervals that resample calibration and test tasks.

| Population | Single tuned | Two tuned | Mixed 50:50 | Mixed cal | Midpoint difference [95% CI] | Calibrated difference [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| UR5e π₀.₅, OOD | 5.822 | 5.497 | 5.148 | 5.465 | −0.349 [−1.271, +0.329] | −0.033 [−1.150, +1.205] |
| IIWA π₀.₅, OOD | 4.499 | 4.254 | 4.023 | 4.568 | −0.231 [−1.230, +0.846] | +0.315 [−1.031, +1.894] |
| Kinova3 π₀.₅, OOD | 5.832 | 5.500 | 5.630 | 6.294 | +0.130 [−0.414, +0.591] | +0.794 [−0.354, +2.197] |
| UR5e π₀, OOD | 9.578 | 9.276 | 9.730 | 10.245 | +0.453 [−0.692, +2.383] | +0.969 [−0.608, +4.396] |
| **Four target panels, OOD** | **6.433** | **6.132** | **6.133** | **6.643** | **+0.001 [−0.719, +0.785]** | **+0.511 [−0.549, +1.970]** |
| Four target panels, familiar shifts | 5.752 | 5.531 | 5.211 | 5.310 | −0.320 [−0.995, +0.350] | −0.222 [−1.030, +0.751] |
| Four target panels, all 12 shifts | 6.206 | 5.932 | 5.825 | 6.198 | −0.106 [−0.689, +0.384] | +0.267 [−0.551, +1.344] |
| Panda source-test | 10.556 | 10.189 | 8.507 | 8.271 | −1.682 [−2.737, −0.761] | −1.918 [−3.403, −0.526] |
| Sawyer source-test | 11.376 | 11.169 | 8.544 | 8.173 | −2.625 [−4.572, −0.910] | −2.996 [−5.544, −0.710] |
| **Two source arms** | **10.966** | **10.679** | **8.525** | **8.222** | **−2.154 [−3.561, −0.904]** | **−2.457 [−4.364, −0.717]** |

The two-tuned midpoint improved on single tuned by **0.301°** on target (paired change −0.301°, 95% CI −0.483° to −0.182°). The mixed midpoint improved by 0.300° (−0.967° to +0.482°). The calibrated mixed ensemble had 0.210° more error than single tuned (−0.822° to +1.668°). On the four compositional shifts, the midpoint mixed-minus-homogeneous difference was −0.892° [−2.308°, +0.246°]. On the four continuous-angle shifts it was +0.894° [+0.175°, +1.748°]. Opposing differences by shift family account for the near-zero pooled point estimate. These subgroups share the same eight tasks and were drawn from previously inspected panels. [Task, shift, panel and seed estimates](score01/metrics.json)

Planar operator error is ‖A Â⁻¹ − I‖F/√2. It uses the labeled gain and yaw and the same estimated gain from the base tuned reader for every method. Gripper sign also comes from that base reader and was correct on all calibration, source-test and target rows. No invalid, antipodal or equal-endpoint case occurred in any ordered pair. [Raw calibration rows](score01/cal-pairs.csv.gz) · [Raw source-test rows](score01/eval-pairs.csv.gz) · [Raw target rows including familiar shifts](score01/target-pairs.csv.gz)

| Mean operator error | Single tuned | Two tuned | Mixed 50:50 | Mixed cal | Midpoint difference [95% CI] | Calibrated difference [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| Target OOD | 0.1660 | 0.1623 | 0.1613 | 0.1677 | −0.0010 [−0.0120, +0.0107] | +0.0053 [−0.0090, +0.0260] |
| Source-test | 0.2255 | 0.2210 | 0.1901 | 0.1863 | −0.0308 [−0.0508, −0.0135] | −0.0346 [−0.0606, −0.0106] |

## Pairing and uncertainty

Each case used the same three five-control chunks for every method. For each of six ordered seed pairs, the base tuned reader supplied yaw, gain and gripper sign. Only its yaw could change. The two-reader methods moved the prediction along the shortest arc from the base yaw to the independent auxiliary yaw. The signed differences below compare mixed with homogeneous on exactly the same cases and seed ordering.

| Base → auxiliary seed | Target single | Target midpoint Δ | Target calibrated Δ | Source single | Source midpoint Δ | Source calibrated Δ |
|---|---:|---:|---:|---:|---:|---:|
| 41 → 42 | 6.412 | +0.013 | +0.461 | 11.359 | −2.526 | −2.834 |
| 41 → 43 | 6.412 | +0.269 | +0.859 | 11.359 | −1.648 | −1.960 |
| 42 → 41 | 6.511 | −0.323 | +0.120 | 11.947 | −2.433 | −2.959 |
| 42 → 43 | 6.511 | +0.067 | +0.652 | 11.947 | −1.585 | −2.023 |
| 43 → 41 | 6.376 | −0.067 | +0.435 | 9.592 | −2.302 | −2.488 |
| 43 → 42 | 6.376 | +0.048 | +0.539 | 9.592 | −2.429 | −2.476 |

The 256 target OOD cases are 4 panels × 8 common tasks × 8 shifts, producing 1,536 ordered comparisons. The full 12-shift export contains 384 cases and 2,304 ordered comparisons. Source calibration contains 224 physical supports over four tasks. Source testing contains 448 supports over eight different tasks (2 arms × 4 starts × 7 shifts per task), producing 2,688 ordered comparisons. A source start recurs across the seven interventions. Each target task recurs across all four panels, and UR5e occurs under both policies. The six pairings reuse three checkpoints per architecture. Reversing two tuned seeds gives the same midpoint yaw even though base gain can change.

The 20,000 paired bootstrap draws resampled whole task IDs and kept all cases, shifts, panels and seeds of each drawn task together. Joint intervals independently resampled the four calibration tasks and used the exact optimum for each of their 35 possible resampling count vectors. With the calibrated weights held fixed, mixed-minus-homogeneous yaw intervals were −0.338° to +1.488° on target and −4.249° to −0.835° on source. Fixed-midpoint intervals coincide across the two analyses. The intervals describe task variation conditional on these readers, arms, support archives and policies. The results file includes paired intervals for every ordered seed pair and arm, and point estimates for every arm-by-seed pair and task. [Bootstrap implementation](score.py) · [Full bootstrap results](score01/metrics.json)

## What was trained and what transferred

| Reader | Historical readers used on target | Restricted readers used on source | Training recipe per reader |
|---|---|---|---|
| Tuned Perceiver | First completed train02pro refits of the original recipe, 1,466 stage-3 Panda/Sawyer trajectories from all 24 source tasks, 128 physical interfaces plus continuous minibatch relabels | 732 trajectories on 12 reader-fit tasks (426 Panda, 306 Sawyer), same 128 interfaces and relabeling recipe | 24 position slots, batch 32, 2,500 updates, 80,000 sampled examples, peak rate 0.0003 and 200 warmup updates |
| Geometric SymMeta | Published Stage 1 seven-shift checkpoints, 1,344 original scale-archive supports from the same 24 source tasks | 672 original scale-archive supports on the same 12 reader-fit tasks, two arms, four starts, seven physically executed shifts | Geometric Procrustes yaw reader with DIS flow and vertical parallax, 32 slots, batch 64, 3,500 updates, 224,000 sampled examples, cosine decay from 0.001 |

The four calibration and eight source-test tasks were excluded from both restricted reader fits. Both sets of task IDs occurred in the historical full-data reader fits. Every reported source score uses the restricted refits. Target tasks were separate, and target evaluation used the original train02pro full-data tuned export and published geometric checkpoints. PR7's earlier training run retained aggregate scores. On UR5e π₀.₅, the archived tuned refits averaged 5.82° OOD yaw error, compared with the original PR7 aggregate of 5.58°. The parent registration fixed which export supplied the comparison. [Full-data export and training sanity](../symmeta-perceiver-retry-20261123-r1/TRAINING.md) · [Restricted reader fits and task membership](../symmeta-omitted-chunk-20261124-r1/INPUTS.md) · [SymMeta manuscript](../../papers/icml-xemb-sysid/main.tex)

Calibration minimized full-yaw circular MAE on the four restricted calibration tasks, giving every task equal mass and every trace and ordered pair within a task equal mass. The source-only optima gave auxiliary weights **1/2** for two tuned readers and **3864318/6085853** for the geometric auxiliary. Calibration MAEs were 5.006° and 4.241°, respectively. The optimizer checks endpoints, every zero-error and antipodal-error crossing, and the midpoint using rational arithmetic. Among equal global minima it chooses the weight nearest ½, then the smaller weight. The homogeneous objective is symmetric because both orders of every seed pair enter. The two weights were saved before opening the source-test labels or target predictions for scoring. The earlier 62% geometric weight came from a grid of same-seed pairs. [Exact solver](calibrate.py) · [Pre-evaluation weights and input digests](score01/weights.json) · [Earlier same-seed registration and result](../symmeta-omitted-chunk-20261124-r1/SCORE.md)

Independent-seed ensembling is an established comparator. Both two-reader controls expend a second reader inference on the same support. Their training histories and per-reader costs differ. For example, two tuned readers consumed 160,000 sampled training examples between them, while a tuned and geometric reader consumed 304,000 from two different archives. The calibrated methods also inherit supervision from the restricted source refits and calibration labels. The source cohorts use the same seven physical shifts on which the geometric recipe trained. The held-out-arm cohorts also change the shift composition, checkpoint training size and, on one panel, the frozen policy. Those changes prevent attribution of the source-to-target loss of advantage to morphology alone. A distinct geometric benefit on native task success would require fresh evidence. This screen measured yaw and operator error only. [Deep Ensembles, Lakshminarayanan et al. (2017)](https://papers.nips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) · [Prior same-seed target results](../symmeta-yaw-blend-20261124-r1/RESULT.md)

## Verification and reproduction

The source readback matched the restricted reader export with the prior compact judge exports, keyed by case and seed, including the two full yaw columns, validity, base gain and sign. The earlier reference decoder replay verified all 16,128 full and omitted predictions. This study used only full predictions. The target readback compared the original four-panel export with 1,152 individually hashed native-pair records and recovered all 2,304 original factor triples exactly. It used the corrections in those records to verify inputs. It never used native success as an endpoint. We reproduced 32 previous target summary entries and four source entries, including the old same-seed source and target midpoints. An independent complex-angle implementation reproduced all 31,680 emitted yaw values within 6 × 10⁻¹⁴ degrees, directly checked the calibration loss for every task multiset, and reconstructed both primary contrasts and both kinds of bootstrap interval on source and target. [Raw readback receipt and artifact hashes](score01/readback.json) · [Independent verifier](readback.py) · [Historical decoder replay](../symmeta-omitted-chunk-20261124-r1/reader-evidence/replay01/replay.json)

Baige CPU cores 26 and 27 ran the registered scorer and verifier at parent-supplied revision `c003bf5cc112261442f9c4613d25710edb8aef16`. The first shell affinity-status command stopped before qualification or empirical access. The corrected launch recorded its affinity, passed the wrap, missing-angle and tie fixtures, and completed at 11:02:19 Shanghai. Scoring and readback used 2.32 and 0.82 process CPU seconds, respectively. Incremental GPU use was zero. The prior restricted reader fits used 0.1097 GPU-hours. Historical full-data fitting, reader inference and native collection were reused. [CPU launch receipt](launch02.txt) · [Known-answer checks](qualify.py) · [Qualification output](qualification.log) · [Completed execution](completed.txt)

The weights file SHA-256 is `4b63a4c8b99d261b6a73c45adf528911993e255a745a1ce75425f5d5372cb23c`. The metrics digest is `740b132d24e9e02a522ccdb275c063e761b2d4508d47aa07ce612dbfc97997e2`. The readback digest is `2b09e8d5e5a458d02151d52c7872290a41cf24f62c5500171c93ace670448ced`. Source archive, checkpoint and native export digests appear in the receipts. To reproduce on the pinned Baige checkout, choose a fresh output path.

```bash
set -eu
cd /mowenjie/fluffy-bassoon-2/setups/symmeta-perceiver-followups-20261123-r1/src-c003bf5cc112
run=outputs/frozen-transfer-ensemble-20261124-r1
out=/mowenjie/fluffy-bassoon-2/$run/replay01
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
taskset -c 26,27 "$py" -B "$run/qualify.py"
taskset -c 26,27 "$py" -B "$run/score.py" --pin "$(git rev-parse HEAD)" --out "$out"
taskset -c 26,27 "$py" -B "$run/readback.py" "$out"
```
