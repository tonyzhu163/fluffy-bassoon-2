# Which directions should a frozen VLA correct after 15 passive controls?

Rank-one truncation gave no consistent predictive advantage over smooth ridge on genuine simulated arm changes. On Kinova3 the matched smooth filter had slightly lower displacement error. The constrained inverse makes a separate, testable choice about the commands issued at deployment.

## Prediction across tasks

Mean Euclidean error in millimetres per five-control chunk. The final column compares the two filters on the same support at exactly the same effective degrees of freedom. Its approximate 95% interval drops each of the eight task identities from both support and query roles. There are 56 ordered task pairs per arm, sharing eight traces. The intervals condition on the source fits and this task panel. Identity issues the frozen command and uses the unadapted source forward law.

| Physical arm | Identity | Planar | Full ridge | Equal-DF ridge | Rank one | Orthogonal rank one | Rank one minus equal-DF ridge [95%] |
|---|---:|---:|---:|---:|---:|---:|---|
| Kinova3 | 3.748 | 3.742 | 3.056 | 3.067 | 3.091 | 3.567 | +0.025 [−0.050, +0.100] |
| IIWA | 2.579 | 2.738 | 2.391 | 2.392 | 2.388 | 2.698 | −0.004 [−0.026, +0.017] |
| UR5e | 3.919 | 2.695 | 3.343 | 3.349 | 3.332 | 4.126 | −0.018 [−0.041, +0.006] |

Using the same-task, same-layout support from each of the other two arms raised rank-one error to 3.907 mm on Kinova3, 2.782 on IIWA and 4.146 on UR5e. Correct minus mismatched support was −0.816 [−1.869, +0.237], −0.394 [−0.824, +0.036] and −0.814 [−1.428, −0.200] mm, respectively. Both its command directions and measured responses change when swapping in a different arm's passive trace. The second singular direction used exactly the same three residual coefficients and one deployable gain as the first, with its own source-selected regularization. The leading direction predicted better on average on every arm, most clearly on IIWA and UR5e. Planar still beat rank one on UR5e by 0.636 mm [0.119, 1.153]. [Paired arm results](evidence/summary.json) · [Task grids including each mismatched support arm](evidence/task-pairs.json.gz)

As an additional control, we gave the second singular direction precisely the same leverage as the first on each trace. This increased second-axis errors to 7.009 mm on Kinova3, 5.137 mm on IIWA and 6.601 mm on UR5e. Rank one minus this equal-DF orthogonal control was −3.917 [−10.763, +2.928], −2.749 [−4.668, −0.830] and −3.269 [−5.892, −0.647] mm. The second direction carries less command energy, so equal leverage amplifies its response noise. The independently source-tuned orthogonal baseline in the first table is the stronger predictive comparator. [Equal-DF orthogonal sensitivity](evidence/orth-df.json)


## What changes in the native command

All forward fits use three canonical translation dimensions. The existing pooled source predicts displacement from the current and preceding native XYZ chunk commands. Each target fit updates only the current-command coefficient. The support design consists of the three five-control chunk means of one passively observed frozen-policy start. This identifies variation between chunk means. Variation within each group of five controls is unavailable in the 24 support files. Panda and Sawyer supply 24 training tasks and four starts per task per arm. Source selection holds out one arm and omits each support's task from its queries. We used each target support only after selecting λ = 1 for leading-axis, full and planar fits, and λ = 0.1 for the second-axis control. [Source selection](evidence/source.json) · [Prior motion and nonlinear comparisons](../frozen-vla-motion-models-20261123-r1/HANDOFF.md)

For singular values σ, the rank-one fit uses leverage d = σ₁²/(σ₁² + λ). The matched ridge solves Σᵢ σᵢ²/(σᵢ² + λₘ) = d separately on each support. With three output coordinates the total effective residual degrees of freedom are 3d for both. The support median stable ranks were 1.007 on Kinova3, 1.006 on IIWA and 1.007 on UR5e. Matched residual degrees of freedom ranged from 0.420 to 0.723 per output across all supports, with maximum mismatch 2.3 × 10⁻¹⁶. This match uses only the observed command spectrum. The target response and reward do not enter the penalty choice. The full and planar estimates also reproduce the sibling's original matrices numerically.

Let v be the leading unit input direction, and let A and B be the source and adapted steady-state forward matrices in row-vector convention. The supported inverse chooses a scalar g that minimizes ‖g vᵀB − vᵀA‖² + η(g−1)², where η is the prior source residual power. It issues the original command x plus (g−1)(x·v)v. At a native box face it shortens this single increment until feasible. This preserves wrist, gripper and perpendicular translation, even when axiswise clipping would change the perpendicular component. The inverse assumes a steady command while the data contain transients. An unrestricted inverse of the very same forward fit isolates that assumption from truncating the estimator. [Adapter implementation](adapter.py) · [24 native calibration records and exposure axes](evidence/adapters.json)

| Held-out command diagnostic | Kinova3 | IIWA | UR5e |
|---|---:|---:|---:|
| Rank-one constrained, mean XYZ change | 0.00941 | 0.00415 | 0.02095 |
| Rank-one unrestricted, mean XYZ change | 0.02080 | 0.01040 | 0.02179 |
| Rank-one unrestricted, mean change perpendicular to support | 0.01508 | 0.00919 | 0.00560 |
| Equal-DF ridge unrestricted, mean change perpendicular to support | 0.01468 | 0.00900 | 0.00557 |

Units are native normalized OSC command units, using held-out five-control means. The constrained inverse kept perpendicular changes at float32 roundoff. Applying the constraint to the matched smooth fit produced mean XYZ changes of 0.00928, 0.00406 and 0.02051 in the same arm order. These crossings attribute most of the difference in command exposure to the **inverse constraint**. [All command, saturation, reversal and mismatch metrics](evidence/summary.json)

## Relation to prior work

Hard truncation and continuous Tikhonov filtering are classical alternatives for weakly identified inverse problems. Frozen-policy action correction using measured target dynamics also has direct precedent in ReGround. SPACE learns robot-specific commands from a Cartesian displacement representation. The experiment here isolates a narrower decision under a single short passive trace on a held-out physical arm. The sibling work already compared full and planar source-shrunk Jacobians, a tiny nonlinear head and fixed active probes, and its Kinova trial tests full and planar control. [Hansen, 1990](https://epubs.siam.org/doi/10.1137/0911028) · [ReGround](https://arxiv.org/abs/2611.20244) · [SPACE](https://arxiv.org/abs/2606.24049) · [Sibling motion evidence](../frozen-vla-motion-models-20261123-r1/HANDOFF.md) · [Sibling native protocol](../frozen-vla-native-transfer-20261123-r1/PLAN.md)

## How much does the commanded trajectory actually change?

We replayed each of six raw Kinova3 π₀.₅ openings on Panda and on each Kinova3 branch through the pinned LIBERO-Pro2 BASE benchmark and robosuite arm binding. Panda received exactly the 15 bounded commands emitted on Kinova3. Its measured response defines an observable source-motion reference. Displacements are relative to each arm's own starting pose. This reference is a proxy for intended motion, and its error can rise or fall independently of native task reward.

| Kinova3 task and layout | Raw error | Supported rank one | Same rank-one fit, unrestricted inverse | Equal-DF ridge | Distance between supported and ridge endpoints after 15 controls |
|---|---:|---:|---:|---:|---:|
| Drawer and bowl, 14 | 4.323 | 2.378 | 2.222 | 2.230 | 2.293 |
| Drawer and bowl, 15 | 5.598 | 3.337 | 3.272 | 3.276 | 2.884 |
| Two moka pots, 14 | 4.634 | 4.287 | 3.813 | 3.847 | 2.134 |
| Two moka pots, 15 | 4.129 | 3.796 | 3.467 | 3.488 | 1.936 |
| Cream cheese and bowl, 14 | 4.825 | 4.955 | 3.872 | 3.878 | 7.386 |
| Cream cheese and bowl, 15 | 3.700 | 3.866 | 4.376 | 4.363 | 6.542 |
| **Mean five-control error (mm)** | **4.535** | **3.770** | **3.504** | **3.514** | |

The raw, constrained and smooth-ridge 15-control endpoint errors from Panda were 12.83, 10.27 and 9.55 mm averaged over starts. Restricting the inverse increased five-control error compared with raw on 2/6 starts, both cream-cheese layouts. Equal-DF ridge increased it on 1/6. Within that task, unrestricted motion helped on layout 14 and hurt on layout 15 relative to the constrained command. Across tasks the supported-minus-smooth mean was +0.256 mm per chunk, approximate three-task 95% t interval [−0.087, +0.599]. The same rank-one forward fit with an unrestricted inverse produced endpoints within 0.03–0.11 mm of smooth ridge. The 1.94–7.39 mm constrained-versus-ridge separation therefore comes primarily from limiting the issued correction. [Case-level replay and three-task intervals](evidence/replay-summary.json)

Applying the four maps to all **1,650** archived raw commands (without feedback) introduced a previously unsaturated coordinate reaching ±1 on 4 controls with the supported inverse, 3 with the unrestricted rank-one inverse and 3 with matched ridge. No method reversed the net translation vector. Across starts, the mean perpendicular command change was 0.01068 for matched ridge and about 1.1 × 10⁻⁸ for the supported inverse. Per-start 95th-percentile supported-versus-ridge command differences converted through the nominal OSC scale ranged from 0.72 to 2.89 mm. The table gives actual executed endpoint separations. It does not use this nominal conversion. [Action anomalies, measured displacements, source tapes and SHA identifiers](evidence/replay.json)

The first replay stopped on a cross-host reset hash mismatch. In the completed replay all within-CPU Kinova branches shared bit-identical physics starts. The archived GPU and CPU cases matched in released row, scene, BDDL and perturbation plan, and replaying the raw commands reproduced each archived opening trajectory to within 0.00049 mm per coordinate. CPU policy inputs during replay were fixed archived commands. The two layouts per task were already selected by the sibling protocol, which limits generalization. [Stopped check](evidence/replay01.log) · [Completed six-start replay](evidence/replay02.log)

## Native decision and reproducibility

**Prepare one paired pilot.** The observed cost of excluding genuinely useful cross-axis corrections can reverse across layouts. Rank-one truncation alone does not justify a native job. The inverse changes physical outcomes on the scale of the original arm mismatch, and success or harm to a previously successful start requires scoring native task reward. The parent submitted the single launcher for all four conditions on the same six Kinova starts. The [submission receipt](submission.jpg) shows one GPU queued. It records each native reward tape, checks the complete within-case reset and opening frozen proposal, and reports rescues and losses with denominators. The CPU fixture passed all four branches. This work used zero GPU-hours. [Exact parent command and matrix SHA](HANDOFF.md) · [Native binding qualification](evidence/fixture.json)

All analysis ran on Baige CPU, constrained to core 0 or the same pair of cores 0–1. The research repository base at intake was `d1d60dbdb0be87fc64ed1340bddf3ac8398c7f5b`. To keep ownership confined to this run, all executable snapshots were committed inside the run and transferred through Git bundles. The original motion implementation pin is `761f4209f4838099e4f80fad2937ad1e0f0cfcc5`. The native episode/binding checkout is `a8a997e7d06bb131d3f9710204c58e106f0f0904`, benchmark release `e5500c690c1498eca1b7050350654964190cdf70`, with the frozen OpenPI pi05_libero checkpoint and five-control replanning. The policy is held fixed throughout. [Executable Git history](evidence/source.bundle) · [Source and split provenance](evidence/source.json)

The following commands reproduce the measurements on Baige with fresh output destinations. The initial screen and parent-only GPU command are in the handoff.

```sh
run=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-supported-correction-20261123-r1
bash "$run/src-cpu/cpu.sh" "$run/cpu-repeat"
bash "$run/src-supp/supplement.sh" "$run/orth-df-repeat.json"
bash "$run/src-replay02/replay.sh" "$run/replay-repeat"
bash "$run/src-native/fixture.sh" "$run/fixture-repeat"
cd "$run"
taskset -c 0 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -Bu \
  src-final/summarize_replay.py replay-repeat/replay.json cpu01/adapters.json replay-repeat/summary.json
```
