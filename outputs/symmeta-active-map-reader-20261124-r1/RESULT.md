# Lateral calibration improved the trained Perceiver’s held command correction

On eight previously unused general interface maps across four paired LIBERO tasks, the direct Perceiver’s held-command error fell from **17.78 to 4.81 mm-equivalent** when its 15 calibration controls probed laterally instead of along the opening command. The paired reduction was **12.96 [1.93, 24.00] mm-equivalent**. All four task differences favored lateral excitation. The corresponding physical forecast improved by **1.04 [−0.10, 2.48] mm**. Its direction remains uncertain across these tasks. [Registered question and falsifiers](PLAN.md) · [Fresh factorial and paired intervals](gpu01/fresh-summary.json)

The explicit body-response fit had no demonstrated advantage on the same lateral measurements. Direct minus incumbent was **−3.42 [−8.23, +0.54] mm-equivalent** on held corrections and **−0.43 [−1.34, +0.41] mm** on achieved-motion forecasts. A simple physical ridge fit with source-selected regularization reached 4.38 mm-equivalent and 6.74 mm. Both lateral differences between direct and tuned ridge also included zero. These four task clusters leave the reader ranking unresolved. [Case-level estimates and endpoints](gpu01/fresh-rows.json.gz)

## Fresh factorial

Each entry gives passive / equal-effort along / lateral in that order. A signed pulse is one separate 15-control attempt. The two signs were averaged at scoring. Command error is RMS Euclidean discrepancy from the oracle’s componentwise bounded inverse on the full independent frozen-policy tape, multiplied by 50 mm per normalized unit. Motion error is RMS Euclidean error forecasting actual world XY grip-site displacement over held five-control chunks. Lower is better for both. [Evaluation code](targets.py) · [All families, signs, arms and seeds](gpu01/fresh-rows.json.gz)

| Fresh map | Reader | Issued command error, mm-equivalent | Achieved motion forecast error, mm |
|---|---|---:|---:|
| **General** | **Direct Perceiver** | **17.72 / 17.78 / 4.81** | **7.89 / 7.89 / 6.84** |
| General | Incumbent physical fit | 27.74 / 23.49 / 8.23 | 9.58 / 9.62 / 7.27 |
| General | Source-selected ridge | 24.01 / 20.79 / 4.38 | 8.10 / 8.86 / 6.74 |
| Native identity | Direct Perceiver | 5.42 / 5.79 / 1.29 | 6.78 / 6.83 / 6.59 |
| Native identity | Incumbent physical fit | 2.16 / 2.54 / 2.64 | 6.45 / 6.52 / 7.42 |
| Native identity | Source-selected ridge | 23.45 / 19.36 / 3.22 | 9.72 / 10.81 / 7.98 |
| Yaw and uniform gain | Direct Perceiver | 24.11 / 17.52 / 4.27 | 7.08 / 7.06 / 6.36 |
| Yaw and uniform gain | Incumbent physical fit | 37.08 / 36.91 / 11.53 | 12.49 / 12.42 / 7.27 |
| Yaw and uniform gain | Source-selected ridge | 15.67 / 21.95 / 4.65 | 7.79 / 11.39 / 6.91 |
| Axis swap | Direct Perceiver | 23.60 / 22.08 / 3.53 | 7.93 / 7.85 / 5.96 |
| Axis swap | Incumbent physical fit | 41.94 / 43.22 / 17.48 | 13.86 / 14.08 / 8.20 |
| Axis swap | Source-selected ridge | 15.19 / 19.87 / 4.60 | 7.92 / 10.27 / 6.27 |

The incumbent’s lateral command error was 4.64 on orientation-preserving general maps and 11.82 on reflecting maps. Direct scored 4.50 and 5.12, and tuned ridge 4.49 and 4.26. The smaller ridge penalty reduced the incumbent’s pull toward identity on reflections. It also raised error on passive native supports from 2.16 to 23.45. Those tradeoffs limit a universal choice of regularization. [Orientation and control breakdowns](gpu01/fresh-summary.json)

| Paired contrast on fresh general maps | Command difference [95% task interval], mm-equivalent | Motion difference [95% task interval], mm |
|---|---:|---:|
| Direct lateral − along | −12.96 [−24.00, −1.93] | −1.04 [−2.48, +0.10] |
| Direct lateral − passive | −12.91 [−25.21, −0.62] | −1.05 [−2.46, +0.13] |
| Incumbent lateral − along | −15.26 [−27.33, −4.81] | −2.34 [−4.09, −0.60] |
| Ridge lateral − along | −16.41 [−29.90, −6.49] | −2.12 [−4.76, +0.52] |
| Direct − incumbent on lateral | −3.42 [−8.23, +0.54] | −0.43 [−1.34, +0.41] |
| Direct − ridge on lateral | +0.43 [−0.65, +1.51] | +0.10 [−0.28, +0.52] |
| Direct − incumbent in the lateral-minus-along effect | +2.30 [−14.19, +19.61] | +1.30 [+0.26, +2.08] |

For the primary lateral reader comparison, direct minus incumbent command differences were −3.88, +1.38, −10.88 and −0.30 on libero_10/8, libero_goal/6, libero_object/8 and libero_spatial/8. Physical differences were −0.45, −0.28, −1.69 and +0.69 mm. By arm, command differences were −3.28 on UR5e and −3.56 on IIWA, with motion differences of −0.94 and +0.08 mm. Direct’s lateral-minus-along command differences on the four tasks were −3.74, −18.36, −0.12 and −29.63. The physical interaction reflects a larger improvement for the incumbent from its poorer along-line forecast. Intervals enumerate all 256 ordered resamples of four whole tasks, keeping bodies, maps, signs and training seeds paired. They describe uncertainty conditional on this finite panel and source fit. [Complete contrasts](gpu01/fresh-summary.json)

## What the strong baseline learned on source

Panda and Sawyer physically executed 1,920 support attempts. Within each of the four suites, four task IDs trained the readers, a fifth validated them, and a sixth tested them. This gave 1,280 actual training, 320 validation and 320 held-test attempts. The 21,467 admissible training views reuse those 1,280 trajectories. The direct reader received actual issued commands, encoder-derived world FK, the common current/lag response design, z offset, normal equations and residual scale, all in three 100-component tokens. The four-entry head predicts the forward map. The physical fits use the same row factors. Actual native controls and controller goals were available only to data-integrity and oracle checks. The comparison assumes known public world geometry. [Feature construction](observations.py) · [Source data inventory](gpu01/source-bank.json) · [Physical source splits and coverage](audit01/source.json)

For an orthogonal change of issued coordinates Q, replacing u by Qu and A by AQᵀ preserves every physical input and leaves the observed world FK unchanged. Training admitted a view only when both boxes and the regenerated causal schedule agreed, including signed pulse and headroom. A separate +11° source lateral replay produced exactly identical float32 OSC controls and all 16 measured FK positions. The historical and fresh general target matrices have distinct absolute determinants from **every** physical training draw. This invariant excludes exact target-map reuse through the augmentation. [Physical replay](audit01/replay.json) · [Draw audit](audit01/map-draws.json)

The eight prospective width/rate/loss combinations competed on original physical validation views. The selected 347,044-parameter Perceiver used width 96, a 0.0003 peak learning rate, 400 warmup updates and cosine decay through 6,000 updates. Map supervision plus measured response loss won on the validation command endpoint. The best command-loss variant scored 13.36 against the selected seed-41 validation score of 8.58. Seeds 42 and 43 scored 16.85 and 16.49 on the same validation metric. Seed 41 did the tuning. All three contribute equally to the reported target estimates. Source validation selected a ridge penalty of 0.01 times the incumbent strength. [Registration](CONFIG.md) · [Selection and checkpoint identities](gpu01/selected.json) · [Training curves](gpu01/learn.log)

On the four untouched **source-test** tasks, general/lateral command errors were 39.85 for identity, 13.51 for incumbent, 8.37 for ridge and **6.84 [5.02, 9.97]** for direct. The direct seed means were 9.06, 4.85 and 6.62. Direct minus incumbent was −6.66 [−11.74, −1.59]. On physical forecasts the respective errors were 35.54, 3.94, 3.49 and **5.29 [3.89, 6.69]** mm. Direct minus incumbent was +1.35 [+0.72, +1.75]. Across *all* families and schedules direct minus incumbent was −4.50 [−8.62, +1.47] on commands and +0.11 [−1.08, +1.13] on motion. These source results establish learning of the bounded full-map inverse and expose a real difference between estimating the injected map and forecasting an arm through the fixed source law. [Held-source estimates](gpu01/source-test.json) · [Source rows by seed](gpu01/source-test-rows.json.gz)

In a separate body holdout, training only on Panda gave direct general/lateral command error 6.70 on Sawyer against 44.35 for identity and 13.66 for incumbent. Training only on Sawyer gave 7.21 on Panda against 35.34 and 10.14. Both diagnostics refit the actuator law using just the training body and training task IDs. Hyperparameters came from joint-source validation. The primary historical law used 192 nominal supports drawn across the source task IDs, including IDs with fresh held views. Its residual power was 4.08 × 10⁻⁵, with a 24-task bootstrap interval of [2.65, 5.86] × 10⁻⁵. Training and validation used independent source query proposals from other starts, available as five-control means. [Panda to Sawyer](gpu01/from-Panda-held-Sawyer.json) · [Sawyer to Panda](gpu01/from-Sawyer-held-Panda.json) · [Law uncertainty](audit01/source.json)

## Physical readbacks and developmental panel

The fresh panel used new layout-17 support resets, two general draws of opposite determinant per task shared across arms, and independent complete nominal frozen-policy tapes on archived layout 14. Across eight task/body queries, 1,524 original executed policy controls were scored. There were 200 physical support attempts, 40 held replay segments, and 120 actual five-control displacements after wash-in. The greatest per-start RMS drift from the archived nominal replay over the prefix was 0.025 mm. The original support motor proposals survive as chunk means, which were repeated and physically issued for five steps each. Neither the support intervention nor the predicted inverse changed the archived query motor proposals. [Raw-input manifest](gpu01/fresh-files.json.gz) · [Physical collector](fixture.py)

Sign-averaged effort was 7.77478 for each active schedule and 7.27714 for passive. The mean weak/strong command singular ratio was 0.077 passive, 0.073 along and 0.282 lateral. The privileged goal-map fit’s largest held correction error over the bank was 0.000074 mm-equivalent. Even with the **true general map**, the fixed source law missed held displacements by **5.82 mm**. The bounded oracle missed desired setpoints by 2.42 mm-equivalent. The direct lateral estimate’s setpoint discrepancy relative to that oracle was 3.31 mm, against 5.85 for incumbent and 3.09 for tuned ridge. Direct seed-specific lateral command errors were 5.18, 4.22 and 5.02, and motion errors were 6.93, 6.53 and 7.08 mm. [Fresh physical and oracle readbacks](gpu01/fresh-rows.json.gz)

Separately, on the **developmental** panel’s equally weighted swap/general maps the unchanged incumbent reproduced its published 34.29 / 30.69 / 14.97 mm-equivalent and 12.08 / 11.71 / 8.00 mm. On the same branches the source-frozen direct reader scored 17.27 / 14.50 / 3.62 and 8.40 / 7.46 / 7.18. Tuned ridge scored 14.12 / 17.54 / 4.97 and 10.28 / 10.97 / 7.88. These tasks provided the motivating observation and remain separate from the primary result. [Developmental factorial](gpu01/developmental-summary.json) · [Earlier study](../symmeta-query-direction-probe-20261124-r1/RESULT.md)

## Research update

1. Better lateral measurements explain a transferable improvement in command correction. It appears with direct attention, the original physical fit and tuned ridge at the same issued effort. The physical forecast interval for the direct reader still permits a small reversal.
2. The registered comparison supplies no evidence that explicit response matching beats a well-trained direct full-map reader on lateral supports. The point estimates favor direct over the historical fit. The joint width of the paired intervals allows advantages in either direction. Equivalence remains open.
3. Source-selected ridge is a competitive small readout on these general maps. A method claim based only on comparison with the fixed historical penalty would miss that control. Related-work precedent places the acquisition result within task-directed experiment design and learned online identification. [Wagenmaker et al.](https://proceedings.mlr.press/v139/wagenmaker21a.html) · [Yu et al.](https://arxiv.org/abs/1702.02453)
4. The true-map residual and the opposing command/forecast ordering on source leave actuator-law mismatch plausible. These data cannot attribute it to contact, posture or missing temporal dynamics, or establish the benefit of a learned residual. A subsequent test can hold the injected map fixed and predict independent physical motions before changing the inverse estimator.

The behavioral follow-up is a paired, freshly reset closed-loop native reward comparison with prospective rescue/loss accounting. This run measured operator correction and physical motion. The score-free release contains the actual seed-41 and incumbent general-map estimates, input fingerprints and inverse conventions for that separate workstream. [Release receipt](release01/receipt.txt) · [Score-free matrices](release01/reader.json)

## Execution and archives

Baige Distributed Training job `job-b5147r3s5ar6` completed on one exclusive NVIDIA RTX 4090. The launcher took 769 seconds, **0.214 GPU-hours**. Fixture Git pin `09faaeebbd639d3906e0c4982efc140273f9f7db` used cores 35, 36, 38 and 39. Its longest subprocess took 654.1 seconds wall and the four together used 1,270.7 CPU seconds. The fit and scorer ran at `589873c103e46b3ced7bda4ca855b3401b171b2d`. Source selections, all three checkpoint hashes and source tests were sealed before opening target metrics. [Device qualification](gpu01/gpu.json) · [Source seal](gpu01/sealed.json) · [GPU exit](gpu01/exit.txt) · [CPU exit](cpu01/exit.txt)

The complete shared root is `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/symmeta-active-map-reader-20261124-r1/`. Physical control/FK tapes and held queries are in `cpu01/`, source-checkpoint files `seed-{41,42,43}.pkl` are in `gpu01/`, and the empirical reparameterization is in `audit01/`. The release is `release01/reader.json`, SHA-256 `3aeebf967bb9680f331b5e6b1b52de53d9ba7648f22a3502568459f47e426764`. The primary metrics, raw estimate rows, manifests and release also accompany this report locally. [Source file hashes](gpu01/source-files.json.gz) · [Fresh file hashes](gpu01/fresh-files.json.gz) · [Developmental file hashes](gpu01/developmental-files.json.gz)
