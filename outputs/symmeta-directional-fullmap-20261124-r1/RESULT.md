# Directional arbitration improved on the direct reader, but its directional advantage remains unresolved

The source-selected directional blend reduced issued-correction error on **all four fresh tasks** relative to the source-selected fixed scalar, which had chosen the direct Perceiver endpoint. The average reduction was **0.49 mm-equivalent**. The command comparison against the fixed physical ridge split two tasks to two. Its motion advantage over the direct reader was also unresolved. Two controls with the same overall amount of shrinkage, a per-case scalar and a reversed spectral assignment, achieved lower mean errors on both endpoints. This panel supports an incremental command improvement over the direct reader and lowers confidence that assigning trust to the prescribed eigenvectors adds value. [Registered question](PLAN.md) · [Paired results](cpu03/target-summary.json)

## Fresh general maps

The primary panel gave equal weight to passive and lateral supports, to the two determinant signs, and to UR5e and IIWA within each task. Both lateral signs and the three frozen reader seeds were averaged before comparing tasks. Command is RMS bounded issued-correction discrepancy on the complete independent policy tape in mm-equivalent. Motion is RMS error forecasting real five-control XY displacement in mm. All 16 primary arm–task–map cells supplied all three measured displacement chunks after wash-in. A positive gain below means the directional estimate had lower error. Brackets enclose the 95% percentile interval from all 256 ordered whole-task bootstrap resamples. [Abbreviations, units and readout](PLAN.md) · [All per-attempt predictions](cpu03/target-rows.json)

| Reader | Command error | Command gain of directional [95%] | Motion error | Motion gain of directional [95%] |
|---|---:|---:|---:|---:|
| Directional | 10.77 | — | 7.28 | — |
| Direct Perceiver | 11.27 | +0.49 [0.23, 0.78] | 7.37 | +0.08 [−0.04, 0.31] |
| Source-selected fixed scalar, α = 1 | 11.27 | +0.49 [0.23, 0.78] | 7.37 | +0.08 [−0.04, 0.31] |
| Fixed physical ridge R | 17.98 | +7.21 [−7.51, 21.93] | 8.43 | +1.15 [0.16, 2.14] |
| Source-selected physical ridge alone | 14.19 | +3.42 [−1.71, 9.67] | 7.42 | +0.14 [−0.32, 0.45] |
| Per-case scalar tr(S)/4 | 9.05 | −1.72 [−4.71, 0.34] | 7.04 | −0.24 [−0.75, 0.25] |
| Reversed assignment, same spectrum | 9.71 | −1.07 [−5.94, 2.60] | 7.16 | −0.13 [−0.88, 0.65] |
| Known map with the fixed lag law | 0.00 | reference | 5.82 | reference |

For the six registered component comparisons jointly, the descriptive simultaneous bands for command gain were [0.18, 0.80] against direct or scalar and [−9.47, 23.89] against fixed R. For motion they were [−0.14, 0.30] and [0.01, 2.28], respectively. The scalar and direct columns coincide because their frozen operators coincide. The large physical command contrast on reflecting maps and the opposite contrast on orientation-preserving maps limit the interpretation of the pooled mean. Four task clusters condition these intervals on the selected models, arm pair, maps and query layouts. [All bands and orientation strata](cpu03/target-summary.json)

| Paired task | Command gain over direct/scalar | Command gain over R | Motion gain over direct/scalar | Motion gain over R |
|---|---:|---:|---:|---:|
| libero_10/8 | +0.51 | +22.06 | +0.42 | +1.89 |
| libero_goal/6 | +0.90 | −8.95 | −0.01 | +0.15 |
| libero_object/8 | +0.43 | +21.79 | −0.05 | +2.39 |
| libero_spatial/8 | +0.13 | −6.07 | −0.03 | +0.16 |

## Information, dose and fit variation

The lateral and along probes had equal sign-averaged perturbation energy to physical recording precision. Lateral increased the weak/strong issued-command singular ratio from 0.073 along to 0.282. With λ fixed on source, the least-observed map direction retained an average direct weight of 0.975 even after a lateral intervention. The best-observed direction retained about 0.50. On the lateral schedule, the source-selected physical ridge alone reached 4.38 mm-equivalent and 6.74 mm, close to the directional blend’s 4.39 and 6.75. On passive supports the trace-matched scalar reached 13.79 mm-equivalent and 7.37 mm, below the directional blend’s 17.16 and 7.82. The acquisition schedule had a larger descriptive effect than the directional-versus-fixed-scalar difference. [Spectra, doses and all alternatives by schedule](cpu03/target-summary.json)

| General-map schedule | Issued XY pulse energy | Mean Gram eigenvalues, weak to strong | Mean direct weights, weak to strong | Directional command / motion | Direct or fixed scalar command / motion | Fixed R command / motion |
|---|---:|---|---|---:|---:|---:|
| Passive | 0 | 0.005, 0.007, 1.716, 1.797 | 0.997, 0.996, 0.508, 0.497 | 17.16 / 7.82 | 17.72 / 7.89 | 27.74 / 9.58 |
| Lateral | 0.498 | 0.042, 0.068, 1.719, 1.799 | 0.975, 0.961, 0.507, 0.496 | 4.39 / 6.75 | 4.81 / 6.84 | 8.23 / 7.27 |
| Along | 0.498 | 0.005, 0.007, 1.756, 1.860 | 0.997, 0.996, 0.502, 0.487 | 17.11 / 7.82 | 17.77 / 7.89 | 23.49 / 9.62 |

On all primary rows, weighted descriptive correlations of the command gain over fixed scalar were −0.060 with log(1 + smallest Gram eigenvalue / k), +0.173 with the direct-weight range and −0.105 with perturbation energy. Corresponding motion correlations were +0.081, −0.370 and +0.060. The interval for the command gain versus fixed R spans both signs, and the dose associations do not identify a region where this directional rule is preferable. [Continuous slopes and correlations](cpu03/target-summary.json)

| Primary subgroup or signed general-map pulse | Directional command / motion | Command gain over direct/scalar | Motion gain over direct/scalar |
|---|---:|---:|---:|
| UR5e | 9.17 / 7.19 | +0.57 | +0.18 |
| IIWA | 12.37 / 7.37 | +0.41 | −0.02 |
| Reader seed 41 | 13.06 / 7.57 | +0.58 | +0.09 |
| Reader seed 42 | 14.32 / 7.32 | +0.41 | +0.02 |
| Reader seed 43 | 4.95 / 6.95 | +0.48 | +0.13 |
| Positive determinant | 10.62 / 7.12 | +0.36 | +0.00 |
| Reflecting | 10.93 / 7.44 | +0.63 | +0.17 |
| Lateral + | 4.46 / 6.76 | +0.48 | +0.11 |
| Lateral − | 4.32 / 6.74 | +0.36 | +0.08 |
| Along + | 22.24 / 8.61 | +0.62 | +0.05 |
| Along − | 11.99 / 7.03 | +0.71 | +0.09 |

The known-map forecast still missed achieved motion by 5.82 mm on average. This residual is compatible with body, state, contact or lag mismatch in the common source law. The trace control motivates a separate hypothesis that per-support variation in overall trust transfers across arms. Testing it would require a fresh comparison against a source-selected fixed scalar. The measured motion is a forecast on an independently executed frozen command tape. Native policy success would require a different experiment. The [complete results table](breakdowns.csv) includes all six comparator readouts, the known-map reference, all five fresh map draws, each schedule, seed summaries and paired task differences.

## Selection and provenance

The registration preceded the original full-map fit. Only the original Panda and Sawyer validation acquisitions (suite task IDs 7/8/6/6) entered arbitration selection. Source general maps supplied 72 relevant supports, 216 seeded predictions and all 48 arm–task–draw–strategy cells. The Perceiver recipe and original standalone ridge had already used these validation sources. The same three final checkpoints were recovered for a validation-only forward. Seed 41 exactly reproduced the upstream validation averages of 8.579312 mm-equivalent and 4.243688 mm over the original full validation design. Separately, the original held-source qualification favored direct over identity by 33.00 mm-equivalent on lateral general maps. Its scores were opened only after our selection for provenance. [Original model selection](gpu01/upstream-selected.json) · [Original held-source control](cpu03/upstream-source-test.json) · [Readback verification](custody.json)

Across seeds, λ = 100k = 1.63245 minimized the registered validation command loss, yielding 12.938 mm-equivalent and 3.227 mm. The best fixed scalar on the 1,001-point grid was α = 1, yielding 13.207 and 4.746. Fixed R yielded 33.301 and 3.676. The standalone source-selected ridge used 0.01k. All blends retained the original k-strength R anchor. The independent source selection hash was recorded at 14:34 Shanghai before target opening. The target replay used the original four task IDs 8/6/8/8 on UR5e and IIWA. It checked the original fixture hashes, three checkpoint hashes and all 600 seed–support predictions against the original scores for both component estimators. The largest difference on direct command scores was 0.0000031 mm-equivalent with the original float32 pseudoinverse. [All source candidates](cpu02/validation-curves.json) · [Independent source seal](cpu02/source-selection.json) · [Target receipt](cpu03/target-receipt.json)

The gate uses the 15 issued controls, encoder-derived FK, the historical lag law and the original model’s estimate at the end of support. The analytic basis and Gram were verified against the direct reader’s tokens on 40 real training supports. The injected map, post-interface controls, controller goals and future policy tape enter only the audit or the endpoint scorer. D and R share observations. This is a post-hoc consistency blend. A Bayesian linear update with an independently specified prior mean would combine its prior precision with a likelihood’s raw sufficient statistics. Feeding an already support-conditioned D into a blend anchored at an already regularized R supplies neither that independence nor a posterior. [Qualification](cpu01/qualification.json) · [Design and primary-work review](PLAN.md)

Baige distributed job `job-m87imrcdjdpq` used one RTX 4090 for 26 s of recorded job-step wall time (0.00722 GPU-hours), including qualification. The forward process used 21.687 s wall and 64.690 s host user-plus-system CPU time. Three single-core stages on idle core 48 used 3, 4 and 6 s wall including the idle checks, totaling 13 core-seconds reserved and 2.949 s timed user-plus-system CPU. These are incremental costs, excluding the separately owned upstream fit and acquisition. The full prediction/forecast readbacks, fitted weights and inputs are preserved in this run locally and under the recorded Baige artifact root. The experiment stopped after its one sealed selection and one fresh target replay. [Timing and pins](custody.json) · [Validation predictions and forecasts](gpu01/validation-forecasts.json) · [Standalone validation inputs](gpu01/validation-cases.npz) · [Standalone target inputs](cpu03/target-cases.npz)
