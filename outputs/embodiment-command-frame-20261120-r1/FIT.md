# Does command-frame breadth improve motion prediction?

This is the analysis contract for nine CPU fits on Panda and Sawyer. The test asks whether replacing six layouts per condition with two additional command mappings improves prediction at the same interaction cost. It measures world-frame wrist motion on two held-out mappings. This worker prepared the fitter on macOS and ran static checks only. The measurements require the completed Linux collection. [Registered comparison](../embodiment-meta-factory-20261120-r1/SCALE.md) · [Run plan](PLAN.md)

## Tapes and training banks

Supply the completed `tapes.jsonl` from the collector, or JSONL shards containing each of its 256 records once. Each record identifies `family`, `setting`, `role` and integer `index`. The model sees a 16 × 7 public command tape and a 16 × 3 tape of measured world XYZ displacements in metres. The query input is one public seven-component command. The setting defines the split. Each support and query role has 12 training indices (0–11) or eight evaluation indices (0–7). The two roles have separate resets even when their indices match. [Collector contract](COLLECTION.md)

The loader checks coverage, unique keys, finite arrays, and identical public commands across families and settings for each split, role and index. When present, the collector's split, layout and probe seeds, and donor fingerprints provide additional checks on assignment, matching and reuse. Given only the required six fields, independent reset provenance comes from the collector.

| Bank | Settings at each Panda layout | Settings at each Sawyer layout | Tapes per arm per setting |
| --- | --- | --- | --- |
| Narrow A | nominal and +90° | nominal and +90° | 12 support + 12 query |
| Narrow B | −90° and gain 0.65 | −90° and gain 0.65 | 12 support + 12 query |
| Broad, even indices | nominal and −90° | +90° and gain 0.65 | 6 support + 6 query across the broad bank |
| Broad, odd indices | +90° and gain 0.65 | nominal and −90° | 6 support + 6 query across the broad bank |

Each fit uses 48 support and 48 query tapes, totalling 1,536 transitions. The broad bank takes its tapes from the narrow banks' union. Each arm has exactly two settings at every layout index in every bank. The first eight preflight pulse tapes count toward the source bank.

The three fit seeds are **41, 42 and 43**. Within a seed, all three fits start from identical weights and reuse the same 100 minibatches of 32 draws. Each minibatch has four draws in each arm × setting-slot × layout-parity stratum. The two slots select the two eligible settings at that layout. A draw chooses a whole support and a separate query independently with replacement within the same parity, then chooses one of the 16 query steps uniformly. Thus every optimizer step uses the same support layout indices, query layout indices, public probes and query steps across banks. It also gives each active arm–setting cell the same weight within a fit.

## Reader and common scale

The reference is the stage-one `ReaderAdapter` deployed at `19804c8895245dcbabd675f9992bf3dbf1dd360d`, read against design base `0fbc5e3167b362bc33b741e6bbfa20724aa93234`. The prediction path uses seven commands, three scaled motions and all 21 command–motion products in command-major order. Each row goes through Linear(31, 48), Tanh, Linear(48, 16), Tanh. The mean of the 16 row embeddings feeds a linear 16-to-24 head. Reshaping its output to 8 × 3 gives seven slope rows and one intercept for predicting scaled XYZ from a query command. Stage-one optimization uses Adam at 0.003 with default betas and gradient-norm clipping at 5. The final update supplies each evaluated model. [Reader and optimizer](../libero-paligemma-crossarm-20261120-r1/adapt_eval.py) · [Pinned stage-one run](../libero-paligemma-crossarm-20261120-r1/evidence/reader01/pretrain/report.json)

Compute one three-component RMS over all 96 source **support** tapes in the union, 1,536 displacement vectors. Accumulate in float64, floor each coordinate at 0.000001 m and freeze in float32. All nine fits share these three source calibration numbers, including the two disjoint narrow banks. Divide support motions, training labels and scoring residuals by the same scale. Public commands retain their original units. The loss and score are mean squared errors over steps and XYZ.

## Paired controls and analytic calibration

Freeze all nine readers before scoring the two held-out blocks, +45° and +90° with gain 0.65. For each of the 32 family–setting–index cases, use the same 16 commands and observed displacements for all controls. **Correct** encodes the same setting's support at that index. **Wrong** encodes the other held-out setting's support from the same arm and index, with identical public commands. **Empty** supplies an exact zero 16-vector to the trained response head, retaining its bias.

Fit an affine least-squares map on the same single support for the analytic comparison. With row vectors, append a one to the seven public commands to obtain matrix X and solve B = X⁺Y for the three motions in metres. Predict X_query B on the same query rows. Repeat for the swapped support. With empty support the minimum-norm coefficient matrix is zero. The learned empty baseline has a trained population response. All predictions use the common scale for scoring.

The first three rows of B form the translational response J. Holding rotation and jaw fixed, a desired incremental world displacement d requires command increment d J⁻¹. The reader supplies the same Jacobian after converting its response head to metres. The report gives command vectors for +1 mm on each world axis, the condition number, and the XY azimuth of the command for world +X. Under an ideal isotropic nominal response, +45° calls for a −45° command rotation. The conjunction calls for −90° and a magnitude factor of 1/0.65 relative to nominal. Signed angular error compares each estimate with the query's ideal inverse. An arm's native response, control lag and layout can also affect this diagnostic. The code evaluates these directions offline.

For support rank r, the analytic residual covariance is EᵀE/(16−r) in m². For an estimable query row x, the diagonal of ‖x X⁺‖² times this covariance gives the variance of the fitted mean. The report emits its per-axis standard errors in metres. This estimate assumes independent, homoscedastic support residuals. Consecutive simulator steps can correlate. Rank, singular values and missing standard errors identify extrapolation outside the observed command subspace. The code reports an analytic inverse only when the support identifies all three translation rows and J has full rank.

## Decision and readout

Within each held-out setting, average correct-context MSE equally over layouts, families and seeds. The narrow reference averages A and B with equal weight. Advance only if broad reduces that reference by **at least 10% in each setting**, the narrow-minus-broad difference is positive for **each paired seed in each setting**, and broad's correct MSE is strictly below both wrong and empty **in each family**, averaging the two settings equally for that control check. A zero narrow reference cannot establish a percentage improvement. A failed condition stops expansion at this budget. The analytic scores and inverse directions help interpret the result. Native task success and transfer to another arm require separate evidence.

The report retains every paired layout score and every per-seed cell mean. For each setting it gives the eight paired gains after averaging seeds and families, their layout-level standard error (sample SD divided by √8), and the three seed gains with their range. The eight layout pairs supply the sampling units. The three starts of the optimizer describe sensitivity to fitting.

## Linux invocation and outputs

Run from the deployed checkout using the allocated Linux NumPy/PyTorch environment. Set `H1_OUT` to the completed collection directory and `H1_FIT_OUT` to a fresh result directory.

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --active --no-sync python outputs/embodiment-command-frame-20261120-r1/fit.py \
    --tapes "${H1_OUT:?}/tapes.jsonl" --out "${H1_FIT_OUT:?}"
```

`report.json` records the input and fitter hashes, checkout revision, shared scale, all training losses, paired scores and gate decisions. Its score axes are seed, bank, family, held-out setting, layout index and control, with labels in the report. `analytic.json` records the support-only coefficients, predictions, covariance, directions and scores. Each `<bank>-<seed>.pt` contains the final reader state and scale. Each `inverse-<bank>-<seed>.json` records the learned correction directions on the three controls. [Fitter](fit.py)
