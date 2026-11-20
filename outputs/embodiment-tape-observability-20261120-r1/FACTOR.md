# Can one support tape recover forward yaw and gain?

This positive control asks whether source factor labels can teach a fixed reader to distinguish the two held-out mappings from one archived 16-command practice sequence. A correctly directed gap close to +45° and small absolute yaw and gain errors on both settings and arms would favor recoverability. A large gap with large angular errors would show donor sensitivity with poor calibration. Separation near the constant baseline or consistently reversed ordering would count against this reader. [Study plan](PLAN.md) · [H1 result](../embodiment-command-frame-20261120-r1/RESULT.md)

## Data and units

The design base is `1967f2b31fd4f64c0c9771fb04df2a268c1e2e41`. The reader requires the original JSONL with SHA-256 `c1207da02bd988de7fb51cab00a894ed29f1a31ca7d7306bfe9c9c7fb05c3ebf`. It checks the complete set of 256 tape keys, 4,096 transitions, 40 distinct layout seeds and scenes, and matching public probes within each block. [Collector contract](../embodiment-command-frame-20261120-r1/COLLECTION.md)

| Use | Tapes | Layout blocks |
| --- | ---: | --- |
| Fit features and targets | 192 | 12 support and 12 query blocks, each shared by two arms and four source settings |
| Estimate held-out factors | 32 | Eight support blocks, each shared by Panda, Sawyer and both held-out settings |
| Inventory held-out queries | 32 | Eight separate query blocks |

The two inputs are pre-mapping public XY commands in normalized controller units and the observed world XY wrist displacements in metres over each 20 Hz control step. A positive forward yaw rotates public +X toward world +Y. Translation gain is the dimensionless multiplier applied to the command. Source setting names supply supervision. Arm, block and donor identifiers serve only to validate and group tapes.

## Fixed reader

For each lag ℓ in 0, 1, 2, 3, pair command at step t with motion at step t + ℓ on the same tape. This gives 16, 15, 14 and 13 pairs. Encode XY as x + iy and subtract each signal's own mean over its overlapping rows. For centered command u and displacement d, the complex feature is `sum(conj(u) * d) / sum(abs(u)**2)`. Its real and imaginary parts measure parallel and quarter-turn response, in metres per public command unit. Command-energy normalization preserves response amplitude. Lag zero uses the motion recorded during the command itself.

Each source target is the real and imaginary parts of `gain * exp(i * yaw_rad)`. The four source settings are (0°, 1), (+90°, 1), (−90°, 1) and (0°, 0.65). Pool both arms and both source roles, with one equally weighted row per tape. Standardize each feature using its source population mean and standard deviation, with divisor one for a constant column. Fit two outputs jointly by minimizing squared residual norms averaged over the 192 tapes plus 0.01 times the sum of squared weights. The intercept is unpenalized. All calculations use float64 components.

| Reader | Inputs |
| --- | --- |
| `lags_0_to_3` | All eight standardized components in lag order, real then imaginary |
| `lag_0` | The identical first two standardized components and the same penalty |
| `source_mean` | The constant mean of the 192 complex source targets, analytically 0.4125 + 0i |

The forward yaw estimate is the argument of the predicted complex number, converted to degrees. Its magnitude is the gain estimate. All three recipes are fixed before held-out scoring.

## Paired readout

Each reader's `cells` in `report.json` contains the eight support estimates and layout seeds for each arm and target. The true donor uses the target setting. The swap donor uses the other setting on the same arm, layout and public commands. Both are scored against the same target. Each cell reports continuous estimates, signed circular errors wrapped to [−180°, 180°), absolute circular errors in degrees and absolute gain errors in dimensionless gain units. Yaw summaries use a circular mean and mean resultant length. Gain and absolute error summaries give the mean, median and range.

The nearest of +45° and +90° is a scoring-only readout of each continuous yaw. An angular-distance tie within 10⁻¹²° splits credit. A zero complex prediction has undefined yaw and receives zero nearest-angle credit. The `paired_angles` table lists the signed wrapped yaw difference from (+45°, 1) to (+90°, 0.65) and its absolute separation for each arm and block. Its ideal signed difference is +45°. Swapping the same two donors in the reverse direction reuses the pair.

The eight shared support layouts are the units of comparison across all four cells. H1 already examined this panel, both arms supplied source data, and the two held settings couple angle to gain. This readout measures decodability under direct source supervision. Query-motion utility is covered by the companion analysis. [Validity review](REVIEW.md)

## Offline Linux execution

From the repository root, use the allocated CPU environment with NumPy and a fresh output directory. The inline check uses two synthetic pulses with a known +90° response delayed by two steps. It checks sign, lag and scale before reading the archive.

```bash
run=outputs/embodiment-tape-observability-20261120-r1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  uv run --active --no-sync python "$run/factor.py" \
  --tapes "${TAPES:?set the path to native02/collection/tapes.jsonl}" \
  --out "${FACTOR_OUT:?set a fresh directory}"
```

The single output, `report.json`, also records the tape and reader hashes, validated counts, fitted coefficients and source standardizers. The implementation handoff covers macOS static checks. Measured estimates require the Linux invocation. [Reader](factor.py)
