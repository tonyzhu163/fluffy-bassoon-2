# Can a 16-command H1 support identify its forward yaw and gain?

H1's broad reader predicted wrist motion almost equally well with matching and swapped supports. This offline test asks whether a pooled model of the wrist's delayed response can recover the difference between the two held-out maps from the same mixed probes. Evidence for observability would include correctly ordered yaw estimates within a layout, low angular and gain errors, and lower independent-query error with the matching donor. Overlapping estimates or similar matching and swapped errors across layouts would count against this explanation. Both outcomes are informative. The method and comparisons are fixed before reading the held-out measurements. [H1 result](../embodiment-command-frame-20261120-r1/RESULT.md)

## Input and pairing

Use the single completed H1 `native02/collection/tapes.jsonl` with SHA-256 `c1207da02bd988de7fb51cab00a894ed29f1a31ca7d7306bfe9c9c7fb05c3ebf`. The implementation base is `1967f2b31fd4f64c0c9771fb04df2a268c1e2e41`. The loader hashes the bytes it parses and checks the complete set of 256 unique family, setting, role and index keys. Each record must have finite numeric arrays of shape 16 × 7 for public pre-map commands and 16 × 3 for measured world wrist displacement in metres. It checks the split, layout ID and seed, probe seed and kind. Every arm and setting within a placement block must have identical public commands and donor fingerprints. The four fingerprint fields must each distinguish all 40 placements. The estimator receives only command XY and displacement XY. [H1 schema and probe generation](../embodiment-command-frame-20261120-r1/COLLECTION.md) · [H1 storage manifest](../embodiment-command-frame-20261120-r1/evidence/native02/raw-sha256.json)

| Use | Source of tapes | Tapes | Response rows per tape |
| --- | --- | ---: | ---: |
| Fit the nominal wrist response | Training supports, nominal, both families × 12 indices | 24 | 13 |
| Fit the fixed common multiplier and set the score scale | Training supports, both families × four source settings × 12 indices, including the nominal tapes | 96 | 13 |
| Estimate each support's map | Held-out supports, both families × two settings × eight indices | 32 | 13 |
| Score independent predictions | Held-out queries, same cells and indices, separate resets and command sequences | 32 | 13 |

The nominal source bank includes its two recorded pulse tapes. Source query roles are used only for input validation. The two held-out settings are +45° with unit gain and +90° with gain 0.65. For each query, the matching and swapped donors share the same family, support layout and command sequence. The swapped donor experienced the other held-out mapping. Family and setting identify records for pairing and reporting. All coefficients are estimated from unlabelled XY arrays after the source bank and donors have been assigned.

## Frozen source response and support fit

Write the public command as `u[t] = x[t] + i y[t]` and the measured displacement as `d[t] = Δx[t] + i Δy[t]`. Positive forward yaw turns public +X toward world +Y. The primary nominal prediction in metres is

```text
z[t] = h[0] u[t] + h[1] u[t−1] + h[2] u[t−2] + h[3] u[t−3].
```

Fit the four **real** coefficients by ordinary least squares on the 24 nominal supports. Stack the X and Y equations with the same coefficients, use zero intercept, and start each tap window within its own tape. Freeze this one filter across families, layouts and held-out settings. The one-tap timing ablation repeats the entire analysis with only `h[0] u[t]` and refits its source coefficients once. Both models fit and score responses at zero-based steps 3 through 15 inclusive. The first three commands supply causal history to the four-tap model.

For a single held-out support, fit the complex scalar that minimizes its 13 squared XY residuals,

```text
c = sum(conj(z[t]) d[t]) / sum(|z[t]|²).
forward yaw = atan2(imag(c), real(c)) in degrees
gain = |c|
```

Each query prediction is `c z_query[t]` from the query's public commands. Query observations enter only the loss. The four conditions use the matching support's scalar, the swapped support's scalar, fixed nominal `c = 1`, and a fixed common scalar. For the last condition, select the continuous least-squares optimum with the same formula, summing over **all 96 source supports**. This optimizes the shared multiplier for the source mixture. Every query uses that frozen value. No candidate held-out angles, gains or arm-specific coefficients enter either fit. Rank-deficient nominal fits, zero prediction energy and an undefined support phase raise an error so a case cannot silently disappear.

## Readouts and sampling units

- **Angle and gain.** Report the real and imaginary coefficients, signed forward yaw in [−180°, 180°), gain, signed wrapped yaw error, signed gain error, and both absolute errors. For scoring, classify by the nearer of +45° and +90° using circular distance. Exact ties count in the denominator as incorrect. Report correct, total and ties. Gain is relative to the pooled nominal response, so an arm with a different response magnitude can have a biased gain estimate.
- **Donor separation.** Record the wrapped correct-minus-swapped angle for each query. Its ideal sign is −45° on a +45° query and +45° on a +90° query. For aggregation across settings, orient the same contrast as estimated yaw at +90° minus estimated yaw at +45°. Report its signed value, its fraction of the ideal 45° separation, and the absolute donor change. The raw signed contrasts cancel across the balanced settings by construction. Correct-minus-swapped absolute yaw error below zero favors the matching donor.
- **Query error.** For each condition, XY MSE in m² is the mean of `(error_x² + error_y²) / 2` over the same 13 steps. Divide by one frozen source scale squared to obtain normalized MSE. The scale is the square root of the mean of `(dx² + dy²) / 2` on the 1,248 eligible source support rows. This isotropic scale applies to both axes, both tap models and every control. Each control-minus-correct MSE is paired on identical observed query rows. Positive values favor matching support. The normalized MSE *between* correct and swapped predictions measures sensitivity even if both predictions are inaccurate. H1's original reader score used XYZ, axis-specific scales and all 16 rows.
- **Aggregation.** Preserve each family × setting × index case. Report slices by family, setting, their intersection and the full panel. An empty slice label in JSON (`null`) pools that factor. Means weight cases equally. Yaw direction uses a circular mean, while errors and oriented contrasts use arithmetic means of the individually wrapped values. Per-index differences average only the cells in the stated slice. Each model has 32 query cases and 416 scored motion vectors. There are 16 family–index donor contrasts sharing **eight** independent support/query layout indices. The two settings give opposite views of the same donor swap. Rows, families and settings within an index stay together.
- **Uncertainty.** For each slice, draw eight layout indices with replacement 10,000 times using NumPy's default generator with seed 20261120. Reuse the same draws across controls, tap counts and slices. Report the 2.5th and 97.5th percentiles of the layout-mean paired MSE differences and oriented angle separation. These exploratory intervals describe variation over the eight observed index pairs conditional on the fitted source model. There is no tuned lag, model-selection gate or multiple-testing claim.

## Linux CPU invocation

The script needs Python and NumPy. From the project checkout in the existing Linux CPU environment, use the synthetic numerical test as a separate invocation. It checks known signed rotations and gains, causal coefficient recovery against independently generated convolutions, ignored warm-up rows, the fixed common fit, circular scoring, and predictions on a different command sequence. It prints its status and writes no evidence.

```bash
run=outputs/embodiment-tape-observability-20261120-r1
export CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -B "$run/estimate.py" --self-test
```

For evidence, supply the pinned archive and a fresh directory. The example reads the H1 archive in place.

```bash
h1=/mowenjie/fluffy-bassoon-2/outputs/embodiment-command-frame-20261120-r1/native02/collection
python -B "$run/estimate.py" --tapes "$h1/tapes.jsonl" --out "$run/evidence01"
```

`report.json` contains the input and script hashes, source scale, counts, both filters and fixed common multipliers. Each model's `cases` gives the raw estimates and paired losses. Its `slices` includes classification counts, query error and angle summaries, intervals, and the per-layout differences.

H1 already examined these eight indices. The two held-out settings couple yaw to gain, so nearest-angle accuracy alone cannot establish independent recovery of both factors. The signed paired angles, absolute calibration errors and query losses show what information transferred on this panel. Wrist dynamics, omitted command channels and short mixed probes can all affect identification. Native task performance and transfer to another arm require separate evidence. [Validity review](REVIEW.md)
