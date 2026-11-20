# Frozen scoring for the probe-order comparison

The primary comparison is the four-tap physical reader’s **blocked minus shuffled matching-query MSE**. A negative value favors blocked practice. The registered preference for blocked requires a negative mean on each arm and a pooled paired 95% interval entirely below zero. The supervised lag-ridge reader gets the same comparisons. Matching practice has predictive value to the extent that it improves on each single-factor donor and the fixed common correction on the same queries. [Registered decision](PLAN.md) · [Frozen design](../embodiment-tape-observability-20261120-r1/RESULT.md)

## Inputs and frozen readers

The source is fixed at `4bd494d188bfdcb757c554bb335b5c19d3bc97e6`. The scorer resolves both reports relative to its own location and checks their SHA-256 digests. It also checks the pinned implementations of the physical reader, the supervised features and the H1 mixed-command generator. [Scorer](score.py) · [Report digests](../embodiment-tape-observability-20261120-r1/evidence/cpu01/SHA256SUMS)

| Quantity | Frozen source |
| --- | --- |
| Four real impulse-response taps and fixed complex common correction | `models.four_tap.fir_m_per_command` and `models.four_tap.fixed_common_complex` in the [physical report](../embodiment-tape-observability-20261120-r1/evidence/cpu01/estimate/report.json) |
| Isotropic XY scale | `source_xy_rms_m` in the physical report, 0.002691755275079319 m per axis RMS from 1,248 source-support rows |
| Ridge intercept, feature means, feature scales and 8 × 2 weights | `readers.lags_0_to_3.fit` in the [supervised report](../embodiment-tape-observability-20261120-r1/evidence/cpu01/factor/report.json) |

Both readers take just the public pre-map command XY and world wrist displacement XY of one support. With X + iY coordinates, a positive forward yaw rotates +X toward +Y. The physical reader computes the causal four-tap prediction `z` from the commands and estimates one complex multiplier `sum(conj(z) * motion) / sum(abs(z)**2)` on support steps 3–15. This per-tape estimate uses the frozen source response. The supervised reader calls the original `features` on all 16 support steps, standardizes the eight real/imaginary lag components in source order, and evaluates `intercept + standardized_features @ weights`. Neither reader clips gain or snaps yaw to a label. Zero estimated magnitude makes yaw undefined and stops scoring. [Physical implementation](../embodiment-tape-observability-20261120-r1/estimate.py) · [Supervised implementation](../embodiment-tape-observability-20261120-r1/factor.py)

## Panel and pairing

The JSONL must cover exactly 384 unique arm × index × yaw × gain × schedule keys, with 128 tapes for each of `blocked`, `shuffled` and `query`. Each carries 16 finite 7-component public controls and 16 finite 3-component world displacements, totaling 6,144 actions. For index `i` in 0–15, both support schedules use donor seed `991200+i` and the query uses donor seed `991300+i`.

The scorer reconstructs the four-command runs of +X, −X, +Y and −Y at amplitude 0.39 with jaw −1. It checks the exact row permutation from seed `2991200+i`, including equality to the recorded blocked rows. It regenerates the mixed query with the original H1 generator and seed `2991300+i`. Each schedule’s commands match across the two families and all four settings. The four donor placement fingerprints must each agree within a seed across arms, settings and applicable schedules and distinguish all 32 seeds. Settled simulator-state and scene fingerprints and the public opening must agree **within each arm and donor**. The scorer checks the native reset against the arm, seed and donor, and checks action bounds, wrist and native receipt lengths, step numbering and early episode endings. It relies on the collector’s live checks of simulator returns, issued controls, wrist differences, collisions and renderer state. [Native collector](collect.py)

## Query comparisons and factor errors

Each arm × index × yaw × gain cell has one independently reset query. All predictions use its frozen causal response and its same 13 observed XY rows, zero-based steps 3–15. That is 1,664 unique query vectors over 128 queries. For multiplier `c`, the normalized MSE is the mean over these steps of `abs(c * z - observed_xy)**2 / (2 * source_xy_rms_m**2)`. Query motion enters only this loss.

| Report key | Multiplier applied to this query |
| --- | --- |
| `matching` | Reader estimate from that setting and schedule |
| `yaw_swap` | Entire reader estimate from the other yaw at the same gain |
| `gain_swap` | Entire reader estimate from the other gain at the same yaw |
| `common` | Fixed complex correction fitted to the archived source supports |
| `nominal` | Identity, 1 + 0i |
| `ideal` | Known forward map, true gain × exp(i × true yaw in radians) |

Both swaps preserve arm, layout index and support schedule. A positive `*_minus_matching_mse` favors the matching estimate. The ideal diagnostic measures the accuracy of the shared wrist-response model when given the true map. Arm-dependent dynamics can allow an estimated multiplier to have smaller query error.

Each raw case preserves the donor identities, complex multipliers, matched continuous estimates and paired losses. Yaw error is wrapped to [−180°, 180°). The signed yaw gap is estimated yaw at 90° minus estimated yaw at 45°, wrapped, at fixed gain. The signed gain gap is estimated gain at 1.0 minus estimated gain at 0.65, at fixed yaw. Their ideal values are +45° and +0.35. Each pair appears with the same orientation in its two target cells. Absolute errors are given both for each target estimate and for each gap relative to its ideal. The fractions with target errors strictly below 22.5° or 0.175, and below both, are descriptive half-spacing references. Values on a boundary receive no credit for being below it.

## Aggregation and uncertainty

Each reader has 128 raw cases, each pairing both schedules. Every summary reports the means for blocked and shuffled plus their within-case blocked-minus-shuffled differences. Each metric includes a percentile 95% interval. A family, yaw or gain value of `null` pools that dimension with equal weight per cell. The eight individual arm/setting cells, both arm means, fixed-gain and fixed-yaw slices, and the overall equal-cell mean are available directly.

Within each slice, average the cells sharing one layout index first. Draw 16 indices with replacement 10,000 times using NumPy’s default generator and seed 20261120. Reuse those draws for every contrast, slice and reader. This keeps both arms, all settings, all predictions and both schedules paired. The repeated view of each donor gap and the repeated scoring of a query do not add independent observations. These intervals describe layout variability conditional on the frozen readers.

## Linux CPU invocation

Run the synthetic self-test separately. It exercises the inclusive native action bound, signed X/Y rotation, gain, causal alignment, source standardization, the support least-squares minimum, both full-donor swaps and the sign of the paired schedule contrast. It reads no trajectories or reports and writes no score file.

```bash
run=outputs/embodiment-probe-order-20261120-r1
uv run --active --no-sync python -B "$run/score.py" --self-test
uv run --active --no-sync python -B "$run/score.py" \
  --tapes "$attempt/collection/tapes.jsonl" --out "$attempt/score"
```

Set `attempt` to the completed Linux collection’s directory. With no arguments, the input is `collection/tapes.jsonl` and the fresh output directory is `score/`, both beside the script. The scorer writes `report.json` after validating and scoring the full panel. It refuses an existing output directory. The report includes the tape and source-report hashes, scorer hash, counts, scoring window, bootstrap settings, raw paired cases, summaries and the physical-reader probe-order decision under the registered rule.
