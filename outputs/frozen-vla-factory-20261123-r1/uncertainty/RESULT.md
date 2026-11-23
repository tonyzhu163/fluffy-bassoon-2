# Frozen-reader disagreement did not improve archived retry outcomes

All eight calibration folds kept SymMeta seed 41 for every query when passive Horn–Schunck was the fallback. The rule replayed 190 successes in 288 previously executed target queries, exactly tying seed 41. The alternative rule that switched to the most central SymMeta seed lost three net successes. These are retrospective selections between archived native binary outcomes. This job ran zero new simulator episodes.

| Policy | Seed-41 coverage | Replayed successes / 288 | Wins / losses vs seed 41 | Paired difference, pp (95% task interval) |
| --- | ---: | ---: | ---: | ---: |
| Always seed 41 | 100% | 190 | 0 / 0 | Reference |
| Source-calibrated Horn–Schunck fallback | 100% | 190 | 0 / 0 | 0 (identical decisions) |
| Force Horn–Schunck above source 90th percentile | 77.1% | 183 | 10 / 17 | −2.43 [−4.51, −0.69] |
| Force Horn–Schunck above source 75th percentile | 59.7% | 171 | 12 / 31 | −6.60 [−9.38, −3.47] |
| Force Horn–Schunck above source median | 42.7% | 160 | 16 / 46 | −10.42 [−13.19, −6.94] |
| Source-calibrated three-seed medoid fallback | 77.8% | 187 | 4 / 7 | −1.04 [−2.78, +0.69] |

**Where rescues existed.** Horn–Schunck alone scored 138/288. Choosing the better recorded outcome in hindsight would score 211/288, with 21 rescues available. Fourteen were on UR5e × π₀, four on Kinova3 × π₀.₅ and three on IIWA × π₀.₅. The source 90th percentile sent 66/288 target supports to the fallback. Its 183 successes exceeded the 179.33 expected for randomly placing the same switches within each task and panel, but its 17 losses still outweighed ten rescues. The calibration panel offered no threshold with a strictly positive net gain in any fold.

**Secondary alternative.** The medoid is an already evaluated seed. Seven source folds favored switching to it at high disagreement. On the target tasks this scored 75/96 on IIWA, 60/96 on Kinova3 and 52/96 on UR5e × π₀, versus fixed seed 41 at 76, 60 and 54. Coverage-matched random placement within task, panel and fallback seed gave 189.04 expected successes. Unconditionally choosing the medoid scored 196/288, the same total as fixed seed 43 (seed 42 scored 191). On the 192 held-out transform combinations or continuous angles, fixed seed 41 and the calibrated Horn–Schunck rule each scored 125, the calibrated medoid 122, the unconditional medoid 124 and fixed seed 43 also 124. [Fold thresholds, task differences and controls](metrics.json)

**Access and leakage audit.** Each route uses the three readers' predicted inverse commands from a single passive support lasting 15 controls (0.75 s). Only UR5e × π₀.₅ outcomes from seven *other* task identities calibrated each fold. The score has no access to the task, hidden transform name or factors, reference errors, query observations, or native labels. The joins use case metadata to align branches and split tasks. All 384 distinct tape digests agreed with the NPZ qualification exports. Every support used layout 11 and every query layout 13. Replacing all three target panels' outcome fields with unparseable values left the routes unchanged. Flipping the held task's calibration-panel outcomes left its target routes unchanged. A separate explicit 4 × 4 matrix calculation reproduced both calibrated policies' switch and success counts. All 288 target seed-41 outcomes matched the independent compressed historical case records. [Design](PLAN.md) · [Replay code](replay.py)

**Limits and decision.** The interval resamples eight whole task identities, retaining all panels, transforms and paired outcomes for each sampled task. It conditions on the fitted routes, observed starts, reader seeds and motor draws. A zero-width interval for identical routes carries no information about uncertainty on new tasks. The archived runs shared an official query schedule. Seeds 42 and 43 ran in separate jobs from seed 41, and the compact outcome table does not certify bitwise opening-proposal equality across jobs. The three-seed diagnostic export re-evaluated the historical frozen reader checkpoints on the same support tapes. A shared model bias can leave all three readers in agreement. The evidence does not support deploying either calibrated switch. If pursuing the 14 available π₀ rescues, test a predictor of *relative fallback benefit* on independent π₀ development starts before allocating a live confirmation run.

## CPU receipt and reproduction

Input revision `757a44cd4374833cad790852df8b87ddf412e638`. One local Python analysis on the lane VM, 0 GPU-hours. From the repository root with this commit's code and the input revision available in Git history, run

```sh
lane=outputs/frozen-vla-factory-20261123-r1/uncertainty
uv run --no-sync python "$lane/replay.py" --output /tmp/frozen-vla-uncertainty.json
cmp "$lane/metrics.json" /tmp/frozen-vla-uncertainty.json
```

Reproduction yielded byte-identical metrics (SHA-256 `018979b62fc40795e57ec6d006396bacc2d8668eb3c32dada7fa56f19a3558ba`). The ordered 288-case decision schedule across both fitted and all three fixed-coverage rules hashes to `84e73a313761f5ec616d00f6919f6ef63fe5b281e3ec032268abcd8146850991`.

The archived inputs are the [frozen SymMeta predictions](../../symmeta-perceiver-retry-20261123-r1/evidence/train02pro/predictions.json) (the tuned Perceiver fields are unused), [export and tape checks](../../symmeta-perceiver-retry-20261123-r1/export.py), [UR5e × π₀.₅ support identities](../../symmeta-camera-command-20261123-r1/evidence/roll.json.gz), [cross-arm tape records](../../symmeta-fk-crossarm-20261123-r1/evidence/summary.json), [paired native outcomes](../../symmeta-paired-statistics-20261123-r1/outcomes.csv), and [original job IDs and evaluator revisions](../../symmeta-paired-statistics-20261123-r1/runtime.txt). The pinned export verifies the remote passive NPZ bytes and three valid five-control chunks per case. This replay consumes the committed hash-keyed exports.
