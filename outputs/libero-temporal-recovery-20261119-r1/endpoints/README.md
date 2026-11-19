# Endpoint audit of the archived bowl and bottle reversals

At control 300, do both ordered practice endpoints agree across the two hidden assignments that require opposite retries? Each assignment uses the same practice commands, two zero native rewards and the same query start as its paired opposite. A qualifying terminal match would have to preserve both camera arrays and all nine public robot-state values at *each* of the two practices.

## Evidence already available

| Published development panel | Eligible groups / candidates | Opposite correction rows | Published RGB native retries |
| --- | ---: | ---: | --- |
| Bowl, cyclic and fixed prompt order | 2 / 4 | 4 | Final dual-camera view 4/4 |
| Bowl, every ordered triple and fixed prompt order | 12 / 24 | 24 | 13-frame differences 24/24 |
| Bottle, cyclic and both prompt orders | 6 / 12 | 12 | Five-frame differences 7/12, swapped 5/12, all assignments 60/96 |

The published deterministic final-view bowl reader solved both sides of each of the two cyclic reversals. Since each side supplied two zero rewards, identical ordered endpoint images would have forced the same decision on both sides. This establishes zero completely RGB-matched reversals among the two groups (zero of four rows) by deduction. A single practice slot could still match. The archived five-frame bottle scores alone leave the bottle endpoint counts undetermined. [Bowl final-view decisions](../../libero-visual-witness-20261119-r1/feedback.json) · [Bowl panels](../../libero-visual-witness-20261119-r1/score.json) · [Bottle five-frame decisions](../../libero-witness-transfer-20261119-r1/rgb.json)

## Baige CPU reproduction

Use an allocated Baige CPU shell and a detached Git checkout containing this audit and the archived witness metadata. The checkout may be sparse if it includes both witness output directories and this script. Set `code` to that checkout and `pin` to its deployed full commit ID. From that shell:

```bash
set -eu
root=/mowenjie/fluffy-bassoon-2
rel=outputs/libero-temporal-recovery-20261119-r1/endpoints
test "$(git -C "$code" rev-parse HEAD)" = "$pin"
mkdir -p "$root/$rel"
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 300s "$root/setups/robocasa365-native-20261101-r1/env/bin/python" -Bu \
  "$code/$rel/audit.py" --pin "$pin" --output "$root/$rel/result.json"
```

The audit verifies its own bytes, the two tape manifests and the published bottle RGB scores against the supplied commit. It checks each of the 32 bowl and 36 bottle NPZ files against its manifest, pairs BASE with D18 by pose and command, verifies practice commands and shared resets, and reconstructs eligibility from the native ever-hit masks. Bowl queries use BASE. Bottle queries use the separately executed QUERY tapes. The strict groups on each panel contribute two dependent rows. The fit and all-ordered bowl panels are labeled separately from the cyclic development panel.

`result.json` lists every exactly equal pair within each archive by modality, every same-command practice pair with its count of differing RGB pixels per view and maximum state discrepancy, and the reversal groups for which both slots match. Cameras use every uint8 RGB value at 96 × 96. Robot state uses exact numeric equality on stored end-effector position, quaternion and both jaw positions. Distinct poses and QUERY tapes enter the unrestricted physical-pair inventory only; matched reversal counts use the registered practice/query roles.

Two terminal readers use only the last observation. The RGB reader uses the published transfer arm's preprocessing and 2 × 2 average grid from each camera. The state reader uses the final nine values scaled by source standard deviations. Both use hit/miss centroids pooled across the same 16 bowl fitting practices on poses 12–15. There is no random sampling or target fitting. Native query credit, same-start swapped strict credit and physical completion are reported. The audit re-scores the archived five-frame RGB practice scores on independently reconstructed bottle assignments and checks every row against the published decisions. The source and bottle candidate panels reuse tapes, so score differences are descriptive.
