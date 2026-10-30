# Native31/90 versus29/90: saved cross-run check

2026-10-30. Read-only comparison of all90 native K0 metadata from the six-worker
student firstwave and two-worker noise screen. No model construction/inference,
new rollouts, tests, GPU use or active-output modification occurred. Only the
four outcome-discordant cases had saved trajectory arrays opened.

Across **all90 tasks**, the following metadata match exactly between runs:
TargetCase, query seed, nominal initial-state hash, actual initial-observation
hash, public-array signature, and frozen motor parameter-state SHA.

|Task |Earlier native |Contemporaneous noise-screen native |First decoded action chunk max absolute difference |
| --- | --- | --- | ---: |
|33 |success |failure |0.0019527674 |
|38 |failure |success |0.0029290915 |
|44 |success |failure |0.0035393238 |
|82 |success |failure |0.0029290915 |

Direct saved-array inspection on these four cases additionally confirms:

- Initial main RGB, wrist RGB and8D proprioception are **bitwise identical**.
- First native Gaussian noise arrays, shape10×32, are **bitwise identical**.
- First decoded/executed action arrays, shape10×7, differ in49–56 values;
  mean absolute differences are0.000302–0.000369 in decoded action units.

The observed divergence is already present at frozen-motor decoding with
matching recorded initial inputs, initial noise and parameter hashes. Changed
initial Gaussian draws or initial rendered observations do not explain these
four flips. This is consistent with the previously observed cross-process
numerical/runtime sensitivity; the check does not identify a particular
kernel, establish the sole cause of each later outcome, or exclude all
unrecorded runtime differences.

Keep both runs separate. The earlier native31/90 is a reference, and native
29/90 is primary for its contemporaneous noise-control comparisons. Do not
pool them as180 identical trials or reuse one as the other's paired control.

`receipt.json` contains mismatch-ID lists for every metadata field, outcome
IDs, both source group/trace SHAs, and full first-chunk/initial-array equality
and delta statistics. `compare_saved.py` reproduces the bounded saved-data
check using standard-library/NumPy operations only.
