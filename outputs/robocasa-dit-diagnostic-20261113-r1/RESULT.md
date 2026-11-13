# DIT first-attempt diagnostic

The saved DIT first-attempt discrepancies are not explained by different starting history. All 250 paired cases across the five tasks have identical empty entry signatures and feedback; 14 have different first-attempt success outcomes. This checks the recorded entry state, not every possible source of nondeterminism.

The raw audit selected all eight CoffeeSetupMug cases in bank-000 before reading their arrays. Both complete archives and all 224 selected members passed their recorded SHA256 checks. Each pair has the same policy-weight digest and identical native noise across all 12 first-attempt chunks. No model or simulator was executed.

| Cases | Earliest observed difference | Outcome |
| --- | --- | --- |
| Slots 0, 1, 4, 5, 6, 7 | None in the compared observations, features, steering, native noise, commands or feedback across the complete first attempt | Paired outcomes agree |
| Slot 2 | After 100 executed controls, one main-camera RGB component differs by one intensity level. The next chunk's features, steering and commands then differ. | Both attempts fail |
| Slot 3 | At reset, 25 main-camera and 35 extra-view RGB components differ by one intensity level; physical/public state and starting history match. Features and commands differ on the first chunk. | Full-history run fails; reset-history run succeeds |

For slot 3, the largest first-chunk feature difference is 0.00872612, steering difference 0.00277883 and requested-command difference 0.00343113. These are descriptive values, not acceptance thresholds. They locate a difference in policy inputs before the divergent action trajectory. The evidence is consistent with rendering variation contributing to this success mismatch; it is not a controlled intervention proving that pixel variation alone caused the outcome, and it does not explain the remaining 13 discordant cases.

This narrows the diagnostic: no recorded initial history leak was found, while even exact initial images can diverge later in the rollout (slot 2). An initial-image check alone would not make full-episode trajectories identical. A matched AD/DIT comparison should control data, initialization, update count and cases, and measure first-attempt repeatability across repetitions alongside the history effect. It must retain actual observations and report residual variation rather than silently replacing policy inputs.

The current recommendation is a controlled RoboCasa source expansion, with the existing three tasks as a fixed baseline and Cabinet/Rack/Counter-to-Stove as additions. See the [scope and caveats](SCALING.md). This diagnostic used zero GPU-hours; no new teacher or transformer job was launched. Existing budget caps remain unchanged.

[Compact raw comparison](evidence/SUMMARY.json) · [All first-attempt metadata](evidence/metadata.json) · [Full array-difference report and archive receipts](evidence/RESULT.json.gz) · [Imported evidence hashes](evidence/IMPORT.json) · [Execution record](EXECUTION.json).
