# Native inference config repair

Attempt `gpu02` passed the allocated-device probe, initialized JAX on the
actual accelerator, and returned from the original parameter restore/model
load. It failed next in `LeRobotRobocasaDataConfig.create`, before any policy
forward or simulator episode. The factory tried to recompute statistics from
the unavailable Human300 training-dataset metadata before the policy loader
could read the checkpoint's bundled normalizer.

The public source already supports supplying `base_config.norm_stats` to
skip that fallback. The next attempt supplies the exact checkpoint-bundled
`assets/norm_stats.json` through that field and the policy loader's existing
`norm_stats` argument. The file SHA256 is
`4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee`,
the same normalizer bound in the original checkpoint and earlier local runs.

No upstream source, tensor, action/observation transform, normalization
formula, model configuration or dataset mixture is changed. This avoids
reading training metadata that is unused during inference. The CPU
qualification checks the real factory with these supplied statistics,
unchanged model/data-directory fields, exact state/action means and standard
deviations, 32-dimensional state padding and all three camera slots.

The fresh attempt also seeds NumPy once before the task loop, matching the
released evaluator's placement. The first two startup attempts produced no
simulator controls or baseline performance measurements. Their receipts and
costs remain separate from subsequent attempts.
