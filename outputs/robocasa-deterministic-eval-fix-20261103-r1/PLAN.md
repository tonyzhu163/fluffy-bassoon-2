# Deterministic evaluation repair

Authorized by the human on November 3, 2026: fix deterministic batching and
matching scenes while preserving observation-conditioned steering and the
existing shared independent Gaussian draws.

New evaluations use fixed row/wave schedules with padded model-only rows,
stable counter-region ordering in their spawned native environments, and a
common cross-arm canonical case directory. Reject physical or initial-pixel
mismatches before dispatch, retaining mismatch evidence. Training collection
and historical pinned attempts retain their existing semantics; no active
worktree, checkpoint or run contract is edited.

Validation: core collector, motor, environment-pairing and checkpoint tests on
allocated Linux CPUs; then a bounded native GPU comparison if a properly
qualified allocation is available. GPU qualification ceiling: one GPU-hour,
10,000 physical controls including reset allowance, no SGD. Use new output
identities for a diagnosed retry and carry costs. No new full scientific PPO
run or replacement 500-case benchmark is authorized by this repair plan.

Qualification must check fixed batch membership under different reset timing,
padding without physical steps, unchanged Gaussian tensors, fresh live
observations and observation-conditioned steering, cross-arm scene checks,
and preservation of completed episode checkpoints on failures. Existing
baseline artifacts remain evidence; do not overwrite them with corrected
measurements.
