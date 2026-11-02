# Native reset pairing diagnosis

The independent dispatch fix is retained. AD readout ad-first-r1 failed its
first-case pairing guard: some repeated OOD seeds have identical robot states
and scene metadata but different visual features and motor chunks. No paired
student/PPO result is admitted. The queued DPT/full phase was stopped before
policy inference. Preserve both allocations' records and charge their costs.

This finite diagnostic uses one allocated GPU, at most20 minutes,24 native
constructor/reset calls and no policy actions or training. Compare exact raw
observations, full MuJoCo state and XML across3 resets of each of3 affected OOD
seeds and one ID seed, then repeat with global Python/NumPy/Torch seeding. Keep
the core68 runtime, native seed/horizon/camera configuration and cohort fixed.
Count constructor controls and physics through the original journal. Global
seeding is a diagnostic condition, not an adopted evaluation protocol.

Fix and requalify the identified cause before a fresh evaluation identity. Do
not weaken pairing checks, select cases using success, edit the active training
worktree, change learner selection, or rerun PPO/source collection.
