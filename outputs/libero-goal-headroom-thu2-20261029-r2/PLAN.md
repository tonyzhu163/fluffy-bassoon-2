# LIBERO-Goal headroom r2 (recovery of r1)

Dated **2026-10-29**. Same question, checkpoint, environment, plan (340 episodes),
seeds and criteria as the
[r1 plan](../libero-goal-headroom-thu2-20261029-r1/PLAN.md). r1 (thu2 job 49204)
failed with a GPU out-of-memory error before the first batch; see
[r1 failure](../libero-goal-headroom-thu2-20261029-r1/FAILURE.md).

Changes, none of which alter what an episode measures:

- 16 renderer workers instead of 34 (each EGL process holds ~456 MiB), 24 CPUs,
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, 3000 s driver cap.
- Each worker keeps the environment built from its first task's BDDL instead of
  rebuilding it whenever the source task changes. All ten BDDLs compile to the
  same seeded scene (CPU sim-check), so this only avoids EGL context churn.
  Native-success parity now refers to the worker's own BDDL (`env_bddl_task`);
  every init state's conditions must still share one settled-state and MJCF hash.
- Per-batch GPU memory is logged. A CPU (OSMesa) smoke of the full run path with
  worker reuse passed (6 two-chunk episodes, parity and pairing checks clean).

This is the second and last GPU job of the approved budget. If it fails, stop and
report rather than relaunch.
