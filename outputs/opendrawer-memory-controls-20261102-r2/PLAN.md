# Matched-control continuation

Continue the original same-checkpoint, same-case, same-p1..p5 comparison after
r1's diagnosed registry mismatch. Source uses the shared driver in
`outputs/opendrawer-memory-controls-20261102-r1/`; set `TASK_MEMORY_OUTPUT` to
this fresh output identity and `TASK_MEMORY_REUSE` to the preserved r1 output.

The original retrospective protocol is OpenDrawer750 / CloseFridge900. Bind
those task horizons only inside this evaluation process; current prospective
training defaults remain600. The binding preflight instantiates both cohort
adapters on allocated Linux CPUs without creating a simulator. Native masks,
observed controls and original initial-state/pixel pairing remain checked in
every real episode. Controller weights, seeds, batching and64-token behavior
are unchanged. No new source histories, fitting or Flora optimization.

Verify all completed r1 rows against the old checkpoint, seeds, saved records
and simulator truth. Expected reuse267 (AD/OOD:184 reset,83 window64), leaving
833 new episodes. Replay carried window64 records to reconstruct the exact
next-attempt memory; skip completed reset endpoints independently. Keep all
incomplete r1 material as failure evidence, and collect its endpoint under
this new identity. The final summary verifies all1100 collected controls
across both allocations plus900 original unchanged endpoints =2000.

Four2-GPU containers,64 CPUs/256GiB RAM/32GiB SHM each,140-minute scheduler
cap (`TASK_MEMORY_LIMIT_SECONDS=8400`,135-minute operational deadline).
r1 conservatively charges17280 GPU-seconds; r2 maximum67200 gives84480 total,
below the original86400 ceiling. Automatic fault retries stay off. Check the
first750-control ID episode promptly, then every20minutes. Diagnose a failure
before considering another identity. Fifty cases per task remain exploratory.
