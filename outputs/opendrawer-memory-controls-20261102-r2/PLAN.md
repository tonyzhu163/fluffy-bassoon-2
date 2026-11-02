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

Eight1-GPU containers,32 CPUs/128GiB RAM/32GiB SHM each,140-minute scheduler
cap (`TASK_MEMORY_LIMIT_SECONDS=8400`,135-minute operational deadline).
r1 conservatively charges17280 GPU-seconds; r2 maximum67200 gives84480 total,
below the original86400 ceiling. Automatic fault retries stay off. Check the
first750-control ID episode promptly, then every20minutes. Diagnose a failure
before considering another identity. Fifty cases per task remain exploratory.

The available cards are fragmented: the scheduler admitted only three2-GPU
instances at preparation. Use `TASK_MEMORY_GPUS_PER_NODE=1` and eight nodes;
logical ranks/case partitioning stay0..7, one singleton motor per GPU. The
driver requires scheduler nodes × requested local GPUs =8 and probes exactly
the requested local count. Total requested CPU/RAM and GPU-time cap are
unchanged. The CPU preflight already verified750/900 and267 carried records
at `cbcffb71`; rerun the binding at the final scheduler-layout source pin.

## Launch and first simulator verification

Deployed through direct HTTPS Git fetch at
`1b9291bdc5729490cf0bf34f00c55a777e8bfe23`; final allocated-Linux CPU binding
verified both750/900 adapters and all267 carried records before submission.
Baige Pro6K / 算法Z组 job `job-dxxkwgvdk4hy`
(`mwj_1102_memory_controls_8x1_r2`) started23:10 Shanghai Nov2, with eight
1-GPU containers and automatic fault retries disabled.

At23:17 Shanghai, all eight allocation probes passed with eight distinct UUIDs.
The shared `MONITOR.json` verified46 completed r2 episodes (18 ID,28 OOD)
against source/checkpoint identity, original cases/seeds, record/truth hashes,
initial pairing reports, terminal records, and reset/window lengths. The first
ID records contain15 terminal-aligned chunks across750 native controls,
confirming the historical horizon fix in real simulation. Combined with267
carried records this is313/1100 collected controls; no final comparison yet.
The20-minute heartbeat follows this job and will verify all2000 endpoints on
completion. Source worktrees and the failed r1 artifacts remain preserved.
