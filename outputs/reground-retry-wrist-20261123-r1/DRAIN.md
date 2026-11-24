# Finish the enrolled wrist query batches

Baige attempt `retry05` is ready for parent submission on one exclusive RTX 4090, ten CPUs, 64 GiB memory and 60 GiB shared memory. The scheduler wall is 90 minutes and the launcher caps the payload at 5,250 seconds.

```bash
bash /mowenjie/fluffy-bassoon-2/outputs/reground-retry-wrist-20261123-r1/drain-src-6e51cd532ec9c5b7ca285a051660bd181374fae8/recovery/drain/launch.sh 6e51cd532ec9c5b7ca285a051660bd181374fae8 retry05
```

The 15 assigned batches are 055–069, covering 60 distinct starts. Batch 054 remains enrolled and parked for separate qualification. The dispatch uses frozen science pin `ac28987f221e336cef2ea64dc16dfffc9f0e74c9` and extends qualified recovery launcher `d6de8c725cfaf2dbb01a6a9c4895197441b8a6af` with operational pin `6e51cd532ec9c5b7ca285a051660bd181374fae8`. The operational launcher hashes to `7638ca269111d3bf505038dc7dbb1bba87c6585c68ba0666a2f68262351a8d53`. Use this hash, the actual scheduler ID and explicit batches 55–69 in the independent producer registry.

Each assigned existing record passes the frozen validator, native journal reconciliation, archive integrity and applicable stored byte hashes before a new rollout starts. The original 54 sealed batches enter the census by directory name. Batches publish by atomic rename on the original filesystem. The producer retains the science pin, and its separate dispatch receipt identifies the operational code and scheduler. Publication directories are authoritative if shutdown interrupts a progress checkpoint. The original journal accounting includes partial physical interactions.

An exact raw camera guard before the first wrist control preserves its exception and attempt files. Dispatch checks the remaining leaves, requires clean native pool teardown and restarts the reader before continuing. An additional state, shape, dtype or other structural difference stops dispatch. Exit 3 means that all assigned batches were handled and at least one remains unpublished after a raw camera guard. Full inference still requires all 70 batches.

The one-core Baige check used the frozen scheduler, validator and guard with synthetic rollout inputs, plus real forkserver worker startup and teardown. It checked enrolled membership and order, shared-lock exclusion, sparse resume, exact failure logging, structural halt and rejection of a byte-altered record. The parent owns scheduling and first live GPU admission. [CPU evidence](evidence/drain-cpu-6e51cd532ec9c5b7ca285a051660bd181374fae8.json) · [Operational Git bundle](evidence/drain-6e51cd532ec9c5b7ca285a051660bd181374fae8.bundle)
