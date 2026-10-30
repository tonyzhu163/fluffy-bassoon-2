# Third trained source round pair

Baige job-l59ep1gqsndb completed Success at01:44:45Shanghai on2026-10-31. Scheduler time371GPU-seconds (01:38:34start), parent353.208856seconds. Same qualified57016ba runtime/c353970 launcher; this continues the original100-round source goal without new pilots, model/optimizer changes or qualification rollouts.

| Learner | Round | CompleteK4 | Source actions | PPO updates / SGD |
| --- | ---: | ---: | ---: | ---: |
| Persistent PPO |55|330|526,663 (+9,220)|55 /275 (+5)|
| Trained reset PPO |57|342|545,156 (+9,486)|57 /285 (+5)|

Added18,706source actions,606recorded decisions and48completeepisodes. Allocated and independent audits passed: exact checkpoint/manifest/input hashes,2×263ZIPmemberCRCs,all606native-masks,12exact-zero likelihood/value replays,durablegroupACK,fullcoordinator/Adam/sampler/RNG/stagesrecovery,unchangedfrozenmotor,zero pending/uncertaincanonicalheads and clean child0/GPUcontext0 teardown. The100-round campaign remains unfinished.

Actual scheduler throughput50.42actions/GPU-second; peak65.55GiB includingSGD. Differences across the three short slices are not a paired causal scalar speedup comparison. Unused2,329operationalGPU-seconds are sealed; unused action allowance does not add trials.

[Independent audit](receipts/independent-final-audit.json) · [Allocated audit](receipts/audit.json) · [Scheduler](receipts/scheduler-terminal.json). Root prepared R4 from these exact final checkpoints while the separate continuous supervisor is corrected and CPU-qualified. Static prepare_continuation.py only builds metadata/argv; actual runtime admission remains allocated Linux and verifies current ownership/locks/checkpoints/configuration/remaining caps beforeCUDA.
