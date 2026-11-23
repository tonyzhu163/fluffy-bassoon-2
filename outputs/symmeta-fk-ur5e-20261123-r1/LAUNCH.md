# Paired UR5e π₀.₅ evaluation

From the parent-deployed, pinned checkout on a single Baige Distributed Training host with eight qualified RTX PRO 6000D devices, run this command. Set `PIN` to the parent-verified full deployment commit and select a fresh attempt name for each invocation.

```bash
bash outputs/symmeta-fk-ur5e-20261123-r1/run.sh "${PIN:?set the verified deployment commit}" paired01 8 2700
```

The final two arguments are the number of allocated GPUs (1–8) and wall seconds for the whole payload. For a one-GPU allocation use `paired01 1 21600` in the same command. Both examples bound this invocation to six launcher GPU-hours. Apply the same wall limit to the scheduler and disable automatic retries to bound the allocation itself. Every invocation runs all eight tasks. Device lane `d` executes shards `d, d + GPU_COUNT, …` in sequence. The deadline includes input verification, the GPU probe, all model loads, simulator children and the strict merge. The launcher rejects an existing attempt and caps the requested GPU count times wall time at eight GPU-hours. Any additional attempt draws from the remaining cumulative pilot budget.

Results go to `/mowenjie/fluffy-bassoon-2/outputs/symmeta-fk-ur5e-20261123-r1/paired01/`. The deployed tree supplies the evaluator and evidence. Both fresh arms use the verified historical Goal 6 recovery checkout, including its official motor, simulator and pre-inference reset callback. Each logical task retains its own reset reference across all twelve transforms. The first create is discarded without taking action. Each arm then has at most three pre-action creates to match the reference. A native rollout is never retried. The baseline runs first, followed by FK. The original Stage 2A query schedule, policy weights, prompt, horizon and float32 noise bank are reused.

## Records and interpretation

- `panel.json` is written before inference. It locks index, task, transform, original support path and SHA-256, manifest hash, support physics, support/query seeds, both yaw estimates and the shared unrounded learned scale and grip. The join checks the original tape bytes on the allocated host. The FK angle is checked independently against the dot and cross products of the issued XY means and measured world displacements.
- The seed-41 tuple comes from the CPU export in `evidence/reader.json`. Each yaw matches the upright native model output in `roll.json.gz`. Model identity, tape digest and case identity must agree. Only the estimated yaw field from `world.json.gz` replaces the learned yaw. The motor receives estimated corrections and a name-only embodiment selector. The simulator uses the selector to instantiate the physical interface. The prior GPU run published rounded predictions; exact historical floating-point parity is unknown.
- Each atomic `shard-N/II.json` retains both entire native records, including step rewards, return, terminal flags, controls, success, reset and opening proposal fingerprints. Each pair must share the complete pre-action reset, explicit noise-bank digest and opening policy proposal. The first completed pair in each shard exercises the real simulator and frozen motor. Progress and rejected reset field differences are in `shard-N.log`.
- `summary.json` requires 96 complete pairs. It reports native successes, FK wins/losses/ties, success-rate differences and task-cluster bootstrap intervals overall and on the 64 OOD cases, plus the eight task effects. Historical SymMeta successes and changed indices are joined after scoring. The raw case files remain available if a shard fails. A nonzero `exit.txt` status or a missing summary means the attempt did not complete.

## Implementation checks

On macOS, Bash syntax and Python 3.11 parsing passed. A readback of the pinned compressed evidence, full-precision reader export and official rows validated all 96 case joins, 96 distinct ordered tape hashes, all 96 measured planar alignments and exact upright learned yaw agreement. The shard allocation covers shards 0–7 exactly once at each requested GPU count. The native runtime and fresh outcomes await the parent launch. Implementation used 0 GPU-hours.
