# Joint three-task training with four GPUs per preparation task

**Closed, Nov5 11:10:32 Shanghai:** AD and DPT converged; DIT and IC-IQL
remain unconverged and are preserved as complete resumable checkpoints.
The two-GPU continuation was manually stopped after the user clarified that
further fitting should use frozen-expert KV and velocity-residual steering.
All GPUs are released; no further direct-action fitting is authorized by this
launcher. The actual continuation cost was0.408889 GPU-hours. Shared cumulative
cost is73.785278/96 GPU-hours, leaving22.214722. This run is an exploratory
baseline and does not establish steering through the frozen action expert.
[Closure](DIRECT-ACTION-CLOSURE.json), [full-state audit](DIRECT-ACTION-CLOSURE-PREFLIGHT.json).

The user requests a new eight-model comparison on the existing three-task
corpus: four objectives using learned KV and the same four using velocity
residuals, followed by the same ID/OOD history readout. Its implementation,
qualification and measured budget belong to a fresh immutable run identity.

The three task jobs only prepare data. Each method then trains **one model on
all three tasks together**: AD, DPT, DIT and IC-IQL use separate objectives and
parameters over the same pooled source. There are four joint direct-action
models. The earlier separate-task fit draft is superseded before any launch.

| Phase | Allocation | Time and maximum reservation |
| --- | --- | --- |
| Prepare OpenDrawer, CloseToasterOvenDoor, CoffeeSetupMug | Three independent 4-GPU jobs; 12 GPUs concurrently | Up to 4 hours each; 48 GPU-hours combined |
| Pool and jointly fit AD, DPT, DIT, IC-IQL | One 4-GPU job; one GPU per method | 8-hour allocation; 32 GPU-hours initially |
| Continue any unconverged joint fits | Same immutable code, data and full resume checkpoints | Remaining cumulative budget; 16 GPU-hours reserved at worst-case preparation cost |

Request Baige Pro6K / 算法Z组 queue `aihcq-rh0wwk9muei4`, 64 CPUs,
256 GiB RAM and 32 GiB SHM per job, subject to current queue limits. Recheck
capacity and actual accelerator type at submission. Automatic retry is off.
Release preparation GPUs before allocating pooled fitting. Mount
`bos://dexmal-mowenjie-data` read-only at `/mnt/bos/adgfyl` for preparation;
the PFS code/runtime/output mount remains `/mowenjie`. The successful template
retains automatic retries OFF. Failed startups reduce the worst-case
continuation reserve from 16 to 15.8133 GPU-hours.

## Data and convergence

Preparation uses four disjoint two-rank source shards per task, B64 on each GPU.
It retains all 130 producer versions, step-130 teacher labels and H600 physical
histories. Pooling namespaces stream IDs by task and preserves each complete
chronological stream. Training has **336 streams** (112/task), validation
**48 streams** (16/task), with the original train/validation slots unchanged.
Equal stream counts and lengths give equal task sampling probability. Task
names in provenance do not become privileged model inputs.

All actor and critic transformers use **width 256, 6 layers and 8 heads**, as
requested by the user. Each actor has 5,579,184 trainable parameters.

Each learner trains with `--until-converged`, at least 1,000 updates, validation
every 100 updates, 16 validation batches, patience 10 and relative delta .001.
DIT and IC-IQL complete the value plateau before policy extraction. The global
batch remains 8 with the existing optimizer/loss settings.

Operational slices last 6,600 seconds and preserve model, optimizer, sampler,
RNG and validation plateau state. The coordinator repeats incomplete slices
inside the allocation. Further allocations resume the same identities and
immutable source until every method declares convergence or the approved
cumulative budget is exhausted. Allocation expiry is recorded as an operational
checkpoint. It never implies convergence. Check cumulative scheduler charges
before continuing; request an increase only if the 96 GPU-hour ceiling would be
exceeded.

## Submission commands

Set `TASK_CODE_DIR` to the clean, Git-deployed source pin that contains this
launcher's final qualification. Run each source task command in a separate
four-GPU, 240-minute scheduler allocation:

```bash
ROBOCASA_META_TASK=OpenDrawer ROBOCASA_META_ATTEMPT=prepare-4gpu-01 bash "$TASK_CODE_DIR/outputs/robocasa-three-task-meta-baseline-20261104-r2/COMMAND.sh"
ROBOCASA_META_TASK=CloseToasterOvenDoor ROBOCASA_META_ATTEMPT=prepare-4gpu-01 bash "$TASK_CODE_DIR/outputs/robocasa-three-task-meta-baseline-20261104-r2/COMMAND.sh"
ROBOCASA_META_TASK=CoffeeSetupMug ROBOCASA_META_ATTEMPT=prepare-4gpu-01 bash "$TASK_CODE_DIR/outputs/robocasa-three-task-meta-baseline-20261104-r2/COMMAND.sh"
```

Once all three preparations are verified and released, submit pooled fitting
as one four-GPU, 480-minute scheduler allocation:

```bash
ROBOCASA_META_TASK=joint ROBOCASA_META_ATTEMPT=fit-4gpu-01 bash "$TASK_CODE_DIR/outputs/robocasa-three-task-meta-baseline-20261104-r2/COMMAND.sh"
```

Use a fresh allocation attempt identity on each diagnosed continuation.
`ROBOCASA_META_WALL_SECONDS=14400` supports a four-hour continuation when that
fits the remaining cumulative budget. The launcher rejects all per-task fits
and refuses to launch while `launch_hold` is true. Every allocation probes all
four GPUs, retains scheduler `NVIDIA_VISIBLE_DEVICES` and binds workers through
the attested CUDA/PCI mapping.

This launcher covers the direct-action four-method baseline. Overnight AD/DPT
ID50 and two-task OOD50 simulator evaluations completed with full-history and
episode-reset controls. These are direct-action results, distinct from learned
steering through the frozen action expert. They show no consistent history
advantage. All continuing work counts against the same96 GPU-hour ceiling.

For this continuation, the immutable wrapper command is:

```bash
BABYSIT_CODE=/mowenjie/fluffy-bassoon-2/sources/babysit-continue-038892d3 BABYSIT_COMMIT=038892d3896b34b96b2a8da0faf835545a2375c7 BABYSIT_METHODS='dit ic_iql' BABYSIT_ATTEMPT=fit-continuation-02 bash /mowenjie/fluffy-bassoon-2/sources/babysit-continue-038892d3/outputs/robocasa-eight-hour-babysit-20261104-r1/CONTINUE-FITS.sh
```

The wrapper preserves the original learner and pooled-data identity. Resource
limits are2 GPUs,32 CPUs,128 GiB RAM,16 GiB SHM and120 minutes. The canonical
shared ledger is `outputs/robocasa-eight-hour-babysit-20261104-r1/META-ALLOCATION-LEDGER.json`
on PFS. Reconcile scheduler release there before any subsequent allocation.

## Pooled-container migration, 2026-11-04

The original zlib writer dominated CPU pooling. The owned CPU-only attempt was
stopped with its partial output preserved as
`joint/preparation-pool-cpu-01-abandoned`. No learner was running. The replacement
`POOL-UNCOMPRESSED.py` is Git-deployed at `6bc69884` and writes the same schema,
arrays, dtypes, task order and split metadata using uncompressed NPZ containers.
All eight objective/split numerical fixtures exactly match the original writer;
only file SHA256 differs. Source receipts and per-task dataset hashes are still
checked, and the learner/source pin stays `d71cb532`.

The fresh output is `joint/preparation-uncompressed-cpu02`. After completion and
verification, `joint/preparation` is a read-only-use path alias to that output;
its manifests bind the actual NPZ paths and hashes. The scientific data, global
batch 8, models and validation-convergence contract are unchanged.
[Migration receipt](POOL-MIGRATION.json),
[numerical qualification](UNCOMPRESSED-NUMERICAL-QUALIFICATION.json).
