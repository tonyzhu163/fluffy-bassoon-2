# Joint three-task training with four GPUs per preparation task

**Status:** Human hold released by “keep going?”; preparing verified deployment and submission. The user approved
more resources/time and clarified joint training until convergence. The approved
cumulative ceiling is **96 GPU-hours**; see [BUDGET.json](BUDGET.json).

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
Release preparation GPUs before allocating pooled fitting.

## Data and convergence

Preparation uses four disjoint two-rank source shards per task, B64 on each GPU.
It retains all 130 producer versions, step-130 teacher labels and H600 physical
histories. Pooling namespaces stream IDs by task and preserves each complete
chronological stream. Training has **336 streams** (112/task), validation
**48 streams** (16/task), with the original train/validation slots unchanged.
Equal stream counts and lengths give equal task sampling probability. Task
names in provenance do not become privileged model inputs.

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

This launcher covers the existing direct-action four-method baseline. Bridge
training and closed-loop evaluation in the broader original plan remain pending
implementation/qualification and must fit the same cumulative approved budget.
The active Stage 1 PPO/GRPO allocation is unchanged.
