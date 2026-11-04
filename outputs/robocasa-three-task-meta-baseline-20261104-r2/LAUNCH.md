# Four GPUs per task, eight-hour submission proposal

**Status:** Code prepared; higher cumulative budget approval and allocated-GPU
qualification pending. No scaled job submitted.

The user requested four GPUs per task and more time. This proposal doubles the
original four-hour scheduler limit to eight hours. Three independent jobs reserve
**12 GPUs concurrently**, **32 GPU-hours per task**, **96 GPU-hours total**.
This exceeds the prior explicit 21 GPU-hour ceiling; the higher ceiling must be
approved before submission, as required by `AGENTS.md`.

| Setting | Proposed value |
| --- | --- |
| Queue | Baige Pro6K / 算法Z组, `aihcq-rh0wwk9muei4`; recheck capacity and limits at submission |
| GPUs | 4 per task, one scheduled container per task |
| CPU / RAM / SHM | Request 64 CPUs / 256 GiB / 32 GiB per task, subject to queue limits |
| Duration | 480 minutes per task; stop sooner if all fits converge |
| Retry | Off |
| Preparation | Four disjoint two-rank source shards, one worker/GPU, B64 per GPU |
| Fitting | GPU 0 AD, GPU 1 DPT, GPU 2 DIT, GPU 3 IC-IQL |
| Identity | New `robocasa-three-task-meta-baseline-20261104-r2` output root |
| Source | Latest reviewed `main`, committed and deployed to a clean detached worktree |

Set `TASK_CODE_DIR` to that deployed worktree and submit each command as its
own four-GPU, 480-minute scheduler job:

```bash
ROBOCASA_META_TASK=OpenDrawer ROBOCASA_META_ATTEMPT=prepare-fit-4gpu-01 bash "$TASK_CODE_DIR/outputs/robocasa-three-task-meta-baseline-20261104-r2/COMMAND.sh"
```

```bash
ROBOCASA_META_TASK=CloseToasterOvenDoor ROBOCASA_META_ATTEMPT=prepare-fit-4gpu-01 bash "$TASK_CODE_DIR/outputs/robocasa-three-task-meta-baseline-20261104-r2/COMMAND.sh"
```

```bash
ROBOCASA_META_TASK=CoffeeSetupMug ROBOCASA_META_ATTEMPT=prepare-fit-4gpu-01 bash "$TASK_CODE_DIR/outputs/robocasa-three-task-meta-baseline-20261104-r2/COMMAND.sh"
```

`BUDGET.json` remains a proposal and the runner refuses to launch until its
approval is recorded. The runner probes all four scheduler-assigned GPUs before
starting children. It preserves `NVIDIA_VISIBLE_DEVICES` and gives each worker
one token from the probe's CUDA/PCI mapping. The full chronological source,
teacher, saved logical batch, train/validation slots, losses and optimizer
schedules stay unchanged. Preparation has a four-hour operational bound; merge
and fitting share the remaining time with checkpoint/release reserves. Each
learner retains its existing convergence rule and resumable checkpoint format.

These commands fit the existing four direct-action learners. Bridge training and
closed-loop evaluation described in the original broader plan require their
own implementation/qualification and must remain inside the approved cumulative
budget. The active Stage 1 eight-GPU PPO/GRPO job is outside these changes.
