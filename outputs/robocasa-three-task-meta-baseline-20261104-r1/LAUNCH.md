# Baige Submission Spec — 3-Task Meta-Learning Baseline (3 × 1-GPU Jobs)

**Approved Budget:** 21.0 GPU-hours ceiling ([`BUDGET.json`](BUDGET.json))  
**GPU Plan:** **3 GPUs total** — three independent **1-GPU** jobs (one per task: `OpenDrawer`, `CloseToasterOvenDoor`, `CoffeeSetupMug`), each chaining Phase A (`PREPARE-STREAM.py` zero-unpack BOS streaming feature + `global_step_130` teacher export at B64) directly into Phase B (`FITS.py` 4-method `ad`/`dpt`/`dit`/`ic_iql` training) in a single allocation.

| Setting | Value |
| --- | --- |
| Queue | `aihcq-rh0wwk9muei4` (Baige Pro6K / 算法Z组) |
| GPUs per job | `1` × `NVIDIA RPBZZZ6` (3 GPUs total across the 3 jobs) |
| CPUs / RAM / SHM | `16` CPUs / `64 GiB` RAM / `16 GiB` SHM |
| Max duration | `240` minutes (`4` hours) |
| Automatic retry | `Off` |
| Pinned worktree | `/mowenjie/fluffy-bassoon-2/sources/three-task-meta-baseline-f125e65c` |

---

## Job 1: `OpenDrawer` (`mwj_1104_meta3_OpenDrawer_1gpu_r1`)

```bash
ROBOCASA_META_TASK=OpenDrawer ROBOCASA_META_STAGE=prepare_and_fit TASK_ATTEMPT=r1 /mowenjie/fluffy-bassoon-2/outputs/robocasa-three-task-meta-baseline-20261104-r1/COMMAND.sh
```

## Job 2: `CloseToasterOvenDoor` (`mwj_1104_meta3_Toaster_1gpu_r1`)

```bash
ROBOCASA_META_TASK=CloseToasterOvenDoor ROBOCASA_META_STAGE=prepare_and_fit TASK_ATTEMPT=r1 /mowenjie/fluffy-bassoon-2/outputs/robocasa-three-task-meta-baseline-20261104-r1/COMMAND.sh
```

## Job 3: `CoffeeSetupMug` (`mwj_1104_meta3_Coffee_1gpu_r1`)

```bash
ROBOCASA_META_TASK=CoffeeSetupMug ROBOCASA_META_STAGE=prepare_and_fit TASK_ATTEMPT=r1 /mowenjie/fluffy-bassoon-2/outputs/robocasa-three-task-meta-baseline-20261104-r1/COMMAND.sh
```
