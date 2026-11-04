# Cross-task drift on matched denoising indices

**Status 2026-11-04: complete; Baige `job-pjgeuietm5of` succeeded and released its GPU at14:21:51 Shanghai.** Owner: Codex chat
`01a1a4de`, on the human's execution request. The dimension predecessor released
at13:05:17 Shanghai with80consumption receipts before this bridge started.
All10common indices qualify:50B16fixtures from10 restored/hash-verified original
Q1actor batches. All80 checkpoint loads /800forwards and the independent800-tensor audit pass;
initial likelihood parity is exact at every index for all five tasks.
[Verified result](RESULT.md), [final audit](FINAL-AUDIT.json). Ceiling remains oneGPU/4GPUh, no automatic retry,
simulator controls, sampling or SGD. [Execution](EXECUTION.json),
[prepared fixtures](PREPARED.json), [Q1 restoration](RESTORED-Q1.json).

## Why

The [dimension-drift readout](../robocasa-flow-sde-dimension-drift-20261104-r1/RESULT.md)
found that each task's fixed B16 cohort sits at one Flow-SDE denoising index:
OpenDrawer 0 (σ 0.300), Coffee 1 (0.285), Toaster and Stove 4 (0.116), and
Drawer→Counter 9 (0.032). A Gaussian log-density change scales with 1/σ², so
Drawer's cohort is ~90× more sensitive than OpenDrawer's. The raw
"Drawer drifts ~100× faster" comparison is therefore confounded and is not
evidence of a Drawer-specific PPO problem.

## Design

[`MATCHED-DRIFT.py`](MATCHED-DRIFT.py) reuses the source run's checkpoint
catalog, cold-storage bridge and hashing helpers unchanged.

- **prepare** (Linux CPU): from each task's retained iteration-1 actor batches
  (policy zero, both actor ranks, 1,536 samples), select 16 samples at **each**
  denoising index 0–9. The selection is seeded (20261104) and recorded by
  position and tensor hash. Indices with fewer than 16 samples in any task
  are dropped from `common_indices`, so every task is scored on the same
  indices.
- **stage** (dev CPU, BOS mount): one private checkpoint copy at a time, in
  the same task order as the GPU loop.
- **run** (one GPU, [`COMMAND.sh`](COMMAND.sh)): for each of the 16 retained
  checkpoints, score every matched B16 batch in eval mode. Step 0 must pass the
  native likelihood-parity check on every batch. The run keeps:
  - each batch's full `[16, 50, 12]` log ratio,
  - raw and variance-normalized means per index (×σ_k²/σ_0²),
  - per-dimension sums,
  - the mean of the normalized values over common indices.

Order: Drawer→Counter, OpenDrawer, Coffee, Toaster, Stove.

CPU checks (synthetic data, macOS):
- The σ formula reproduces the native-sampler values recorded in
  `COHORT-READOUT.json`.
- A uniform residual shift gives raw drift of −6, −40 and −540 at indices 0, 4
  and 9, and an identical −6 after normalization.
- Matched selection drops an index that one task lacks.

Budget: R3 took 4,285 GPU-s for 80 checkpoint loads. This adds up to 10 B16
forwards per load. Ceiling one GPU × 240 minutes (4 GPU-h), no automatic retry.

Precondition: the Q1 `training-rollouts/iteration-0001-*` actor batches must
still be on PFS. The production corpus reclaim only touched `production-r1`.
If they are gone, `prepare` fails before any GPU work. Restore those two files
per task from the Q1 corpus archive first.

## Readout

Primary: final (step-130) drift per task **at the same index**, and the mean
normalized drift over common indices, each with its across-sample spread.
Secondary: the curve over checkpoints and the per-dimension split at matched
indices.

- **Drawer within the other tasks' range at matched indices:** the confound
  explains the anomaly. Drop drift as a Drawer-specific explanation. The
  remaining question is whether its evaluation regression is real, which the
  checkpoint-selection sweep's 128-case TEST answers.
- **Drawer still clearly larger at every index (beyond the across-sample
  spread):** a real task-specific excess. Then compare at matched indices with
  the per-update KL (Drawer ~1.6× the improving tasks) and inspect late
  rollouts before any recipe or task change.

This remains a conditional-transition likelihood on policy-zero samples. It
is not a full-action marginal KL or a behavioural test.

## Independent closeout audit

The separately Git-deployed [FINAL-AUDIT.py](FINAL-AUDIT.py), source `d1850311`,
does not modify the active executable checkout. Its Linux CPU partial audit
checks all 480 index tensors for the completed Drawer, OpenDrawer and Coffee
tasks, selected fixture contents against the original Q1 actor records,
checkpoint marker/READY/CONSUMED joins, variance normalization and dimension
sums. The maximum reduction difference is 1.56e-15; this is not a whole-run
closeout. [Partial audit](PARTIAL-AUDIT-20261104-1400.json).

After the GPU clock exits successfully and staging has all 80 receipts, run
the following with the allocated Linux environment's Python. The destination
must be new; omit `--allow-incomplete` for the final audit.

```bash
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/robocasa-flow-sde-matched-denoise-drift-20261104-r1
TASK_META=$TASK_ROOT/sources/stove-drift-endpoint512-d13dc18d
"$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python" -B \
  "$TASK_RUN/FINAL-AUDIT-d1850311.py" --root "$TASK_RUN" \
  --source-script "$TASK_META/outputs/robocasa-flow-sde-matched-denoise-drift-20261104-r1/MATCHED-DRIFT.py" \
  --destination "$TASK_RUN/FINAL-AUDIT.json"
```

The audit also requires every temporary private model copy to have been
reclaimed by its own stager. Scheduler terminal/release accounting and the
scientific five-task comparison remain separate closeout steps.
