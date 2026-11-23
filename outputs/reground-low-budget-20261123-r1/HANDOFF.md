# Baige launcher and evaluation panels

**Executable pin** `200647539605980ceb81bb10d32cc003a6f736dd`. The detached checkout, sealed inputs for all 15 cells, and original frozen runtime are on shared Baige storage. CPU receipts are separate from production. This work used zero GPU-hours. The parent owns scheduler submissions and the cumulative allocation ledger. [Feasibility and access accounting](RESULT.md) · [Launcher](launch.sh)

## First panel

Select one exclusive RTX 4090 or RTX PRO 6000D in Baige Distributed Training. The following is the job command for seed 0. To cover the opposite small-budget training source, use seed 3 and a fresh attempt ID in a second job. Each job fits three target model pairs and evaluates 96 episodes. Each sees a union of 1,000 archived target transitions across the three fits. Each budget scores the same 32 starts, four per task. Two seeds give 192 low-budget assignments on 32 distinct task–start pairs and 2,000 distinct calibration transitions.

```bash
pin=200647539605980ceb81bb10d32cc003a6f736dd
run=outputs/reground-low-budget-20261123-r1
work=/mowenjie/fluffy-bassoon-2/$run
entry="$work/src-$pin/$run/launch.sh"
bash "$entry" "$pin" panel-s0-01 3600 fit-panel32 0 15 100 1000
```

The one-hour wall is a reservation ceiling per card. The native 10k controller took 223 and 300 summed rollout seconds on these 32 starts at seeds 0 and 3. Multiplying by three budgets gives **0.436 GPU-hours** across the two panels at the archived rollout durations. Scaling each original batch to its maximum horizon at its observed rate gives **0.806 GPU-hours** across the two panels. Both estimates exclude weight loading, fitting, cache verification, environment initialization and scheduler overhead. The example reserves at most **2 launcher GPU-hours** for the two one-card jobs. The finalizer records elapsed launcher time. Preserve the scheduler receipt for total billing. [Batch timing and exact assignments](evidence/reference-panels.json)

| Seed | Source of gradient updates at 15 and 100 | Archived 10k success / unsafe | Archived Base success / unsafe |
| --- | --- | --- | --- |
| 0 | Policy-driven | 32/32 · 15/32 | 0/32 · 22/32 |
| 3 | Babbling | 26/32 · 15/32 | 0/32 · 22/32 |

The selection rule uses the first four-episode batch of the first native shard beginning with four episodes from each task. It depends only on the released grid order. Those are shards 0, 3, 5, 7, 9, 11, 14 and 16. The controls were counted after defining the rule. The two seeds cover the two observed split orientations. They also have different collection and model seeds, so comparing them cannot isolate a causal effect of training source.

## Expansion, resumption and scoring

- `fit` publishes complete native target models individually. `panel32` evaluates existing GPU fits. `fit-panel32` fits and evaluates. `full` evaluates all 280 native assignments of one specified seed for each supplied budget. All phases use the same six arguments shown above, with any nonempty selection from 15, 100 and 1000. Run a fit phase before a stand-alone evaluation.
- Use fresh attempt IDs with the same pin. Preserve the shared production directory. Within a cell, each completed four-environment batch stores its rows, traces and post-batch CEM generator state. Full-grid expansion reuses all 32 panel episodes and adds 248 per seed/budget. Each of 17 full shards contains 16 episodes and the last contains eight. Restarting after a partial fit reuses completed models and refits the unfinished model from the saved original initialization. The release has no mid-optimizer checkpoint.
- Example for expanding seed 0 after its panel: `bash "$entry" "$pin" full-s0-01 10800 full 0 15 100 1000`. This reserves at most three launcher GPU-hours. A completed full grid is 4,200 low-budget assignments, 3 × 5 × 8 × 35, on 280 distinct starts. At original rollout lengths the three complete grids extrapolate to 6.29 pure rollout GPU-hours, or 11.46 at the per-batch full-horizon scenario. Historical 10k evaluation actually charged 3.348 launcher GPU-hours for one 1,400-assignment grid. Budget future launches against actual wall limits and elapsed charges.
- Check `$work/attempts/<id>/{attempt.json,gpu.json,run.log,exit.json}` and `$work/production/seed<S>/b<B>/{models,eval}`. Under `eval/<contract>/`, `panel32/` and `full/` contain verified native shards and share `batches/`. The first publication and subsequent 20-minute checks belong to the parent. For a killed container that leaves a cell's `active/`, verify that the recorded host, PID, attempt and scheduler job are terminal, then archive that stale directory before starting a fresh attempt. A normal failure or SIGTERM releases the claim. Never run two writers in the same cell.
- For post-rollout comparison on Linux CPU, source the existing runtime. Use `taskset -c 0,1 "$REGROUND_RUNTIME_ROOT/env/bin/python" -B "$work/src-$pin/$run/panel.py" --seed 0 --budget 15 --panel panel32 > "$work/panel-s0-b15.json"`. The reader validates low-budget shards, pairs every key and noise seed against archived Base and 10k ReGround, and reports successes, unsafe episodes, native steps and opening mismatches. It compares the initial robot state and up to the first ten proposed actions. It opens the archived test traces only after scoring. GPU simulator fixture histories and later policy proposals can differ even with fixed starts. Inspect mismatch sensitivity on paired outcomes. [Previous opening audit](../reground-rotation-reference-20261123-r1/prefix-audit/RESULT.md)
- Report absolute success and unsafe counts, paired gains/losses by task and seed and the corresponding uncertainty. For expansion, resample starts within task and carry all seeds and budgets for that start together. Four starts per task on the small panel give limited precision. Use the original release IQM and its interval on the complete five-seed grid. The frozen Base and 10k controls already cover all assignments.

## Reproducing the CPU checks

The target tape is materialized already. To audit it again, use the release environment and the pinned scripts on the two CPU cores. The prepare command rereads original archives, verifies their published hashes and asserts agreement with the existing immutable tapes. Production fitting and evaluation have separate entry points.

```bash
export REGROUND_RUNTIME_ROOT=/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1
source "$REGROUND_RUNTIME_ROOT/runtime/activate.sh"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py="$REGROUND_RUNTIME_ROOT/env/bin/python"
scripts="$work/src-$pin/$run"
taskset -c 0,1 "$py" -Bu "$scripts/prepare.py"
taskset -c 0,1 "$py" -Bu "$scripts/runner.py" cpu-qualify 0 15 100 1000
taskset -c 0,1 "$py" -Bu "$scripts/check_transport.py"
```

The final check compares a newly fitted 15-transition model pair with the literal CPU fit preserved at `$work/literal-seed0-b15/models/`. That independent fit set `babble.RUNS` to the original `reference-native/runs`, set `cache.RUNS` to the private literal run directory, and called the released `cache.load_or_fit(spec, "cpu")` for forward and IDM at budget 15 and mixture 0.6. Before constructing each model it restored the corresponding seed-0 10k target model's `fit-starts/<spec-hash>.pt` CPU RNG state. Its metrics include the five disjoint VLA holdouts. [Literal fit log](evidence/literal-15-seed0.log) · [Independent final-pin readback](evidence/transport-cpu.json)
