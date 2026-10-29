# Two-source-goal subset qualification

The existing assembler and bounded four-controller trainer already accept
multiple source indices. Use them for Spatial00 plus LIBERO10:00. Independent
validation collection for **both** source goals is required before assembly;
the LIBERO10 training history cannot serve as its own validation history.
This remains an explicitly admitted small pilot. Production all40 admission
and the full90/Plus/Pro target protocol are unchanged.

The qualification command adds only checks missing from that generic route:
four explicit source-index file hashes, the exact two-goal membership, four
independent learner seeds, the published500/256/20 recipe, executed
`sac_online` behavior after actual warmup, chronological changing teacher
clocks, and a checksum for each complete source recovery. The existing
`load_source`/`assemble` path checks exact motor/features/noise provenance,
whole chronological cases, local train states0..39 versus validation
states40..49, actual nominal-state hash separation, source-only data,
paired AD/DPT arrays and complete collection costs. Validation comes from the
40..49 bank; the receipt reports the actual states visited without implying
that all10 have been collected.

Run qualification and focused tests on an allocated Linux resource. Set
`TASK_L00_VALIDATION` only after that independent collector has completed.
The source paths below refer to complete original histories, not target probes.

```bash
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_SPATIAL_TRAIN=$TASK_ROOT/outputs/libero40-1m-sac-20261029-s00-train-r1/experiment/index.json
TASK_SPATIAL_VALIDATION=$TASK_ROOT/outputs/libero40-pi05-source-smoke-20261029-s00-validation-r1/experiment/index.json
TASK_L00_TRAIN=$TASK_ROOT/outputs/libero40-1m-sac-20261029-l00-train-r1/experiment/index.json
TASK_L00_VALIDATION=/absolute/path/to/completed/independent-l00-validation/experiment/index.json
TASK_ASSEMBLY=$TASK_ROOT/outputs/FRESH_TWO_GOAL_RUN/assembly
TASK_ST_HASH=$(sha256sum "$TASK_SPATIAL_TRAIN" | cut -d' ' -f1)
TASK_SV_HASH=$(sha256sum "$TASK_SPATIAL_VALIDATION" | cut -d' ' -f1)
TASK_LT_HASH=$(sha256sum "$TASK_L00_TRAIN" | cut -d' ' -f1)
TASK_LV_HASH=$(sha256sum "$TASK_L00_VALIDATION" | cut -d' ' -f1)
python -m fluffy_bassoon.meta_rl.vla_source40_two_goal_qualification \
  --qualification-subset \
  --train-index "$TASK_SPATIAL_TRAIN" "$TASK_L00_TRAIN" \
  --train-sha256 "$TASK_ST_HASH" "$TASK_LT_HASH" \
  --validation-index "$TASK_SPATIAL_VALIDATION" "$TASK_L00_VALIDATION" \
  --validation-sha256 "$TASK_SV_HASH" "$TASK_LV_HASH" \
  --output-dir "$TASK_ASSEMBLY"
```

Read the immutable `two-goal-qualification.json` before launching training.
It identifies all original actor checkpoints, complete learner recoveries,
actual learned-behavior counts and full source costs. No actor is called an
expert. This command executes no simulator actions and does not initialize
models. Its `assembly.json` is the standard existing assembly format.

Use the existing qualified OpenPI/cu128 student environment, actual allocation
GPU probe, exact clean source worktree and bounded subset trainer. No training
stack or architecture change is required:

```bash
python -m fluffy_bassoon.meta_rl.vla_source40_subset_training \
  --qualification-subset \
  --train-index "$TASK_SPATIAL_TRAIN" "$TASK_L00_TRAIN" \
  --validation-index "$TASK_SPATIAL_VALIDATION" "$TASK_L00_VALIDATION" \
  --assembly "$TASK_ASSEMBLY/assembly.json" \
  --assembly-file-sha256 "$(sha256sum "$TASK_ASSEMBLY/assembly.json" | cut -d' ' -f1)" \
  --output "$TASK_ROOT/outputs/FRESH_TWO_GOAL_RUN/experiment" \
  --source-commit EXACT_REVIEWED_COMMIT --expected-hostname "$(hostname)" \
  --scales 25 100 500 --seed 0 \
  --batch-size 8 --width 128 --layers 2 --heads 4 \
  --context-length 300 --max-window 128 \
  --max-controller-seconds 600 --max-total-seconds 2700
```

The trainer retains one continuous500-update optimizer trajectory for each
AD/DPT full/static controller and immutable25/100/500 checkpoints, including
optimizer/sampler/global RNG recovery. The existing data-scale helper can
still produce per-goal whole-case diagnostics; it must not concatenate
overlapping prefixes or remove full final-teacher collection costs. This
two-goal run uses each latest complete source history once.

The current trainer's singular `frozen_source_sac` compatibility field selects
the **first training index**, Spatial00 in this command. It is the original
Spatial00 actor and must not be described as a jointly trained two-goal DSRL
baseline. The qualification receipt preserves both independent source actors.
Target robot results, with real support/query budgets and fixed parameters,
are required to assess whether adding the second source goal helps.
