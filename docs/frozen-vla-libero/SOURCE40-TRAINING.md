# Source40 history assembly and matched controller training

`fluffy_bassoon.meta_rl.vla_source40_training` connects the persistent source
collector to the existing causal history trainer. It consumes only completed,
qualified OpenPI `pi05_libero`40-task source histories with the original motor
receipt and prefix-v2 feature contract. It rejects target-suite indices and
RLinf130 motor/features.

Assembly chooses exactly one latest committed paired AD/DPT prefix from each
source task and partition. Earlier prefixes contain overlapping interactions
and are not concatenated. The selected prefix must cover the entire completed
history. Actor checkpoint and dataset hashes, actual complete episode receipts,
source contract, dimensions and shared AD/DPT context are checked. Source
teacher candidates are not presumed expert.

Both partitions cover the same40 source tasks. Validation uses independent
source learners and whole cases with disjoint `(task, initial-state ID)` and
actual nominal-state hashes. Case and history identities are disjoint too.
Default assembly requires all40 tasks and actual completion of every requested
case. `--allow-small-pilot` explicitly permits a smaller task set and bounded
chunk-target qualification histories; this status remains in every dataset and
must be explicitly accepted again by the training command.

```bash
python -m fluffy_bassoon.meta_rl.vla_source40_training assemble \
  --train-index <train-task-00>/index.json <train-task-01>/index.json <...> \
  --validation-index <validation-task-00>/index.json <validation-task-01>/index.json <...> \
  --output-dir <fresh-assembly-directory>
```

Run training inside an allocated Linux job, with a GPU probe and exact clean
source commit checked by the launcher. The command below trains AD/full,
AD/static, DPT/full and DPT/static sequentially for each explicitly supplied
seed. Each receives identical architecture, sampling, optimizer and update
budgets. Normalization is fit only to training observations by the generic
trainer. Full histories retain their actual chronological episode boundaries;
source interaction lengths do not determine the target shot budget.

```bash
python -m fluffy_bassoon.meta_rl.vla_source40_training train \
  --assembly <assembly-directory>/assembly.json \
  --output-dir <fresh-matrix-directory> --seeds 0 1 2 --updates 1000 \
  --source-commit <exact-reviewed-commit> --expected-hostname <allocated-host> \
  --max-elapsed-seconds 1800
```

The elapsed ceiling is cumulative per controller. The full matrix ceiling is
four times this value per seed. These example budgets are not an authorization
for a larger run. Choose explicit budgets in the actual run contract.

Each controller preserves `latest.pt` with policy, optimizer, sampler and
Python/NumPy/Torch/CUDA RNG state. `candidate.pt` is selected by validation NLL;
`final.pt` is exported separately at the fixed requested final update and is
the primary candidate for a matched comparison. A time-limited controller that
has not reached the requested update count keeps its recovery and stops the
matrix without promoting a final export. `--resume` continues the original
matrix identity and contract; completed final exports are verified and reused.
The matrix index records per-controller updates, time, final and recovery
hashes. Source interaction costs are recorded once per partition in
`assembly.json`, separate from optimizer reuse across four controllers.

Imitation NLL and saved candidates establish neither robotic success nor
in-context adaptation. The next scientific comparison uses separately counted
support/query episodes on the full90/Plus/Pro target suites, fixed deployment
weights, and matched memory controls.
