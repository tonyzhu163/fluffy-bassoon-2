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

Training covers all40 source tasks. The original complete-case route also uses
all40 tasks for validation. The action-budget campaign instead uses a fixed
eight-task source-local validation bank: task IDs00 and01 in each of Spatial,
Object, Goal and LIBERO10. This reduces independent validation-teacher warmup
cost while retaining every training task and the full90/Plus/Pro target suites.
These validation tasks are chosen by index, with no semantic filtering.

Validation uses independent source learners and whole cases with disjoint
`(task, initial-state ID)` and actual nominal-state hashes. Case and history
identities are disjoint too. A fixed-eight validation bank requires an explicit
`--campaign` file and its byte-level `--campaign-sha256`. The file declares
`format=libero40_source_1m_campaign_v1`, `maximum_total_actions=1000000`, all40
`source_training_task_keys`, the fixed8 `source_validation_task_keys` and a
`campaign_sha256` computed from the remaining content. Assembly records both
hashes and the exact per-partition task coverage.

For `complete_cases`, every requested case must complete. For `physical_budget`,
the selected final prefix must contain all actual complete cases, its physical
ledger must include known and uncertain primitive/settling actions, and it must
stop when another worst-case complete case cannot fit or the candidate bank is
exhausted. Unused candidate seeds are not missing required interactions. The
new campaign has a shared1M-action ceiling across SAC, fresh PPO, validation,
settling and retries; assembly admission does not allocate extra interactions.
Raw-only histories that never clear warmup cannot enter AD/DPT assembly.

`--allow-small-pilot` explicitly permits a smaller task set and bounded
chunk-target qualification histories; this status remains in every dataset
and must be explicitly accepted again by the training command.

```bash
python -m fluffy_bassoon.meta_rl.vla_source40_training assemble \
  --train-index <train-task-00>/index.json <train-task-01>/index.json <...> \
  --validation-index <validation-task-00>/index.json <validation-task-01>/index.json <...> \
  --output-dir <fresh-assembly-directory>
```

For the action-budget campaign append
`--campaign <campaign.json> --campaign-sha256 <exact-file-SHA256>` and supply the
eight declared validation indices. Earlier all40/all40 assembled datasets and
explicit technical-pilot datasets remain readable under their original roles.

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
