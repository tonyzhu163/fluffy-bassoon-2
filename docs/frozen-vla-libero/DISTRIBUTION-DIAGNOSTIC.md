# Fixed9 frozen mean/dispersion diagnostic

Preparation only. Root must reconcile and explicitly fund the proposed27k
simulator-action/one-GPU-hour hold from unused original campaign time before
launch. Source runs, checkpoints, the source campaign and previous ledgers
remain immutable. No target data enters pretraining or checkpoint selection.

Use the existing nine LIBERO90 cases0,10,…80, nominal state0/scene seed123,
K0 only. Seven methods share the same motor process per case:

|Method |Applied mean |Applied std before clipping |
| --- | --- | --- |
|Native |0, independent10×32 flow noise |1 |
|AD full original |Original source500 AD mean |Original learned global channel std |
|DPT full original |Original source500 DPT mean |Original learned global channel std |
|AD zero mean, same std |0 |Same AD learned std |
|DPT zero mean, same std |0 |Same DPT learned std |
|AD original mean, std1 |Original AD mean |1 |
|DPT original mean, std1 |Original DPT mean |1 |

The six steering methods retain actual observation/feedback/cache evolution.
They all clip the proposed32D coordinate to[−1,1] and repeat it over H10.
The distribution operation changes no base parameter or buffer bytes.
This is a diagnostic of learned conditional mean versus dispersion, not a new
algorithm, checkpoint selection or meta-learning performance claim.

`DistributionController` calls `base.step(deterministic=True)` for its actual
mean/cache, then makes **one** random draw with the original shape, dtype,
device and qualified per-episode Torch Generator. It applies the declared mean
and std operations to that draw. Allocated tests compare the original adapter
bitwise against current stochastic `FrozenHistoryController` actions, feedback,
cache evolution and RNG state. A second check uses both actual loaded500-update
policies on the allocated GPU inside the first planned reset; its fixed
synthetic feedback is qualification data only and is never executed or learned
from. Qualification adds zero simulator resets or actions.

Each learned chunk records base mean, learned std, applied mean/std, epsilon,
unbounded proposal, expected clipped noise and actual executed coordinate and
feedback. The mean/std/epsilon/proposal are also journaled **before RPC**.
Native mean0/std1 and no clipping are explicit in its contract; actual epsilon
and proposed flow noise are the saved `native_initial_noise` array. Full raw
RGB/wrist/proprioception, decoded actions, rewards/boundaries, global/controller/
motor RNG and complete context/cursors remain preserved by the shared driver.
Actual initial arrays and query seed must match within the common process.
Motor and controller parameter-state hashes are compared before and after.

63 physical queries ×(400 cutoff+10 settling) cost25,830 nominal actions;
27,000 is the joint ceiling. One original3,600-second cumulative worker hold
includes setup, focused allocated tests, model loading, qualification, execution,
failed attempts and teardown. The initial3,540-second slice plus60-second
scheduler allowance fit that hold. An explicit graceful deadline precedes the
outer timeout, and partial queries retain their raw charges/recovery without
becoming successes. Exact-source retries use the original output/remaining
clock and lose strict common-runtime status when the process changes.

```sh
python -m fluffy_bassoon.meta_rl.vla_distribution_plan \
  --worktree EXACT_WORKTREE --source-commit EXACT_SOURCE \
  --matrix ORIGINAL_FINAL500_MATRIX --matrix-sha256 MATRIX_FILE_SHA \
  --assembly ORIGINAL_ALL40_ASSEMBLY --assembly-sha256 ASSEMBLY_FILE_SHA \
  --output FRESH_DIAGNOSTIC_PLAN
bash scripts/diagnostics/vla_distribution_diagnostic.sh \
  EXACT_SOURCE RUN_ID DIAGNOSTIC_PLAN TARGET_LEDGER 3540
```

The ledger format is `source40_distribution_diagnostic_budget_v1`, with
`diagnostic_sha256`, `maximum_total_actions:27000`,
`maximum_gpu_seconds:3600`, `source_campaign_immutable:true` and a human-readable
`scope_amendment`. Its single `reservations` row, keyed by run ID, binds
`run_id`, exact `source_commit`, `diagnostic_sha256`,
`maximum_total_actions:27000`, `worker_seconds:3600`,
`purpose:target_evaluation`, `source_pretraining_data:false` and
`fresh_ppo_training_data:false`. Neither planner nor runner creates it.
The actual original all40 final500 matrix, all four source controller/recovery
hashes and the assembly are verified; only full AD/DPT policies are intervened
on. Separately trained static or reset weights are never substituted.
