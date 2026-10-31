# Approved PPO and frozen-validation execution

The human approved32 additional LIBERO GPU-hours on2026-10-30, within the
existing1M simulator-action cap: eight SAC workers ×2h, full PPO up to6h,
separately trained episode-reset PPO up to6h, and frozen validation up to4h.
RoboTwin has separate ownership and is outside this approval.

The canonical Baige PPO ledger was read before preparing these commands. It
contains19,500 qualification actions and70,000 target-evaluation actions;
no full/reset172k or56k frozen-validation reservation existed then. Adding
those approved entries totals489,500 of the490k pool, leaving500 actions.
The old3944-prepared worktrees are preserved and are not these new runs.

Fresh pinned full/reset workers, inside separate approved6h allocations:

```bash
bash scripts/diagnostics/vla_source40_multitask_ppo_collect.sh EXACT_COMMIT \
  libero40-ppo-full-20261030-r1 172000 100000 full 21480
bash scripts/diagnostics/vla_source40_multitask_ppo_collect.sh EXACT_COMMIT \
  libero40-ppo-reset-20261030-r1 172000 100000 episode 21480
```

Each arm has one policy/optimizer shared across all40 tasks in outcome-independent
round robin. K4 retries use one fixed physical case. Source states0–39 are
sampled with replacement; actual task/state/hash exposure is reported. PPO uses
fresh on-policy data, behavior-policy likelihood replay, width128/two layers/
four heads/context300, LR1e-4/two epochs and one whole K4 trial per update.
This is an explicit pilot batch rather than a claim to reproduce VLRLA's batch.

The cumulative worker ceiling is21,600seconds per arm. Every attempt reserves
its slice plus60seconds for startup/shutdown uncertainty; completed attempts
charge actual elapsed time plus that allowance, and unclosed attempts retain
their full reservation. The driver/resource ceiling stays fixed while retry
slice lengths may shrink. Resume uses the same source/run/action cap/update
ceiling/architecture, with `--resume`. No later attempt receives another172k
actions or another6h. Scheduler start/end receipts must also be reconciled by
the launch owner; the worker ledger does not replace scheduler billing evidence.

Raw rollouts, pre-tanh samples, bounded actions, old log-probabilities/values,
reset masks and recurrent initial states are saved. Atomic complete-trial
recovery includes policy, optimizer, sampler, Python/NumPy/Torch/CUDA RNG,
motor RNG and task cursor. Partial attempts remain separately charged and
completed prior artifacts stay recoverable.

After both source arms are terminal and models are ready, frozen validation
runs in one common motor process, avoiding a separate motor load per method:

```bash
bash scripts/diagnostics/vla_source40_frozen_validation.sh EXACT_COMMIT \
  libero40-ppo-validation-20261030-r1 \
  libero40-ppo-full-20261030-r1 libero40-ppo-reset-20261030-r1 14280
```

The launcher snapshots hashes of both final `learner/latest.pt` recoveries
while holding each source launcher's lock. It checks their6h time ledgers and
refuses live writers. The allocated runner validates complete-trial boundaries,
all40 actual task exposures, expected172k source/motor/feature/architecture/
optimizer contracts and nonzero optimizer advancement. It exports immutable
frozen copies directly from those recoveries. This avoids accidentally using
a stale `candidate.pt` after an interrupted source invocation. Missing all40
coverage or incomplete policy/RNG state is an explicit readiness failure.

Validation uses the fixed eight source goals, states40 and41, two K4 cases for
each model/goal and3,500 actions per slot. The same cases and per-episode seeds
are paired across full/reset. Each slot reserves3,280 worst-case trial actions
plus qualification;210 actions remain for retry overhead. The joint cap is
56,000 and the cumulative validation time ledger is14,400seconds, including
setup/shutdown/retries. It creates no optimizer and performs zero updates.

Complete-trial validation recovery binds immutable model hashes, the source
case plan, slot/trial cursor, global RNG and motor RNG. Raw complete episodes
and partial RPC inputs/outcomes remain available. Exact-source resumptions
retain the same model/input hashes,4h ledger and56k action cap, and may use a
shorter scheduler slice by appending `--resume`. If time or action reserve is
insufficient, report incomplete slots; do not silently enlarge either budget.

These are seen source goals on disjoint physical cases. Their validation
outcomes do not establish LIBERO90/Plus/Pro transfer or novel-goal adaptation.
