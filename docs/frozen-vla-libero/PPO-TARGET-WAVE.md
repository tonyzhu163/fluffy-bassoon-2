# Frozen final-source PPO target wave

This future wave deploys the two immutable final all40 source PPO policies
**after successful terminal source-local validation**. Qualification means
all32 K4 trials completed, weights unchanged, likelihood replay passed and
full recovery/exports remain hash-bound. It does not impose a success threshold
or select a different checkpoint using validation or target outcomes.

PhaseA uses all90 LIBERO90 tasks, nominal state0 and scene seed123. Native,
full-history source PPO and separately trained episode-reset PPO each execute
one K0 query. PhaseB uses the predeclared nine IDs0,10,…80; full PPO, a
same-weight episode-reset intervention, and separately trained reset PPO each
receive one actual support episode and a separate fresh query. Parameters
remain fixed and deployment performs zero optimizer updates. Native K0 is
contemporaneous. Same-weight reset K0 explicitly references the full policy's
K0 query; separately trained reset is its own method.

|Phase |Groups |Physical episodes |Worst actions including settling |
| --- | ---: | ---: | ---: |
|A, three K0 methods ×90 |270 |270 |110,700 |
|B, three K1 methods ×nine |27 |54 |22,140 |
|Total |297 |324 |132,840 |

Each episode has H400 and10 settling actions. Two workers each receive70,000
actions and3,600 cumulative GPU-seconds:140k/two hours jointly. Funding and
reservations remain root's future decision from unused original validation
holds within the approved32-hour campaign. No allocation/reservation is made
by the planner or runner, and previous source/noise/student ledgers stay fixed.
The nine panel IDs appear first, followed by remaining numeric IDs; then
`cases[i::2]` gives worker0 five panel cases and worker1 four. Their nominal
loads are165/159 episodes and67,650/65,190 actions. Simple numeric parity
would put all nine panel IDs on one worker and exceed its70k cap.

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_ppo_wave_plan \
  --worktree EXACT_SOURCE_WORKTREE --source-commit EXACT_SOURCE \
  --validation-root /mowenjie/fluffy-bassoon-2/outputs/libero40-ppo-source-validation-20261030-r1 \
  --validation-index-sha256 VERIFIED_TERMINAL_INDEX_FILE_SHA \
  --output FRESH_PLAN_JSON
bash scripts/diagnostics/vla_source40_ppo_wave.sh \
  EXACT_SOURCE WORKER_RUN WORKER_INDEX PLAN_JSON TARGET_LEDGER_JSON \
  /mowenjie/fluffy-bassoon-2/outputs/libero40-source-1m-20261029-r1/schedule.json 3540
```

The separate read-only target ledger format is
`source40_frozen_ppo_target_wave_budget_v1`. It binds `ppo_wave_sha256`,
`maximum_total_actions:140000`, `maximum_gpu_seconds:7200`,
`source_campaign_immutable:true`, and a human-readable `scope_amendment`.
Exactly two `reservations`, keyed by run ID, contain `run_id`, `worker`0/1,
exact `source_commit`, `ppo_wave_sha256`, `maximum_total_actions:70000`,
`worker_seconds:3600`, `purpose:target_evaluation`,
`source_pretraining_data:false`, and `fresh_ppo_training_data:false`.

Each worker copies the **same validated export bytes** to its own output,
retaining the original source-validation model SHA. Existing `load_models`
re-admits full terminal R3 recoveries, both migrations, original R1 reservations,
source optimizer/cursor state and common4104-feature/motor contract, then
verifies the existing copied exports. No new export is written to the original
source-validation run. Both policies are loaded once per worker. Same-weight
reset shares the full-policy object; trained reset uses the separate policy.

Every comparator for a case shares one FrozenOpenPI object and legacy
`jax.random.PRNGKey(seed)` motor keys. PPO sampling uses its qualified global
Torch CPU/CUDA episode RNG with `generator=None`. Complete support context,
global/sampler/motor RNG, episode cursors, actual RGB/wrist/proprioception,
noises, VLA actions, rewards/boundaries and before-RPC journals are retained.
Actual initial arrays and paired query seeds must match. Motor parameter-state
hashes are recorded before/after. Source costs remain explicit and counted
once per independently trained model; the same-weight reset shares full cost.

The3,540-second slice plus60-second scheduler allowance fit the original
hour. Setup, tests, model loading, inference, teardown and failed attempts
consume the same cumulative clock. An explicit graceful deadline precedes the
outer timeout by at least60seconds. A deadline writes an incomplete index,
retains complete recoveries/raw partial charges, and never fabricates a query.
Exact-source retries use the same output with `--resume` and an affordable
remaining slice. Fresh-process continuations lose strict common-runtime status.

All90 reporting covers one initialization and seed per task; the fixed9 panel
reports one-shot support/query behavior. This wave adds no Plus/Pro repeats,
new source data, semantic goal exclusions or target checkpoint selection.
