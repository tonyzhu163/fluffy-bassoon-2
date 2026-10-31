# Paired LIBERO90 noise-interface controls

The completed source40 screen needs an interface diagnosis before interpreting
learned steering quality. This diagnostic evaluates all90 LIBERO90 task IDs,
nominal state0 and scene seed123. It does not select checkpoints, learn on
targets or change the source campaign. Each case runs four K0 query episodes
in one frozen motor process, with the same actual initialization and query RNG:

| Method | Initial flow noise |
| --- | --- |
|Native |Independent unbounded standard normal over10×32 |
|Zero |Zero32D coordinate repeated over H10 |
|Tied Gaussian |Unbounded standard normal32D coordinate repeated over H10 |
|Tied Gaussian clipped |The same32D Gaussian draw, clipped to[-1,1], repeated over H10 |

The unbounded tied control matches the actual source warmup:
`OnlineSourceSimulator.choose` calls `sampler.standard_normal(32).astype(float32)`
while source updates are zero, and the collector calls `run_episode` with
`bounded=False`. There is no noise rescaling. After source optimization starts,
SAC coordinates are clipped[-1,1], as are deployed learned student coordinates.
The separate clipped control factors the bounding effect from temporal tying.
Every saved noise array is checked against an exact reconstruction of its
recorded episode sampler, including the clipping branch; plausible statistics
alone do not establish the correct distribution.

360 physical queries cost at most147,600 executed actions including400-step
cutoffs and10 settling actions per episode. Two disjoint case-strided workers
own `cases[i::2]`, each with76,000 actions and3,600 cumulative GPU seconds.
The joint cap is152,000 actions/two GPU-hours, drawn only from root's explicitly
released target allowance inside the approved32-hour campaign. Qualification
uses the first planned reset with no extra physical reset or episode.

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_noise_plan \
  --worktree EXACT_WORKER_WORKTREE --source-commit EXACT_SOURCE \
  --output FRESH_NOISE_PLAN_JSON
bash scripts/diagnostics/vla_source40_noise_controls.sh \
  EXACT_SOURCE WORKER_RUN_ID WORKER_INDEX NOISE_PLAN_JSON TARGET_LEDGER_JSON 3540
```

The read-only ledger format is `libero90_noise_controls_budget_v1`, with top
`noise_plan_sha256`, `maximum_total_actions:152000`,
`maximum_gpu_seconds:7200`, `source_campaign_immutable:true`, and an explicit
human-readable `scope_amendment`. Its two `reservations` are keyed by run ID.
Each row contains `run_id`, `worker`0/1, exact `source_commit`,
`noise_plan_sha256`, `maximum_total_actions:76000`, `worker_seconds:3600`,
`purpose:target_evaluation`, `source_pretraining_data:false`, and
`fresh_ppo_training_data:false`. The runner never creates a reservation.

The initial3,540-second slice and60-second scheduler allowance fit the original
hour. Setup, probe/tests, model loading, inference, shutdown and failed attempts
consume the same cumulative worker clock. An exact-source retry uses the same
run/output with `--resume` and only an affordable remaining slice. Unclosed
attempts remain fully charged. Full raw RGB/wrist/proprioception, proposed and
executed noise, decoded actions, rewards/boundaries, sampler/global/motor RNG,
before-RPC intents and complete-episode cursors remain recoverable. Nominal
state, actual initial frames and query seed are checked across controls.
A fresh-process continuation is explicitly non-primary for strict common-runtime
claims. Frozen motor parameter state is hashed before and after execution.

Each invocation also records an explicit absolute graceful stop deadline. It is
validated against the active original cumulative slice,125seconds before the
slice end and at least60seconds before the shell's outer timeout. That margin
is for closing the owned simulator, parameter-state checks, writing an incomplete
final index and recording worker costs. A deadline reached during an episode
preserves the complete-episode cursor and all raw partial RPC costs; it does not
count a partial query as complete. Other runtime failures still propagate.
The deadline is invocation metadata and never extends a resume's original cap.

Read-only terminal aggregation is available as `vla_source40_noise_report`.
It refuses live launcher locks, nonzero exits and partial indices. It verifies
all360 group/trace/context/recovery hashes; reconstructs every actual noise
array from its recorded RNG; checks raw reset/trace initial RGB/wrist/proprio
and nominal **and post-settle** physics parity; verifies one process per worker
and unchanged/common motor parameter SHA; and reconciles actual journals and
cumulative clocks. It reports per-task outcomes and all six paired2×2 tables,
including discordant task IDs. Contemporaneous native is primary; the earlier
31/90 firstwave native is a separately labeled reference. No model/framework
is imported or constructed. Only saved arrays/metadata are read. Nothing is
written to the workers, and no waiting/polling occurs.

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_noise_report \
  --plan /mowenjie/fluffy-bassoon-2/outputs/libero40-noise-controls-plan-20261030-r1/plan.json \
  --plan-sha256 9f5f86d6bffec1922952cf0da8e3dc657beccd361db2b0fda132837c5e9e0c7d \
  --ledger /mowenjie/fluffy-bassoon-2/outputs/libero40-noise-controls-plan-20261030-r1/target-ledger.json \
  --ledger-sha256 0de51e4e6fec4c4929cc801f64ea043bc530f115fc68ff2da989d0ed2571b4c2 \
  --worker-root /mowenjie/fluffy-bassoon-2/outputs/libero40-noise-controls-20261030-w00-r1 \
  --worker-root /mowenjie/fluffy-bassoon-2/outputs/libero40-noise-controls-20261030-w01-r1 \
  --output /absolute/new-report-outside-worker-roots.json
```

The output argument is optional; stdout contains the compact summary. When
provided, it creates a fresh report artifact outside both immutable worker
roots and refuses an existing destination. Run saved-data verification once
both allocations are terminal; this command submits no allocation or tests.
