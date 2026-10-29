# Reviewable collection expansion

Status: prepared, explicit32-GPU-hour approval requested in this chat. The1M LIBERO simulator
action cap is already approved; this document does not enlarge it. RoboTwin
has its own owner and action/compute accounting. No target evaluation is
included in this source collection request.

Reviewed source: `3944ff79cc25b314dfffdbb7ce2fbf4d48d6fccf`, incorporating
latest origin/main `acca706`. Allocated Linux preflight221passed/2JAX
probe-environment skips in `job-y48o6pc3928k`. Its separate4,000-action
width128 qualification exited0:8 actual episodes over two tasks,80 chunks,
two PPO iterations/four optimizer steps,744 primitives+90 settling=834actions.
Behavior likelihood replay error0, finite losses, full recovery saved.

## Requested compute ceiling:32 additional GPU-hours

| Work | Initial allocation shape | Maximum GPU-hours |
| --- | --- | ---: |
|48 SAC source slots, one already completed | Eight independent1GPU workers, up to2h each;6 slots/worker |16 |
|Shared full-memory PPO,197k actions | One1GPU worker, up to6h cumulative |6 |
|Separately trained episode-reset PPO,197k actions | One1GPU worker, up to6h cumulative |6 |
|Frozen source validation of both PPO models,8 goals each | Up to4h cumulative; zero optimizer updates |4 |
|Total new allocation ceiling | Existing bounded qualifications excluded and recorded separately |32 |

Each allocation requests16CPU,96GiB RAM and8GiB shared memory per GPU.
Independent workers may overlap within available scheduler capacity. Each
SAC child has a40-minute ceiling inside its worker. The requested2h parent
allocation can leave later slots incomplete; do not imply six are guaranteed
to finish. Use exact saved source/recovery to resume within the same32h pool.
If that pool cannot finish the declared collection, report the actual
remaining streams/actions and ask before adding GPU-hours. Runtime safety
ceilings of4h/SAC worker and24h/PPO stream are protective code limits, not
authority to spend those larger amounts.

Measured SAC stream:7,892actions in723.91worker seconds. A simple extrapolation
to476k new reserved SAC actions is12.13GPUh; one stream does not establish
other-task throughput. The smaller shared PPO check measured1,293 training
actions in80.58update-window seconds, about16.05actions/s; initial motor
qualification is additional. Width128 measured824 training/settling actions
in48.14 update-window seconds, about17.12actions/s; approximately3.25GPUh
per200k before initial setup and qualification. Two easy source tasks cannot
establish all-task throughput. These are estimates, not
completion guarantees. Global action caps govern even if compute remains.

Storage reservation: up to421GB for the selected48 SAC streams,128GiB for
the two PPO streams plus64GiB for frozen validation/receipts. This is a
conservative approximately630GB allowance, not observed data size. Available
project storage at23:44Shanghai was1.50TB. Current measured capped SAC output
was588MiB and the smaller shared PPO qualification34MiB. Keep original
raw images and complete recoveries; do not delete active artifacts to fit.

## Fixed action reservations

The [campaign](campaign.json) and [plan](PLAN.md) retain all40 source training
goals and8 fixed independent source-validation goals. SAC380k train+96k
validation+12,944 earlier qualification actions=488,944; fresh-PPO pool490k;
unassigned general reserve21,056 =>1M.

The490k PPO pool is divided before full launches:

| Use | Actions reserved |
| --- | ---: |
|Original single/shared qualification reservations |15,500 |
|New width128 qualification |4,000 |
|Full-memory training |197,000 |
|Separately trained episode-reset control |197,000 |
|Frozen source validation,8×2×4,000 |64,000 |
|Targeted controller evaluation escrow, no target data in source training |12,000 |
|Remaining PPO contingency |500 |

Original qualification reservations remain retained even when actual cost
was lower. Settling, qualifications, failed/recollected attempts and unknown
RPC intents all count. A run starts another case/trial only if its entire
worst-case action reserve fits; final unused remainders are expected. No
extra source data is generated merely to hit exactly1M.

## Executable immutable workers

Preparation creates clean detached worktrees and writes receipts, without
simulator/model computation. Eight shard preparations cover all48 slots;
Spatial00 retains its existing8ada0c0 run/source and is skipped as completed.

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_campaign_worker prepare \
  --source-commit 3944ff79cc25b314dfffdbb7ce2fbf4d48d6fccf \
  --worker <0-through-7> --workers 8 --output <fresh-preparation-receipt>
```

Inside each approved allocation, choose an unused worker-attempt receipt:

```sh
bash scripts/diagnostics/vla_source40_campaign_worker.sh \
  3944ff79cc25b314dfffdbb7ce2fbf4d48d6fccf <0-through-7> <fresh-worker-receipt> 7200
```

The full/reset PPO runs use separate fresh pinned worktrees and equal caps:

```sh
bash scripts/diagnostics/vla_source40_multitask_ppo_collect.sh \
  3944ff79cc25b314dfffdbb7ce2fbf4d48d6fccf libero40-pi05-ppo-full-20261029-r1 197000 100000 full 3600
bash scripts/diagnostics/vla_source40_multitask_ppo_collect.sh \
  3944ff79cc25b314dfffdbb7ce2fbf4d48d6fccf libero40-pi05-ppo-reset-20261029-r1 197000 100000 episode 3600
```

Subsequent equal-sized PPO slices pass `--resume` on the exact same source,
run, cap, optimizer target and architecture. Record actual scheduler time
across all slices within the6h/stream requested budget. Per-attempt tests,
GPU probes, logs, behavior checkpoints and policy/optimizer/sampler/motor
RNG/task cursor recovery remain available. Interrupted actions stay charged.

Architecture128/2layers/4heads/context300, LR1e-4/two epochs, K4 and one
complete trial/update are explicit pilot choices; this is not VLRLA's larger
rollout batch. Outcome-independent round-robin covers tasks actually reached;
states0–39 are sampled with replacement. Report actual state IDs/hashes and
do not claim all40 states/task are guaranteed. Frozen validation uses40–49.

Offline AD/DPT training can reuse the saved SAC corpus without more simulator
actions. Its substantial training budget and LIBERO90/Plus/Pro support/query
evaluation will be metered separately after usable source coverage is known.
