# Fixed500 supervised-controller target first wave

This planner prepares metadata for an explicitly declared robot evaluation.
It launches no jobs and reserves no actions or GPU time. The runner must bind
the actual source40 matrix/assembly/checkpoint hashes and its independently
authorized target budget before executing it. The official frozen OpenPI40
motor receipt remains fixed; no RLinf130 weights, source records or features
are admitted as substitutes.

```sh
python -m fluffy_bassoon.meta_rl.vla_source40_target_firstwave_plan \
  --worktree <explicit-pinned-run-worktree> --source-commit <exact-SHA> \
  --output <fresh-target-plan.json>
```

PhaseA evaluates native, AD full/static and DPT full/static at K0 on all
LIBERO90 tasks0–89, nominal state0, seed123. It also uses the two existing
official Plus variants667/290 at state IDs0/1 (four cases), and the existing
Pro swap/appearance variants at state0 with seeds123/124 (four seeded cases).
The latter are repeats across **two variants**, not four distinct tasks or
four nominal state vectors. All Plus/Pro cases retain development scope.
There is no semantic-goal filter or outcome-based case admission.

PhaseB is fixed before results: LIBERO90 IDs0,10,…80 and all eight shifted
cases. AD full/episode-reset and DPT full/episode-reset each receive one actual
support episode, followed by a separate query episode on the same case.
Weights remain fixed. An episode-reset K0 result references the same full
controller's PhaseA result; it does not reuse the separately trained static
controller and incurs no additional episode.

All4 learned candidates are their fixed500-update exports. This first wave
does not include ordinary DSRL or PPO; those remain necessary for the full
requested method comparison. The source dataset's actual exposure, prior
target-development use, and backbone's40-task fine-tuning must be disclosed;
planning full90 evaluation does not establish semantic novelty or useful
adaptation.

| Allocation | Physical episodes | Nominal maximum actions including per-episode settling |
| --- | ---: | ---: |
|PhaseA |490 |196,000 |
|PhaseB |136 |47,920 |
|Total |626 |243,920 |

LIBERO90 horizon400+settle10; Plus280+10; Pro280+5. These are declared
evaluation horizons, not a claim of matching every full leaderboard protocol.
Qualification uses the first planned reset and adds no physical reset. Six workers
own `cases[i::6]`, so a case's A+B groups share one frozen motor process.
Each worker receives its nominal action sum+1,000 headroom and at most3,600
GPU-seconds. Worker action caps total249,920, within a250,000-action global
ceiling; six worker GPU ceilings total21,600seconds/six GPU-hours. Root's
budget ledger decides whether and when that allocation is authorized.

The hash-bound plan includes `contract`,98 `cases`,558 flattened `groups`,
six `workers`, exact counts and `firstwave_sha256`. A case uses the existing
`source40_ppo_target_case_v1` wrapper with its `case` TargetCase dictionary,
simulator module/Python/import paths/arguments, expected metadata and complete
method groups. Dynamic `--output` and `--port` arguments belong to the runner.
Import paths begin with the explicit pinned worktree's `src`, never an agent's
local editing path. The Plus/Pro declaration files are read as restricted AST
literals, without importing a simulator or model.

The executable is `vla_source40_target_firstwave` and the allocated launcher is
`scripts/diagnostics/vla_source40_target_firstwave.sh`. It reads a separate
`libero_target_wave_budget_v1` ledger with `wave_sha256`,
`maximum_total_actions:250000`, `maximum_gpu_seconds:21600`,
`source_campaign_immutable:true`, a human-readable `scope_amendment`, and
exactly six `reservations`. Each row binds `run_id`, `worker`, `source_commit`,
`wave_sha256`, `maximum_total_actions` from that worker's plan, `worker_seconds:3600`,
`purpose:target_evaluation`, `source_pretraining_data:false`, and
`fresh_ppo_training_data:false`. Neither the runner nor launcher creates a
reservation. The old source campaign and action ledger remain immutable.

```sh
bash scripts/diagnostics/vla_source40_target_firstwave.sh \
  EXACT_SOURCE WORKER_RUN_ID WORKER_INDEX \
  /absolute/firstwave-plan.json /absolute/final500/index.json \
  /absolute/assembly.json /absolute/target-ledger.json 3540
```

Run from each prepared exact-source worker worktree. Root schedules six
separate allocations. The first invocation has a3,540-second slice plus a
60-second scheduler allowance inside the original3,600-second ceiling.
`worker-time.json` starts before setup. Actual probe/tests, model loading,
execution, teardown, failed attempts and scheduler allowance are charged.
A retry uses the same worker output with `--resume` and a **smaller** slice
whose reservation fits the remaining original hour. Unclosed attempts retain
their full reserved charge. No retry receives a new hour or another action cap.

Each group archives actual initial RGB/wrist/proprioception, decoded actions,
noise, rewards, episode boundaries, controller context, full sampler/global
and motor RNG, before-RPC intents and a complete-episode cursor. Unknown RPC
costs stay conservatively charged. Nominal initialization and query seed
checks remain mandatory. Raw initial-observation equality is reported within
and across groups; camera differences prevent a strict causal memory claim
while the nominal-case screen remains reportable. A fresh-process continuation
is also marked non-primary for strict shared-runtime claims. Frozen motor
parameter hashes are compared before and after the entire worker process.

Frozen PPO deployment is a separate path. `vla_source40_ppo_lineage` admits
only explicitly supplied successful terminal R3 roots, binds both R1→R2→R3
migration receipts, complete recovery SHA, raw-evidence catalogs and all
cumulative time/physical charges, and retains the original R1 reservation.
It never falls back to a failed40-update partial checkpoint. AD-only deployment
requires no PPO manifest and therefore does not wait for PPO completion.
