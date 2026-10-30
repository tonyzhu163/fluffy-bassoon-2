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
One qualification reset per case adds up to960 settling actions. Six workers
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
