# Frozen source-PPO target support/query execution

The original `FrozenRL2Controller` supplies act/observe memory semantics, but the
generic target evaluator previously assigned it a local Torch Generator that
its RL2 sampler explicitly rejects. The target adapter now seeds the qualified
global Torch sampler per actual episode and snapshots/restores recurrent cache,
previous executed noise/reward/boundary, Python/NumPy/Torch/CUDA RNG and reset
flags. It never creates an optimizer or changes motor/controller weights.

The target runner uses the existing fixed-case support/query evaluator and
qualified4104D prefix features. Its PPO controls are:

- `ppo_full`: separately source-trained full-context policy.
- `ppo_episode_reset`: the same full-policy weights with context reset each
  episode, preserving the memory-lesion control.
- `ppo_trained_episode`: the separately source-trained episode-reset policy.

It defaults to these three controls plus native, with0/1/2/4/8 actual support
episodes, each followed by a separate reset/query episode. Each support budget
starts fresh controller context. Query RNG is paired by physical case and
role, independent of support count. Support and query action costs and success
remain separate. Interrupted RPCs retain conservative charges; exact resumes
restore the last complete episode context/cursor and keep the original cap.

Final source recoveries use the frozen-validation admission/export path:
complete K4 boundary, actual all40 task coverage, expected full/episode mode,
172k source contract, qualified motor/features, matching architecture,
optimizer advancement and the original6h time ledger. Only immutable frozen
copies enter deployment. Source interactions are counted once per trained
model; the full/reset lesion shares the full model's source cost.

Optional existing AD/DPT controls use their original loader and matched feature
contract in this same motor process. Pass `--students-index`,
`--students-index-sha256`, `--assembly`, and the existing `--student-seed`.
Their default grid and implementations remain unchanged. Ordinary target-online
DSRL remains the separate published-recipe implementation; PPO and frozen SAC
must not be relabeled as that baseline.

One hash-bound case plan declares the physical case and its already qualified
simulator runtime. For an original LIBERO90 case:

```json
{
  "format": "source40_ppo_target_case_v1",
  "case": {
    "benchmark": "libero90",
    "case_id": "libero90:00/state-0/seed-123",
    "suite": "libero_90",
    "task_id": 0,
    "initial_state": 0,
    "scene_seed": 123
  },
  "simulator": {
    "python": "QUALIFIED_SIMULATOR_PYTHON",
    "pythonpath": "EXACT_EVALUATION_WORKTREE/src:PINNED_LIBERO:QUALIFIED_CLIENT_DEPENDENCIES",
    "libero_root": "PINNED_ORIGINAL_LIBERO_ROOT"
  }
}
```

Every LIBERO90 task0–89 is admissible without semantic filtering. Plus plans
use `benchmark=plus`, `suite=libero_object`, its runtime case index as task_id,
and `simulator.plus_root`; Pro plans use their declared family/task1,
`simulator.pro_root`, `data_root` and `pro_case`. The current shifted workers
only implement their existing declared development slices. This route does not
claim full Plus/Pro enumeration. Pro now returns the actual nominal-state hash
required by the shared evaluator; physical behavior is unchanged.

After models are ready and the root has recorded a concrete target-only action
escrow and compute budget, run inside the identified allocation:

```bash
bash scripts/diagnostics/vla_source40_ppo_target.sh EXACT_COMMIT FRESH_RUN \
  FINAL_MODELS_MANIFEST CASE_PLAN_JSON ACTION_CAP TOTAL_GPU_SECONDS SLICE_SECONDS CANONICAL_LEDGER
```

The launcher reads that existing reservation and does not create one or submit
an allocation. Its cumulative time ledger includes setup/shutdown/retries and
retains the same model/case/action/time inputs on `--resume`. Optional explicit
`--support-budgets`/`--methods` select a declared bounded slice; outputs record
whether the complete protocol shot grid ran.

A full default one-case grid is80 episodes:32,800 worst-case actions on
LIBERO90 at400+10,23,200 on the existing Plus280+10 worker, and22,800 on the
existing Pro280+5 worker. These figures are planning costs, not new approved
reservations. Full-suite/future target compute must fit the root's remaining
recorded escrow; this code does not expand the existing1M benchmark cap.
