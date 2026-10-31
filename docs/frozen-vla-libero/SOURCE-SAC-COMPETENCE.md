# Frozen per-task source SAC competence

This bounded diagnostic asks whether the actual source teachers outperform
the frozen motor on disjoint source-local states. It makes no target transfer,
expert-teacher or jointly trained40-task SAC claim.

Use the existing fixed source goals: `libero_spatial:00/01`,
`libero_object:00/01`, `libero_goal:00/01`, `libero_10:00/01`; states40/41.
Each of the16 cases receives three separate fresh query episodes, no support:

| Method | Noise fed to the frozen OpenPI motor |
| --- | --- |
| native | Independent unbounded Gaussian `[10,32]` |
| mean | Corresponding task's `eval_actions`;32D clipped[-1,1], repeated10slots |
| sample | Corresponding task's stochastic `sample_actions`; same clipping/repetition |

The latter two use the published pixel64/proprio8 conversion and the same
per-task actor. Mean is the deterministic teacher used for DPT relabeling;
sample matches the authors' post-update evaluation rule. Native Gaussian is
distinct from the tied32D Gaussian used during source warmup. Pair case,
motor/query seed and physical reset across methods. Preserve the explicit
source40 reset profile: exact nominal/full simulator state/proprio, raw
uint8 camera changes at most1LSB, no image replacement. Retain comparison
statistics and all raw observations.

The canonical source ledger selects only `/train` entries for these eight
goals. Bind each completed index, contract and final committed publication
checkpoint by SHA256. Restore the entire original SAC learner: actor,
critics/targets, temperature, optimizers, replay and RNG/sampler state.
The immutable checkpoint holds those full states; diagnostic recovery binds
it by hash and saves query cursor, actor RNG, motor RNG and global RNG.
Never append diagnostic observations to replay or call an update. Compare
all actor/critic/optimizer state bytes, update counts and replay hashes
before/after every episode. Random sampling keys may advance and are saved.

Keep the legacy source horizon400 and10 settling actions.48 episodes cost at
most19,680 actions. Qualification uses the first planned reset of each task,
so it adds zero physical actions. A20,000-action ceiling includes any retries
and unresolved RPC ceilings. Raw inference inputs/noises/proposals and
physical outcomes are journaled, including unfinished episodes. Resume only
at an actual complete-query checkpoint; orphaned physical actions remain
charged. No future suite-horizon recipe is introduced by this diagnostic.

This spends only a reconciled remainder of the original source-validation
56k-action/14,400-GPU-second envelope. It reserves nothing itself. After the
PPO validation is terminal, successful and complete, the budget owner writes
an immutable `source40_validation_remainder_alias_v1` JSON containing:

```text
run_id, source_commit
maximum_total_actions:20000
maximum_worker_seconds:at most3600 and at most14400 minus parent charge
parent_output_root
parent_terminal_index_sha256, parent_worker_time_sha256
parent_total_action_charge, parent_worker_charge_seconds
canonical_ppo_ledger_path
original_action_ceiling:56000, original_worker_seconds:14400
```

Admission verifies those terminal hashes, canonical original56k reservation,
combined action/time ceilings and absence of a duplicate child reservation.
Exact-source child retries retain the same alias and cumulative worker ledger.
The child's actual charge must be reconciled with the parent; no second56k
or4h hold exists.

After review, deploy a fresh worktree and use its exact commit in the alias:

```sh
bash scripts/diagnostics/vla_source40_sac_diagnostic.sh \
  EXACT_SHA FRESH_RUN_ID /absolute/remainder-alias.json ALIAS_SHA 3540
```

The proposed3540s slice needs a3600s alias including60s scheduler allowance.
A shorter remaining hold uses a smaller slice and may end with an honest
partial48-query matrix. Setup/model loading consume that same clock; the
driver stops before its outer timeout and reserves180s before a new query.

Use one qualified GPU,16 CPU and the already qualified96GiB runtime allowance.
At roughly15 actions/s, full-length episodes alone take21–22min. Allow
about30–60min GPU time including eight task switches, full learner restores,
JIT compilation and artifact writes; ceiling1GPUh/16CPUh. This estimate is
conditional on the actual unused parent hold. Budget at most2GiB of new raw
diagnostic artifacts; source checkpoints are referenced, not copied.

The launcher checks the allocated GPU and runs focused Linux semantics tests.
It performs no allocation, canonical reservation or model computation on the
development host. Report per-method successes/16, primitive costs and paired
case outcomes; the small source-local sample cannot establish generalization.
