# RoboCasa 1.0.0 / 1.0.1 and the official π0.5 evaluation

Source inspection and static verification, 2026-11-09. No reproduced success
rate is claimed in this document.

The narrow 1.0.1 release changes the evaluation time allowance, rather than
the controller, physics, scene distribution or success predicates. More
importantly for our reproduction, the original OpenPI evaluator already
multiplied the old registry horizons by 1.5. The coordinated releases move
that multiplier from the evaluator into RoboCasa's registry. Switching only
the package version does not, by itself, change the effective allowance.

## Exact release boundary

The `v1.0` tag is `8f3c96ec8d1bfcd8126cad2bca887da98d30e997`. The version
remained 1.0.0 over subsequent commits. Immediately before the horizon
change, the source is `00dd765dc868be7ca78f14adf706ce713f9f868c`.

[RoboCasa commit 29f7ce8](https://github.com/robocasa/robocasa/commit/29f7ce8814c1547f5af762a0997fbd4b64848dd7)
on May 12 changes exactly four files:

| File | Change |
| --- | --- |
| `robocasa/utils/dataset_registry.py` | Every one of the 317 configured horizons becomes exactly 1.5× its previous value |
| `robocasa/__init__.py` | Version 1.0.0 → 1.0.1 |
| `setup.py` | Package version 1.0.0 → 1.0.1 |
| `README.md` | Release notice |

The static [Git/AST audit](research/HORIZON-AUDIT.json) checks all 317 ratios,
unchanged task-set membership, and all 50 leaderboard tasks (18/16/16).
Examples: OpenDrawer 500→750, CloseToasterOvenDoor 300→450,
CoffeeSetupMug 400→600, TurnOffStove 500→750.
No simulator, fixture, controller, action mapping, reset, renderer or success
implementation is changed by that release commit.

[OpenPI commit b175bb9](https://github.com/robocasa-benchmark/openpi/commit/b175bb90771ccbdeb25080e71b604ba68d133b81),
merged as `5a6beda9` the same day, removes the evaluator's `* 1.5`.
Consequently these are the matching pairs:

| RoboCasa registry | OpenPI evaluator | Effective OpenDrawer horizon |
| --- | --- | ---: |
| 1.0.0, 500 | Submission-linked `ca4c6d7`, ×1.5 | 750 |
| 1.0.1, 750 | Updated `5a6beda9`, ×1 | 750 |
| 1.0.1, 750 | Old `ca4c6d7` without adjustment | 1125 — double scaling |

The [official benchmark documentation](https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html)
also describes the horizon increase and the two scene/object splits.

## Later commits still called 1.0.1

Our previous simulator was pinned to `3443cdf3`, not the narrow release
commit. Comparing it with `29f7ce8` exposes later changes. The saved
[runtime diff](research/LATER-RUNTIME-CHANGES.diff) includes optional depth
camera support, additional ArrangeDrinkware layout exclusions, and removal
of a duplicate sink method. The removed method is textually the same as the
earlier definition, so that removal does not change its implementation.
The camera-depth option defaults to false. This is not a complete claim that
every later commit is behaviorally irrelevant; it is why the reproduction
uses a fresh source pinned to the narrow release instead of changing shared
production sources.

## Reference evaluation and discrepancies in our earlier measurements

The [official π0.5 submission](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions/standard/pi05_2026-04-02.json)
names the same Human300 step-75000 checkpoint we downloaded, version 1.0.0,
and OpenPI `ca4c6d7`. Its references are 39.6% Atomic-Seen, 7.1%
Composite-Seen and 1.2% Composite-Unseen, giving 16.912% overall by task count.

| Setting | Released reference evaluator | Earlier local task evaluations |
| --- | --- | --- |
| Native prediction / executed controls | Predict 50, execute 5, then replan | Typically predict and execute 50 |
| Policy implementation | Original OpenPI JAX available | Converted RLinf PyTorch |
| Scene/object split | `pretrain` | Task-focused `target` panels |
| Task coverage | 50 target task identities, split 18/16/16 | Selected tasks and headroom screens |
| Trials | 50 per task, seed 7 | Various 8/20/50/64/512 case banks |
| Success | First success, terminate | First and end success over complete episodes |
| Horizon | Old registry ×1.5, or new registry ×1 | Native task horizons or shorter fixed 400/600 panels |

`atomic_seen` describes task identity; `pretrain` describes scene/object
distribution. These must not be conflated. The reference uses the 50 target
task identities in pretraining kitchens/objects.

The released code uses 224-pixel resize-with-padding, left/wrist/right RGB,
the relative-eef/base/gripper state order, 10 ODE denoising steps, and native
binary gripper/base-mode conversion. Its `Args.task_set` / `args.task_soup`
CLI typo is bypassed by explicitly selecting the saved task list. No action
or observation transformation is changed to fix that typo.

There is a remaining public-provenance limitation: the submission date is
April 2, while its linked code commit is dated May 4 and already includes the
1.5 multiplier. The public submission gives aggregates, not its actual
per-case logs or environment/dependency commit. Record success before both
the old and extended horizons; do not assert that the metadata alone proves
which allowance generated those percentages.

## Prepared execution

On Baige, static source parsing verified all horizons. The native OpenPI
config imports in the existing Linux runtime and resolves to π0.5,
action dimension 32, prediction horizon 50, token length 200, BF16 and
discrete state input. Original checkpoint and shared assets remain in place.
Fresh source checkouts use the protected assets through a local symlink;
the tracked asset metadata was preserved separately.

The next measurement is a one-GPU, one-hour maximum qualification with
native original JAX loading and three real pretrain-split episodes. Its
result determines batch/worker capacity and the 18-task sweep allocation.
See [the full contract](PLAN.md). No version downgrade or benchmark parity
is claimed before the corresponding measurements.
