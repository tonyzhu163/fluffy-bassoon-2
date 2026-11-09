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

The original `v1.0` tag predates the immediate 1.0.0 parent by nine commits.
That intervening [tag-to-release diff](research/V100-TAG-TO-V101.diff) also
includes asset/dataset download mirrors changing from Hugging Face to Box,
demo/recording utilities, empty package initializers, and an optional enclosing
wall render wrapper used by demos. No task physics/controller/reset/success
implementation changed in that interval. The four-file statement above is
specifically the final version-bump commit, not every change since the tag.
The fallback source is prepared separately at the actual `v1.0` tag.

The protected shared assets were checked against all 210 tracked release
asset files: 209 match Git blob hashes exactly; only the dataset download URL
index differs. No scene, fixture or geometry metadata differed. See
[asset metadata audit](research/ASSET-METADATA-AUDIT.json). External downloaded
asset archives remain the existing protected cache; the original submission
does not provide archive checksums for a full historical asset comparison.

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
task identities in pretraining kitchens/objects. This is independently confirmed
by the [official multi-task protocol](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html),
which lists the exact 39.6/7.1/1.2 π0.5 reference and explicitly evaluates OpenPI
on `--args.split pretrain`. The separate foundation-model fine-tuning benchmark
uses target scenes; its scene split must not be applied to this checkpoint's
multi-task reference.

A [May 4 maintainer reply](https://github.com/robocasa-benchmark/openpi/issues/1)
also explicitly confirms that the reported experiments kept the default
`replan_steps=5`. This supports H5 independently of the released script's
argument defaults. The same thread records the public π0.5-config release.

The released code uses 224-pixel resize-with-padding, left/wrist/right RGB,
the relative-eef/base/gripper state order, 10 ODE denoising steps, and native
binary gripper/base-mode conversion. Its `Args.task_set` / `args.task_soup`
CLI typo is bypassed by explicitly selecting the saved task list. No action
or observation transformation is changed to fix that typo.

The submission date is April 2, while its linked code commit is dated May 4.
Following the evaluator's file history resolves much of the horizon question:
the initial RoboCasa365 evaluator at
[`43fd9da0`, February 17](https://github.com/robocasa-benchmark/openpi/blob/43fd9da0bdad94a2d409f5601b4b878f5ca59070/examples/robocasa/main.py)
already used `int(task_horizon * 1.5)`, explicitly to give the policy more
time, along with replan5/pretrain/50 trials. The April 16 π0.5-support
commit retains that allowance and adds the right-camera input; see the
[saved file-history diff](research/EVALUATOR-PI05-HISTORY.diff).
Thus the extended allowance is the primary released-recipe comparison, and
the original-registry cutoff is a secondary diagnostic. The public submission
still provides aggregates rather than its original per-case logs and full
dependency pins; the source history does not prove exact episode replay.

## Prepared execution

On Baige, static source parsing verified all horizons. The native OpenPI
config imports in the existing Linux runtime and resolves to π0.5,
action dimension 32, prediction horizon 50, token length 200, BF16 and
discrete state input. Original checkpoint and shared assets remain in place.
Fresh source checkouts use the protected assets through a local symlink;
the tracked asset metadata was preserved separately.

Native qualification completed three episodes, followed by four concurrent
simulator clients against the official batch-one WebSocket policy server.
The latter completed 2,832 controls / 567 native predictions in 75.52 seconds
of collection (37.50 controls/second), with a matching saved sampler count,
first-input hashes and zero training updates. These seven diagnostic episodes
qualify execution and throughput, not benchmark performance. See
[the qualification audit](GPU04-AUDIT.json) and [campaign contract](CAMPAIGN.md).
