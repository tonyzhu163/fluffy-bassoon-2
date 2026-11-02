# Stage 1 handoff review

Reviewed origin/main `64179b9e`; merged latest remote into main, preserving both interface APIs. Merge source `79d0a93e` passes all 47 affected interface, batched motor, environment and reward tests on allocated Baige Linux CPU, with CUDA masked. [Test output](CPU-TESTS-79d0a93e.txt). The initial merged fixture omitted its configurable layer count; that fixture was restored before the passing run. No numerical computation ran on macOS; no simulator or GPU action was launched by this review.

## Handoff scope versus the latest human-approved joint test

The [handoff](../robocasa-stage1-interface-ppo-20261102-r1/HANDOFF.md) describes the earlier OpenDrawer-only two-zero-init-learner pilot: 30 x 64 TRAIN episodes, a combined 20 GPU-hour cap, calibration before update 1, DEV every five updates and final paired 50-case ID test. It explicitly says a driver still needs writing; it is not a submitted run.

`conditioning_all` shares one 32-dimensional code across all cache layers and perturbs values only. Its name does not mean both keys and values are changed. The separate `kv_all_layers` interface implements the latest requirement: independent rank-32 blocks for K and V at each of 18 layers, 1152 learned coefficients. Both base VLM and action expert remain frozen. This new map still needs real GPU zero-parity, per-layer influence and scale calibration; CPU qualification does not establish those.

The latest joint scope is recorded in the [5M contract](../robocasa-joint-steering-5m-20261102-r1/PLAN.json): two joint five-task arms, fixed DEV every five rounds and a separate ten-task final paired evaluation. Do not launch the older single-task pilot in parallel as an extra experiment. Fresh steering/value policies start at native identity; the closed full-expert Flow checkpoint is not their initializer.

## Accounting correction before launch

The original 104 x 80 x 600 schedule left only 8000 TRAIN controls for resets. At the observed three reset controls per episode it would exceed the human-approved 5M ceiling. The schedule is now 103 balanced rounds (4,944,000 policy controls), leaving 56,000 for resets and qualification. All actual, failed or unresolved physical calls remain charged. Planned evaluation policy controls are 1,173,000; reserve 27,000 more for evaluation resets/qualification, for a separate 1,200,000 ceiling. No TRAIN ceiling is raised.

The existing 20 GPU-hour combined allocation cap is retained. The data-volume approval is not recorded as an unrequested 128 GPU-hour allowance. A counted update and measured throughput must determine whether the 5M-per-arm target fits; if not, a complete checkpoint and concrete resource decision are required.

No joint PPO job is queued or running. Master-driver integration, frozen initializer/recovery checks and actual GPU qualification remain.
